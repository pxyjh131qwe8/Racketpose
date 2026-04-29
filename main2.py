# # main_pose.py
# import copy
# import csv
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import warnings
# from argparse import ArgumentParser
# import datetime
# import math

# import torch
# import torch.nn.functional as F
# import tqdm
# import yaml
# from torch.utils.tensorboard import SummaryWriter

# from nets import nn_center_and_normal as nn
# from utils import util
# from datasets.racketpose2 import build_loader  # pose 版 build_loader：返回 (loader, sampler, dataset)

# warnings.filterwarnings("ignore")


# # -------------------- ckpt utils --------------------
# def _torch_load_any(path: str, map_location="cpu"):
#     try:
#         return torch.load(path, map_location=map_location, weights_only=False)
#     except TypeError:
#         return torch.load(path, map_location=map_location)


# def _extract_state_dict(ckpt_obj):
#     if isinstance(ckpt_obj, dict):
#         for k in ["model", "state_dict", "ema", "model_ema", "ema_state_dict"]:
#             if k in ckpt_obj:
#                 v = ckpt_obj[k]
#                 if hasattr(v, "state_dict"):
#                     return v.state_dict()
#                 if isinstance(v, dict):
#                     return v
#         if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
#             return ckpt_obj
#     if hasattr(ckpt_obj, "state_dict"):
#         return ckpt_obj.state_dict()
#     raise ValueError(f"Unrecognized checkpoint format: {type(ckpt_obj)}")


# def _strip_prefix(sd, prefixes=("module.", "model.")):
#     out = {}
#     for k, v in sd.items():
#         nk = k
#         for p in prefixes:
#             if nk.startswith(p):
#                 nk = nk[len(p):]
#         out[nk] = v
#     return out


# def load_weights_safely(model: torch.nn.Module, weight_path: str, strict: bool = False):
#     ckpt = _torch_load_any(weight_path, map_location="cpu")
#     sd = _strip_prefix(_extract_state_dict(ckpt))

#     msd = model.state_dict()
#     filtered, skipped = {}, []
#     for k, v in sd.items():
#         if k in msd and msd[k].shape == v.shape:
#             filtered[k] = v
#         else:
#             skipped.append(k)

#     missing, unexpected = model.load_state_dict(filtered, strict=strict)

#     print(f"[LOAD] weights = {weight_path}")
#     print(f"[LOAD] loaded={len(filtered)} skipped(shape)={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}")
#     if skipped:
#         print(f"[LOAD] example skipped: {skipped[:8]}{' ...' if len(skipped) > 8 else ''}")
#     return ckpt


# # -------------------- Pose MIL Loss --------------------
# class PoseMILLoss(torch.nn.Module):
#     """
#     outputs (train): list of 3 tensors, each [B, 6+nc, H, W] (cls is logits, NOT sigmoid)
#     targets: dict with keys
#       - label: [B] long
#       - center_norm: [B,3] float  (standardized)
#       - normal: [B,3] float (unit)
#     """
#     def __init__(
#         self,
#         nc: int,
#         tau: float = 1.0,
#         w_cls: float = 1.0,
#         w_center: float = 1.0,
#         w_normal: float = 1.0,
#         smoothl1_beta: float = 1.0,
#     ):
#         super().__init__()
#         self.nc = int(nc)
#         self.tau = float(tau)
#         self.w_cls = float(w_cls)
#         self.w_center = float(w_center)
#         self.w_normal = float(w_normal)
#         self.beta = float(smoothl1_beta)

#     def _flatten_train_outputs(self, outputs):
#         assert isinstance(outputs, (list, tuple)), "Train outputs must be list/tuple of feature maps."
#         B = outputs[0].shape[0]
#         C = outputs[0].shape[1]
#         x = torch.cat([o.view(B, C, -1) for o in outputs], dim=2)  # [B, 6+nc, N]
#         return x

#     def forward(self, outputs, targets):
#         x = self._flatten_train_outputs(outputs)  # [B,6+nc,N]
#         B = x.shape[0]

#         reg = x[:, 0:6, :]                 # [B,6,N]
#         cls_logits = x[:, 6:6+self.nc, :]  # [B,nc,N] logits

#         y = targets["label"].long()                 # [B]
#         center_gt = targets["center_norm"].float()  # [B,3] standardized
#         normal_gt = targets["normal"].float()       # [B,3] unit

#         # --- cls loss (image-level MIL pooling) ---
#         img_logits = torch.logsumexp(cls_logits, dim=2)  # [B,nc]
#         loss_cls = F.cross_entropy(img_logits, y)

#         # --- attention weights from GT class ---
#         idx = torch.arange(B, device=x.device)
#         g = cls_logits[idx, y, :]                 # [B,N]
#         w = torch.softmax(g / self.tau, dim=1)    # [B,N]

#         # --- center ---
#         center_loc = reg[:, 0:3, :]                              # [B,3,N] standardized
#         center_pred = (center_loc * w.unsqueeze(1)).sum(dim=2)   # [B,3]
#         loss_center = F.smooth_l1_loss(center_pred, center_gt, beta=self.beta)

#         # --- normal ---
#         normal_loc = reg[:, 3:6, :]                   # [B,3,N] raw
#         normal_loc = torch.tanh(normal_loc)
#         normal_loc = F.normalize(normal_loc, dim=1, eps=1e-6)
#         normal_pred = (normal_loc * w.unsqueeze(1)).sum(dim=2)   # [B,3]
#         normal_pred = F.normalize(normal_pred, dim=1, eps=1e-6)

#         cos = (normal_pred * normal_gt).sum(dim=1).clamp(-1.0, 1.0)
#         loss_normal = (1.0 - cos).mean()

#         loss = self.w_cls * loss_cls + self.w_center * loss_center + self.w_normal * loss_normal
#         return loss, {
#             "loss_cls": loss_cls.detach(),
#             "loss_center": loss_center.detach(),
#             "loss_normal": loss_normal.detach(),
#         }


# # -------------------- evaluation --------------------
# @torch.no_grad()
# def eval_pose(
#     model,
#     loader,
#     nc: int,
#     tau: float = 1.0,
#     device="cuda",
#     # 阈值集合：center(m), angle(deg)
#     pose_thresholds=((0.05, 5.0), (0.05, 10.0)),
#     # 单项阈值
#     center_thresholds=(0.05, 0.10),
#     angle_thresholds=(5.0, 10.0),
# ):
#     """
#     eval mode outputs: [B, 6+nc, N]
#       center: meters (若 head.denorm_inference=True)
#       normal: unit-ish
#       cls: sigmoid prob
#     targets:
#       label [B], center_m [B,3], normal [B,3]
#     """
#     model.eval()

#     total = 0
#     correct = 0
#     correct_count = 0  # cls 正确样本数

#     # --- sums (all) ---
#     sum_center_l2 = 0.0
#     sum_center_mae = torch.zeros(3, dtype=torch.float64, device=device)
#     sum_ang = 0.0

#     # --- sums (cls-correct only) ---
#     sum_center_l2_c = 0.0
#     sum_center_mae_c = torch.zeros(3, dtype=torch.float64, device=device)
#     sum_ang_c = 0.0

#     # --- threshold counts ---
#     ok_center = {thr: 0 for thr in center_thresholds}
#     ok_center_with_cls = {thr: 0 for thr in center_thresholds}

#     ok_angle = {thr: 0 for thr in angle_thresholds}
#     ok_angle_with_cls = {thr: 0 for thr in angle_thresholds}

#     ok_pose = {(d, a): 0 for (d, a) in pose_thresholds}
#     ok_pose_with_cls = {(d, a): 0 for (d, a) in pose_thresholds}

#     eps = 1e-6

#     for imgs, targets in tqdm.tqdm(loader, desc="Eval", leave=False):
#         imgs = imgs.to(device, non_blocking=True).float()
#         y = targets["label"].to(device, non_blocking=True).long()             # [B]
#         center_gt = targets["center_m"].to(device, non_blocking=True).float() # [B,3] meters
#         normal_gt = targets["normal"].to(device, non_blocking=True).float()   # [B,3] unit

#         with torch.amp.autocast("cuda", dtype=torch.float16):
#             out = model(imgs)  # [B, 6+nc, N]

#         B = out.shape[0]
#         N = out.shape[2]

#         center_loc = out[:, 0:3, :]                          # [B,3,N] meters
#         normal_loc = out[:, 3:6, :]                          # [B,3,N]
#         cls_prob = out[:, 6:6+nc, :].clamp(eps, 1.0 - eps)   # [B,nc,N]

#         # -------- image-level class prob (noisy-or) --------
#         log_not = torch.log1p(-cls_prob)                      # log(1-p)
#         img_prob = 1.0 - torch.exp(log_not.sum(dim=2))        # [B,nc]
#         pred_y = img_prob.argmax(dim=1)                       # [B]

#         cls_ok = (pred_y == y)                                # [B]
#         correct += cls_ok.sum().item()
#         correct_count += cls_ok.sum().item()
#         total += B

#         # -------- attention weights from GT class prob --------
#         idx = torch.arange(B, device=device)
#         g = cls_prob[idx, y, :]                               # [B,N]
#         w = torch.softmax(torch.log(g + eps) / tau, dim=1)    # [B,N]

#         center_pred = (center_loc * w.unsqueeze(1)).sum(dim=2)  # [B,3]
#         normal_pred = (normal_loc * w.unsqueeze(1)).sum(dim=2)  # [B,3]
#         normal_pred = F.normalize(normal_pred, dim=1, eps=1e-6)

#         # -------- center metrics --------
#         diff = center_pred - center_gt
#         l2 = diff.norm(dim=1)                                 # [B]
#         sum_center_l2 += l2.sum().item()
#         sum_center_mae += diff.abs().sum(dim=0).double()

#         if cls_ok.any():
#             l2_c = l2[cls_ok]
#             diff_c = diff[cls_ok]
#             sum_center_l2_c += l2_c.sum().item()
#             sum_center_mae_c += diff_c.abs().sum(dim=0).double()

#         # -------- normal angle metrics --------
#         dot = (normal_pred * normal_gt).sum(dim=1).clamp(-1.0, 1.0)
#         ang = torch.acos(dot) * (180.0 / math.pi)             # [B]
#         sum_ang += ang.sum().item()
#         if cls_ok.any():
#             sum_ang_c += ang[cls_ok].sum().item()

#         # -------- single-threshold success rates --------
#         for thr in center_thresholds:
#             ok = (l2 < thr)
#             ok_center[thr] += ok.sum().item()
#             ok_center_with_cls[thr] += (ok & cls_ok).sum().item()

#         for thr in angle_thresholds:
#             ok = (ang < thr)
#             ok_angle[thr] += ok.sum().item()
#             ok_angle_with_cls[thr] += (ok & cls_ok).sum().item()

#         # -------- composite pose success (center+normal) --------
#         for (d_thr, a_thr) in pose_thresholds:
#             ok = (l2 < d_thr) & (ang < a_thr)
#             ok_pose[(d_thr, a_thr)] += ok.sum().item()
#             ok_pose_with_cls[(d_thr, a_thr)] += (ok & cls_ok).sum().item()

#     if total == 0:
#         return {}

#     # ---- means (all) ----
#     acc = correct / total
#     center_l2_mean = sum_center_l2 / total
#     center_mae_mean = (sum_center_mae / total).detach().cpu().tolist()
#     ang_mean = sum_ang / total

#     # ---- means (cls-correct only) ----
#     if correct_count > 0:
#         center_l2_mean_c = sum_center_l2_c / correct_count
#         center_mae_mean_c = (sum_center_mae_c / correct_count).detach().cpu().tolist()
#         ang_mean_c = sum_ang_c / correct_count
#     else:
#         center_l2_mean_c = float("nan")
#         center_mae_mean_c = [float("nan")] * 3
#         ang_mean_c = float("nan")

#     metrics = {
#         # cls
#         "acc": float(acc),
#         "acc_count": int(correct),
#         "total": int(total),

#         # errors (all)
#         "center_l2_m": float(center_l2_mean),
#         "center_mae_x_m": float(center_mae_mean[0]),
#         "center_mae_y_m": float(center_mae_mean[1]),
#         "center_mae_z_m": float(center_mae_mean[2]),
#         "normal_deg": float(ang_mean),

#         # errors (cls-correct only)
#         "center_l2_m_cls": float(center_l2_mean_c),
#         "center_mae_x_m_cls": float(center_mae_mean_c[0]),
#         "center_mae_y_m_cls": float(center_mae_mean_c[1]),
#         "center_mae_z_m_cls": float(center_mae_mean_c[2]),
#         "normal_deg_cls": float(ang_mean_c),
#     }

#     # threshold metrics (all + with-cls prerequisite as overall fraction)
#     for thr in center_thresholds:
#         metrics[f"center@{int(thr*100)}cm"] = float(ok_center[thr] / total)
#         metrics[f"center+cls@{int(thr*100)}cm"] = float(ok_center_with_cls[thr] / total)

#     for thr in angle_thresholds:
#         metrics[f"normal@{int(thr)}deg"] = float(ok_angle[thr] / total)
#         metrics[f"normal+cls@{int(thr)}deg"] = float(ok_angle_with_cls[thr] / total)

#     for (d_thr, a_thr) in pose_thresholds:
#         metrics[f"pose@{int(d_thr*100)}cm_{int(a_thr)}deg"] = float(ok_pose[(d_thr, a_thr)] / total)
#         metrics[f"pose+cls@{int(d_thr*100)}cm_{int(a_thr)}deg"] = float(ok_pose_with_cls[(d_thr, a_thr)] / total)

#     return metrics


# # -------------------- train / test --------------------
# def train(args, params):
#     device = "cuda"
#     nc = len(params["names"])

#     model = nn.yolo_v11_x(nc).to(device)

#     start_epoch = 0
#     best_score = float("-inf")

#     accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
#     params["weight_decay"] *= args.batch_size * args.world_size * accumulate / 64

#     optimizer = torch.optim.SGD(
#         util.set_params(model, params["weight_decay"]),
#         params["min_lr"],
#         params["momentum"],
#         nesterov=True
#     )
#     amp_scale = torch.amp.GradScaler()

#     # -------- loaders (train computes stats; val/test reuse train stats) --------
#     train_loader, train_sampler, train_set = build_loader("train", args, params, shuffle=True, center_stats=None)
#     train_stats = (train_set.center_mean, train_set.center_std)

#     eval_split = "test" if getattr(args, "eval_split", "test") == "test" else "val"
#     eval_loader, _, _ = build_loader(eval_split, args, params, shuffle=False, center_stats=train_stats)

#     # 把统计量写入 head（推理阶段自动输出 meters）
#     m0 = model.module if hasattr(model, "module") else model
#     if hasattr(m0, "head") and hasattr(m0.head, "set_center_stats"):
#         m0.head.set_center_stats(train_set.center_mean.tolist(), train_set.center_std.tolist(), denorm_inference=True)

#     # -------- resume / pretrained --------
#     if args.resume is not None:
#         ckpt = load_weights_safely(model, args.resume, strict=(not args.no_strict))
#         if isinstance(ckpt, dict):
#             start_epoch = int(ckpt.get("epoch", 0))
#             best_score = float(ckpt.get("best_score", best_score))
#             if "optimizer" in ckpt:
#                 try:
#                     optimizer.load_state_dict(ckpt["optimizer"])
#                     print("[RESUME] optimizer state loaded.")
#                 except Exception as e:
#                     print(f"[RESUME] optimizer load failed: {e}")
#             if "scaler" in ckpt:
#                 try:
#                     amp_scale.load_state_dict(ckpt["scaler"])
#                     print("[RESUME] scaler state loaded.")
#                 except Exception as e:
#                     print(f"[RESUME] scaler load failed: {e}")
#         print(f"[RESUME] start_epoch={start_epoch} best_score={best_score}")

#     elif args.pretrained is not None:
#         _ = load_weights_safely(model, args.pretrained, strict=(not args.no_strict))
#         print("[PRETRAIN] loaded model weights only (no optimizer/scaler).")

#     # -------- ddp --------
#     if args.distributed:
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#         model = torch.nn.parallel.DistributedDataParallel(
#             module=model, device_ids=[args.local_rank], output_device=args.local_rank
#         )

#     # -------- ema / writer --------
#     writer = SummaryWriter(log_dir=args.tb_dir) if args.local_rank == 0 else None
#     ema = util.EMA(model) if args.local_rank == 0 else None

#     num_steps = len(train_loader)
#     scheduler = util.LinearLR(args, params, num_steps)

#     # -------- loss hyper --------
#     tau = float(getattr(args, "tau", 1.0))
#     w_cls = float(getattr(args, "w_cls", params.get("w_cls", 1.0)))
#     w_center = float(getattr(args, "w_center", params.get("w_center", 1.0)))
#     w_normal = float(getattr(args, "w_normal", params.get("w_normal", 1.0)))
#     beta = float(getattr(args, "smoothl1_beta", params.get("smoothl1_beta", 1.0)))

#     criterion = PoseMILLoss(
#         nc=nc, tau=tau,
#         w_cls=w_cls, w_center=w_center, w_normal=w_normal,
#         smoothl1_beta=beta
#     )

#     # -------- thresholds (把原 10cm10deg 替换为 5cm5deg，不新增) --------
#     pose_thresholds = (
#         (args.pose_center_thr, args.pose_angle_thr),         # 5cm 10deg
#         (args.pose_center_thr, args.pose_angle_thr_small),   # 5cm 5deg  (替代原 10cm10deg)
#     )
#     center_thresholds = (args.pose_center_thr, args.pose_center_thr2)          # 5cm / 10cm 单项
#     angle_thresholds = (args.pose_angle_thr_small, args.pose_angle_thr)        # 5deg / 10deg 单项

#     best_key = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg"  # 用 5cm5deg 选 best

#     # -------- logging csv --------
#     step_csv = os.path.join(args.save_dir, "step.csv")
#     with open(step_csv, "w", newline="") as log:
#         if args.local_rank == 0:
#             logger = csv.DictWriter(
#                 log,
#                 fieldnames=[
#                     "epoch",
#                     "loss", "loss_cls", "loss_center", "loss_normal",
#                     "acc",
#                     "center_l2_m", "center_l2_m_cls",
#                     "center_mae_x_m", "center_mae_y_m", "center_mae_z_m",
#                     "center_mae_x_m_cls", "center_mae_y_m_cls", "center_mae_z_m_cls",
#                     "normal_deg", "normal_deg_cls",
#                     # 单项阈值（all + cls）
#                     f"center@{int(args.pose_center_thr*100)}cm", f"center+cls@{int(args.pose_center_thr*100)}cm",
#                     f"center@{int(args.pose_center_thr2*100)}cm", f"center+cls@{int(args.pose_center_thr2*100)}cm",
#                     f"normal@{int(args.pose_angle_thr_small)}deg", f"normal+cls@{int(args.pose_angle_thr_small)}deg",
#                     f"normal@{int(args.pose_angle_thr)}deg", f"normal+cls@{int(args.pose_angle_thr)}deg",
#                     # 组合阈值（all + cls）(只保留两组：5cm10deg 和 5cm5deg)
#                     f"pose@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg",
#                     f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg",
#                     f"pose@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg",
#                     f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg",
#                 ]
#             )
#             logger.writeheader()

#         for epoch in range(start_epoch, args.epochs):
#             model.train()
#             if args.distributed:
#                 train_sampler.set_epoch(epoch)

#             p_bar = enumerate(train_loader)
#             if args.local_rank == 0:
#                 print(("\n" + "%10s" * 6) % ("epoch", "memory", "loss", "cls", "center", "normal"))
#                 p_bar = tqdm.tqdm(p_bar, total=num_steps)

#             optimizer.zero_grad(set_to_none=True)
#             avg_loss = util.AverageMeter()
#             avg_cls = util.AverageMeter()
#             avg_center = util.AverageMeter()
#             avg_normal = util.AverageMeter()

#             for i, (samples, targets) in p_bar:
#                 step = i + num_steps * epoch
#                 scheduler.step(step, optimizer)

#                 samples = samples.to(device, non_blocking=True).float()
#                 targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

#                 with torch.amp.autocast("cuda", dtype=torch.float16):
#                     outputs = model(samples)  # train -> list of maps
#                     loss, ld = criterion(outputs, targets)

#                 avg_loss.update(loss.item(), samples.size(0))
#                 avg_cls.update(float(ld["loss_cls"]), samples.size(0))
#                 avg_center.update(float(ld["loss_center"]), samples.size(0))
#                 avg_normal.update(float(ld["loss_normal"]), samples.size(0))

#                 amp_scale.scale(loss).backward()

#                 if step % accumulate == 0:
#                     amp_scale.step(optimizer)
#                     amp_scale.update()
#                     optimizer.zero_grad(set_to_none=True)
#                     if ema:
#                         ema.update(model)

#                 if writer is not None:
#                     writer.add_scalar("train/loss", avg_loss.avg, step)
#                     writer.add_scalar("train/loss_cls", avg_cls.avg, step)
#                     writer.add_scalar("train/loss_center", avg_center.avg, step)
#                     writer.add_scalar("train/loss_normal", avg_normal.avg, step)
#                     writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

#                 torch.cuda.synchronize()

#                 if args.local_rank == 0:
#                     memory = f"{torch.cuda.memory_reserved() / 1E9:.4g}G"
#                     s = ("%10s" * 2 + "%10.4g" * 4) % (
#                         f"{epoch+1}/{args.epochs}", memory,
#                         avg_loss.avg, avg_cls.avg, avg_center.avg, avg_normal.avg
#                     )
#                     p_bar.set_description(s)

#             # -------- eval & save --------
#             if args.local_rank == 0:
#                 eval_model = ema.ema if ema else model

#                 metrics = eval_pose(
#                     eval_model,     # ✅ 不要 .half()，避免把模型永久变成 fp16
#                     eval_loader,
#                     nc=nc,
#                     tau=tau,
#                     device=device,
#                     pose_thresholds=pose_thresholds,
#                     center_thresholds=center_thresholds,
#                     angle_thresholds=angle_thresholds,
#                 )

#                 score = metrics.get(best_key, float("-inf"))

#                 if writer is not None:
#                     writer.add_scalar(f"{eval_split}/acc", metrics["acc"], epoch)
#                     writer.add_scalar(f"{eval_split}/center_l2_m", metrics["center_l2_m"], epoch)
#                     writer.add_scalar(f"{eval_split}/center_l2_m_cls", metrics["center_l2_m_cls"], epoch)
#                     writer.add_scalar(f"{eval_split}/normal_deg", metrics["normal_deg"], epoch)
#                     writer.add_scalar(f"{eval_split}/normal_deg_cls", metrics["normal_deg_cls"], epoch)
#                     for k, v in metrics.items():
#                         if k.startswith(("center@", "center+cls@", "normal@", "normal+cls@", "pose@", "pose+cls@")):
#                             writer.add_scalar(f"{eval_split}/{k}", v, epoch)
#                     writer.add_scalar(f"{eval_split}/best_score", score, epoch)

#                 print(
#                     f"[{eval_split.upper()}] "
#                     f"acc={metrics['acc']:.3f} | "
#                     f"center_l2={metrics['center_l2_m']:.4f}m (cls:{metrics['center_l2_m_cls']:.4f}m) | "
#                     f"normal={metrics['normal_deg']:.2f}° (cls:{metrics['normal_deg_cls']:.2f}°) | "
#                     f"{best_key}={score:.3f}"
#                 )

#                 # 保存（last 始终包含“当前 best_score”）
#                 new_best = max(best_score, score)
#                 save = {
#                     "epoch": epoch + 1,
#                     "best_score": new_best,
#                     "model": copy.deepcopy((ema.ema if ema else model).module.float()
#                                            if hasattr((ema.ema if ema else model), "module")
#                                            else (ema.ema if ema else model).float()),
#                     "optimizer": optimizer.state_dict(),
#                     "scaler": amp_scale.state_dict(),
#                 }
#                 torch.save(save, os.path.join(args.save_dir, "last.pt"))

#                 if score > best_score:
#                     best_score = score
#                     save["best_score"] = best_score
#                     torch.save(save, os.path.join(args.save_dir, "best.pt"))
#                     print(f"[Best] {best_key}={best_score:.4f} saved best.pt")
#                 else:
#                     best_score = new_best

#                 # 写 step.csv
#                 row = {
#                     "epoch": epoch + 1,
#                     "loss": avg_loss.avg,
#                     "loss_cls": avg_cls.avg,
#                     "loss_center": avg_center.avg,
#                     "loss_normal": avg_normal.avg,
#                     "acc": metrics["acc"],
#                     "center_l2_m": metrics["center_l2_m"],
#                     "center_l2_m_cls": metrics["center_l2_m_cls"],
#                     "center_mae_x_m": metrics["center_mae_x_m"],
#                     "center_mae_y_m": metrics["center_mae_y_m"],
#                     "center_mae_z_m": metrics["center_mae_z_m"],
#                     "center_mae_x_m_cls": metrics["center_mae_x_m_cls"],
#                     "center_mae_y_m_cls": metrics["center_mae_y_m_cls"],
#                     "center_mae_z_m_cls": metrics["center_mae_z_m_cls"],
#                     "normal_deg": metrics["normal_deg"],
#                     "normal_deg_cls": metrics["normal_deg_cls"],
#                 }
#                 for k, v in metrics.items():
#                     if k in logger.fieldnames:
#                         row[k] = v
#                 logger.writerow(row)
#                 log.flush()

#     if writer is not None:
#         writer.close()


# @torch.no_grad()
# def test(args, params, model=None):
#     device = "cuda"
#     nc = len(params["names"])
#     tau = float(getattr(args, "tau", 1.0))

#     pose_thresholds = (
#         (args.pose_center_thr, args.pose_angle_thr),         # 5cm10deg
#         (args.pose_center_thr, args.pose_angle_thr_small),   # 5cm5deg
#     )
#     center_thresholds = (args.pose_center_thr, args.pose_center_thr2)
#     angle_thresholds = (args.pose_angle_thr_small, args.pose_angle_thr)

#     if model is None:
#         ckpt = torch.load(args.weight, map_location="cuda")
#         model = ckpt["model"].float().fuse()

#     # 尝试从 head 中取 stats，传给 dataset（让 center_norm 对齐训练）
#     stats = None
#     m = model.module if hasattr(model, "module") else model
#     if hasattr(m, "head") and hasattr(m.head, "center_mean") and hasattr(m.head, "center_std"):
#         try:
#             cm = m.head.center_mean.detach().cpu()
#             cs = m.head.center_std.detach().cpu()
#             if cm.numel() == 3 and cs.numel() == 3:
#                 stats = (cm, cs)
#         except Exception:
#             stats = None

#     test_loader, _, _ = build_loader("test", args, params, shuffle=False, center_stats=stats)

#     model = model.eval()

#     metrics = eval_pose(
#         model,
#         test_loader,
#         nc=nc,
#         tau=tau,
#         device=device,
#         pose_thresholds=pose_thresholds,
#         center_thresholds=center_thresholds,
#         angle_thresholds=angle_thresholds,
#     )

#     k_pose_5cm5 = f"pose@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg"
#     k_posecls_5cm5 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg"
#     k_pose_5cm10 = f"pose@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"
#     k_posecls_5cm10 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"

#     print(f"[TEST] acc={metrics['acc']:.3f}")
#     print(f"[TEST] center_l2={metrics['center_l2_m']:.4f}m | center_l2_cls={metrics['center_l2_m_cls']:.4f}m")
#     print(f"[TEST] normal={metrics['normal_deg']:.2f}deg | normal_cls={metrics['normal_deg_cls']:.2f}deg")
#     print(f"[TEST] {k_pose_5cm5}={metrics.get(k_pose_5cm5, float('nan')):.3f} | {k_posecls_5cm5}={metrics.get(k_posecls_5cm5, float('nan')):.3f}")
#     print(f"[TEST] {k_pose_5cm10}={metrics.get(k_pose_5cm10, float('nan')):.3f} | {k_posecls_5cm10}={metrics.get(k_posecls_5cm10, float('nan')):.3f}")

#     return metrics


# def main():
#     parser = ArgumentParser()
#     parser.add_argument("--input-size", default=640, type=int)
#     parser.add_argument("--batch-size", default=20, type=int)
#     parser.add_argument("--epochs", default=300, type=int)

#     parser.add_argument("--local-rank", default=0, type=int)
#     parser.add_argument("--local_rank", default=0, type=int)

#     parser.add_argument("--data-root", default="/root/autodl-tmp/racketpose2.0/data", type=str)

#     parser.add_argument("--train", action="store_true")
#     parser.add_argument("--test", action="store_true")

#     # ---- eval thresholds ----
#     parser.add_argument("--pose-center-thr", type=float, default=0.05)        # 5cm
#     parser.add_argument("--pose-center-thr2", type=float, default=0.10)       # 10cm（只用于单项 center@10cm）
#     parser.add_argument("--pose-angle-thr", type=float, default=10.0)         # 10deg
#     parser.add_argument("--pose-angle-thr-small", type=float, default=5.0)    # 5deg

#     # weights
#     parser.add_argument("--pretrained", type=str, default=None)
#     parser.add_argument("--resume", type=str, default=None)
#     parser.add_argument("--no-strict", action="store_true")

#     # loss hyper
#     parser.add_argument("--tau", type=float, default=1.0, help="attention temperature")
#     parser.add_argument("--w-cls", type=float, default=1.0)
#     parser.add_argument("--w-center", type=float, default=1.0)
#     parser.add_argument("--w-normal", type=float, default=1.0)
#     parser.add_argument("--smoothl1-beta", type=float, default=1.0)

#     # test weight
#     parser.add_argument("--weight", type=str, default=None, help="best.pt/last.pt for test()")

#     args = parser.parse_args()

#     args.local_rank = int(os.getenv("LOCAL_RANK", 0))
#     args.world_size = int(os.getenv("WORLD_SIZE", 1))
#     args.distributed = int(os.getenv("WORLD_SIZE", 1)) > 1
#     if args.distributed:
#         torch.cuda.set_device(device=args.local_rank)
#         torch.distributed.init_process_group(backend="nccl", init_method="env://")

#     with open("utils/args.yaml", errors="ignore") as f:
#         params = yaml.safe_load(f)

#     util.setup_seed()
#     util.setup_multi_processes()

#     ts = datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S")
#     run_root = os.path.join("runs", f"pose-{ts}")
#     if args.local_rank == 0:
#         os.makedirs(run_root, exist_ok=True)
#         os.makedirs(os.path.join(run_root, "weights"), exist_ok=True)
#         os.makedirs(os.path.join(run_root, "tb"), exist_ok=True)

#     args.run_root = run_root
#     args.save_dir = os.path.join(run_root, "weights")
#     args.tb_dir = os.path.join(run_root, "tb")

#     if args.train:
#         train(args, params)
#     if args.test:
#         assert args.weight is not None, "--test 需要 --weight 指定 ckpt"
#         test(args, params)

#     if args.distributed:
#         torch.distributed.destroy_process_group()
#     torch.cuda.empty_cache()


# if __name__ == "__main__":
#     main()


# main_pose.py
import copy
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
from argparse import ArgumentParser
import datetime
import math

import torch
import torch.nn.functional as F
import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter

from nets import nn_center_and_normal as nn
from utils import util
from datasets.racketpose2 import build_loader  # pose 版 build_loader：返回 (loader, sampler, dataset)

warnings.filterwarnings("ignore")


def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m


# -------------------- ckpt utils --------------------
def _torch_load_any(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["model", "state_dict", "ema", "model_ema", "ema_state_dict"]:
            if k in ckpt_obj:
                v = ckpt_obj[k]
                if hasattr(v, "state_dict"):
                    return v.state_dict()
                if isinstance(v, dict):
                    return v
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    if hasattr(ckpt_obj, "state_dict"):
        return ckpt_obj.state_dict()
    raise ValueError(f"Unrecognized checkpoint format: {type(ckpt_obj)}")


def _strip_prefix(sd, prefixes=("module.", "model.")):
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def load_weights_safely(model: torch.nn.Module, weight_path: str, strict: bool = False):
    ckpt = _torch_load_any(weight_path, map_location="cpu")
    sd = _strip_prefix(_extract_state_dict(ckpt))

    msd = model.state_dict()
    filtered, skipped = {}, []
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=strict)

    print(f"[LOAD] weights = {weight_path}")
    print(f"[LOAD] loaded={len(filtered)} skipped(shape)={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}")
    if skipped:
        print(f"[LOAD] example skipped: {skipped[:8]}{' ...' if len(skipped) > 8 else ''}")
    return ckpt


# -------------------- Pose MIL Loss --------------------
class PoseMILLoss(torch.nn.Module):
    """
    outputs (train): list of 3 tensors, each [B, 6+nc, H, W] (cls is logits, NOT sigmoid)
    targets: dict with keys
      - label: [B] long
      - center_norm: [B,3] float  (standardized)
      - normal: [B,3] float (unit)
    """
    def __init__(
        self,
        nc: int,
        tau: float = 1.0,
        w_cls: float = 1.0,
        w_center: float = 1.0,
        w_normal: float = 1.0,
        smoothl1_beta: float = 1.0,
    ):
        super().__init__()
        self.nc = int(nc)
        self.tau = float(tau)
        self.w_cls = float(w_cls)
        self.w_center = float(w_center)
        self.w_normal = float(w_normal)
        self.beta = float(smoothl1_beta)

    def _flatten_train_outputs(self, outputs):
        assert isinstance(outputs, (list, tuple)), "Train outputs must be list/tuple of feature maps."
        B = outputs[0].shape[0]
        C = outputs[0].shape[1]
        x = torch.cat([o.view(B, C, -1) for o in outputs], dim=2)  # [B, 6+nc, N]
        return x

    def forward(self, outputs, targets):
        x = self._flatten_train_outputs(outputs)  # [B,6+nc,N]
        B = x.shape[0]

        reg = x[:, 0:6, :]                 # [B,6,N]
        cls_logits = x[:, 6:6+self.nc, :]  # [B,nc,N] logits

        y = targets["label"].long()                 # [B]
        center_gt = targets["center_norm"].float()  # [B,3] standardized
        normal_gt = targets["normal"].float()       # [B,3] unit

        # --- cls loss (image-level MIL pooling) ---
        img_logits = torch.logsumexp(cls_logits, dim=2)  # [B,nc]
        loss_cls = F.cross_entropy(img_logits, y)

        # --- attention weights from GT class ---
        idx = torch.arange(B, device=x.device)
        g = cls_logits[idx, y, :]                 # [B,N]
        w = torch.softmax(g / self.tau, dim=1)    # [B,N]

        # --- center ---
        center_loc = reg[:, 0:3, :]                              # [B,3,N] standardized
        center_pred = (center_loc * w.unsqueeze(1)).sum(dim=2)   # [B,3]
        loss_center = F.smooth_l1_loss(center_pred, center_gt, beta=self.beta)

        # --- normal ---
        normal_loc = reg[:, 3:6, :]                   # [B,3,N] raw
        normal_loc = torch.tanh(normal_loc)
        normal_loc = F.normalize(normal_loc, dim=1, eps=1e-6)
        normal_pred = (normal_loc * w.unsqueeze(1)).sum(dim=2)   # [B,3]
        normal_pred = F.normalize(normal_pred, dim=1, eps=1e-6)

        cos = (normal_pred * normal_gt).sum(dim=1).clamp(-1.0, 1.0)
        loss_normal = (1.0 - cos).mean()

        loss = self.w_cls * loss_cls + self.w_center * loss_center + self.w_normal * loss_normal
        return loss, {
            "loss_cls": loss_cls.detach(),
            "loss_center": loss_center.detach(),
            "loss_normal": loss_normal.detach(),
        }


# -------------------- evaluation --------------------
@torch.no_grad()
def eval_pose(
    model,
    loader,
    nc: int,
    tau: float = 1.0,
    device="cuda",
    pose_thresholds=((0.05, 5.0), (0.05, 10.0)),
    center_thresholds=(0.05, 0.10),
    angle_thresholds=(5.0, 10.0),
):
    model.eval()

    total = 0
    correct = 0
    correct_count = 0

    sum_center_l2 = 0.0
    sum_center_mae = torch.zeros(3, dtype=torch.float64, device=device)
    sum_ang = 0.0

    sum_center_l2_c = 0.0
    sum_center_mae_c = torch.zeros(3, dtype=torch.float64, device=device)
    sum_ang_c = 0.0

    ok_center = {thr: 0 for thr in center_thresholds}
    ok_center_with_cls = {thr: 0 for thr in center_thresholds}
    ok_angle = {thr: 0 for thr in angle_thresholds}
    ok_angle_with_cls = {thr: 0 for thr in angle_thresholds}
    ok_pose = {(d, a): 0 for (d, a) in pose_thresholds}
    ok_pose_with_cls = {(d, a): 0 for (d, a) in pose_thresholds}

    eps = 1e-6

    for imgs, targets in tqdm.tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device, non_blocking=True).float()
        y = targets["label"].to(device, non_blocking=True).long()
        center_gt = targets["center_m"].to(device, non_blocking=True).float()
        normal_gt = targets["normal"].to(device, non_blocking=True).float()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(imgs)  # [B, 6+nc, N]

        B = out.shape[0]

        center_loc = out[:, 0:3, :]
        normal_loc = out[:, 3:6, :]
        cls_prob = out[:, 6:6+nc, :].clamp(eps, 1.0 - eps)

        log_not = torch.log1p(-cls_prob)
        img_prob = 1.0 - torch.exp(log_not.sum(dim=2))
        pred_y = img_prob.argmax(dim=1)

        cls_ok = (pred_y == y)
        correct += cls_ok.sum().item()
        correct_count += cls_ok.sum().item()
        total += B

        idx = torch.arange(B, device=device)
        g = cls_prob[idx, y, :]
        w = torch.softmax(torch.log(g + eps) / tau, dim=1)

        center_pred = (center_loc * w.unsqueeze(1)).sum(dim=2)
        normal_pred = (normal_loc * w.unsqueeze(1)).sum(dim=2)
        normal_pred = F.normalize(normal_pred, dim=1, eps=1e-6)

        diff = center_pred - center_gt
        l2 = diff.norm(dim=1)
        sum_center_l2 += l2.sum().item()
        sum_center_mae += diff.abs().sum(dim=0).double()

        if cls_ok.any():
            l2_c = l2[cls_ok]
            diff_c = diff[cls_ok]
            sum_center_l2_c += l2_c.sum().item()
            sum_center_mae_c += diff_c.abs().sum(dim=0).double()

        dot = (normal_pred * normal_gt).sum(dim=1).clamp(-1.0, 1.0)
        ang = torch.acos(dot) * (180.0 / math.pi)
        sum_ang += ang.sum().item()
        if cls_ok.any():
            sum_ang_c += ang[cls_ok].sum().item()

        for thr in center_thresholds:
            ok = (l2 < thr)
            ok_center[thr] += ok.sum().item()
            ok_center_with_cls[thr] += (ok & cls_ok).sum().item()

        for thr in angle_thresholds:
            ok = (ang < thr)
            ok_angle[thr] += ok.sum().item()
            ok_angle_with_cls[thr] += (ok & cls_ok).sum().item()

        for (d_thr, a_thr) in pose_thresholds:
            ok = (l2 < d_thr) & (ang < a_thr)
            ok_pose[(d_thr, a_thr)] += ok.sum().item()
            ok_pose_with_cls[(d_thr, a_thr)] += (ok & cls_ok).sum().item()

    if total == 0:
        return {}

    acc = correct / total
    center_l2_mean = sum_center_l2 / total
    center_mae_mean = (sum_center_mae / total).detach().cpu().tolist()
    ang_mean = sum_ang / total

    if correct_count > 0:
        center_l2_mean_c = sum_center_l2_c / correct_count
        center_mae_mean_c = (sum_center_mae_c / correct_count).detach().cpu().tolist()
        ang_mean_c = sum_ang_c / correct_count
    else:
        center_l2_mean_c = float("nan")
        center_mae_mean_c = [float("nan")] * 3
        ang_mean_c = float("nan")

    metrics = {
        "acc": float(acc),
        "acc_count": int(correct),
        "total": int(total),

        "center_l2_m": float(center_l2_mean),
        "center_mae_x_m": float(center_mae_mean[0]),
        "center_mae_y_m": float(center_mae_mean[1]),
        "center_mae_z_m": float(center_mae_mean[2]),
        "normal_deg": float(ang_mean),

        "center_l2_m_cls": float(center_l2_mean_c),
        "center_mae_x_m_cls": float(center_mae_mean_c[0]),
        "center_mae_y_m_cls": float(center_mae_mean_c[1]),
        "center_mae_z_m_cls": float(center_mae_mean_c[2]),
        "normal_deg_cls": float(ang_mean_c),
    }

    for thr in center_thresholds:
        metrics[f"center@{int(thr*100)}cm"] = float(ok_center[thr] / total)
        metrics[f"center+cls@{int(thr*100)}cm"] = float(ok_center_with_cls[thr] / total)

    for thr in angle_thresholds:
        metrics[f"normal@{int(thr)}deg"] = float(ok_angle[thr] / total)
        metrics[f"normal+cls@{int(thr)}deg"] = float(ok_angle_with_cls[thr] / total)

    for (d_thr, a_thr) in pose_thresholds:
        metrics[f"pose@{int(d_thr*100)}cm_{int(a_thr)}deg"] = float(ok_pose[(d_thr, a_thr)] / total)
        metrics[f"pose+cls@{int(d_thr*100)}cm_{int(a_thr)}deg"] = float(ok_pose_with_cls[(d_thr, a_thr)] / total)

    return metrics


# -------------------- train / test --------------------
def train(args, params):
    device = "cuda"
    nc = len(params["names"])

    model = nn.yolo_v11_x(nc).to(device)

    start_epoch = 0
    best_loss = float("inf")  # ✅ best 按 loss 最低

    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params["weight_decay"] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(
        util.set_params(model, params["weight_decay"]),
        params["min_lr"],
        params["momentum"],
        nesterov=True
    )
    amp_scale = torch.amp.GradScaler()

    # -------- loaders (train computes stats; val/test reuse train stats) --------
    train_loader, train_sampler, train_set = build_loader("train", args, params, shuffle=True, center_stats=None)
    train_stats = (train_set.center_mean, train_set.center_std)

    eval_split = "test" if getattr(args, "eval_split", "test") == "test" else "val"
    eval_loader, _, _ = build_loader(eval_split, args, params, shuffle=False, center_stats=train_stats)

    # -------- resume / pretrained --------
    if args.resume is not None:
        ckpt = load_weights_safely(model, args.resume, strict=(not args.no_strict))
        if isinstance(ckpt, dict):
            start_epoch = int(ckpt.get("epoch", 0))
            # 兼容旧字段 best_score
            best_loss = float(ckpt.get("best_loss", ckpt.get("best_score", best_loss)))
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    print("[RESUME] optimizer state loaded.")
                except Exception as e:
                    print(f"[RESUME] optimizer load failed: {e}")
            if "scaler" in ckpt:
                try:
                    amp_scale.load_state_dict(ckpt["scaler"])
                    print("[RESUME] scaler state loaded.")
                except Exception as e:
                    print(f"[RESUME] scaler load failed: {e}")
        print(f"[RESUME] start_epoch={start_epoch} best_loss={best_loss}")

    elif args.pretrained is not None:
        _ = load_weights_safely(model, args.pretrained, strict=(not args.no_strict))
        print("[PRETRAIN] loaded model weights only (no optimizer/scaler).")

    # ✅ 统计量写入 head：放在加载权重之后，避免被 ckpt 覆盖
    m0 = _unwrap_model(model)
    if hasattr(m0, "head") and hasattr(m0.head, "set_center_stats"):
        m0.head.set_center_stats(train_set.center_mean.tolist(), train_set.center_std.tolist(), denorm_inference=True)

    # -------- ddp --------
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # -------- ema / writer --------
    writer = SummaryWriter(log_dir=args.tb_dir) if args.local_rank == 0 else None
    ema = util.EMA(model) if args.local_rank == 0 else None

    num_steps = len(train_loader)
    scheduler = util.LinearLR(args, params, num_steps)

    # -------- loss hyper --------
    tau = float(getattr(args, "tau", 1.0))
    w_cls = float(getattr(args, "w_cls", params.get("w_cls", 1.0)))
    w_center = float(getattr(args, "w_center", params.get("w_center", 1.0)))
    w_normal = float(getattr(args, "w_normal", params.get("w_normal", 1.0)))
    beta = float(getattr(args, "smoothl1_beta", params.get("smoothl1_beta", 1.0)))

    criterion = PoseMILLoss(
        nc=nc, tau=tau,
        w_cls=w_cls, w_center=w_center, w_normal=w_normal,
        smoothl1_beta=beta
    )

    # -------- thresholds --------
    pose_thresholds = (
        (args.pose_center_thr, args.pose_angle_thr),         # 5cm 10deg
        (args.pose_center_thr, args.pose_angle_thr_small),   # 5cm 5deg
    )
    center_thresholds = (args.pose_center_thr, args.pose_center_thr2)
    angle_thresholds = (args.pose_angle_thr_small, args.pose_angle_thr)

    # -------- logging csv --------
    step_csv = os.path.join(args.save_dir, "step.csv")
    with open(step_csv, "w", newline="") as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(
                log,
                fieldnames=[
                    "epoch",
                    "loss", "loss_cls", "loss_center", "loss_normal",
                    "acc",
                    "center_l2_m", "center_l2_m_cls",
                    "center_mae_x_m", "center_mae_y_m", "center_mae_z_m",
                    "center_mae_x_m_cls", "center_mae_y_m_cls", "center_mae_z_m_cls",
                    "normal_deg", "normal_deg_cls",
                    f"center@{int(args.pose_center_thr*100)}cm", f"center+cls@{int(args.pose_center_thr*100)}cm",
                    f"center@{int(args.pose_center_thr2*100)}cm", f"center+cls@{int(args.pose_center_thr2*100)}cm",
                    f"normal@{int(args.pose_angle_thr_small)}deg", f"normal+cls@{int(args.pose_angle_thr_small)}deg",
                    f"normal@{int(args.pose_angle_thr)}deg", f"normal+cls@{int(args.pose_angle_thr)}deg",
                    f"pose@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg",
                    f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg",
                    f"pose@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg",
                    f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg",
                ]
            )
            logger.writeheader()

        for epoch in range(start_epoch, args.epochs):
            model.train()
            if args.distributed:
                train_sampler.set_epoch(epoch)

            p_bar = enumerate(train_loader)
            if args.local_rank == 0:
                print(("\n" + "%10s" * 6) % ("epoch", "memory", "loss", "cls", "center", "normal"))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad(set_to_none=True)
            avg_loss = util.AverageMeter()
            avg_cls = util.AverageMeter()
            avg_center = util.AverageMeter()
            avg_normal = util.AverageMeter()

            for i, (samples, targets) in p_bar:
                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.to(device, non_blocking=True).float()
                targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(samples)  # train -> list of maps
                    loss, ld = criterion(outputs, targets)

                avg_loss.update(loss.item(), samples.size(0))
                avg_cls.update(float(ld["loss_cls"]), samples.size(0))
                avg_center.update(float(ld["loss_center"]), samples.size(0))
                avg_normal.update(float(ld["loss_normal"]), samples.size(0))

                amp_scale.scale(loss).backward()

                if step % accumulate == 0:
                    amp_scale.step(optimizer)
                    amp_scale.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema:
                        ema.update(model)

                if writer is not None:
                    writer.add_scalar("train/loss", avg_loss.avg, step)
                    writer.add_scalar("train/loss_cls", avg_cls.avg, step)
                    writer.add_scalar("train/loss_center", avg_center.avg, step)
                    writer.add_scalar("train/loss_normal", avg_normal.avg, step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

                torch.cuda.synchronize()

                if args.local_rank == 0:
                    memory = f"{torch.cuda.memory_reserved() / 1E9:.4g}G"
                    s = ("%10s" * 2 + "%10.4g" * 4) % (
                        f"{epoch+1}/{args.epochs}", memory,
                        avg_loss.avg, avg_cls.avg, avg_center.avg, avg_normal.avg
                    )
                    p_bar.set_description(s)

            # -------- eval & save --------
            if args.local_rank == 0:
                eval_model = ema.ema if ema else model

                metrics = eval_pose(
                    eval_model,
                    eval_loader,
                    nc=nc,
                    tau=tau,
                    device=device,
                    pose_thresholds=pose_thresholds,
                    center_thresholds=center_thresholds,
                    angle_thresholds=angle_thresholds,
                )

                # ✅ best 依据：本 epoch 的训练 loss
                epoch_loss = float(avg_loss.avg)

                if writer is not None:
                    writer.add_scalar(f"{eval_split}/acc", metrics["acc"], epoch)
                    writer.add_scalar(f"{eval_split}/center_l2_m", metrics["center_l2_m"], epoch)
                    writer.add_scalar(f"{eval_split}/center_l2_m_cls", metrics["center_l2_m_cls"], epoch)
                    writer.add_scalar(f"{eval_split}/normal_deg", metrics["normal_deg"], epoch)
                    writer.add_scalar(f"{eval_split}/normal_deg_cls", metrics["normal_deg_cls"], epoch)
                    for k, v in metrics.items():
                        if k.startswith(("center@", "center+cls@", "normal@", "normal+cls@", "pose@", "pose+cls@")):
                            writer.add_scalar(f"{eval_split}/{k}", v, epoch)
                    writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
                    writer.add_scalar("train/best_loss", best_loss, epoch)

                # 打印同时给出 5cm10deg
                k_pose_5cm10 = f"pose@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"
                k_posecls_5cm10 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"

                print(
                    f"[{eval_split.upper()}] "
                    f"epoch_loss={epoch_loss:.6f} | best_loss={best_loss:.6f} | "
                    f"acc={metrics['acc']:.3f} | "
                    f"center_l2={metrics['center_l2_m']:.4f}m | normal={metrics['normal_deg']:.2f}° | "
                    f"{k_pose_5cm10}={metrics.get(k_pose_5cm10, float('nan')):.3f} | "
                    f"{k_posecls_5cm10}={metrics.get(k_posecls_5cm10, float('nan')):.3f}"
                )

                # 保存 last
                model_to_save = copy.deepcopy(_unwrap_model(eval_model).float())
                save = {
                    "epoch": epoch + 1,
                    "best_loss": float(best_loss),
                    "model": model_to_save,
                    "optimizer": optimizer.state_dict(),
                    "scaler": amp_scale.state_dict(),
                }
                torch.save(save, os.path.join(args.save_dir, "last.pt"))

                # ✅ 保存 best：loss 更低就覆盖 best.pt
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save["best_loss"] = float(best_loss)
                    torch.save(save, os.path.join(args.save_dir, "best.pt"))
                    print(f"[Best] epoch_loss={best_loss:.6f} saved best.pt")

                # 写 step.csv
                row = {
                    "epoch": epoch + 1,
                    "loss": avg_loss.avg,
                    "loss_cls": avg_cls.avg,
                    "loss_center": avg_center.avg,
                    "loss_normal": avg_normal.avg,
                    "acc": metrics["acc"],
                    "center_l2_m": metrics["center_l2_m"],
                    "center_l2_m_cls": metrics["center_l2_m_cls"],
                    "center_mae_x_m": metrics["center_mae_x_m"],
                    "center_mae_y_m": metrics["center_mae_y_m"],
                    "center_mae_z_m": metrics["center_mae_z_m"],
                    "center_mae_x_m_cls": metrics["center_mae_x_m_cls"],
                    "center_mae_y_m_cls": metrics["center_mae_y_m_cls"],
                    "center_mae_z_m_cls": metrics["center_mae_z_m_cls"],
                    "normal_deg": metrics["normal_deg"],
                    "normal_deg_cls": metrics["normal_deg_cls"],
                }
                for k, v in metrics.items():
                    if k in logger.fieldnames:
                        row[k] = v
                logger.writerow(row)
                log.flush()

    if writer is not None:
        writer.close()


@torch.no_grad()
def test(args, params, model=None):
    device = "cuda"
    nc = len(params["names"])
    tau = float(getattr(args, "tau", 1.0))

    pose_thresholds = (
        (args.pose_center_thr, args.pose_angle_thr),         # 5cm10deg
        (args.pose_center_thr, args.pose_angle_thr_small),   # 5cm5deg
    )
    center_thresholds = (args.pose_center_thr, args.pose_center_thr2)
    angle_thresholds = (args.pose_angle_thr_small, args.pose_angle_thr)

    if model is None:
        ckpt = torch.load(args.weight, map_location="cuda")
        model = ckpt["model"].float().fuse()

    # 尝试从 head 取 stats，传给 dataset
    stats = None
    m = _unwrap_model(model)
    if hasattr(m, "head") and hasattr(m.head, "center_mean") and hasattr(m.head, "center_std"):
        try:
            cm = m.head.center_mean.detach().cpu()
            cs = m.head.center_std.detach().cpu()
            if cm.numel() == 3 and cs.numel() == 3:
                stats = (cm, cs)
        except Exception:
            stats = None

    test_loader, _, _ = build_loader("test", args, params, shuffle=False, center_stats=stats)

    model = model.eval()

    metrics = eval_pose(
        model,
        test_loader,
        nc=nc,
        tau=tau,
        device=device,
        pose_thresholds=pose_thresholds,
        center_thresholds=center_thresholds,
        angle_thresholds=angle_thresholds,
    )

    # ✅ 明确打印 5cm10deg（以及 +cls）
    k_pose_5cm10 = f"pose@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"
    k_posecls_5cm10 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"

    print(f"[TEST] acc={metrics['acc']:.3f}")
    print(f"[TEST] center_l2={metrics['center_l2_m']:.4f}m | center_l2_cls={metrics['center_l2_m_cls']:.4f}m")
    print(f"[TEST] normal={metrics['normal_deg']:.2f}deg | normal_cls={metrics['normal_deg_cls']:.2f}deg")
    print(f"[TEST] {k_pose_5cm10}={metrics.get(k_pose_5cm10, float('nan')):.3f} | {k_posecls_5cm10}={metrics.get(k_posecls_5cm10, float('nan')):.3f}")

    return metrics


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--epochs", default=300, type=int)

    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument("--data-root", default="/root/autodl-tmp/racketpose2.0/data", type=str)

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    # ---- eval thresholds ----
    parser.add_argument("--pose-center-thr", type=float, default=0.05)        # 5cm
    parser.add_argument("--pose-center-thr2", type=float, default=0.10)       # 10cm（单项用）
    parser.add_argument("--pose-angle-thr", type=float, default=10.0)         # 10deg
    parser.add_argument("--pose-angle-thr-small", type=float, default=5.0)    # 5deg

    # weights
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-strict", action="store_true")

    # loss hyper
    parser.add_argument("--tau", type=float, default=1.0, help="attention temperature")
    parser.add_argument("--w-cls", type=float, default=0.2)
    parser.add_argument("--w-center", type=float, default=5)
    parser.add_argument("--w-normal", type=float, default=2)
    parser.add_argument("--smoothl1-beta", type=float, default=1.0)

    # test weight
    parser.add_argument("--weight", type=str, default=None, help="best.pt/last.pt for test()")

    args = parser.parse_args()

    args.local_rank = int(os.getenv("LOCAL_RANK", 0))
    args.world_size = int(os.getenv("WORLD_SIZE", 1))
    args.distributed = int(os.getenv("WORLD_SIZE", 1)) > 1
    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    with open("utils/args.yaml", errors="ignore") as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    ts = datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S")
    run_root = os.path.join("runs", f"pose-{ts}")
    if args.local_rank == 0:
        os.makedirs(run_root, exist_ok=True)
        os.makedirs(os.path.join(run_root, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_root, "tb"), exist_ok=True)

    args.run_root = run_root
    args.save_dir = os.path.join(run_root, "weights")
    args.tb_dir = os.path.join(run_root, "tb")

    if args.train:
        train(args, params)
    if args.test:
        assert args.weight is not None, "--test 需要 --weight 指定 ckpt"
        test(args, params)

    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
