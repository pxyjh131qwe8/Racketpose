# # main_pose_roi.py
# import copy
# import csv
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import warnings
# from argparse import ArgumentParser
# import datetime
# import math
# from typing import List, Tuple

# import torch
# import torch.nn.functional as F
# import tqdm
# import yaml
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.ops import nms

# from nets import nn_roi_pose as pose_nn
# from nets import nn as det_nn  # detector 文件
# from utils import util
# from datasets.racketpose2 import build_loader

# warnings.filterwarnings("ignore")


# def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
#     return m.module if hasattr(m, "module") else m


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


# # ---------------- ROI Pose Loss ----------------
# class PoseROILoss(torch.nn.Module):
#     def __init__(self, w_cls=1.0, w_center=1.0, w_normal=1.0, smoothl1_beta=1.0):
#         super().__init__()
#         self.w_cls = float(w_cls)
#         self.w_center = float(w_center)
#         self.w_normal = float(w_normal)
#         self.beta = float(smoothl1_beta)

#     def forward(self, pred, targets):
#         y = targets["label"].long()
#         center_gt = targets["center_norm"].float()
#         normal_gt = targets["normal"].float()

#         cls_logits = pred["cls_logits"]
#         center_pred = pred["center_norm"]
#         normal_pred = pred["normal"]

#         loss_cls = F.cross_entropy(cls_logits, y)
#         loss_center = F.smooth_l1_loss(center_pred, center_gt, beta=self.beta)

#         cos = (normal_pred * normal_gt).sum(dim=1).clamp(-1.0, 1.0)
#         loss_normal = (1.0 - cos).mean()

#         loss = self.w_cls * loss_cls + self.w_center * loss_center + self.w_normal * loss_normal
#         return loss, {
#             "loss_cls": loss_cls.detach(),
#             "loss_center": loss_center.detach(),
#             "loss_normal": loss_normal.detach(),
#         }


# # ---------------- Detector decode (你的 detector 专用) ----------------
# def _cxcywh_to_xyxy(box_cxcywh: torch.Tensor) -> torch.Tensor:
#     cx, cy, w, h = box_cxcywh.unbind(-1)
#     x1 = cx - w / 2
#     y1 = cy - h / 2
#     x2 = cx + w / 2
#     y2 = cy + h / 2
#     return torch.stack([x1, y1, x2, y2], dim=-1)


# @torch.no_grad()
# def detector_top1_boxes_from_yolo_v11(det_model, imgs: torch.Tensor, nc: int,
#                                      conf_thres: float = 0.25, iou_thres: float = 0.7,
#                                      roi_scale: float = 1.0) -> List[torch.Tensor]:
#     """
#     适配你贴的 detector：
#       det_out: [B, 4+nc, N], box=(cx,cy,w,h) in pixels, cls_prob in [0,1]
#     返回 list[B]，每个 [1,4] xyxy pixels
#     """
#     det_model.eval()
#     B, _, H, W = imgs.shape
#     full = torch.tensor([[0.0, 0.0, float(W - 1), float(H - 1)]], device=imgs.device)

#     with torch.amp.autocast("cuda", dtype=torch.float16):
#         det_out = det_model(imgs)  # [B,4+nc,N]

#     assert isinstance(det_out, torch.Tensor) and det_out.dim() == 3, f"det_out type/shape invalid: {type(det_out)}"
#     assert det_out.shape[1] == 4 + nc, f"expected C=4+nc, got {det_out.shape}"

#     box = det_out[:, 0:4, :].permute(0, 2, 1).contiguous().float()      # [B,N,4] cxcywh(px)
#     cls = det_out[:, 4:4+nc, :].permute(0, 2, 1).contiguous().float()   # [B,N,nc] prob

#     scores, _ = cls.max(dim=2)  # [B,N] 取所有类中最大 prob 作为 obj score

#     boxes_list = []
#     for b in range(B):
#         s = scores[b]
#         bx = box[b]

#         keep = s >= conf_thres
#         bx = bx[keep]
#         s2 = s[keep]
#         if bx.numel() == 0:
#             boxes_list.append(full)
#             continue

#         # cxcywh -> xyxy
#         xyxy = _cxcywh_to_xyxy(bx)

#         # 可选：放大 ROI（更稳一点，建议 1.1~1.3）
#         if roi_scale != 1.0:
#             cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
#             cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
#             w = (xyxy[:, 2] - xyxy[:, 0]) * roi_scale
#             h = (xyxy[:, 3] - xyxy[:, 1]) * roi_scale
#             xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)

#         # clamp
#         xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, W - 1)
#         xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, H - 1)

#         keep2 = nms(xyxy, s2, iou_thres)
#         top1 = xyxy[keep2[:1]]  # [1,4]
#         boxes_list.append(top1)

#     return boxes_list


# # ---------------- evaluation ----------------
# @torch.no_grad()
# def eval_pose_roi(model, det_model, loader, nc: int, device="cuda",
#                   conf_thres=0.25, iou_thres=0.7, roi_scale=1.0,
#                   pose_thresholds=((0.05, 5.0), (0.05, 10.0))):
#     model.eval()
#     det_model.eval()

#     total = 0
#     correct = 0
#     sum_center_l2 = 0.0
#     sum_ang = 0.0

#     ok_pose = {(d, a): 0 for (d, a) in pose_thresholds}
#     ok_pose_with_cls = {(d, a): 0 for (d, a) in pose_thresholds}

#     for imgs, targets in tqdm.tqdm(loader, desc="Eval", leave=False):
#         imgs = imgs.to(device, non_blocking=True).float()
#         y = targets["label"].to(device, non_blocking=True).long()
#         center_gt = targets["center_m"].to(device, non_blocking=True).float()
#         normal_gt = targets["normal"].to(device, non_blocking=True).float()

#         boxes = detector_top1_boxes_from_yolo_v11(
#             det_model, imgs, nc=nc, conf_thres=conf_thres, iou_thres=iou_thres, roi_scale=roi_scale
#         )

#         with torch.amp.autocast("cuda", dtype=torch.float16):
#             out = model(imgs, boxes)

#         center_pred = out["center_m"].float()
#         normal_pred = out["normal"].float()
#         cls_prob = out["cls_prob"].float()

#         pred_y = cls_prob.argmax(dim=1)
#         cls_ok = (pred_y == y)
#         correct += cls_ok.sum().item()

#         diff = center_pred - center_gt
#         l2 = diff.norm(dim=1)
#         sum_center_l2 += l2.sum().item()

#         dot = (normal_pred * normal_gt).sum(dim=1).clamp(-1.0, 1.0)
#         ang = torch.acos(dot) * (180.0 / math.pi)
#         sum_ang += ang.sum().item()

#         for (d_thr, a_thr) in pose_thresholds:
#             ok = (l2 < d_thr) & (ang < a_thr)
#             ok_pose[(d_thr, a_thr)] += ok.sum().item()
#             ok_pose_with_cls[(d_thr, a_thr)] += (ok & cls_ok).sum().item()

#         total += imgs.size(0)

#     if total == 0:
#         return {}

#     metrics = {
#         "acc": float(correct / total),
#         "center_l2_m": float(sum_center_l2 / total),
#         "normal_deg": float(sum_ang / total),
#     }
#     for (d_thr, a_thr) in pose_thresholds:
#         metrics[f"pose@{int(d_thr*100)}cm_{int(a_thr)}deg"] = float(ok_pose[(d_thr, a_thr)] / total)
#         metrics[f"pose+cls@{int(d_thr*100)}cm_{int(a_thr)}deg"] = float(ok_pose_with_cls[(d_thr, a_thr)] / total)
#     return metrics


# # ---------------- train / test ----------------
# def train(args, params):
#     device = "cuda"
#     nc = len(params["names"])

#     # pose model
#     model = pose_nn.roi_pose_v11_x(num_classes=nc, img_size=args.input_size, roi_ch=args.roi_ch).to(device)
#     # model = pose_nn.roi_pose_v11_x(
#     #     num_classes=nc, img_size=args.input_size, roi_ch=args.roi_ch,
#     #     use_global=args.use_global, global_from=args.global_from
#     # ).to(device)

#     # detector (frozen)
#     det = det_nn.yolo_v11_x(nc).to(device)
#     ckpt_det = _torch_load_any(args.det_weight, map_location="cpu")
#     sd_det = _strip_prefix(_extract_state_dict(ckpt_det))
#     det.load_state_dict(sd_det, strict=False)
#     det.eval()
#     for p in det.parameters():
#         p.requires_grad_(False)
#     print(f"[DET] loaded detector weights: {args.det_weight}")

#     start_epoch = 0
#     best_loss = float("inf")

#     accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
#     params["weight_decay"] *= args.batch_size * args.world_size * accumulate / 64

#     optimizer = torch.optim.SGD(
#         util.set_params(model, params["weight_decay"]),
#         params["min_lr"],
#         params["momentum"],
#         nesterov=True
#     )
#     amp_scale = torch.amp.GradScaler()

#     train_loader, train_sampler, train_set = build_loader("train", args, params, shuffle=True, center_stats=None)
#     train_stats = (train_set.center_mean, train_set.center_std)

#     eval_split = "test" if getattr(args, "eval_split", "test") == "test" else "val"
#     eval_loader, _, _ = build_loader(eval_split, args, params, shuffle=False, center_stats=train_stats)

#     # 写入 center stats：保证 eval 输出是 meters
#     _unwrap(model).set_center_stats(train_set.center_mean.tolist(), train_set.center_std.tolist(), denorm_inference=True)

#     if args.resume is not None:
#         ckpt = load_weights_safely(model, args.resume, strict=(not args.no_strict))
#         if isinstance(ckpt, dict):
#             start_epoch = int(ckpt.get("epoch", 0))
#             best_loss = float(ckpt.get("best_loss", best_loss))
#             if "optimizer" in ckpt:
#                 try:
#                     optimizer.load_state_dict(ckpt["optimizer"])
#                     print("[RESUME] optimizer loaded.")
#                 except Exception as e:
#                     print(f"[RESUME] optimizer load failed: {e}")
#             if "scaler" in ckpt:
#                 try:
#                     amp_scale.load_state_dict(ckpt["scaler"])
#                     print("[RESUME] scaler loaded.")
#                 except Exception as e:
#                     print(f"[RESUME] scaler load failed: {e}")
#         print(f"[RESUME] start_epoch={start_epoch} best_loss={best_loss}")

#     if args.distributed:
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#         model = torch.nn.parallel.DistributedDataParallel(
#             module=model, device_ids=[args.local_rank], output_device=args.local_rank
#         )

#     writer = SummaryWriter(log_dir=args.tb_dir) if args.local_rank == 0 else None
#     ema = util.EMA(model) if args.local_rank == 0 else None

#     num_steps = len(train_loader)
#     scheduler = util.LinearLR(args, params, num_steps)

#     criterion = PoseROILoss(
#         w_cls=args.w_cls, w_center=args.w_center, w_normal=args.w_normal, smoothl1_beta=args.smoothl1_beta
#     )

#     pose_thresholds = (
#         (args.pose_center_thr, args.pose_angle_thr_small),  # 5cm5deg
#         (args.pose_center_thr, args.pose_angle_thr),        # 5cm10deg
#     )

#     step_csv = os.path.join(args.save_dir, "step.csv")
#     with open(step_csv, "w", newline="") as log:
#         logger = None
#         if args.local_rank == 0:
#             logger = csv.DictWriter(log, fieldnames=[
#                 "epoch",
#                 "loss", "loss_cls", "loss_center", "loss_normal",
#                 "acc", "center_l2_m", "normal_deg",
#                 f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg",
#                 f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg",
#             ])
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

#                 boxes = detector_top1_boxes_from_yolo_v11(
#                     det, samples, nc=nc,
#                     conf_thres=args.det_conf, iou_thres=args.det_iou,
#                     roi_scale=args.roi_scale
#                 )

#                 with torch.amp.autocast("cuda", dtype=torch.float16):
#                     pred = model(samples, boxes)
#                     # print
#                     # with torch.no_grad():
#                     #     nrm = pred["normal"].norm(dim=1).mean().item()
#                     #     print("normal_pred norm avg:", nrm)

#                     loss, ld = criterion(pred, targets)

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

#             if args.local_rank == 0:
#                 # eval_model = ema.ema if ema else model
#                 eval_model = model

#                 metrics = eval_pose_roi(
#                     _unwrap(eval_model), det, eval_loader, nc=nc, device=device,
#                     conf_thres=args.det_conf, iou_thres=args.det_iou, roi_scale=args.roi_scale,
#                     pose_thresholds=pose_thresholds
#                 )

#                 epoch_loss = float(avg_loss.avg)

#                 k5 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg"
#                 k10 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"
#                 print(f"[{eval_split.upper()}] epoch_loss={epoch_loss:.6f} | best_loss={best_loss:.6f} | "
#                       f"acc={metrics['acc']:.3f} | center_l2={metrics['center_l2_m']:.4f}m | "
#                       f"normal={metrics['normal_deg']:.2f}° | {k5}={metrics.get(k5, float('nan')):.3f} | "
#                       f"{k10}={metrics.get(k10, float('nan')):.3f}")

#                 model_to_save = copy.deepcopy(_unwrap(eval_model).float())
#                 save = {
#                     "epoch": epoch + 1,
#                     "best_loss": float(best_loss),
#                     "model": model_to_save,
#                     "optimizer": optimizer.state_dict(),
#                     "scaler": amp_scale.state_dict(),
#                 }
#                 torch.save(save, os.path.join(args.save_dir, "last.pt"))

#                 if epoch_loss < best_loss:
#                     best_loss = epoch_loss
#                     save["best_loss"] = float(best_loss)
#                     torch.save(save, os.path.join(args.save_dir, "best.pt"))
#                     print(f"[Best] epoch_loss={best_loss:.6f} saved best.pt")

#                 if logger is not None:
#                     row = {
#                         "epoch": epoch + 1,
#                         "loss": avg_loss.avg,
#                         "loss_cls": avg_cls.avg,
#                         "loss_center": avg_center.avg,
#                         "loss_normal": avg_normal.avg,
#                         "acc": metrics["acc"],
#                         "center_l2_m": metrics["center_l2_m"],
#                         "normal_deg": metrics["normal_deg"],
#                         k5: metrics.get(k5, float("nan")),
#                         k10: metrics.get(k10, float("nan")),
#                     }
#                     logger.writerow(row)
#                     log.flush()

#     if writer is not None:
#         writer.close()


# @torch.no_grad()
# def test(args, params):
#     device = "cuda"
#     nc = len(params["names"])

#     ckpt = torch.load(args.weight, map_location="cuda", weights_only=False)
#     model = ckpt["model"].float().fuse().to(device).eval()

#     det = det_nn.yolo_v11_x(nc).to(device)
#     ckpt_det = _torch_load_any(args.det_weight, map_location="cpu")
#     sd_det = _strip_prefix(_extract_state_dict(ckpt_det))
#     det.load_state_dict(sd_det, strict=False)
#     det.eval()
#     for p in det.parameters():
#         p.requires_grad_(False)

#     stats = None
#     if hasattr(model, "center_mean") and hasattr(model, "center_std"):
#         stats = (model.center_mean.detach().cpu().view(3), model.center_std.detach().cpu().view(3))

#     test_loader, _, _ = build_loader("test", args, params, shuffle=False, center_stats=stats)

#     pose_thresholds = (
#         (args.pose_center_thr, args.pose_angle_thr_small),
#         (args.pose_center_thr, args.pose_angle_thr),
#     )

#     metrics = eval_pose_roi(
#         model, det, test_loader, nc=nc, device=device,
#         conf_thres=args.det_conf, iou_thres=args.det_iou, roi_scale=args.roi_scale,
#         pose_thresholds=pose_thresholds
#     )

#     k5 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg"
#     k10 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"
#     print(f"[TEST] acc={metrics['acc']:.3f}")
#     print(f"[TEST] center_l2={metrics['center_l2_m']:.4f}m | normal={metrics['normal_deg']:.2f}deg")
#     print(f"[TEST] {k5}={metrics.get(k5, float('nan')):.3f}")
#     print(f"[TEST] {k10}={metrics.get(k10, float('nan')):.3f}")
#     return metrics


# def main():
#     parser = ArgumentParser()
#     parser.add_argument("--input-size", default=640, type=int)
#     parser.add_argument("--batch-size", default=64, type=int)
#     parser.add_argument("--epochs", default=250, type=int)

#     parser.add_argument("--local-rank", default=0, type=int)
#     parser.add_argument("--local_rank", default=0, type=int)

#     parser.add_argument("--data-root", default="/data1/wangqiurui/pxy/Dataset/racketpose2.0/data", type=str)
#     # parser.add_argument("--data-root", default="/data1/wangqiurui/pxy/Dataset/racketpose2.0/test_by_racket/T", type=str)

#     parser.add_argument("--train", action="store_true")
#     parser.add_argument("--test", action="store_true")

#     # detector
#     parser.add_argument("--det-weight", type=str, default="/data1/wangqiurui/pxy/yolov11-detect/weights/detect/det-2026_02_07-170201/weights/best.pt")
#     parser.add_argument("--det-conf", type=float, default=0.25)
#     parser.add_argument("--det-iou", type=float, default=0.7)

#     # 默认 1.2：ROI 稍微放大一点，回归更稳、更容易冲 5cm/5°
#     parser.add_argument("--roi-scale", type=float, default=2.0)

#     # roi
#     parser.add_argument("--roi-ch", type=int, default=256)
    
#     # global
#     parser.add_argument("--use-global", action="store_true")
#     parser.add_argument("--global-from", type=str, default="p5", choices=["p5","p345"])

#     # thresholds
#     parser.add_argument("--pose-center-thr", type=float, default=0.05)
#     parser.add_argument("--pose-angle-thr", type=float, default=10.0)
#     parser.add_argument("--pose-angle-thr-small", type=float, default=5.0)

#     # weights
#     parser.add_argument("--resume", type=str, default=None)
#     parser.add_argument("--no-strict", action="store_true")

#     # loss
#     parser.add_argument("--w-cls", type=float, default=1.0)
#     parser.add_argument("--w-center", type=float, default=5.0)
#     parser.add_argument("--w-normal", type=float, default=2.0)
#     parser.add_argument("--smoothl1-beta", type=float, default=1.0)

#     # test
#     parser.add_argument("--weight", type=str, default=None)

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
#     run_root = os.path.join("runs", f"pose-roi-{ts}")
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


# main_pose_roi.py
import copy
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
from argparse import ArgumentParser
import datetime
import math
from typing import List, Tuple, Dict
from collections import Counter

import torch
import torch.nn.functional as F
import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import nms

from nets import nn_roi_pose as pose_nn
from nets import nn as det_nn  # detector 文件
from utils import util
from datasets.racketpose2 import build_loader

warnings.filterwarnings("ignore")


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m


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


# ---------------- ROI Pose Loss ----------------
class PoseROILoss(torch.nn.Module):
    def __init__(self, w_cls=1.0, w_center=1.0, w_normal=1.0, smoothl1_beta=1.0):
        super().__init__()
        self.w_cls = float(w_cls)
        self.w_center = float(w_center)
        self.w_normal = float(w_normal)
        self.beta = float(smoothl1_beta)

    def forward(self, pred, targets):
        y = targets["label"].long()
        center_gt = targets["center_norm"].float()
        normal_gt = targets["normal"].float()

        cls_logits = pred["cls_logits"]
        center_pred = pred["center_norm"]
        normal_pred = pred["normal"]

        loss_cls = F.cross_entropy(cls_logits, y)
        loss_center = F.smooth_l1_loss(center_pred, center_gt, beta=self.beta)

        cos = (normal_pred * normal_gt).sum(dim=1).clamp(-1.0, 1.0)
        loss_normal = (1.0 - cos).mean()

        loss = self.w_cls * loss_cls + self.w_center * loss_center + self.w_normal * loss_normal
        return loss, {
            "loss_cls": loss_cls.detach(),
            "loss_center": loss_center.detach(),
            "loss_normal": loss_normal.detach(),
        }


# ---------------- Detector decode (你的 detector 专用) ----------------
def _cxcywh_to_xyxy(box_cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box_cxcywh.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.no_grad()
def detector_top1_boxes_from_yolo_v11(det_model, imgs: torch.Tensor, nc: int,
                                     conf_thres: float = 0.25, iou_thres: float = 0.7,
                                     roi_scale: float = 1.0) -> List[torch.Tensor]:
    """
    适配你贴的 detector：
      det_out: [B, 4+nc, N], box=(cx,cy,w,h) in pixels, cls_prob in [0,1]
    返回 list[B]，每个 [1,4] xyxy pixels
    """
    det_model.eval()
    B, _, H, W = imgs.shape
    full = torch.tensor([[0.0, 0.0, float(W - 1), float(H - 1)]], device=imgs.device)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        det_out = det_model(imgs)  # [B,4+nc,N]

    assert isinstance(det_out, torch.Tensor) and det_out.dim() == 3, f"det_out type/shape invalid: {type(det_out)}"
    assert det_out.shape[1] == 4 + nc, f"expected C=4+nc, got {det_out.shape}"

    box = det_out[:, 0:4, :].permute(0, 2, 1).contiguous().float()      # [B,N,4] cxcywh(px)
    cls = det_out[:, 4:4+nc, :].permute(0, 2, 1).contiguous().float()   # [B,N,nc] prob

    scores, _ = cls.max(dim=2)  # [B,N] 取所有类中最大 prob 作为 obj score

    boxes_list = []
    for b in range(B):
        s = scores[b]
        bx = box[b]

        keep = s >= conf_thres
        bx = bx[keep]
        s2 = s[keep]
        if bx.numel() == 0:
            boxes_list.append(full)
            continue

        # cxcywh -> xyxy
        xyxy = _cxcywh_to_xyxy(bx)

        # 可选：放大 ROI（更稳一点，建议 1.1~1.3）
        if roi_scale != 1.0:
            cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
            cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
            w = (xyxy[:, 2] - xyxy[:, 0]) * roi_scale
            h = (xyxy[:, 3] - xyxy[:, 1]) * roi_scale
            xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)

        # clamp
        xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, W - 1)
        xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, H - 1)

        keep2 = nms(xyxy, s2, iou_thres)
        top1 = xyxy[keep2[:1]]  # [1,4]
        boxes_list.append(top1)

    return boxes_list


# ---------------- FPN assignment stats ----------------
def assign_fpn_levels_from_boxes(
    boxes: List[torch.Tensor],
    k_min: int = 3,
    k_max: int = 5,
    canonical_scale: int = 224,
    canonical_level: int = 4,
):
    """
    模拟 MultiScaleRoIAlign 的 level assignment 逻辑。
    输入:
        boxes: list[B], each tensor [Ni,4] in xyxy pixel coords
    返回:
        levels_all: 所有 roi 的层编号列表，例如 [3,3,4,5,...]
        stats: 每个 roi 的详细信息，后续可导出 csv / 作图
    """
    levels_all = []
    stats = []

    for img_idx, b in enumerate(boxes):
        if b.numel() == 0:
            continue

        w = (b[:, 2] - b[:, 0]).clamp(min=1.0)
        h = (b[:, 3] - b[:, 1]).clamp(min=1.0)
        area = w * h
        size = torch.sqrt(area)

        lvl = torch.floor(
            canonical_level + torch.log2(size / canonical_scale + 1e-6)
        )
        lvl = torch.clamp(lvl, min=k_min, max=k_max).to(torch.int64)

        levels_all.extend(lvl.cpu().tolist())

        for i in range(len(lvl)):
            stats.append({
                "img_id": int(img_idx),
                "x1": float(b[i, 0].item()),
                "y1": float(b[i, 1].item()),
                "x2": float(b[i, 2].item()),
                "y2": float(b[i, 3].item()),
                "w": float(w[i].item()),
                "h": float(h[i].item()),
                "area": float(area[i].item()),
                "size": float(size[i].item()),
                "level": int(lvl[i].item()),
            })

    return levels_all, stats


def summarize_fpn_levels(levels_all: List[int]) -> Dict[str, float]:
    cnt = Counter(levels_all)
    total = sum(cnt.values())

    if total == 0:
        return {
            "total": 0,
            "p3": 0, "p4": 0, "p5": 0,
            "ratio_p3": 0.0, "ratio_p4": 0.0, "ratio_p5": 0.0,
        }

    return {
        "total": total,
        "p3": cnt.get(3, 0),
        "p4": cnt.get(4, 0),
        "p5": cnt.get(5, 0),
        "ratio_p3": cnt.get(3, 0) / total,
        "ratio_p4": cnt.get(4, 0) / total,
        "ratio_p5": cnt.get(5, 0) / total,
    }


def save_fpn_stats_csv(stats: List[dict], save_path: str):
    if len(stats) == 0:
        print(f"[FPN] no stats to save: {save_path}")
        return

    fieldnames = list(stats[0].keys())
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)
    print(f"[FPN] saved roi-level stats to: {save_path}")


# ---------------- evaluation ----------------
@torch.no_grad()
def eval_pose_roi(model, det_model, loader, nc: int, device="cuda",
                  conf_thres=0.25, iou_thres=0.7, roi_scale=1.0,
                  pose_thresholds=((0.05, 5.0), (0.05, 10.0)),
                  collect_fpn_stats: bool = False):
    model.eval()
    det_model.eval()

    total = 0
    correct = 0
    sum_center_l2 = 0.0
    sum_ang = 0.0

    ok_pose = {(d, a): 0 for (d, a) in pose_thresholds}
    ok_pose_with_cls = {(d, a): 0 for (d, a) in pose_thresholds}

    all_fpn_levels = []
    all_fpn_stats = []

    for imgs, targets in tqdm.tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device, non_blocking=True).float()
        y = targets["label"].to(device, non_blocking=True).long()
        center_gt = targets["center_m"].to(device, non_blocking=True).float()
        normal_gt = targets["normal"].to(device, non_blocking=True).float()

        boxes = detector_top1_boxes_from_yolo_v11(
            det_model, imgs, nc=nc, conf_thres=conf_thres, iou_thres=iou_thres, roi_scale=roi_scale
        )

        if collect_fpn_stats:
            levels_all, stats = assign_fpn_levels_from_boxes(boxes)
            all_fpn_levels.extend(levels_all)
            all_fpn_stats.extend(stats)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(imgs, boxes)

        center_pred = out["center_m"].float()
        normal_pred = out["normal"].float()
        cls_prob = out["cls_prob"].float()

        pred_y = cls_prob.argmax(dim=1)
        cls_ok = (pred_y == y)
        correct += cls_ok.sum().item()

        diff = center_pred - center_gt
        l2 = diff.norm(dim=1)
        sum_center_l2 += l2.sum().item()

        dot = (normal_pred * normal_gt).sum(dim=1).clamp(-1.0, 1.0)
        ang = torch.acos(dot) * (180.0 / math.pi)
        sum_ang += ang.sum().item()

        for (d_thr, a_thr) in pose_thresholds:
            ok = (l2 < d_thr) & (ang < a_thr)
            ok_pose[(d_thr, a_thr)] += ok.sum().item()
            ok_pose_with_cls[(d_thr, a_thr)] += (ok & cls_ok).sum().item()

        total += imgs.size(0)

    if total == 0:
        return {}

    metrics = {
        "acc": float(correct / total),
        "center_l2_m": float(sum_center_l2 / total),
        "normal_deg": float(sum_ang / total),
    }
    for (d_thr, a_thr) in pose_thresholds:
        metrics[f"pose@{int(d_thr*100)}cm_{int(a_thr)}deg"] = float(ok_pose[(d_thr, a_thr)] / total)
        metrics[f"pose+cls@{int(d_thr*100)}cm_{int(a_thr)}deg"] = float(ok_pose_with_cls[(d_thr, a_thr)] / total)

    if collect_fpn_stats:
        fpn_summary = summarize_fpn_levels(all_fpn_levels)
        for k, v in fpn_summary.items():
            metrics[f"fpn_{k}"] = v
        metrics["_fpn_stats"] = all_fpn_stats

    return metrics


# ---------------- train / test ----------------
def train(args, params):
    device = "cuda"
    nc = len(params["names"])

    # pose model
    # 如果你想启用 global-local，就用下面这一版
    model = pose_nn.roi_pose_v11_x(
        num_classes=nc,
        img_size=args.input_size,
        roi_ch=args.roi_ch,
        use_global=args.use_global,
        global_from=args.global_from
    ).to(device)

    # detector (frozen)
    det = det_nn.yolo_v11_x(nc).to(device)
    ckpt_det = _torch_load_any(args.det_weight, map_location="cpu")
    sd_det = _strip_prefix(_extract_state_dict(ckpt_det))
    det.load_state_dict(sd_det, strict=False)
    det.eval()
    for p in det.parameters():
        p.requires_grad_(False)
    print(f"[DET] loaded detector weights: {args.det_weight}")

    start_epoch = 0
    best_loss = float("inf")

    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params["weight_decay"] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(
        util.set_params(model, params["weight_decay"]),
        params["min_lr"],
        params["momentum"],
        nesterov=True
    )
    amp_scale = torch.amp.GradScaler()

    train_loader, train_sampler, train_set = build_loader("train", args, params, shuffle=True, center_stats=None)
    train_stats = (train_set.center_mean, train_set.center_std)

    eval_split = "test" if getattr(args, "eval_split", "test") == "test" else "val"
    eval_loader, _, _ = build_loader(eval_split, args, params, shuffle=False, center_stats=train_stats)

    # 写入 center stats：保证 eval 输出是 meters
    _unwrap(model).set_center_stats(train_set.center_mean.tolist(), train_set.center_std.tolist(), denorm_inference=True)

    if args.resume is not None:
        ckpt = load_weights_safely(model, args.resume, strict=(not args.no_strict))
        if isinstance(ckpt, dict):
            start_epoch = int(ckpt.get("epoch", 0))
            best_loss = float(ckpt.get("best_loss", best_loss))
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    print("[RESUME] optimizer loaded.")
                except Exception as e:
                    print(f"[RESUME] optimizer load failed: {e}")
            if "scaler" in ckpt:
                try:
                    amp_scale.load_state_dict(ckpt["scaler"])
                    print("[RESUME] scaler loaded.")
                except Exception as e:
                    print(f"[RESUME] scaler load failed: {e}")
        print(f"[RESUME] start_epoch={start_epoch} best_loss={best_loss}")

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    writer = SummaryWriter(log_dir=args.tb_dir) if args.local_rank == 0 else None
    ema = util.EMA(model) if args.local_rank == 0 else None

    num_steps = len(train_loader)
    scheduler = util.LinearLR(args, params, num_steps)

    criterion = PoseROILoss(
        w_cls=args.w_cls, w_center=args.w_center, w_normal=args.w_normal, smoothl1_beta=args.smoothl1_beta
    )

    pose_thresholds = (
        (args.pose_center_thr, args.pose_angle_thr_small),  # 5cm5deg
        (args.pose_center_thr, args.pose_angle_thr),        # 5cm10deg
    )

    step_csv = os.path.join(args.save_dir, "step.csv")
    with open(step_csv, "w", newline="") as log:
        logger = None
        if args.local_rank == 0:
            fieldnames = [
                "epoch",
                "loss", "loss_cls", "loss_center", "loss_normal",
                "acc", "center_l2_m", "normal_deg",
                f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg",
                f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg",
            ]
            if args.collect_fpn_stats:
                fieldnames += [
                    "fpn_total", "fpn_p3", "fpn_p4", "fpn_p5",
                    "fpn_ratio_p3", "fpn_ratio_p4", "fpn_ratio_p5",
                ]
            logger = csv.DictWriter(log, fieldnames=fieldnames)
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

                boxes = detector_top1_boxes_from_yolo_v11(
                    det, samples, nc=nc,
                    conf_thres=args.det_conf, iou_thres=args.det_iou,
                    roi_scale=args.roi_scale
                )

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pred = model(samples, boxes)
                    loss, ld = criterion(pred, targets)

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

            if args.local_rank == 0:
                # eval_model = ema.ema if ema else model
                eval_model = model

                metrics = eval_pose_roi(
                    _unwrap(eval_model), det, eval_loader, nc=nc, device=device,
                    conf_thres=args.det_conf, iou_thres=args.det_iou, roi_scale=args.roi_scale,
                    pose_thresholds=pose_thresholds,
                    collect_fpn_stats=args.collect_fpn_stats
                )

                epoch_loss = float(avg_loss.avg)

                k5 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg"
                k10 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"
                print(f"[{eval_split.upper()}] epoch_loss={epoch_loss:.6f} | best_loss={best_loss:.6f} | "
                      f"acc={metrics['acc']:.3f} | center_l2={metrics['center_l2_m']:.4f}m | "
                      f"normal={metrics['normal_deg']:.2f}° | {k5}={metrics.get(k5, float('nan')):.3f} | "
                      f"{k10}={metrics.get(k10, float('nan')):.3f}")

                if args.collect_fpn_stats:
                    print(
                        f"[FPN] total={int(metrics.get('fpn_total', 0))} | "
                        f"p3={int(metrics.get('fpn_p3', 0))} ({metrics.get('fpn_ratio_p3', 0.0):.3f}) | "
                        f"p4={int(metrics.get('fpn_p4', 0))} ({metrics.get('fpn_ratio_p4', 0.0):.3f}) | "
                        f"p5={int(metrics.get('fpn_p5', 0))} ({metrics.get('fpn_ratio_p5', 0.0):.3f})"
                    )

                model_to_save = copy.deepcopy(_unwrap(eval_model).float())
                save = {
                    "epoch": epoch + 1,
                    "best_loss": float(best_loss),
                    "model": model_to_save,
                    "optimizer": optimizer.state_dict(),
                    "scaler": amp_scale.state_dict(),
                }
                torch.save(save, os.path.join(args.save_dir, "last.pt"))

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save["best_loss"] = float(best_loss)
                    torch.save(save, os.path.join(args.save_dir, "best.pt"))
                    print(f"[Best] epoch_loss={best_loss:.6f} saved best.pt")

                if args.collect_fpn_stats and "_fpn_stats" in metrics:
                    fpn_csv_path = os.path.join(
                        args.save_dir, f"fpn_stats_epoch_{epoch+1:03d}.csv"
                    )
                    save_fpn_stats_csv(metrics["_fpn_stats"], fpn_csv_path)

                if logger is not None:
                    row = {
                        "epoch": epoch + 1,
                        "loss": avg_loss.avg,
                        "loss_cls": avg_cls.avg,
                        "loss_center": avg_center.avg,
                        "loss_normal": avg_normal.avg,
                        "acc": metrics["acc"],
                        "center_l2_m": metrics["center_l2_m"],
                        "normal_deg": metrics["normal_deg"],
                        k5: metrics.get(k5, float("nan")),
                        k10: metrics.get(k10, float("nan")),
                    }
                    if args.collect_fpn_stats:
                        row.update({
                            "fpn_total": metrics.get("fpn_total", 0),
                            "fpn_p3": metrics.get("fpn_p3", 0),
                            "fpn_p4": metrics.get("fpn_p4", 0),
                            "fpn_p5": metrics.get("fpn_p5", 0),
                            "fpn_ratio_p3": metrics.get("fpn_ratio_p3", 0.0),
                            "fpn_ratio_p4": metrics.get("fpn_ratio_p4", 0.0),
                            "fpn_ratio_p5": metrics.get("fpn_ratio_p5", 0.0),
                        })
                    logger.writerow(row)
                    log.flush()

    if writer is not None:
        writer.close()


@torch.no_grad()
def test(args, params):
    device = "cuda"
    nc = len(params["names"])

    ckpt = torch.load(args.weight, map_location="cuda", weights_only=False)
    model = ckpt["model"].float().fuse().to(device).eval()

    det = det_nn.yolo_v11_x(nc).to(device)
    ckpt_det = _torch_load_any(args.det_weight, map_location="cpu")
    sd_det = _strip_prefix(_extract_state_dict(ckpt_det))
    det.load_state_dict(sd_det, strict=False)
    det.eval()
    for p in det.parameters():
        p.requires_grad_(False)

    stats = None
    if hasattr(model, "center_mean") and hasattr(model, "center_std"):
        stats = (model.center_mean.detach().cpu().view(3), model.center_std.detach().cpu().view(3))

    test_loader, _, _ = build_loader("test", args, params, shuffle=False, center_stats=stats)

    pose_thresholds = (
        (args.pose_center_thr, args.pose_angle_thr_small),
        (args.pose_center_thr, args.pose_angle_thr),
    )

    metrics = eval_pose_roi(
        model, det, test_loader, nc=nc, device=device,
        conf_thres=args.det_conf, iou_thres=args.det_iou, roi_scale=args.roi_scale,
        pose_thresholds=pose_thresholds,
        collect_fpn_stats=args.collect_fpn_stats
    )

    k5 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg"
    k10 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"
    print(f"[TEST] acc={metrics['acc']:.3f}")
    print(f"[TEST] center_l2={metrics['center_l2_m']:.4f}m | normal={metrics['normal_deg']:.2f}deg")
    print(f"[TEST] {k5}={metrics.get(k5, float('nan')):.3f}")
    print(f"[TEST] {k10}={metrics.get(k10, float('nan')):.3f}")

    if args.collect_fpn_stats:
        print(
            f"[FPN] total={int(metrics.get('fpn_total', 0))} | "
            f"p3={int(metrics.get('fpn_p3', 0))} ({metrics.get('fpn_ratio_p3', 0.0):.3f}) | "
            f"p4={int(metrics.get('fpn_p4', 0))} ({metrics.get('fpn_ratio_p4', 0.0):.3f}) | "
            f"p5={int(metrics.get('fpn_p5', 0))} ({metrics.get('fpn_ratio_p5', 0.0):.3f})"
        )

        if "_fpn_stats" in metrics:
            fpn_csv_path = os.path.join(args.run_root, "fpn_stats_test.csv")
            save_fpn_stats_csv(metrics["_fpn_stats"], fpn_csv_path)

    return metrics


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=250, type=int)

    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)

    # parser.add_argument("--data-root", default="/data1/wangqiurui/pxy/Dataset/racketpose2.0/data", type=str)
    # parser.add_argument("--data-root", default="/data1/wangqiurui/pxy/Dataset/racketpose2.0/test_by_racket/T", type=str)
    parser.add_argument("--data-root", default="/data1/wangqiurui/pxy/Dataset/racketpose2.0/data/blur_split/sharp", type=str)

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    # detector
    parser.add_argument("--det-weight", type=str, default="/data1/wangqiurui/pxy/yolov11-detect/weights/detect/det-2026_02_07-170201/weights/best.pt")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.7)

    # 默认 1.2：ROI 稍微放大一点，回归更稳、更容易冲 5cm/5°
    parser.add_argument("--roi-scale", type=float, default=1.2)

    # roi
    parser.add_argument("--roi-ch", type=int, default=256)

    # global
    parser.add_argument("--use-global", action="store_true")
    parser.add_argument("--global-from", type=str, default="p5", choices=["p5", "p345"])

    # fpn stats
    parser.add_argument("--collect-fpn-stats", action="store_true")

    # thresholds
    parser.add_argument("--pose-center-thr", type=float, default=0.05)
    parser.add_argument("--pose-angle-thr", type=float, default=10.0)
    parser.add_argument("--pose-angle-thr-small", type=float, default=5.0)

    # weights
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-strict", action="store_true")

    # loss
    parser.add_argument("--w-cls", type=float, default=1.0)
    parser.add_argument("--w-center", type=float, default=5.0)
    parser.add_argument("--w-normal", type=float, default=2.0)
    parser.add_argument("--smoothl1-beta", type=float, default=1.0)

    # test
    parser.add_argument("--weight", type=str, default=None)

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
    run_root = os.path.join("runs", f"pose-roi-{ts}")
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