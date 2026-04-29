# main_pose_roi.py
import copy
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings
from argparse import ArgumentParser
import datetime
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import nms

from nets import nn_roi_pose_gl as pose_nn
from nets import nn as det_nn  # detector 文件（yolo_v11_x）
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


# ---------------- Detector decode ----------------
def _cxcywh_to_xyxy(box_cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box_cxcywh.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _scale_xyxy(xyxy: torch.Tensor, scale: float, H: int, W: int) -> torch.Tensor:
    if scale == 1.0:
        out = xyxy
    else:
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
        ww = (xyxy[:, 2] - xyxy[:, 0]) * scale
        hh = (xyxy[:, 3] - xyxy[:, 1]) * scale
        out = torch.stack([cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2], dim=1)

    out[:, 0::2] = out[:, 0::2].clamp(0, W - 1)
    out[:, 1::2] = out[:, 1::2].clamp(0, H - 1)
    return out


@torch.no_grad()
def detector_top1_boxes_from_yolo_v11(
    det_model,
    imgs: torch.Tensor,
    nc: int,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    roi_scale: float = 1.0,
    # ✅ 如果指定 class_id，则只用该类分数（用于 COCO person）
    class_id: Optional[int] = None,
    fallback_boxes: Optional[List[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    """
    det_out: [B, 4+nc, N], box=(cx,cy,w,h) in pixels, cls_prob in [0,1]（sigmoid）
    返回 list[B]，每个 [1,4] xyxy pixels
    """
    det_model.eval()
    B, _, H, W = imgs.shape
    full = torch.tensor([[0.0, 0.0, float(W - 1), float(H - 1)]], device=imgs.device)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        det_out = det_model(imgs)  # [B,4+nc,N]

    assert isinstance(det_out, torch.Tensor) and det_out.dim() == 3
    assert det_out.shape[1] == 4 + nc, f"expected C=4+nc, got {det_out.shape}"

    box = det_out[:, 0:4, :].permute(0, 2, 1).contiguous().float()      # [B,N,4] cxcywh(px)
    cls = det_out[:, 4:4+nc, :].permute(0, 2, 1).contiguous().float()   # [B,N,nc] prob

    if class_id is None:
        scores, _ = cls.max(dim=2)  # [B,N]
    else:
        assert 0 <= int(class_id) < nc, f"class_id={class_id} out of range for nc={nc}"
        scores = cls[:, :, int(class_id)]  # [B,N] 只用该类

    boxes_list = []
    for b in range(B):
        s = scores[b]
        bx = box[b]

        keep = s >= conf_thres
        bx = bx[keep]
        s2 = s[keep]
        if bx.numel() == 0:
            if fallback_boxes is not None:
                boxes_list.append(fallback_boxes[b].to(imgs.device))
            else:
                boxes_list.append(full)
            continue

        xyxy = _cxcywh_to_xyxy(bx)
        xyxy = _scale_xyxy(xyxy, roi_scale, H, W)

        keep2 = nms(xyxy, s2, iou_thres)
        top1 = xyxy[keep2[:1]]  # [1,4]
        boxes_list.append(top1)

    return boxes_list


def union_boxes_xyxy_list(
    a_list: List[torch.Tensor],
    b_list: List[torch.Tensor],
    H: int,
    W: int,
    scale: float = 1.0,
) -> List[torch.Tensor]:
    out = []
    for a, b in zip(a_list, b_list):
        a = a.float()
        b = b.float()
        x1 = torch.minimum(a[:, 0], b[:, 0])
        y1 = torch.minimum(a[:, 1], b[:, 1])
        x2 = torch.maximum(a[:, 2], b[:, 2])
        y2 = torch.maximum(a[:, 3], b[:, 3])
        u = torch.stack([x1, y1, x2, y2], dim=1)
        u = _scale_xyxy(u, scale, H, W)
        out.append(u)
    return out


# ---------------- evaluation ----------------
@torch.no_grad()
def eval_pose_roi(
    model,
    racket_det,
    person_det,
    loader,
    nc_pose: int,
    device="cuda",
    racket_det_nc=4,
    person_det_nc=80,
    person_class_id=0,   # ✅ COCO80: 通常 person=0
    conf_thres_r=0.25,
    conf_thres_p=0.25,
    iou_thres=0.7,
    roi_scale=1.2,
    global_scale=1.05,
    pose_thresholds=((0.05, 5.0), (0.05, 10.0)),
):
    model.eval()
    racket_det.eval()
    person_det.eval()

    total = 0
    correct = 0
    sum_center_l2 = 0.0
    sum_ang = 0.0

    ok_pose = {(d, a): 0 for (d, a) in pose_thresholds}
    ok_pose_with_cls = {(d, a): 0 for (d, a) in pose_thresholds}

    for imgs, targets in tqdm.tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device, non_blocking=True).float()
        y = targets["label"].to(device, non_blocking=True).long()
        center_gt = targets["center_m"].to(device, non_blocking=True).float()
        normal_gt = targets["normal"].to(device, non_blocking=True).float()

        B, _, H, W = imgs.shape

        # local: racket
        racket_boxes = detector_top1_boxes_from_yolo_v11(
            racket_det, imgs, nc=racket_det_nc,
            conf_thres=conf_thres_r, iou_thres=iou_thres,
            roi_scale=roi_scale,
            class_id=None,
            fallback_boxes=None
        )

        # person: COCO80，只取 person 类分数；没检测到就用 racket 兜底
        person_boxes = detector_top1_boxes_from_yolo_v11(
            person_det, imgs, nc=person_det_nc,
            conf_thres=conf_thres_p, iou_thres=iou_thres,
            roi_scale=1.0,
            class_id=person_class_id,
            fallback_boxes=racket_boxes
        )

        # global: union(racket, person)
        global_boxes = union_boxes_xyxy_list(racket_boxes, person_boxes, H, W, scale=global_scale)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(imgs, racket_boxes, global_boxes)

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
    return metrics


# ---------------- train / test ----------------
def train(args, params):
    device = "cuda"
    nc_pose = len(params["names"])  # 你的姿态分类类别数（球拍4类）

    model = pose_nn.roi_pose_v11_x(
        num_classes=nc_pose,
        img_size=args.input_size,
        roi_ch=args.roi_ch,
        use_global=True
    ).to(device)

    # racket det (nc=4)
    racket_det = det_nn.yolo_v11_x(args.racket_det_nc).to(device)
    ckpt_r = _torch_load_any(args.racket_det_weight, map_location="cpu")
    sd_r = _strip_prefix(_extract_state_dict(ckpt_r))
    racket_det.load_state_dict(sd_r, strict=False)
    racket_det.eval()
    for p in racket_det.parameters():
        p.requires_grad_(False)
    print(f"[RACKET DET] nc={args.racket_det_nc} weight={args.racket_det_weight}")

    # person det (COCO80)
    person_det = det_nn.yolo_v11_x(args.person_det_nc).to(device)
    ckpt_p = _torch_load_any(args.person_det_weight, map_location="cpu")
    sd_p = _strip_prefix(_extract_state_dict(ckpt_p))
    person_det.load_state_dict(sd_p, strict=False)
    person_det.eval()
    for p in person_det.parameters():
        p.requires_grad_(False)
    print(f"[PERSON DET] nc={args.person_det_nc} weight={args.person_det_weight} person_cls_id={args.person_class_id}")

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
            k5 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg"
            k10 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"
            logger = csv.DictWriter(log, fieldnames=[
                "epoch",
                "loss", "loss_cls", "loss_center", "loss_normal",
                "acc", "center_l2_m", "normal_deg",
                k5, k10,
            ])
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
                B, _, H, W = samples.shape

                racket_boxes = detector_top1_boxes_from_yolo_v11(
                    racket_det, samples, nc=args.racket_det_nc,
                    conf_thres=args.racket_det_conf, iou_thres=args.det_iou,
                    roi_scale=args.roi_scale,
                    class_id=None,
                    fallback_boxes=None
                )
                person_boxes = detector_top1_boxes_from_yolo_v11(
                    person_det, samples, nc=args.person_det_nc,
                    conf_thres=args.person_det_conf, iou_thres=args.det_iou,
                    roi_scale=1.0,
                    class_id=args.person_class_id,
                    fallback_boxes=racket_boxes
                )
                global_boxes = union_boxes_xyxy_list(racket_boxes, person_boxes, H, W, scale=args.global_scale)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pred = model(samples, racket_boxes, global_boxes)
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
                eval_model = ema.ema if ema else model

                metrics = eval_pose_roi(
                    _unwrap(eval_model),
                    racket_det,
                    person_det,
                    eval_loader,
                    nc_pose=nc_pose,
                    device=device,
                    racket_det_nc=args.racket_det_nc,
                    person_det_nc=args.person_det_nc,
                    person_class_id=args.person_class_id,
                    conf_thres_r=args.racket_det_conf,
                    conf_thres_p=args.person_det_conf,
                    iou_thres=args.det_iou,
                    roi_scale=args.roi_scale,
                    global_scale=args.global_scale,
                    pose_thresholds=pose_thresholds
                )

                epoch_loss = float(avg_loss.avg)
                k5 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg"
                k10 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"
                print(f"[{eval_split.upper()}] epoch_loss={epoch_loss:.6f} | best_loss={best_loss:.6f} | "
                      f"acc={metrics['acc']:.3f} | center_l2={metrics['center_l2_m']:.4f}m | "
                      f"normal={metrics['normal_deg']:.2f}° | {k5}={metrics.get(k5, float('nan')):.3f} | "
                      f"{k10}={metrics.get(k10, float('nan')):.3f}")

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
                    logger.writerow(row)
                    log.flush()

    if writer is not None:
        writer.close()


@torch.no_grad()
def test(args, params):
    device = "cuda"
    nc_pose = len(params["names"])

    ckpt = torch.load(args.weight, map_location="cuda", weights_only=False)
    model = ckpt["model"].float().fuse().to(device).eval()

    racket_det = det_nn.yolo_v11_x(args.racket_det_nc).to(device)
    ckpt_r = _torch_load_any(args.racket_det_weight, map_location="cpu")
    sd_r = _strip_prefix(_extract_state_dict(ckpt_r))
    racket_det.load_state_dict(sd_r, strict=False)
    racket_det.eval()
    for p in racket_det.parameters():
        p.requires_grad_(False)

    person_det = det_nn.yolo_v11_x(args.person_det_nc).to(device)
    ckpt_p = _torch_load_any(args.person_det_weight, map_location="cpu")
    sd_p = _strip_prefix(_extract_state_dict(ckpt_p))
    person_det.load_state_dict(sd_p, strict=False)
    person_det.eval()
    for p in person_det.parameters():
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
        model, racket_det, person_det, test_loader,
        nc_pose=nc_pose, device=device,
        racket_det_nc=args.racket_det_nc,
        person_det_nc=args.person_det_nc,
        person_class_id=args.person_class_id,
        conf_thres_r=args.racket_det_conf,
        conf_thres_p=args.person_det_conf,
        iou_thres=args.det_iou,
        roi_scale=args.roi_scale,
        global_scale=args.global_scale,
        pose_thresholds=pose_thresholds
    )

    k5 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg"
    k10 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"
    print(f"[TEST] acc={metrics['acc']:.3f}")
    print(f"[TEST] center_l2={metrics['center_l2_m']:.4f}m | normal={metrics['normal_deg']:.2f}deg")
    print(f"[TEST] {k5}={metrics.get(k5, float('nan')):.3f}")
    print(f"[TEST] {k10}={metrics.get(k10, float('nan')):.3f}")
    return metrics


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=250, type=int)

    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument("--data-root", default="/data1/wangqiurui/pxy/Dataset/racketpose2.0/data", type=str)

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    # detectors
    parser.add_argument("--racket-det-weight", type=str, default="/data1/wangqiurui/pxy/yolov11-detect/weights/detect/det-2026_02_07-170201/weights/best.pt")
    parser.add_argument("--person-det-weight", type=str, default="/data1/wangqiurui/pxy/yolov11-detect/weights/offical/v11_x.pt")

    parser.add_argument("--racket-det-nc", type=int, default=4)
    parser.add_argument("--person-det-nc", type=int, default=80)   #  COCO80
    parser.add_argument("--person-class-id", type=int, default=0)  #  COCO80 person 通常=0

    parser.add_argument("--racket-det-conf", type=float, default=0.25)
    parser.add_argument("--person-det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.7)

    # ROI scales
    parser.add_argument("--roi-scale", type=float, default=2.0)     # local(racket)
    parser.add_argument("--global-scale", type=float, default=1.05) # union(racket,person)

    # roi
    parser.add_argument("--roi-ch", type=int, default=256)

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
    run_root = os.path.join("runs", f"pose-roi-person-{ts}")
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