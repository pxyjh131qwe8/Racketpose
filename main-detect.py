# import copy
# import csv
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# import warnings
# from argparse import ArgumentParser
# import datetime
# import math

# import torch
# import tqdm
# import yaml
# from torch.utils.tensorboard import SummaryWriter

# from nets import nn
# from utils import util
# from datasets.detect_racket import build_loader  # 现在是 det-only build_loader

# warnings.filterwarnings("ignore")

# try:
#     from pycocotools.coco import COCO
#     from pycocotools.cocoeval import COCOeval
#     _COCO_OK = True
# except Exception:
#     _COCO_OK = False


# def _xyxy_to_xywh(xyxy: torch.Tensor):
#     if xyxy.numel() == 0:
#         return xyxy
#     x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
#     w = (x2 - x1).clamp_min(0)
#     h = (y2 - y1).clamp_min(0)
#     return torch.stack([x1, y1, w, h], dim=1)


# def _build_coco_structs(preds_accum, gts_accum, num_classes: int):
#     info = {
#         "description": "Auto-generated for evaluation",
#         "version": "1.0",
#         "year": 2026,
#         "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     }
#     licenses = [{"id": 1, "name": "Unknown", "url": ""}]

#     images, annotations = [], []
#     categories = [{"id": int(c), "name": str(c), "supercategory": ""} for c in range(num_classes)]
#     dt = []

#     ann_id = 1
#     seen = set()

#     for gt in gts_accum:
#         img_id = int(gt["image_id"])
#         if img_id not in seen:
#             H, W = gt.get("size", (640, 640))
#             images.append({"id": img_id, "width": int(W), "height": int(H), "license": 1})
#             seen.add(img_id)

#         if gt["boxes"].numel():
#             xywh = _xyxy_to_xywh(gt["boxes"]).cpu().numpy().tolist()
#             cls = gt["labels"].cpu().tolist()
#             for b, c in zip(xywh, cls):
#                 w, h = float(b[2]), float(b[3])
#                 annotations.append({
#                     "id": ann_id,
#                     "image_id": img_id,
#                     "category_id": int(c),
#                     "bbox": [float(b[0]), float(b[1]), w, h],
#                     "area": max(0.0, w * h),
#                     "iscrowd": 0,
#                 })
#                 ann_id += 1

#     for pred in preds_accum:
#         img_id = int(pred["image_id"])
#         if pred["boxes"].numel():
#             xywh = _xyxy_to_xywh(pred["boxes"]).cpu().numpy().tolist()
#             cls = pred["labels"].cpu().tolist()
#             scr = pred["scores"].cpu().numpy().tolist()
#             for b, c, s in zip(xywh, cls, scr):
#                 dt.append({
#                     "image_id": img_id,
#                     "category_id": int(c),
#                     "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
#                     "score": float(s)
#                 })

#     coco_gt_dict = {
#         "info": info,
#         "licenses": licenses,
#         "images": images,
#         "annotations": annotations,
#         "categories": categories
#     }
#     return coco_gt_dict, dt


# def _coco_eval_from_accum(preds_accum, gts_accum, num_classes: int):
#     if not _COCO_OK:
#         return None
#     import tempfile, json
#     coco_gt_dict, dt_list = _build_coco_structs(preds_accum, gts_accum, num_classes)

#     with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f_gt, \
#          tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f_dt:
#         json.dump(coco_gt_dict, f_gt); f_gt.flush()
#         json.dump(dt_list if len(dt_list) else [], f_dt); f_dt.flush()
#         coco_gt = COCO(f_gt.name)
#         coco_dt = coco_gt.loadRes(f_dt.name)

#     coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
#     coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

#     return {
#         "mAP@[.5:.95]": float(coco_eval.stats[0]),
#         "AP@0.5":       float(coco_eval.stats[1]),
#         "AP@0.75":      float(coco_eval.stats[2]),
#     }


# def train(args, params):
#     model = nn.yolo_v11_x(len(params["names"])).cuda()

#     accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
#     params["weight_decay"] *= args.batch_size * args.world_size * accumulate / 64

#     optimizer = torch.optim.SGD(
#         util.set_params(model, params["weight_decay"]),
#         params["min_lr"],
#         params["momentum"],
#         nesterov=True
#     )

#     writer = SummaryWriter(log_dir=args.tb_dir) if args.local_rank == 0 else None
#     ema = util.EMA(model) if args.local_rank == 0 else None

#     loader, sampler = build_loader("train", args, params, shuffle=True)

#     num_steps = len(loader)
#     scheduler = util.LinearLR(args, params, num_steps)

#     if args.distributed:
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#         model = torch.nn.parallel.DistributedDataParallel(
#             module=model, device_ids=[args.local_rank], output_device=args.local_rank
#         )

#     amp_scale = torch.amp.GradScaler()
#     criterion = util.ComputeLoss(model, params)

#     with open(os.path.join(args.save_dir, "step.csv"), "w") as log:
#         if args.local_rank == 0:
#             logger = csv.DictWriter(log, fieldnames=["epoch", "box", "cls", "dfl", "Recall", "Precision", "mAP@50", "mAP"])
#             logger.writeheader()

#         best_map = getattr(train, "_best_map", float("-inf"))

#         for epoch in range(args.epochs):
#             model.train()
#             if args.distributed:
#                 sampler.set_epoch(epoch)

#             p_bar = enumerate(loader)
#             if args.local_rank == 0:
#                 print(("\n" + "%10s" * 5) % ("epoch", "memory", "box", "cls", "dfl"))
#                 p_bar = tqdm.tqdm(p_bar, total=num_steps)

#             optimizer.zero_grad()
#             avg_box = util.AverageMeter()
#             avg_cls = util.AverageMeter()
#             avg_dfl = util.AverageMeter()

#             for i, (samples, targets) in p_bar:
#                 step = i + num_steps * epoch
#                 scheduler.step(step, optimizer)

#                 samples = samples.cuda(non_blocking=True).float()
#                 # targets dict -> to cuda
#                 targets = {k: v.cuda(non_blocking=True) for k, v in targets.items()}

#                 with torch.amp.autocast("cuda"):
#                     outputs = model(samples)
#                     loss_box, loss_cls, loss_dfl = criterion(outputs, targets)
#                     loss = loss_box + loss_cls + loss_dfl

#                 avg_box.update(loss_box.item(), samples.size(0))
#                 avg_cls.update(loss_cls.item(), samples.size(0))
#                 avg_dfl.update(loss_dfl.item(), samples.size(0))

#                 amp_scale.scale(loss).backward()

#                 if writer is not None:
#                     writer.add_scalar("train/loss_box", float(loss_box.detach()), step)
#                     writer.add_scalar("train/loss_cls", float(loss_cls.detach()), step)
#                     writer.add_scalar("train/loss_dfl", float(loss_dfl.detach()), step)
#                     writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

#                 if step % accumulate == 0:
#                     amp_scale.step(optimizer)
#                     amp_scale.update()
#                     optimizer.zero_grad()
#                     if ema:
#                         ema.update(model)

#                 torch.cuda.synchronize()

#                 if args.local_rank == 0:
#                     memory = f"{torch.cuda.memory_reserved() / 1E9:.4g}G"
#                     s = ("%10s" * 2 + "%10.3g" * 4) % (
#                         f"{epoch+1}/{args.epochs}", memory,
#                         avg_box.avg, avg_cls.avg, avg_dfl.avg,
#                         optimizer.param_groups[0]["lr"]
#                     )
#                     p_bar.set_description(s)

#             if args.local_rank == 0:
#                 map_all, map50, rec, pre, coco_res = test(args, params, ema.ema if ema else model)

#                 if writer is not None:
#                     writer.add_scalar("val/mAP", map_all, epoch)
#                     writer.add_scalar("val/mAP50", map50, epoch)
#                     writer.add_scalar("val/Recall", rec, epoch)
#                     writer.add_scalar("val/Precision", pre, epoch)
#                     if coco_res is not None:
#                         writer.add_scalar("val/COCO_mAP", coco_res["mAP@[.5:.95]"], epoch)

#                 print(f"Val | mAP:{map_all:.3f} | mAP@50:{map50:.3f} | R:{rec:.3f} | P:{pre:.3f}")

#                 save = {"epoch": epoch + 1, "model": copy.deepcopy(ema.ema if ema else model)}
#                 torch.save(save, os.path.join(args.save_dir, "last.pt"))

#                 if map_all > best_map:
#                     torch.save(save, os.path.join(args.save_dir, "best.pt"))
#                     best_map = map_all
#                     train._best_map = best_map
#                     print(f"[Best] mAP 提升为 {best_map:.4f}，已保存 best.pt")

#                 del save

#     if writer is not None:
#         writer.close()


# @torch.no_grad()
# def test(args, params, model=None):
#     loader, _ = build_loader("test", args, params, shuffle=False)

#     if model is None:
#         ckpt = torch.load(os.path.join(args.save_dir, "best.pt"), map_location="cuda", weights_only=False)
#         model = model['model'].float().fuse()

#     model = model.half().eval()

#     preds_accum, gts_accum = [], []
#     global_img_idx = 0

#     num_classes = len(params["names"])
#     iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()
#     n_iou = iou_v.numel()

#     metrics = []
#     m_pre = m_rec = map50 = mean_ap = 0.0

#     p_bar = tqdm.tqdm(loader, desc=("%10s" * 5) % ("", "precision", "recall", "mAP50", "mAP"))
#     for samples, targets in p_bar:
#         samples = samples.cuda().half()
#         _, _, h, w = samples.shape
#         input_size = torch.tensor([h, w], device=samples.device, dtype=torch.float32)

#         # NMS outputs: list[B] each [N,6]=xyxy conf cls
#         out = model(samples)
#         outputs = util.non_max_suppression(out)

#         # targets dict -> cuda
#         targets = {k: v.cuda(non_blocking=True) for k, v in targets.items()}

#         # 将 dict targets 变回每张图的 GT：cls + xyxy(pixel)
#         # targets["box"] 是 cxcywh normalized -> 转 xyxy pixel
#         idx_all = targets["idx"].view(-1).long() if targets["idx"].numel() else torch.zeros((0,), dtype=torch.long, device=samples.device)
#         cls_all = targets["cls"].view(-1)        if targets["cls"].numel() else torch.zeros((0,), device=samples.device)
#         box_all = targets["box"]                 if targets["box"].numel() else torch.zeros((0,4), device=samples.device)

#         if box_all.numel():
#             cx, cy, bw, bh = box_all[:, 0], box_all[:, 1], box_all[:, 2], box_all[:, 3]
#             x1 = (cx - bw / 2) * w
#             y1 = (cy - bh / 2) * h
#             x2 = (cx + bw / 2) * w
#             y2 = (cy + bh / 2) * h
#             gt_xyxy_pix = torch.stack([x1, y1, x2, y2], dim=1)
#         else:
#             gt_xyxy_pix = torch.zeros((0, 4), device=samples.device)

#         for i, output in enumerate(outputs):
#             mask = (idx_all == i)
#             if mask.any():
#                 gt_cls_i = cls_all[mask].view(-1, 1)
#                 gt_box_i = gt_xyxy_pix[mask]
#                 target_i = torch.cat([gt_cls_i, gt_box_i], dim=1)  # [N,5]

#                 metric = util.compute_metric(output[:, :6], target_i, iou_v)
#                 metrics.append((metric, output[:, 4], output[:, 5], gt_cls_i.squeeze(-1)))

#                 gts_accum.append({
#                     "image_id": global_img_idx,
#                     "boxes": gt_box_i.clone(),
#                     "labels": gt_cls_i.squeeze(1).long().clone(),
#                     "size": (h, w)
#                 })
#             else:
#                 metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=samples.device)
#                 metrics.append((metric, *torch.zeros((2, 0), device=samples.device), torch.zeros((0,), device=samples.device)))
#                 gts_accum.append({
#                     "image_id": global_img_idx,
#                     "boxes": torch.zeros((0, 4), device=samples.device),
#                     "labels": torch.zeros((0,), dtype=torch.long, device=samples.device),
#                     "size": (h, w)
#                 })

#             if output is not None and output.numel():
#                 preds_accum.append({
#                     "image_id": global_img_idx,
#                     "boxes": output[:, 0:4].clone(),
#                     "scores": output[:, 4].clone(),
#                     "labels": output[:, 5].long().clone()
#                 })
#             else:
#                 preds_accum.append({
#                     "image_id": global_img_idx,
#                     "boxes": torch.zeros((0, 4), device=samples.device),
#                     "scores": torch.zeros((0,), device=samples.device),
#                     "labels": torch.zeros((0,), dtype=torch.long, device=samples.device)
#                 })

#             global_img_idx += 1

#     metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)] if len(metrics) else None
#     if metrics and metrics[0].any():
#         tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics, plot=False, names=params["names"])

#     coco_res = None
#     if len(preds_accum) and len(gts_accum):
#         coco_res = _coco_eval_from_accum(preds_accum, gts_accum, num_classes)
#         if coco_res is None:
#             print("[COCO] pycocotools 未安装，跳过 COCO mAP。")
#         else:
#             print(f"[COCO] mAP@[.5:.95]: {coco_res['mAP@[.5:.95]']:.3f} | "
#                   f"AP@0.5: {coco_res['AP@0.5']:.3f} | AP@0.75: {coco_res['AP@0.75']:.3f}")

#     print(("%10s" + "%10.3g" * 4) % ("", m_pre, m_rec, map50, mean_ap))
#     return float(mean_ap), float(map50), float(m_rec), float(m_pre), coco_res


# def main():
#     parser = ArgumentParser()
#     parser.add_argument("--input-size", default=640, type=int)
#     parser.add_argument("--batch-size", default=20, type=int)
#     parser.add_argument("--epochs", default=100, type=int)

#     parser.add_argument("--local-rank", default=0, type=int)
#     parser.add_argument("--local_rank", default=0, type=int)

#     parser.add_argument("--data-root", default="/root/autodl-tmp/racketpose2.0/detect/finetune", type=str)

#     parser.add_argument("--train", action="store_true")
#     parser.add_argument("--test", action="store_true")

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
#     run_root = os.path.join("runs", f"det-{ts}")
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
#         test(args, params)

#     if args.distributed:
#         torch.distributed.destroy_process_group()
#     torch.cuda.empty_cache()


# if __name__ == "__main__":
#     main()



import copy
import csv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import warnings
from argparse import ArgumentParser
import datetime

import torch
import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter

from nets import nn
from utils import util
from datasets.detect_racket import build_loader  # det-only build_loader

warnings.filterwarnings("ignore")

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    _COCO_OK = True
except Exception:
    _COCO_OK = False


# -------------------- ckpt utils --------------------
def _torch_load_any(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(ckpt_obj):
    """
    兼容：
      - 纯 state_dict
      - {'model': <nn.Module>} / {'model': state_dict}
      - {'state_dict': state_dict}
      - {'ema': ...}
    """
    if isinstance(ckpt_obj, dict):
        for k in ["model", "state_dict", "ema", "model_ema", "ema_state_dict"]:
            if k in ckpt_obj:
                v = ckpt_obj[k]
                if hasattr(v, "state_dict"):
                    return v.state_dict()
                if isinstance(v, dict):
                    return v
        # dict 本身就是 state_dict？
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    # 直接是模型对象
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


def load_weights_safely(model: torch.nn.Module, weight_path: str, strict: bool = False, device="cuda"):
    """
    加载权重：自动跳过 shape 不匹配的 key（即“头不适配就不加载那部分”）
    """
    ckpt = _torch_load_any(weight_path, map_location="cpu")
    sd = _strip_prefix(_extract_state_dict(ckpt))

    msd = model.state_dict()
    filtered = {}
    skipped = []
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

    return ckpt  # 可能包含 epoch/optimizer 等信息


# -------------------- coco eval helpers --------------------
def _xyxy_to_xywh(xyxy: torch.Tensor):
    if xyxy.numel() == 0:
        return xyxy
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    w = (x2 - x1).clamp_min(0)
    h = (y2 - y1).clamp_min(0)
    return torch.stack([x1, y1, w, h], dim=1)


def _build_coco_structs(preds_accum, gts_accum, num_classes: int):
    info = {
        "description": "Auto-generated for evaluation",
        "version": "1.0",
        "year": 2026,
        "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    licenses = [{"id": 1, "name": "Unknown", "url": ""}]

    images, annotations = [], []
    categories = [{"id": int(c), "name": str(c), "supercategory": ""} for c in range(num_classes)]
    dt = []

    ann_id = 1
    seen = set()

    for gt in gts_accum:
        img_id = int(gt["image_id"])
        if img_id not in seen:
            H, W = gt.get("size", (640, 640))
            images.append({"id": img_id, "width": int(W), "height": int(H), "license": 1})
            seen.add(img_id)

        if gt["boxes"].numel():
            xywh = _xyxy_to_xywh(gt["boxes"]).cpu().numpy().tolist()
            cls = gt["labels"].cpu().tolist()
            for b, c in zip(xywh, cls):
                w, h = float(b[2]), float(b[3])
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(c),
                    "bbox": [float(b[0]), float(b[1]), w, h],
                    "area": max(0.0, w * h),
                    "iscrowd": 0,
                })
                ann_id += 1

    for pred in preds_accum:
        img_id = int(pred["image_id"])
        if pred["boxes"].numel():
            xywh = _xyxy_to_xywh(pred["boxes"]).cpu().numpy().tolist()
            cls = pred["labels"].cpu().tolist()
            scr = pred["scores"].cpu().numpy().tolist()
            for b, c, s in zip(xywh, cls, scr):
                dt.append({
                    "image_id": img_id,
                    "category_id": int(c),
                    "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "score": float(s)
                })

    coco_gt_dict = {
        "info": info,
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    return coco_gt_dict, dt


def _coco_eval_from_accum(preds_accum, gts_accum, num_classes: int):
    if not _COCO_OK:
        return None
    import tempfile, json
    coco_gt_dict, dt_list = _build_coco_structs(preds_accum, gts_accum, num_classes)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f_gt, \
         tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f_dt:
        json.dump(coco_gt_dict, f_gt); f_gt.flush()
        json.dump(dt_list if len(dt_list) else [], f_dt); f_dt.flush()
        coco_gt = COCO(f_gt.name)
        coco_dt = coco_gt.loadRes(f_dt.name)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

    return {
        "mAP@[.5:.95]": float(coco_eval.stats[0]),
        "AP@0.5":       float(coco_eval.stats[1]),
        "AP@0.75":      float(coco_eval.stats[2]),
    }


# -------------------- train / test --------------------
def train(args, params):
    model = nn.yolo_v11_x(len(params["names"])).cuda()

    # optimizer / scaler / resume state
    start_epoch = 0
    best_map = float("-inf")

    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params["weight_decay"] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(
        util.set_params(model, params["weight_decay"]),
        params["min_lr"],
        params["momentum"],
        nesterov=True
    )

    amp_scale = torch.amp.GradScaler()

    # 可选：加载权重（resume 或 pretrained）
    if args.resume is not None:
        ckpt = load_weights_safely(model, args.resume, strict=(not args.no_strict), device="cuda")
        if isinstance(ckpt, dict):
            start_epoch = int(ckpt.get("epoch", 0))
            best_map = float(ckpt.get("best_map", best_map))
            # 如果你以后保存了 optimizer/scaler，也能自动恢复
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
        print(f"[RESUME] start_epoch={start_epoch} best_map={best_map}")

    elif args.pretrained is not None:
        # 打印
        sd = torch.load(args.pretrained, map_location="cpu")
        sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        keys = list(sd.keys()) if isinstance(sd, dict) else list(sd.state_dict().keys())
        print([k for k in keys if "head.cls.0" in k][:50])

        
        _ = load_weights_safely(model, args.pretrained, strict=(not args.no_strict), device="cuda")
        print("[PRETRAIN] loaded model weights only (no optimizer/scaler).")

    writer = SummaryWriter(log_dir=args.tb_dir) if args.local_rank == 0 else None
    ema = util.EMA(model) if args.local_rank == 0 else None

    loader, sampler = build_loader("train", args, params, shuffle=True)
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    criterion = util.ComputeLoss(model, params)

    with open(os.path.join(args.save_dir, "step.csv"), "w") as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=["epoch", "box", "cls", "dfl", "Recall", "Precision", "mAP@50", "mAP"])
            logger.writeheader()

        # 允许从 resume 的 best_map 继续
        train._best_map = best_map

        for epoch in range(start_epoch, args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)

            p_bar = enumerate(loader)
            if args.local_rank == 0:
                print(("\n" + "%10s" * 5) % ("epoch", "memory", "box", "cls", "dfl"))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box = util.AverageMeter()
            avg_cls = util.AverageMeter()
            avg_dfl = util.AverageMeter()

            for i, (samples, targets) in p_bar:
                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda(non_blocking=True).float()
                targets = {k: v.cuda(non_blocking=True) for k, v in targets.items()}

                with torch.amp.autocast("cuda"):
                    outputs = model(samples)
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)
                    loss = loss_box + loss_cls + loss_dfl

                avg_box.update(loss_box.item(), samples.size(0))
                avg_cls.update(loss_cls.item(), samples.size(0))
                avg_dfl.update(loss_dfl.item(), samples.size(0))

                amp_scale.scale(loss).backward()

                if writer is not None:
                    writer.add_scalar("train/loss_box", float(loss_box.detach()), step)
                    writer.add_scalar("train/loss_cls", float(loss_cls.detach()), step)
                    writer.add_scalar("train/loss_dfl", float(loss_dfl.detach()), step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

                if step % accumulate == 0:
                    amp_scale.step(optimizer)
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                if args.local_rank == 0:
                    memory = f"{torch.cuda.memory_reserved() / 1E9:.4g}G"
                    s = ("%10s" * 2 + "%10.3g" * 4) % (
                        f"{epoch+1}/{args.epochs}", memory,
                        avg_box.avg, avg_cls.avg, avg_dfl.avg,
                        optimizer.param_groups[0]["lr"]
                    )
                    p_bar.set_description(s)

            if args.local_rank == 0:
                map_all, map50, rec, pre, coco_res = test(args, params, ema.ema if ema else model)

                if writer is not None:
                    writer.add_scalar("val/mAP", map_all, epoch)
                    writer.add_scalar("val/mAP50", map50, epoch)
                    writer.add_scalar("val/Recall", rec, epoch)
                    writer.add_scalar("val/Precision", pre, epoch)
                    if coco_res is not None:
                        writer.add_scalar("val/COCO_mAP", coco_res["mAP@[.5:.95]"], epoch)

                print(f"Val | mAP:{map_all:.3f} | mAP@50:{map50:.3f} | R:{rec:.3f} | P:{pre:.3f}")

                best_map = getattr(train, "_best_map", float("-inf"))

                save = {
                    "epoch": epoch + 1,
                    "best_map": best_map,
                    "model": copy.deepcopy(ema.ema if ema else model),
                    "optimizer": optimizer.state_dict(),
                    "scaler": amp_scale.state_dict(),
                }
                torch.save(save, os.path.join(args.save_dir, "last.pt"))

                if map_all > best_map:
                    torch.save(save, os.path.join(args.save_dir, "best.pt"))
                    best_map = map_all
                    train._best_map = best_map
                    print(f"[Best] mAP 提升为 {best_map:.4f}，已保存 best.pt")

                del save

    if writer is not None:
        writer.close()


@torch.no_grad()
def test(args, params, model=None):
    loader, _ = build_loader("test", args, params, shuffle=False)

    if model is None:
        # ckpt = torch.load(os.path.join(args.save_dir, "best.pt"), map_location="cuda", weights_only=False)
        ckpt = torch.load("/root/autodl-tmp/yolov11-detect/runs/det-2026_02_07-170201/weights/best.pt", map_location="cuda")
        model = ckpt["model"].float().fuse()

    model = model.half().eval()

    preds_accum, gts_accum = [], []
    global_img_idx = 0

    num_classes = len(params["names"])
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()
    n_iou = iou_v.numel()

    metrics = []
    m_pre = m_rec = map50 = mean_ap = 0.0

    p_bar = tqdm.tqdm(loader, desc=("%10s" * 5) % ("", "precision", "recall", "mAP50", "mAP"))
    for samples, targets in p_bar:
        samples = samples.cuda().half()
        _, _, h, w = samples.shape

        out = model(samples)
        outputs = util.non_max_suppression(out)

        targets = {k: v.cuda(non_blocking=True) for k, v in targets.items()}

        idx_all = targets["idx"].view(-1).long() if targets["idx"].numel() else torch.zeros((0,), dtype=torch.long, device=samples.device)
        cls_all = targets["cls"].view(-1) if targets["cls"].numel() else torch.zeros((0,), device=samples.device)
        box_all = targets["box"] if targets["box"].numel() else torch.zeros((0, 4), device=samples.device)

        if box_all.numel():
            cx, cy, bw, bh = box_all[:, 0], box_all[:, 1], box_all[:, 2], box_all[:, 3]
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            gt_xyxy_pix = torch.stack([x1, y1, x2, y2], dim=1)
        else:
            gt_xyxy_pix = torch.zeros((0, 4), device=samples.device)

        for i, output in enumerate(outputs):
            mask = (idx_all == i)
            if mask.any():
                gt_cls_i = cls_all[mask].view(-1, 1)
                gt_box_i = gt_xyxy_pix[mask]
                target_i = torch.cat([gt_cls_i, gt_box_i], dim=1)

                metric = util.compute_metric(output[:, :6], target_i, iou_v)
                metrics.append((metric, output[:, 4], output[:, 5], gt_cls_i.squeeze(-1)))

                gts_accum.append({
                    "image_id": global_img_idx,
                    "boxes": gt_box_i.clone(),
                    "labels": gt_cls_i.squeeze(1).long().clone(),
                    "size": (h, w)
                })
            else:
                metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=samples.device)
                metrics.append((metric, *torch.zeros((2, 0), device=samples.device), torch.zeros((0,), device=samples.device)))
                gts_accum.append({
                    "image_id": global_img_idx,
                    "boxes": torch.zeros((0, 4), device=samples.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=samples.device),
                    "size": (h, w)
                })

            if output is not None and output.numel():
                preds_accum.append({
                    "image_id": global_img_idx,
                    "boxes": output[:, 0:4].clone(),
                    "scores": output[:, 4].clone(),
                    "labels": output[:, 5].long().clone()
                })
            else:
                preds_accum.append({
                    "image_id": global_img_idx,
                    "boxes": torch.zeros((0, 4), device=samples.device),
                    "scores": torch.zeros((0,), device=samples.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=samples.device)
                })

            global_img_idx += 1

    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)] if len(metrics) else None
    if metrics and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics, plot=False, names=params["names"])

    coco_res = None
    if len(preds_accum) and len(gts_accum):
        coco_res = _coco_eval_from_accum(preds_accum, gts_accum, num_classes)
        if coco_res is None:
            print("[COCO] pycocotools 未安装，跳过 COCO mAP。")
        else:
            print(f"[COCO] mAP@[.5:.95]: {coco_res['mAP@[.5:.95]']:.3f} | "
                  f"AP@0.5: {coco_res['AP@0.5']:.3f} | AP@0.75: {coco_res['AP@0.75']:.3f}")

    print(("%10s" + "%10.3g" * 4) % ("", m_pre, m_rec, map50, mean_ap))
    return float(mean_ap), float(map50), float(m_rec), float(m_pre), coco_res


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--epochs", default=400, type=int)

    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument("--data-root", default="/root/autodl-tmp/racketpose2.0/data", type=str)

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    # ===== 新增：加载权重选项 =====
    parser.add_argument("--pretrained", type=str, default=None, help="只加载模型权重用于 warm start（不恢复优化器）")
    parser.add_argument("--resume", type=str, default=None, help="从某个 last.pt/best.pt 继续训练（尝试恢复 epoch/optim/scaler）")
    parser.add_argument("--no-strict", action="store_true", help="非严格加载（默认：会自动跳过 shape 不匹配的层）")

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
    run_root = os.path.join("runs", f"det-{ts}")
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
        test(args, params)

    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
