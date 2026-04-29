import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Iterable, Mapping

from torchvision.ops import batched_nms
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ---------- 坐标变换 ----------
def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def box_xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = b.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w  = (x2 - x1).clamp_min(1e-6)
    h  = (y2 - y1).clamp_min(1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)

# ---------- IoU / GIoU ----------
def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a:[Na,4], b:[Nb,4]
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp_min(1e-12)

def generalized_box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a:[Na,4], b:[Nb,4], xyxy
    iou = box_iou_xyxy(a, b)
    tl = torch.min(a[:, None, :2], b[None, :, :2])
    br = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    area_c = wh[..., 0] * wh[..., 1]  # C enclosing area
    tl2 = torch.max(a[:, None, :2], b[None, :, :2])
    br2 = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh2 = (br2 - tl2).clamp(min=0)
    inter = wh2[..., 0] * wh2[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    giou = iou - (area_c - union) / area_c.clamp_min(1e-12)
    return giou

def giou_loss_xyxy(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor, reduction='mean'):
    # pred/tgt: [...,4]
    Na = pred_xyxy.shape[0]
    if Na == 0:
        return pred_xyxy.sum() * 0.0
    giou = generalized_box_iou(pred_xyxy, tgt_xyxy).diagonal()  # pairwise aligned
    loss = 1.0 - giou
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def iou_loss_xyxy(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor, reduction='mean'):
    if pred_xyxy.shape[0] == 0:
        return pred_xyxy.sum() * 0.0
    iou = box_iou_xyxy(pred_xyxy, tgt_xyxy).diagonal()
    loss = 1.0 - iou
    return loss.mean() if reduction == 'mean' else loss.sum()

# ---------- 简易 NMS（不依赖 torchvision.ops） ----------
def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float):
    idxs = scores.argsort(descending=True)
    keep = []
    while idxs.numel() > 0:
        i = idxs[0].item()
        keep.append(i)
        if idxs.numel() == 1:
            break
        cur = boxes[i].unsqueeze(0)         # [1,4]
        rest = boxes[idxs[1:]]              # [N-1,4]
        ious = box_iou_xyxy(cur, rest).squeeze(0)  # [N-1]
        idxs = idxs[1:][ious <= iou_thr]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

# ---------- mAP 计算（VOC/COCO） ----------
def compute_ap(recall: torch.Tensor, precision: torch.Tensor) -> float:
    # 插值法求 PR 曲线面积（近似 COCO 的 AUC）
    mrec = torch.cat([torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)])
    mpre = torch.cat([torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    # 查分求面积
    i = (mrec[1:] != mrec[:-1]).nonzero().squeeze(1)
    ap = ( (mrec[i + 1] - mrec[i]) * mpre[i + 1] ).sum().item()
    return float(ap)

def eval_map_per_class(
    preds: List[Dict], gts: List[Dict], num_classes: int, iou_thr_list: List[float]
) -> Dict[str, float]:
    """
    preds/gts:
      - 按图像聚合后的列表，每一项是 dict：
        preds: {"image_id": id, "boxes": Tensor[N,4](xyxy,像素), "scores": Tensor[N], "labels": Long[N]}
        gts  : {"image_id": id, "boxes": Tensor[M,4](xyxy,像素), "labels": Long[M]}
    返回:
      {"mAP@0.5": x, "mAP@[.5:.95]": y}
    """
    iou_thr_tensor = torch.tensor(iou_thr_list, dtype=torch.float32)
    device = preds[0]["boxes"].device if preds and preds[0]["boxes"].numel() else torch.device("cpu")

    aps_thr = []
    ap50_list = []

    for thr in iou_thr_list:
        ap_c_sum = 0.0
        valid_c  = 0
        for c in range(num_classes):
            # 收集该类的预测与 GT
            cls_preds = []
            cls_gts = {}
            for p in preds:
                m = (p["labels"] == c)
                if m.any():
                    boxes = p["boxes"][m]
                    scores= p["scores"][m]
                    imgid = p["image_id"]
                    for i in range(boxes.size(0)):
                        cls_preds.append((imgid, scores[i].item(), boxes[i]))
            # GT dict: image_id -> boxes for class c, and matched flags
            for g in gts:
                m = (g["labels"] == c)
                if m.any():
                    cls_gts.setdefault(g["image_id"], [])
                    cls_gts[g["image_id"]].append(g["boxes"][m])
            # flatten GT per image
            gt_per_img = {}
            for imgid, lst in cls_gts.items():
                b = torch.cat(lst, dim=0)
                gt_per_img[imgid] = {"boxes": b, "matched": torch.zeros((b.size(0),), dtype=torch.bool, device=b.device)}

            if len(cls_preds) == 0:
                continue
            # 按分数排序
            cls_preds.sort(key=lambda x: x[1], reverse=True)
            tp = torch.zeros((len(cls_preds),), device=device)
            fp = torch.zeros((len(cls_preds),), device=device)
            for i, (imgid, score, box) in enumerate(cls_preds):
                if imgid not in gt_per_img:
                    fp[i] = 1.0
                    continue
                gt_boxes = gt_per_img[imgid]["boxes"]
                matched = gt_per_img[imgid]["matched"]
                if gt_boxes.numel() == 0:
                    fp[i] = 1.0
                    continue
                ious = box_iou_xyxy(box.unsqueeze(0), gt_boxes).squeeze(0)  # [Ng]
                max_iou, j = ious.max(dim=0)
                if max_iou.item() >= thr and not matched[j]:
                    tp[i] = 1.0
                    matched[j] = True
                else:
                    fp[i] = 1.0
            # PR & AP
            if (tp.sum() + fp.sum()) == 0:
                continue
            tp_cum = torch.cumsum(tp, dim=0)
            fp_cum = torch.cumsum(fp, dim=0)
            rec = tp_cum / max(1.0, sum( (gt_per_img[k]["boxes"].size(0) for k in gt_per_img) ))
            prec= tp_cum / torch.clamp(tp_cum + fp_cum, min=1.0)
            ap_c = compute_ap(rec, prec)
            ap_c_sum += ap_c
            valid_c += 1
        mAP_thr = (ap_c_sum / max(1, valid_c)) if valid_c > 0 else 0.0
        aps_thr.append(mAP_thr)
        if abs(thr - 0.5) < 1e-6:
            ap50_list.append(mAP_thr)

    coco_map = float(sum(aps_thr) / max(1, len(aps_thr)))
    voc_map50 = ap50_list[0] if ap50_list else 0.0
    return {"mAP@0.5": voc_map50, "mAP@[.5:.95]": coco_map}



def decode_and_nms_one_image(
    pred_boxes_b,      # (4, Hf, Wf)  cxcywh in [0,1]
    pred_logits_b,     # (K, Hf, Wf)  sigmoid
    pred_obj_b,        # (1, Hf, Wf)  sigmoid
    img_h, img_w,
    conf_thresh=0.03,
    iou_thr=0.5,
    max_dets=300,
    per_class_topk=1000,   # 每类预筛 topK（降复杂度）
):
    K, Hf, Wf = pred_logits_b.shape
    # 平铺
    pb = pred_boxes_b.permute(1, 2, 0).reshape(-1, 4)     # (Hf*Wf, 4) cxcywh
    # 转 xyxy 像素
    cxcywh = pb
    x1y1x2y2 = torch.empty_like(cxcywh)
    x1y1x2y2[:, 0] = (cxcywh[:, 0] - 0.5*cxcywh[:, 2]) * img_w
    x1y1x2y2[:, 1] = (cxcywh[:, 1] - 0.5*cxcywh[:, 3]) * img_h
    x1y1x2y2[:, 2] = (cxcywh[:, 0] + 0.5*cxcywh[:, 2]) * img_w
    x1y1x2y2[:, 3] = (cxcywh[:, 1] + 0.5*cxcywh[:, 3]) * img_h
    x1y1x2y2 = x1y1x2y2.clamp(min=0)

    obj = pred_obj_b.reshape(-1)                           # (Hf*Wf)
    boxes_all, scores_all, labels_all = [], [], []

    # 先按类阈值+TopK做预筛，减少 NMS 复杂度
    for c in range(K):
        cls = pred_logits_b[c].reshape(-1)                 # (Hf*Wf)
        sc = (cls * obj)                                   # per-class confidence
        m = sc >= conf_thresh
        if not m.any():
            continue
        sc = sc[m]
        bxs = x1y1x2y2[m]
        if per_class_topk is not None and sc.numel() > per_class_topk:
            topk = torch.topk(sc, per_class_topk)
            sc = topk.values
            bxs = bxs[topk.indices]
        boxes_all.append(bxs)
        scores_all.append(sc)
        labels_all.append(torch.full((sc.numel(),), c, dtype=torch.long, device=sc.device))

    if len(boxes_all) == 0:
        return (torch.zeros((0,4), device=pred_boxes_b.device),
                torch.zeros((0,), device=pred_boxes_b.device),
                torch.zeros((0,), dtype=torch.long, device=pred_boxes_b.device))

    boxes_all  = torch.cat(boxes_all,  dim=0)  # (N,4) xyxy
    scores_all = torch.cat(scores_all, dim=0)  # (N,)
    labels_all = torch.cat(labels_all, dim=0)  # (N,)

    keep = batched_nms(boxes_all, scores_all, labels_all, iou_thr)
    keep = keep[:max_dets]
    return boxes_all[keep], scores_all[keep], labels_all[keep]





def _xyxy_to_xywh(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    w = (x2 - x1).clamp_min(0.0)
    h = (y2 - y1).clamp_min(0.0)
    return torch.stack([x1, y1, w, h], dim=-1)

def _build_coco_gt(gts: List[Dict], num_classes: int, category_start: int = 1) -> "COCO":
    """
    gts[i] = {"image_id": int, "boxes": Float[M,4] (xyxy, 像素), "labels": Long[M]}
    """
    assert COCO is not None, "pycocotools 未安装，请先: pip install pycocotools"

    dataset = {
        "info": {"description": "auto-generated GT for evaluation"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": cid, "name": f"cls_{cid}"} 
                       for cid in range(category_start, category_start + num_classes)]
    }
    ann_id = 1
    seen_img = set()
    for g in gts:
        img_id = int(g["image_id"])
        if img_id not in seen_img:
            dataset["images"].append({"id": img_id})
            seen_img.add(img_id)

        boxes = g["boxes"]
        labels = g["labels"]
        if boxes is None or boxes.numel() == 0:
            continue
        boxes = boxes.detach().cpu().float()
        labels = labels.detach().cpu().long()

        xywh = _xyxy_to_xywh(boxes)
        for i in range(xywh.size(0)):
            x, y, w, h = xywh[i].tolist()
            # label(0..K-1) -> COCO category_id(1..K)
            cid = int(labels[i].item()) + (category_start - 0)
            cid = cid + (1 - category_start) + category_start  # 保持表达清晰，实际就是 + (category_start)
            cid = int(labels[i].item()) + category_start
            dataset["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "iscrowd": 0,
                "area": float(max(w, 0.0) * max(h, 0.0)),
                "bbox": [float(x), float(y), float(w), float(h)],
            })
            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    return coco_gt

def _build_coco_dt_from_preds(coco_gt: "COCO", preds: List[Dict], category_start: int = 1) -> List[Dict]:
    """
    preds[i] = {"image_id": int, "boxes": Float[N,4] (xyxy, 像素), "scores": Float[N], "labels": Long[N]}
    返回可直接给 coco_gt.loadRes 的 list[dict]
    """
    results = []
    for p in preds:
        img_id = int(p["image_id"])
        boxes = p["boxes"]; scores = p["scores"]; labels = p["labels"]
        if boxes is None or boxes.numel() == 0:
            continue
        boxes  = boxes.detach().cpu().float()
        scores = scores.detach().cpu().float()
        labels = labels.detach().cpu().long()

        xywh = _xyxy_to_xywh(boxes)
        for i in range(xywh.size(0)):
            x, y, w, h = xywh[i].tolist()
            sc  = float(scores[i].item())
            cid = int(labels[i].item()) + category_start
            results.append({
                "image_id": img_id,
                "category_id": cid,
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": sc
            })
    return results

def eval_map_coco(
    preds: List[Dict],
    gts: List[Dict],
    num_classes: int,
    iou_type: str = "bbox",
    iou_thrs: Optional[List[float]] = None,  # 例如只算 VOC@0.5: [0.5]
    max_dets: Optional[List[int]] = None,    # 例如固定 [100]
    category_start: int = 1,                 # COCO 类别 ID 起始(默认1)
) -> Dict[str, float]:
    """
    用 pycocotools 评估 COCO 指标。
    返回(按 COCO 官方 summarize 含义)：
      stats[0] AP @[.5:.95]  area=all, maxDets=100
      stats[1] AP @0.5
      stats[2] AP @0.75
      stats[6] AR @100
    """
    assert COCO is not None, "pycocotools 未安装，请先: pip install pycocotools"

    coco_gt = _build_coco_gt(gts, num_classes, category_start=category_start)
    coco_dt_list = _build_coco_dt_from_preds(coco_gt, preds, category_start=category_start)
    if len(coco_dt_list) == 0:
        return {"mAP@[.5:.95]": 0.0, "mAP@0.5": 0.0, "mAP@0.75": 0.0, "AR@100": 0.0}

    coco_dt = coco_gt.loadRes(coco_dt_list)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    # 覆盖 IoU / maxDets（可选）
    if iou_thrs is not None and len(iou_thrs) > 0:
        import numpy as np
        coco_eval.params.iouThrs = np.array(iou_thrs, dtype=float)
    if max_dets is not None and len(max_dets) > 0:
        coco_eval.params.maxDets = max_dets

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "mAP@[.5:.95]": float(stats[0]),
        "mAP@0.5": float(stats[1]),
        "mAP@0.75": float(stats[2]),
        "AR@100": float(stats[6]),
    }





def _average_precision_from_scores(y_true_bool, y_socre):
    """按分数降序，AP = 正样位置的 precision 之和 / 正样个数""" 
    Np = int(y_true_bool.sum().item()) 
    if Np == 0:
        return 0.0 
    
    order = torch.argsort(y_socre, descending=True) 
    t = y_true_bool[order] 
    tp = torch.cumsum(t.to(torch.float32), dim=0) 
    fp = torch.cumsum((~t).to(torch.float32), dim=0) 
    precsion = tp / torch.clamp(tp + fp, min=1.0) 
    ap = (precsion[t].sum() / max(1, Np)).item() 
    return float(ap)


def eval_angle_map_by_thresholds_det(
    preds,
    gts,
    num_classes,
    angle_ok_by_img,
    thresholds=(1.0, 2.0, 5.0, 10.0)
):
    """
    基于“检测头”的 mAP@角度（检测框×类别 评分 = 置信度分数；正负由 角度达标 & 是否有该类GT 决定）
    - preds[i] = {"image_id": int, "boxes": Float[N,4] (xyxy, 像素), "scores": Float[N], "labels": Long[N]}
    - gts[i] = {"image_id": int, "boxes": Float[M,4] (xyxy, 像素), "labels": Long[M]}
    - angle_ok_by_img: {image_id -> {thr(float)->bool}}  # 该图在阈值下角度是否“合格”
    - thresholds: 角度阈值（度）

    返回: { "mAP@angle1": x, "mAP@angle2": y, ... }
    """
    # 收集 image_ids
    image_ids = set()
    for p in preds: image_ids.add(int(p["image_id"]))
    for g in gts:   image_ids.add(int(g["image_id"]))
    if not image_ids:
        return {f"mAP@angle{int(t)}": 0.0 for t in thresholds}
    image_ids = sorted(image_ids)

    # 每图每类最高分（来自检测头）
    cls_topscore_by_img: Dict[int, Dict[int, float]] = {}
    for p in preds:
        imgid = int(p["image_id"])
        tops = cls_topscore_by_img.setdefault(imgid, {})
        if p["scores"] is None or p["labels"] is None or p["scores"].numel() == 0:
            continue
        labels = p["labels"].detach().cpu().long()
        scores = p["scores"].detach().cpu().float()
        for cid in labels.unique().tolist():
            mask = (labels == cid)
            if mask.any():
                smax = float(scores[mask].max().item())
                tops[cid] = max(tops.get(cid, 0.0), smax)
    for imgid in image_ids:
        cls_topscore_by_img.setdefault(imgid, {})

    # 每图 GT 类集合
    gt_classes_by_img: Dict[int, set] = {}
    for g in gts:
        imgid = int(g["image_id"])
        labs = g.get("labels", None)
        s = gt_classes_by_img.setdefault(imgid, set())
        if labs is not None and labs.numel() > 0:
            for cid in labs.detach().cpu().long().unique().tolist():
                s.add(int(cid))
    for imgid in image_ids:
        gt_classes_by_img.setdefault(imgid, set())

    # 逐阈值计算 mAP（宏平均各类）
    result = {}
    for thr in thresholds:
        ap_sum, valid_c = 0.0, 0
        for c in range(num_classes):
            scores_c, ytrue_c = [], []
            for imgid in image_ids:
                score   = float(cls_topscore_by_img[imgid].get(c, 0.0))
                has_gt  = (c in gt_classes_by_img[imgid])
                ok      = bool(angle_ok_by_img.get(imgid, {}).get(float(thr), False))
                scores_c.append(score)
                ytrue_c.append(1 if (has_gt and ok) else 0)
            t = torch.tensor(ytrue_c, dtype=torch.bool)
            s = torch.tensor(scores_c, dtype=torch.float32)
            ap_c = _average_precision_from_scores(t, s)
            ap_sum += ap_c; valid_c += 1
        result[f"mAP@angle{int(thr)}"] = float(ap_sum / max(1, valid_c)) if valid_c > 0 else 0.0
    return result



def eval_angle_map_by_thresholds_img(
    img_scores_by_img,
    gts,
    num_classes,
    angle_ok_by_img,
    thresholds,
    use_softmax=False,
):
    """
    基于“图像级分类头”的 mAP@角度（图像×类别 评分 = 图像级分类分数；正负由 角度达标 & 是否有该类GT 决定）
    - img_scores_by_img: {image_id -> Tensor[C]}  (logits 或 prob)
    - gts[i] : {"image_id": int, "boxes": Float[M,4], "labels": Long[M]}  # 用于判定该图是否含某类 GT
    - angle_ok_by_img: {image_id -> {thr(float)->bool}}  # 该图在阈值下角度是否“合格”
    - thresholds: 角度阈值（度）

    返回: { "mAP@angle1": x, "mAP@angle2": y, ... }
    """
    
    # 收集image_id全部
    image_ids = set(img_scores_by_img.keys()) 
    for g in gts:
        image_ids.add(int(g["image_id"]))
    image_ids = sorted(image_ids)
    if len(image_ids) == 0:
        return {f"mAP@{thr}": 0.0 for thr in thresholds} 
    
    # 每图gt类集合
    gt_classes_by_img = {} 
    for g in gts:
        imgid = int(g["image_id"]) 
        labs = g.get("labels", None) 
        s = gt_classes_by_img.setdefault(imgid, set()) 
        if labs is not None and labs.numel() > 0:
            for cid in labs.detach().cpu().long().unique().tolist():
                s.add(int(cid))
    for imgid in image_ids:
        gt_classes_by_img.setdefault(imgid, set())
    
    # 保证每图都有分数向量（没有就置0）
    # 如 use_softmax=True，则先做 softmax
    zero_vec = None
    for imgid in image_ids:
        vec = img_scores_by_img.get(imgid, None)
        if vec is None:
            if zero_vec is None:
                zero_vec = torch.zeros((num_classes,), dtype=torch.float32)
            img_scores_by_img[imgid] = zero_vec.clone()
        else:
            v = vec.detach().cpu().float()
            if v.numel() != num_classes:
                raise ValueError(f"img {imgid} scores dim {v.numel()} != num_classes {num_classes}")
            img_scores_by_img[imgid] = v.softmax(dim=-1) if use_softmax else v

    # 计算各阈值的 mAP（宏平均各类）
    result = {}
    for thr in thresholds:
        ap_sum, valid_c = 0.0, 0
        for c in range(num_classes):
            scores_c = []
            ytrue_c  = []
            for imgid in image_ids:
                score = float(img_scores_by_img[imgid][c].item())
                has_gt_c = (c in gt_classes_by_img[imgid])
                angle_ok = bool(angle_ok_by_img.get(imgid, {}).get(float(thr), False))
                scores_c.append(score)
                ytrue_c.append(1 if (has_gt_c and angle_ok) else 0)
            t = torch.tensor(ytrue_c, dtype=torch.bool)
            s = torch.tensor(scores_c, dtype=torch.float32)
            ap_c = _average_precision_from_scores(t, s)
            ap_sum += ap_c
            valid_c += 1
        result[f"mAP@angle{int(thr)}"] = float(ap_sum / max(1, valid_c)) if valid_c > 0 else 0.0

    return result    