#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, random, argparse
from types import SimpleNamespace as NS
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader

from boxes_utils import box_cxcywh_to_xyxy, giou_loss_xyxy, nms_xyxy, eval_map_per_class, decode_and_nms_one_image, eval_map_coco, eval_angle_map_by_thresholds_img, eval_angle_map_by_thresholds_det

class AccHistory:
    def __init__(self):
        self.val_acc10 = []
        self.best_val_acc10 = float("-inf")
        self.best_epoch = None

    def append_acc(self, epoch, acc10: float):
        self.val_acc10.append((epoch, acc10))
        # if acc10 > self.best_val_acc10:
        #     self.best_val_acc10 = acc10
        #     self.best_epoch = epoch


acc_history = AccHistory()

def multi_task_fit_one_epoch(
    model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
    gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0,
    acc_thresholds=(1.0, 2.0, 5.0, 10.0),
    # ---- 检测相关系数（可调）----
    det_lambda_obj: float = 1.0,
    det_lambda_cls: float = 1.0,
    det_lambda_box: float = 5.0,
    det_neg_weight: float = 0.05,   # 负样本 obj 权重
):

    # ----------------- utils -----------------
    def _deg_to_norm(deg): return (deg + 180.0) / 360.0
    def _norm_to_deg(x):   return x * 360.0 - 180.0
    def _wrap_diff_deg(pred_deg, tgt_deg):
        return (pred_deg - tgt_deg + 180.0).remainder(360.0) - 180.0
    def _huber(x, y, delta):
        d = (x - y).abs()
        return torch.where(d < delta, 0.5 * (d**2) / delta, d - 0.5 * delta).mean()
    def _circular_huber_on_norm(pred_norm, tgt_norm, delta=0.05):
        d = (pred_norm - tgt_norm).abs()
        d = torch.minimum(d, 1.0 - d)
        return torch.where(d < delta, 0.5 * (d**2) / delta, d - 0.5 * delta).mean()
    def _log1p(x): return torch.log1p(torch.clamp_min(x, 0.0))
    def _expm1_clamp(x): return torch.expm1(x).clamp_min(0.0)
    def _acc_thresholds_angles(abs_err_deg, thresholds):
        tot = abs_err_deg.numel()
        return {f"acc@{int(t)}": float((abs_err_deg <= t).sum().item()) / max(1, tot) for t in thresholds}
    def get_lr(opt):
        for pg in opt.param_groups:
            if 'lr' in pg: return pg['lr']
        return None

    # mAP: per-class AP（one-vs-rest），scores:[N,K] (softmax prob 或 logits均可，推荐prob)，labels:[N] int
    # 此mAP仅分类模式用
    def _average_precision(y_true_bool: torch.Tensor, y_score: torch.Tensor) -> float:
        # y_true_bool: [N] bool; y_score: [N] float
        Np = int(y_true_bool.sum().item())
        if Np == 0:
            return 0.0
        order = torch.argsort(y_score, descending=True)
        y_true_sorted = y_true_bool[order]
        tp = torch.cumsum(y_true_sorted.to(torch.float32), dim=0)
        fp = torch.cumsum((~y_true_sorted).to(torch.float32), dim=0)
        prec = tp / torch.clamp(tp + fp, min=1.0)
        # AP = sum over positions where y_true=1 of precision / num_pos
        ap = (prec[y_true_sorted].sum() / max(1, Np)).item()
        return float(ap)

    def _compute_map(all_scores: torch.Tensor, all_labels: torch.Tensor):
        # all_scores: [N,K] (prob)，all_labels:[N] long
        K = all_scores.size(1)
        ap_list = []
        per_class_ap = {}
        for c in range(K):
            y_true = (all_labels == c)
            ap_c = _average_precision(y_true, all_scores[:, c])
            ap_list.append(ap_c)
            per_class_ap[c] = ap_c
        # 宏平均（包含无样本类时会是0，这里保留原样）
        mAP = float(sum(ap_list) / len(ap_list)) if len(ap_list) > 0 else 0.0
        return mAP, per_class_ap
    
    # 检测目标构建
    def _boxes_to_targets(
        boxes_list, labels_list, feat_hw, img_hw, num_classes=4
    ):
        """
        输入:
          - boxes_list: List[Tensor[Ni,4]]  每个为 xyxy（像素 或 0~1 归一化）
          - labels_list: List[Tensor[Ni]]
          - feat_hw: (H, W)  检测头输出网格大小
          - img_hw : (H_img, W_img)  输入图片大小（用于像素->归一化）
          - num_classes: int  类别数
        输出:
          obj_t:  [B, 1, H, W]  {0,1}
          cls_t:  [B, K, H, W]  one-hot（仅正样处为1）
          box_t:  [B, 4, H, W]  目标(cx,cy,w,h) in [0,1]（仅正样处有效）
          pos_mask: [B,1,H,W]  正样 mask
        """
        B = len(boxes_list) 
        H, W = feat_hw 
        K = num_classes
        
        obj_t   = torch.zeros((B, 1, H, W), dtype=torch.float32, device=boxes_list[0].device)
        cls_t   = torch.zeros((B, K, H, W), dtype=torch.float32, device=obj_t.device)
        box_t   = torch.zeros((B, 4, H, W), dtype=torch.float32, device=obj_t.device)
        pos_m   = torch.zeros((B, 1, H, W), dtype=torch.bool,    device=obj_t.device)
        
        Himg, Wimg = img_hw 
        for b in range(B):
            boxes = boxes_list[b] #[Ni, 4] 
            labs = labels_list[b] #[Ni] 
            if boxes.numel() == 0: 
                continue 
            # 像素 -> 归一化（空框安全）
            if boxes.numel() > 0 and boxes.max().item() > 1.5:
                boxes_norm = boxes.clone()
                boxes_norm[:, [0, 2]] /= float(Wimg)
                boxes_norm[:, [1, 3]] /= float(Himg)
            else:
                boxes_norm = boxes

            
            # xyxy -> cxcywh 
            cx = (boxes_norm[:,0] + boxes_norm[:,2]) * 0.5 
            cy = (boxes_norm[:,1] + boxes_norm[:,3]) * 0.5 
            w = (boxes_norm[:,2] - boxes_norm[:,0]).clamp_min(1e-6) 
            h = (boxes_norm[:,3] - boxes_norm[:,1]).clamp_min(1e-6)     
            
            # 量化到网格
            gx = torch.clamp((cx * W).long(), 0, W-1)
            gy = torch.clamp((cy * H).long(), 0, H-1)

            for i in range(cx.size(0)):
                y, x = int(gy[i].item()), int(gx[i].item())
                obj_t[b, 0, y, x] = 1.0
                pos_m[b, 0, y, x] = True
                cls_id = int(labs[i].item())
                if 0 <= cls_id < K:
                    cls_t[b, cls_id, y, x] = 1.0
                box_t[b, :, y, x] = torch.tensor([cx[i], cy[i], w[i], h[i]], device=box_t.device)

        return obj_t, cls_t, box_t, pos_m
        
        
        

    # --------- 聚合量 ---------
    # 角度
    tr_ang_abs_sum = va_ang_abs_sum = 0.0
    tr_ang_sq_sum  = va_ang_sq_sum  = 0.0
    tr_ang_cnt_el  = va_ang_cnt_el  = 0
    tr_acc_cnts = {f"acc@{int(t)}": 0.0 for t in acc_thresholds}
    va_acc_cnts = {f"acc@{int(t)}": 0.0 for t in acc_thresholds}
    # 距离
    tr_d_abs_sum = va_d_abs_sum = 0.0
    tr_d_sq_sum  = va_d_sq_sum  = 0.0
    tr_d_cnt     = va_d_cnt     = 0
    # 分类（为算 mAP 需要缓存全量scores/labels）
    tr_scores, tr_labels = [], []
    va_scores, va_labels = [], []
    # Loss
    tr_loss_sum = va_loss_sum = 0.0

    # --------- Train ---------
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    is_detect = bool(getattr(model_train, "detect", False))
    
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        # # 支持 (imgs, reg, cls) / dict
        # if isinstance(batch, dict):
        #     imgs = batch.get("pixel_values", batch.get("images"))
        #     reg  = batch.get("reg")
        #     cls  = batch.get("cls")
        # else:
        #     if len(batch) == 3:
        #         imgs, reg, cls = batch
        #     else:
        #         imgs, pair = batch
        #         reg, cls = pair
        # 解析 batch：支持 (imgs, reg, cls[, boxes, box_labels])
        if isinstance(batch, dict):
            imgs = batch.get("pixel_values", batch.get("images"))
            reg  = batch.get("reg")
            cls  = batch.get("cls")
            boxes_list = batch.get("boxes")
            box_labels_list = batch.get("box_labels", cls)
        else:
            if len(batch) == 5:
                imgs, reg, cls, boxes_list, box_labels_list = batch
            elif len(batch) == 4:
                imgs, reg, cls, boxes_list = batch
                box_labels_list = cls
            elif len(batch) == 3:
                imgs, reg, cls = batch
                boxes_list = box_labels_list = None
            else:
                imgs, pair = batch
                reg, cls = pair
                boxes_list = box_labels_list = None

        # labels 整理
        dist_t = reg[:, 0]     # [B]
        ang_t  = reg[:, 1:4]   # [B,3] (deg)

        # 分类标签稳健处理 -> [B] long
        if cls.ndim > 1:
            if cls.size(-1) == 1:
                cls = cls.squeeze(-1)
            else:
                cls = cls.argmax(dim=-1)
        cls = cls.long()

        if cuda:
            imgs   = imgs.cuda(local_rank, non_blocking=True)
            dist_t = dist_t.cuda(local_rank, non_blocking=True)
            ang_t  = ang_t.cuda(local_rank, non_blocking=True)
            cls    = cls.cuda(local_rank, non_blocking=True)
            if boxes_list is not None:
                boxes_list = [b.cuda(local_rank, non_blocking=True) for b in boxes_list]
            if box_labels_list is not None and isinstance(box_labels_list, (list, tuple)):
                box_labels_list = [l.cuda(local_rank, non_blocking=True) for l in box_labels_list]

        optimizer.zero_grad(set_to_none=True)

        if not fp16:
            out = model_train(imgs)
            # if isinstance(out, dict):
            #     logits, angle_norm, dist_log = out["logits"], out["angle_norm"], out["dist_log"]
            # else:
            #     raise RuntimeError("Model must return a dict with keys: logits, angle_norm, dist_log")
            # loss_cls  = F.cross_entropy(logits, cls)
            # loss_ang  = _circular_huber_on_norm(angle_norm, _deg_to_norm(ang_t), delta=0.05)
            # loss_dist = _huber(dist_log.squeeze(-1), _log1p(dist_t), delta=0.2)
            # loss = loss_cls + loss_ang + loss_dist
            # loss.backward()
            # optimizer.step()
            if is_detect:
                # 检测分支 + 回归
                pred_logits = out["pred_logits"]   # (B,K,H,W)
                pred_obj    = out["pred_obj"]      # (B,1,H,W)
                pred_boxes  = out["pred_boxes"]    # (B,4,H,W)
                angle_norm  = out["angle_norm"]    # (B,3)
                dist_log    = out["dist_log"]      # (B,1)

                # 构建目标
                B, _, H, W = pred_obj.shape
                Himg = imgs.shape[2]; Wimg = imgs.shape[3]
                if boxes_list is None:
                    raise RuntimeError("detect=True 需要 dataloader 提供 boxes_list")
                if box_labels_list is None:
                    box_labels_list = [cls for _ in range(B)]

                obj_t, cls_t, box_t, pos_m = _boxes_to_targets(
                    boxes_list, box_labels_list, (H, W), (Himg, Wimg)
                )

                # 损失
                # obj：全图 BCE（负样本放缩）
                # 权重用 float
                pos = pos_m.float()
                neg = (~pos_m).float()
                wmap = pos + det_neg_weight * neg
                loss_obj = F.binary_cross_entropy(pred_obj, obj_t, weight=wmap, reduction='sum') / wmap.sum().clamp_min(1.0)

                # 索引用 bool 掩码
                pos_any = pos_m.any()
                if pos_any:
                    cls_mask  = pos_m.expand_as(pred_logits)   # [B,1,H,W] -> [B,K,H,W]  (bool)
                    box_mask  = pos_m.expand_as(pred_boxes)    # [B,1,H,W] -> [B,4,H,W]  (bool)

                    loss_cls_det = F.binary_cross_entropy(
                        pred_logits[cls_mask],
                        cls_t[cls_mask],
                        reduction='mean'
                    )

                    pb = pred_boxes[box_mask].view(-1, 4)      # [Npos,4], in [0,1]
                    tb = box_t[box_mask].view(-1, 4)
                    pb_xyxy = box_cxcywh_to_xyxy(pb)
                    tb_xyxy = box_cxcywh_to_xyxy(tb)
                    scale = torch.tensor([Wimg, Himg, Wimg, Himg], device=pb.device, dtype=pb.dtype)
                    pb_xyxy_pix = pb_xyxy * scale
                    tb_xyxy_pix = tb_xyxy * scale
                    loss_box = giou_loss_xyxy(pb_xyxy_pix, tb_xyxy_pix, reduction='mean')
                else:
                    loss_cls_det = pred_logits.sum() * 0.0
                    loss_box = pred_boxes.sum() * 0.0


                loss_ang  = _circular_huber_on_norm(angle_norm, _deg_to_norm(ang_t), delta=0.05)
                loss_dist = _huber(dist_log.squeeze(-1), _log1p(dist_t), delta=0.2)

                loss = det_lambda_obj * loss_obj + det_lambda_cls * loss_cls_det + det_lambda_box * loss_box \
                       + loss_ang + loss_dist
                loss.backward()
                optimizer.step()
        else:
            # from torch.cuda.amp import autocast
            # with autocast():
            #     out = model_train(imgs)
            #     if isinstance(out, dict):
            #         logits, angle_norm, dist_log = out["logits"], out["angle_norm"], out["dist_log"]
            #     else:
            #         raise RuntimeError("Model must return a dict with keys: logits, angle_norm, dist_log")
            #     loss_cls  = F.cross_entropy(logits, cls)
            #     loss_ang  = _circular_huber_on_norm(angle_norm, _deg_to_norm(ang_t), delta=0.05)
            #     loss_dist = _huber(dist_log.squeeze(-1), _log1p(dist_t), delta=0.2)
            #     loss = loss_cls + loss_ang + loss_dist
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            with autocast(device_type="cuda"):
                out = model_train(imgs)
                if not isinstance(out, dict):
                    raise RuntimeError("Model must return a dict")

                if is_detect:
                    pred_logits = out["pred_logits"]; pred_obj = out["pred_obj"]; pred_boxes = out["pred_boxes"]
                    angle_norm  = out["angle_norm"];  dist_log  = out["dist_log"]
                    B, _, H, W = pred_obj.shape
                    Himg = imgs.shape[2]; Wimg = imgs.shape[3]
                    if boxes_list is None:
                        raise RuntimeError("detect=True 需要 dataloader 提供 boxes_list")
                    if box_labels_list is None:
                        box_labels_list = [cls for _ in range(B)]
                    obj_t, cls_t, box_t, pos_m = _boxes_to_targets(
                        boxes_list, box_labels_list, (H, W), (Himg, Wimg)
                    )
                    
                    pos = pos_m.float()
                    neg = (~pos_m).float()
                    wmap = pos + det_neg_weight * neg
                    loss_obj = F.binary_cross_entropy(pred_obj, obj_t, weight=wmap, reduction='sum') / wmap.sum().clamp_min(1.0)

                    pos_any = pos_m.any()
                    if pos_any:
                        cls_mask = pos_m.expand_as(pred_logits)
                        box_mask = pos_m.expand_as(pred_boxes)

                        loss_cls_det = F.binary_cross_entropy(
                            pred_logits[cls_mask],
                            cls_t[cls_mask],
                            reduction='mean'
                        )

                        pb = pred_boxes[box_mask].view(-1,4)
                        tb = box_t[box_mask].view(-1,4)
                        pb_xyxy = box_cxcywh_to_xyxy(pb)
                        tb_xyxy = box_cxcywh_to_xyxy(tb)
                        scale = torch.tensor([Wimg, Himg, Wimg, Himg], device=pb.device, dtype=pb.dtype)
                        pb_xyxy_pix = pb_xyxy * scale
                        tb_xyxy_pix = tb_xyxy * scale
                        loss_box = giou_loss_xyxy(pb_xyxy_pix, tb_xyxy_pix, reduction='mean')
                    else:
                        loss_cls_det = pred_logits.sum() * 0.0
                        loss_box = pred_boxes.sum() * 0.0

                    loss_ang  = _circular_huber_on_norm(angle_norm, _deg_to_norm(ang_t), delta=0.05)
                    loss_dist = _huber(dist_log.squeeze(-1), _log1p(dist_t), delta=0.2)
                    loss = det_lambda_obj * loss_obj + det_lambda_cls * loss_cls_det + det_lambda_box * loss_box \
                           + loss_ang + loss_dist
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ---- metrics (train) ----
        with torch.no_grad():
            tr_loss_sum += loss.item()

            # # 缓存分类 scores/labels 用于 mAP
            # probs = logits.softmax(dim=-1).detach().float().cpu()
            # tr_scores.append(probs)
            # tr_labels.append(cls.detach().cpu())
            # 分类 mAP 只在非检测模式下积累
            if not is_detect:
                probs = out["logits"].softmax(dim=-1).detach().float().cpu()
                tr_scores.append(probs)
                tr_labels.append(cls.detach().cpu())

            # 角度
            ang_pred = _norm_to_deg(angle_norm)  # [B,3] deg
            ang_diff = _wrap_diff_deg(ang_pred, ang_t)
            ang_abs  = ang_diff.abs()
            tr_ang_abs_sum += ang_abs.sum().item()
            tr_ang_sq_sum  += (ang_diff ** 2).sum().item()
            tr_ang_cnt_el  += ang_abs.numel()
            accs = _acc_thresholds_angles(ang_abs, list(acc_thresholds))
            for k, v in accs.items():
                tr_acc_cnts[k] += v * ang_abs.numel()

            # 距离
            dist_pred = _expm1_clamp(dist_log.squeeze(-1))
            d_abs = (dist_pred - dist_t).abs()
            tr_d_abs_sum += d_abs.sum().item()
            tr_d_sq_sum  += ((dist_pred - dist_t) ** 2).sum().item()
            tr_d_cnt     += dist_t.numel()

        if local_rank == 0:
            tr_maae_ang = tr_ang_abs_sum / max(1, tr_ang_cnt_el)
            tr_rmse_ang = math.sqrt(tr_ang_sq_sum / max(1, tr_ang_cnt_el))
            tr_mae_dist = tr_d_abs_sum / max(1, tr_d_cnt)
            tr_rmse_dist= math.sqrt(tr_d_sq_sum / max(1, tr_d_cnt))
            pbar.set_postfix(**{
                'loss': tr_loss_sum / (iteration + 1),
                'A-MAAE(deg)': f'{tr_maae_ang:.3f}',
                'A-RMSE(deg)': f'{tr_rmse_ang:.3f}',
                'D-MAE(m)': f'{tr_mae_dist:.3f}',
                'D-RMSE(m)': f'{tr_rmse_dist:.3f}',
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # --------- val ---------
    model_train.eval()
    preds_accum, gts_accum = [], []
    # 新增map@角度相关缓存
    angle_ok_by_img = dict()       # image_id -> {thr(float): bool}
    # img_scores_by_img = dict()     # image_id -> Tensor[C]  (img_logits 或 softmax)


    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break

            # if isinstance(batch, dict):
            #     imgs = batch.get("pixel_values", batch.get("images"))
            #     reg  = batch.get("reg")
            #     cls  = batch.get("cls")
            # else:
            #     if len(batch) == 3:
            #         imgs, reg, cls = batch
            #     else:
            #         imgs, pair = batch
            #         reg, cls = pair
            if isinstance(batch, dict):
                imgs = batch.get("pixel_values", batch.get("images"))
                reg  = batch.get("reg")
                cls  = batch.get("cls")
                boxes_list = batch.get("boxes")
                box_labels_list = batch.get("box_labels", cls)
            else:
                if len(batch) == 5:
                    imgs, reg, cls, boxes_list, box_labels_list = batch
                elif len(batch) == 4:
                    imgs, reg, cls, boxes_list = batch
                    box_labels_list = cls
                elif len(batch) == 3:
                    imgs, reg, cls = batch
                    boxes_list = box_labels_list = None
                else:
                    imgs, pair = batch
                    reg, cls = pair
                    boxes_list = box_labels_list = None

            dist_t = reg[:, 0]
            ang_t  = reg[:, 1:4]

            if cls.ndim > 1:
                if cls.size(-1) == 1:
                    cls = cls.squeeze(-1)
                else:
                    cls = cls.argmax(dim=-1)
            cls = cls.long()

            if cuda:
                imgs   = imgs.cuda(local_rank, non_blocking=True)
                dist_t = dist_t.cuda(local_rank, non_blocking=True)
                ang_t  = ang_t.cuda(local_rank, non_blocking=True)
                cls    = cls.cuda(local_rank, non_blocking=True)
                if boxes_list is not None:
                    boxes_list = [b.cuda(local_rank, non_blocking=True) for b in boxes_list]
                if box_labels_list is not None and isinstance(box_labels_list, (list, tuple)):
                    box_labels_list = [l.cuda(local_rank, non_blocking=True) for l in box_labels_list]


            out = model_train(imgs)
            # if isinstance(out, dict):
            #     logits, angle_norm, dist_log = out["logits"], out["angle_norm"], out["dist_log"]
            # else:
            #     raise RuntimeError("Model must return a dict with keys: logits, angle_norm, dist_log")

            # loss_cls  = F.cross_entropy(logits, cls)
            # loss_ang  = _circular_huber_on_norm(angle_norm, _deg_to_norm(ang_t), delta=0.05)
            # loss_dist = _huber(dist_log.squeeze(-1), _log1p(dist_t), delta=0.2)
            # loss = loss_cls + loss_ang + loss_dist
            # va_loss_sum += loss.item()

            # # 缓存分类 scores/labels
            # probs = logits.softmax(dim=-1).detach().float().cpu()
            # va_scores.append(probs)
            # va_labels.append(cls.detach().cpu())
            if not isinstance(out, dict):
                raise RuntimeError("Model must return a dict")

            if is_detect:
                pred_logits = out["pred_logits"]; pred_obj = out["pred_obj"]; pred_boxes = out["pred_boxes"]
                angle_norm  = out["angle_norm"];  dist_log  = out["dist_log"]
                B, _, H, W = pred_obj.shape
                Himg = imgs.shape[2]; Wimg = imgs.shape[3]
                if boxes_list is None:
                    # 没有验证框时，只评估回归损失
                    loss_obj = pred_obj.sum()*0.0
                    loss_cls_det = pred_logits.sum()*0.0
                    loss_box = pred_boxes.sum()*0.0
                else:
                    if box_labels_list is None:
                        box_labels_list = [cls for _ in range(B)]
                    obj_t, cls_t, box_t, pos_m = _boxes_to_targets(
                        boxes_list, box_labels_list, (H, W), (Himg, Wimg)
                    )
                   
                    pos = pos_m.float()
                    neg = (~pos_m).float()
                    wmap = pos + det_neg_weight * neg
                    loss_obj = F.binary_cross_entropy(pred_obj, obj_t, weight=wmap, reduction='sum') / wmap.sum().clamp_min(1.0)

                    pos_any = pos_m.any()
                    if pos_any:
                        cls_mask = pos_m.expand_as(pred_logits)
                        box_mask = pos_m.expand_as(pred_boxes)

                        loss_cls_det = F.binary_cross_entropy(
                            pred_logits[cls_mask],
                            cls_t[cls_mask],
                            reduction='mean'
                        )

                        pb = pred_boxes[box_mask].view(-1,4)
                        tb = box_t[box_mask].view(-1,4)
                        pb_xyxy = box_cxcywh_to_xyxy(pb)
                        tb_xyxy = box_cxcywh_to_xyxy(tb)
                        scale = torch.tensor([Wimg, Himg, Wimg, Himg], device=pb.device, dtype=pb.dtype)
                        pb_xyxy_pix = pb_xyxy * scale
                        tb_xyxy_pix = tb_xyxy * scale
                        loss_box = giou_loss_xyxy(pb_xyxy_pix, tb_xyxy_pix, reduction='mean')
                    else:
                        loss_cls_det = pred_logits.sum() * 0.0
                        loss_box = pred_boxes.sum() * 0.0

                loss_ang  = _circular_huber_on_norm(angle_norm, _deg_to_norm(ang_t), delta=0.05)
                loss_dist = _huber(dist_log.squeeze(-1), _log1p(dist_t), delta=0.2)
                loss = det_lambda_obj * loss_obj + det_lambda_cls * loss_cls_det + det_lambda_box * loss_box \
                       + loss_ang + loss_dist
                va_loss_sum += loss.item()
                
                # 检测分支计算完loss以后收集评估所需
                # ========= 收集检测预测用于 mAP =========
                # 解码预测为像素 xyxy
                B, K, Hf, Wf = pred_logits.shape
                # B_det = pred_logits.shape[0]                 # 来自检测头
                # B_obj = pred_obj.shape[0]
                # B_box = pred_boxes.shape[0]
                # B_ang = angle_norm.shape[0]                  # 角度回归
                # B_in  = imgs.shape[0]                        # 输入
                # B     = min(B_det, B_obj, B_box, B_ang, B_in)  # 安全上限
                
                # print(f"[VAL] B_in={B_in}, B_det={B_det}, B_obj={B_obj}, B_box={B_box}, B_ang={B_ang} "
                #     f"len(boxes_list)={len(boxes_list) if boxes_list is not None else None}, "
                #     f"len(box_labels_list)={len(box_labels_list) if box_labels_list is not None else None}")

                # # 强约束（稳定后可去掉）
                # assert B_in == B_det == B_obj == B_box == B_ang, "Batch mismatch among model outputs / input"  #已验证全部相等
                
                Himg = imgs.shape[2]; Wimg = imgs.shape[3]

                for b in range(B):
                    boxes_img, scores_img, labels_img = decode_and_nms_one_image(
                        pred_boxes[b], pred_logits[b], pred_obj[b],
                        img_h=imgs.shape[2], img_w=imgs.shape[3],
                        conf_thresh=0.05, iou_thr=0.5, max_dets=300, per_class_topk=1000
                    )
                    preds_accum.append({
                        "image_id": iteration * B + b,
                        "boxes": boxes_img,
                        "scores": scores_img,
                        "labels": labels_img,
                    })

                    # GT（xyxy 像素）
                    if boxes_list is None:
                        g_boxes = torch.zeros((0,4), device=imgs.device)
                        g_labels= torch.zeros((0,), dtype=torch.long, device=imgs.device)
                    else:
                        g_boxes = boxes_list[b]
                        if g_boxes.numel() > 0 and g_boxes.max().item() <= 1.0:
                            g_boxes = g_boxes.clone()
                            g_boxes[:, [0,2]] *= Wimg
                            g_boxes[:, [1,3]] *= Himg
                        g_labels = box_labels_list[b] if box_labels_list is not None else cls[b].repeat(g_boxes.size(0))
                    gts_accum.append({
                        "image_id": iteration * B + b,
                        "boxes": g_boxes,
                        "labels": g_labels.long()
                    })
                # ========= 收集结束 =========
                    img_id = iteration * B + b
                    ang_pred_b = _norm_to_deg(angle_norm[b])  # [3] deg 不能用训练里面的ang_pred
                    ang_t_b    = ang_t[b]
                    ang_err_b  = (ang_pred_b - ang_t_b + 180.0).remainder(360.0) - 180.0
                    ang_err_b  = ang_err_b.abs()
                    ok_map = angle_ok_by_img.setdefault(img_id, {})
                    for thr in acc_thresholds:
                        ok_map[float(thr)] = bool((ang_err_b <= thr).all().item())
                
                    # 新增map@角度
                    # img_id = iteration * B + b

                    # # 记录图像级分类分数（建议用 softmax 概率）
                    # img_logits_b = out["img_logits"][b].detach().float().cpu()  # [C]
                    # img_scores_by_img[img_id] = img_logits_b.softmax(dim=-1)    # 做好softmax

                    # # 记录角度是否达标（三个角度都 ≤ 阈值） 
                    # ang_pred_b = ang_pred[b]   # [3]
                    # ang_t_b    = ang_t[b]      # [3]
                    # ang_err_b  = (ang_pred_b - ang_t_b + 180.0).remainder(360.0) - 180.0
                    # ang_err_b  = ang_err_b.abs()

                    # ok_map = angle_ok_by_img.setdefault(img_id, {})
                    # for thr in acc_thresholds:
                    #     ok_map[float(thr)] = bool((ang_err_b <= thr).all().item())  # 不可以直接用ang_abs，会跨batch
                    
            else:
                logits     = out["logits"]
                angle_norm = out["angle_norm"]
                dist_log   = out["dist_log"]
                loss_cls  = F.cross_entropy(logits, cls)
                loss_ang  = _circular_huber_on_norm(angle_norm, _deg_to_norm(ang_t), delta=0.05)
                loss_dist = _huber(dist_log.squeeze(-1), _log1p(dist_t), delta=0.2)
                loss = loss_cls + loss_ang + loss_dist
                va_loss_sum += loss.item()

                # 缓存分类 scores/labels
                probs = logits.softmax(dim=-1).detach().float().cpu()
                va_scores.append(probs)
                va_labels.append(cls.detach().cpu())

            # 角度
            ang_pred = _norm_to_deg(angle_norm)
            ang_diff = _wrap_diff_deg(ang_pred, ang_t)
            ang_abs  = ang_diff.abs()
            va_ang_abs_sum += ang_abs.sum().item()
            va_ang_sq_sum  += (ang_diff ** 2).sum().item()
            va_ang_cnt_el  += ang_abs.numel()
            accs = _acc_thresholds_angles(ang_abs, list(acc_thresholds))
            for k, v in accs.items():
                va_acc_cnts[k] += v * ang_abs.numel()

            # 距离
            dist_pred = _expm1_clamp(dist_log.squeeze(-1))
            d_abs = (dist_pred - dist_t).abs()
            va_d_abs_sum += d_abs.sum().item()
            va_d_sq_sum  += ((dist_pred - dist_t) ** 2).sum().item()
            va_d_cnt     += dist_t.numel()

            if local_rank == 0:
                va_maae_ang = va_ang_abs_sum / max(1, va_ang_cnt_el)
                va_rmse_ang = math.sqrt(va_ang_sq_sum / max(1, va_ang_cnt_el))
                va_mae_dist = va_d_abs_sum / max(1, va_d_cnt)
                va_rmse_dist= math.sqrt(va_d_sq_sum / max(1, va_d_cnt))
                pbar.set_postfix(**{
                    'val_loss': va_loss_sum / (iteration + 1),
                    'val_A-MAAE(deg)': f'{va_maae_ang:.3f}',
                    'val_A-RMSE(deg)': f'{va_rmse_ang:.3f}',
                    'val_D-MAE(m)': f'{va_mae_dist:.3f}',
                    'val_D-RMSE(m)': f'{va_rmse_dist:.3f}',
                    'lr': get_lr(optimizer)
                })
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish validation')

        # ---- 角度聚合 ----
        train_maae_ang = tr_ang_abs_sum / max(1, tr_ang_cnt_el)
        train_rmse_ang = math.sqrt(tr_ang_sq_sum / max(1, tr_ang_cnt_el))
        train_mae_dist= tr_d_abs_sum / max(1, tr_d_cnt)
        train_rmse_dist= math.sqrt(tr_d_sq_sum / max(1, tr_d_cnt))
        val_mae_dist   = va_d_abs_sum / max(1, va_d_cnt)
        val_rmse_dist  = math.sqrt(va_d_sq_sum / max(1, va_d_cnt))
        val_maae_ang   = va_ang_abs_sum / max(1, va_ang_cnt_el)
        val_rmse_ang   = math.sqrt(va_ang_sq_sum / max(1, va_ang_cnt_el))
        train_accs = {k: tr_acc_cnts[k] / max(1, tr_ang_cnt_el) for k in tr_acc_cnts}
        val_accs   = {k: va_acc_cnts[k] / max(1, va_ang_cnt_el) for k in va_acc_cnts}

        # # ---- 分类 mAP（macro） ----
        # tr_scores_cat = torch.cat(tr_scores, dim=0) if len(tr_scores) else torch.empty(0, device='cpu')
        # tr_labels_cat = torch.cat(tr_labels, dim=0) if len(tr_labels) else torch.empty(0, dtype=torch.long, device='cpu')
        # va_scores_cat = torch.cat(va_scores, dim=0) if len(va_scores) else torch.empty(0, device='cpu')
        # va_labels_cat = torch.cat(va_labels, dim=0) if len(va_labels) else torch.empty(0, dtype=torch.long, device='cpu')

        # train_mAP, train_ap_per_class = _compute_map(tr_scores_cat, tr_labels_cat) if tr_scores_cat.numel() else (0.0, {})
        # val_mAP,   val_ap_per_class   = _compute_map(va_scores_cat, va_labels_cat) if va_scores_cat.numel() else (0.0, {})
        # 分类 mAP（detect 模式下为空 → 0）
        tr_scores_cat = torch.cat(tr_scores, dim=0) if len(tr_scores) else torch.empty(0, device='cpu')
        tr_labels_cat = torch.cat(tr_labels, dim=0) if len(tr_labels) else torch.empty(0, dtype=torch.long, device='cpu')
        va_scores_cat = torch.cat(va_scores, dim=0) if len(va_scores) else torch.empty(0, device='cpu')
        va_labels_cat = torch.cat(va_labels, dim=0) if len(va_labels) else torch.empty(0, dtype=torch.long, device='cpu')
        train_mAP, train_ap_per_class = (0.0, {}) if tr_scores_cat.numel()==0 else _compute_map(tr_scores_cat, tr_labels_cat)
        val_mAP,   val_ap_per_class   = (0.0, {}) if va_scores_cat.numel()==0 else _compute_map(va_scores_cat, va_labels_cat)

        # 仍沿用原来的 loss_history 记录
        loss_history.append_loss(epoch + 1, tr_loss_sum / epoch_step, va_loss_sum / epoch_step_val)

        det_metrics_str = ""
        # if bool(getattr(model_train, "detect", False)):
        #     # VOC@0.5 与 COCO@[0.5:0.95]
        #     ious_voc  = [0.5]
        #     ious_coco = [0.5 + 0.05 * i for i in range(10)]  # 0.50:0.05:0.95
        #     num_classes = int(getattr(model_train, "num_classes", 1))
        #     if len(preds_accum) > 0 and len(gts_accum) > 0:
        #         res_voc  = eval_map_per_class(preds_accum, gts_accum, num_classes, ious_voc)
        #         res_coco = eval_map_per_class(preds_accum, gts_accum, num_classes, ious_coco)
        #         det_metrics_str = f" | mAP@0.5(det): {res_voc['mAP@0.5']:.3f} | mAP@[.5:.95](det): {res_coco['mAP@[.5:.95]']:.3f}"
        #     else:
        #         det_metrics_str = " | mAP@0.5(det): 0.000 | mAP@[.5:.95](det): 0.000"
        if bool(getattr(model_train, "detect", False)):
            num_classes = int(getattr(model_train, "num_classes", 1))
            if len(preds_accum) > 0 and len(gts_accum) > 0:
                # COCO 官方默认 IoU 0.50:0.95 步长 0.05，maxDets=[1,10,100]
                res_coco = eval_map_coco(preds_accum, gts_accum, num_classes, iou_type="bbox")
                det_metrics_str = (
                    f" | COCO mAP@[.5:.95]: {res_coco['mAP@[.5:.95]']:.3f}"
                    f" | AP@0.5: {res_coco['mAP@0.5']:.3f}"
                    f" | AP@0.75: {res_coco['mAP@0.75']:.3f}"
                )
            else:
                det_metrics_str = " | COCO mAP@[.5:.95]: 0.000 | AP@0.5: 0.000 | AP@0.75: 0.000"

        # 新增角度map评估
        angle_metrics_str = ""
        if bool(getattr(model_train, "detect", False)) and len(preds_accum) > 0:
            angle_map_dict = eval_angle_map_by_thresholds_det(
                preds=preds_accum,
                gts=gts_accum,
                num_classes=int(getattr(model_train, "num_classes", 1)),
                angle_ok_by_img=angle_ok_by_img,
                thresholds=acc_thresholds
            )
            angle_metrics_str = " | " + " | ".join(
                [f"{k}: {angle_map_dict[k]:.3f}" for k in sorted(angle_map_dict.keys(), key=lambda x:int(x.split('angle')[1]))]
            )
        # if bool(getattr(model_train, "detect", False)) and len(img_scores_by_img) > 0:
        #     angle_map_dict = eval_angle_map_by_thresholds_img(
        #         img_scores_by_img=img_scores_by_img,
        #         gts=gts_accum,
        #         num_classes=int(getattr(model_train, "num_classes", 4)),
        #         angle_ok_by_img=angle_ok_by_img,
        #         thresholds=acc_thresholds,
        #         use_softmax=False  # 上面已经做了 softmax 就 False；若这里传 logits 就 True
        #     )
        #     angle_metrics_str = " | " + " | ".join(
        #         [f"{k}: {angle_map_dict[k]:.3f}" for k in sorted(angle_map_dict.keys(), key=lambda x:int(x.split('angle')[1]))]
        #     )

        det_metrics_str = det_metrics_str + angle_metrics_str
        
        
        # 控制台输出
        def _ap_str(per_class_ap):
            # 简短打印每类 AP
            return ", ".join([f"c{c}:{ap:.3f}" for c, ap in per_class_ap.items()])
        msg = (
            f"Epoch: {epoch+1}/{Epoch} | "
            f"Loss: {tr_loss_sum/epoch_step:.3f} | val Loss: {va_loss_sum/epoch_step_val:.3f} | "
            f"A-MAAE: {train_maae_ang:.3f} | val A-MAAE: {val_maae_ang:.3f} | "
            f"A-RMSE: {train_rmse_ang:.3f} | val A-RMSE: {val_rmse_ang:.3f} | "
            f"D-MAE: {train_mae_dist:.3f} | val D-MAE: {val_mae_dist:.3f} | "
            f"D-RMSE: {train_rmse_dist:.3f} | val D-RMSE: {val_rmse_dist:.3f} | "
            f"mAP-train: {train_mAP:.3f} ({_ap_str(train_ap_per_class)}) | "
            f"mAP-val: {val_mAP:.3f} ({_ap_str(val_ap_per_class)}) | "
            + " | ".join([f"{k}: {val_accs[k]*100:.1f}%" for k in sorted(val_accs.keys(), key=lambda x:int(x.split('@')[1]))])
            + det_metrics_str
        )

        print(msg)

        # # ---------- 保存权重（仍按 val 角度 MAAE） ----------
        # if len(loss_history.val_loss) <= 1 or (val_maae_ang <= getattr(loss_history, "best_val_maae", float('inf'))):
        #     print('Save best model to best_epoch_weights.pth (by val A-MAAE)')
        #     torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
        #     loss_history.best_val_maae = val_maae_ang

        # # 周期性保存
        # if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
        #     fname = f'ep{epoch+1:03d}-loss{tr_loss_sum/epoch_step:.3f}-val_loss{va_loss_sum/epoch_step_val:.3f}-val_amaa{val_maae_ang:.3f}-map{val_mAP:.3f}.pth'
        #     torch.save(model.state_dict(), os.path.join(save_dir, fname))
        
        # ---------- 保存权重（按 val acc@10 最高） ----------
        # 取出当前验证集 acc@10（若不存在则回退为 0.0）
        cur_acc10 = float(val_accs.get("acc@10", 0.0))
        # 记录 acc@10
        acc_history.append_acc(epoch + 1, cur_acc10)

        if cur_acc10 > acc_history.best_val_acc10:
            print(
                f"Save best model to best_epoch_weights.pth "
                f"(by val acc@10: {cur_acc10:.4f}, prev best: {acc_history.best_val_acc10 if acc_history.best_val_acc10 != float('-inf') else 'N/A'})"
            )
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            acc_history.best_val_acc10 = cur_acc10

        # 周期性保存（带上 acc@10）
        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            fname = (
                f'ep{epoch+1:03d}'
                f'-loss{tr_loss_sum/epoch_step:.3f}'
                f'-val_loss{va_loss_sum/epoch_step_val:.3f}'
                f'-val_amaa{val_maae_ang:.3f}'
                f'-map{val_mAP:.3f}'
                f'-acc10{cur_acc10:.4f}.pth'
            )
            torch.save(model.state_dict(), os.path.join(save_dir, fname))
        # 永远更新 last
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        