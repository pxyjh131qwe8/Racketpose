import os, math, random, argparse
from types import SimpleNamespace as NS
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from boxes_utils import eval_map_per_class, decode_and_nms_one_image, eval_map_coco, eval_angle_map_by_thresholds_img, eval_angle_map_by_thresholds_det




def multi_task_test(
    model_train, epoch, epoch_step_test,
    gen_test, Epoch, cuda, local_rank=0,
    acc_thresholds=(1.0, 2.0, 5.0, 10.0),
    # 评估相关阈值（仅检测模式使用）
    det_conf_thresh: float = 0.05,
    det_nms_iou: float = 0.6,
    det_max_dets: int = 300,
    det_per_class_topk: int = 1000,
):
    # ----------------- utils -----------------
    def _norm_to_deg(x):   return x * 360.0 - 180.0
    def _wrap_diff_deg(pred_deg, tgt_deg):
        return (pred_deg - tgt_deg + 180.0).remainder(360.0) - 180.0
    def _expm1_clamp(x): return torch.expm1(x).clamp_min(0.0)
    def _acc_thresholds_angles(abs_err_deg, thresholds):
        tot = abs_err_deg.numel()
        return {f"acc@{int(t)}": float((abs_err_deg <= t).sum().item()) / max(1, tot) for t in thresholds}

    # mAP（仅分类模式）
    def _average_precision(y_true_bool: torch.Tensor, y_score: torch.Tensor) -> float:
        Np = int(y_true_bool.sum().item())
        if Np == 0:
            return 0.0
        order = torch.argsort(y_score, descending=True)
        y_true_sorted = y_true_bool[order]
        tp = torch.cumsum(y_true_sorted.to(torch.float32), dim=0)
        fp = torch.cumsum((~y_true_sorted).to(torch.float32), dim=0)
        prec = tp / torch.clamp(tp + fp, min=1.0)
        ap = (prec[y_true_sorted].sum() / max(1, Np)).item()
        return float(ap)

    def _compute_map(all_scores: torch.Tensor, all_labels: torch.Tensor):
        K = all_scores.size(1)
        ap_list = []
        per_class_ap = {}
        for c in range(K):
            y_true = (all_labels == c)
            ap_c = _average_precision(y_true, all_scores[:, c])
            ap_list.append(ap_c)
            per_class_ap[c] = ap_c
        mAP = float(sum(ap_list) / len(ap_list)) if len(ap_list) > 0 else 0.0
        return mAP, per_class_ap

    # --------- 聚合量 ---------
    # 回归
    test_acc_cnts = {f"acc@{int(t)}": 0.0 for t in acc_thresholds}
    test_ang_abs_sum = 0.0
    test_ang_sq_sum = 0.0
    test_ang_cnt_el = 0
    test_d_abs_sum = 0.0
    test_d_sq_sum  = 0.0
    test_d_cnt     = 0
    # 分类（仅在 detect=False）
    test_scores, test_labels = [], []
    # 检测（仅在 detect=True）
    preds_accum, gts_accum = [], []

    if local_rank == 0:
        print('Start Test')
        pbar = tqdm(total=epoch_step_test, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    is_detect = bool(getattr(model_train, "detect", False))
    
    # 新增map@角度相关缓存
    angle_ok_by_img = dict()       # image_id -> {thr(float): bool}

    with torch.no_grad():
        for iteration, batch in enumerate(gen_test):
            if iteration >= epoch_step_test:
                break

            # ------- 解析 batch：支持是否带框 -------
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
            if not isinstance(out, dict):
                raise RuntimeError("Model must return a dict")

            # ------ 公共：角度/距离指标 ------
            angle_norm = out["angle_norm"]    # (B,3)
            dist_log   = out["dist_log"]      # (B,1)
            ang_pred = _norm_to_deg(angle_norm)
            ang_diff = _wrap_diff_deg(ang_pred, ang_t)
            ang_abs  = ang_diff.abs()
            test_ang_abs_sum += ang_abs.sum().item()
            test_ang_sq_sum  += (ang_diff ** 2).sum().item()
            test_ang_cnt_el  += ang_abs.numel()
            accs = _acc_thresholds_angles(ang_abs, list(acc_thresholds))
            for k, v in accs.items():
                test_acc_cnts[k] += v * ang_abs.numel()

            dist_pred = _expm1_clamp(dist_log.squeeze(-1))
            d_abs = (dist_pred - dist_t).abs()
            test_d_abs_sum += d_abs.sum().item()
            test_d_sq_sum  += ((dist_pred - dist_t) ** 2).sum().item()
            test_d_cnt     += dist_t.numel()

            # ------ 分模式处理 ------
            if is_detect:
                # 收集检测预测（用 batched_nms 封装后的解码）
                pred_logits = out["pred_logits"]  # (B,K,Hf,Wf)
                pred_obj    = out["pred_obj"]     # (B,1,Hf,Wf)
                pred_boxes  = out["pred_boxes"]   # (B,4,Hf,Wf)

                B, K, Hf, Wf = pred_logits.shape
                Himg = imgs.shape[2]; Wimg = imgs.shape[3]

                for b in range(B):
                    boxes_img, scores_img, labels_img = decode_and_nms_one_image(
                        pred_boxes[b], pred_logits[b], pred_obj[b],
                        img_h=Himg, img_w=Wimg,
                        conf_thresh=det_conf_thresh, iou_thr=det_nms_iou,
                        max_dets=det_max_dets, per_class_topk=det_per_class_topk
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
                        if g_boxes.numel() > 0 and (g_boxes.max() <= 1.0).item():
                            g_boxes = g_boxes.clone()
                            g_boxes[:, [0,2]] *= Wimg
                            g_boxes[:, [1,3]] *= Himg
                        g_labels = box_labels_list[b] if box_labels_list is not None else cls[b].repeat(g_boxes.size(0))
                    gts_accum.append({
                        "image_id": iteration * B + b,
                        "boxes": g_boxes,
                        "labels": g_labels.long()
                    })
                
                    # 新增map@角度
                    img_id = iteration * B + b
                    ang_pred_b = _norm_to_deg(angle_norm[b])  # [3] deg 不能用训练里面的ang_pred
                    ang_t_b    = ang_t[b]
                    ang_err_b  = (ang_pred_b - ang_t_b + 180.0).remainder(360.0) - 180.0
                    ang_err_b  = ang_err_b.abs()
                    ok_map = angle_ok_by_img.setdefault(img_id, {})
                    for thr in acc_thresholds:
                        ok_map[float(thr)] = bool((ang_err_b <= thr).all().item())   

            else:
                # 分类 mAP
                logits = out["logits"]
                probs = logits.softmax(dim=-1).detach().float().cpu()
                test_scores.append(probs)
                test_labels.append(cls.detach().cpu())

            if local_rank == 0:
                test_maae_ang = test_ang_abs_sum / max(1, test_ang_cnt_el)
                test_rmse_ang = math.sqrt(test_ang_sq_sum / max(1, test_ang_cnt_el))
                test_mae_dist = test_d_abs_sum / max(1, test_d_cnt)
                test_rmse_dist= math.sqrt(test_d_sq_sum / max(1, test_d_cnt))
                pbar.set_postfix(**{
                    'test_A-MAAE(deg)': f'{test_maae_ang:.3f}',
                    'test_A-RMSE(deg)': f'{test_rmse_ang:.3f}',
                    'test_D-MAE(m)': f'{test_mae_dist:.3f}',
                    'test_D-RMSE(m)': f'{test_rmse_dist:.3f}',
                })
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Test')

        # ---- 回归聚合 ----
        test_mae_dist   = test_d_abs_sum / max(1, test_d_cnt)
        test_rmse_dist  = math.sqrt(test_d_sq_sum / max(1, test_d_cnt))
        test_maae_ang   = test_ang_abs_sum / max(1, test_ang_cnt_el)
        test_rmse_ang   = math.sqrt(test_ang_sq_sum / max(1, test_ang_cnt_el))
        test_accs   = {k: test_acc_cnts[k] / max(1, test_ang_cnt_el) for k in test_acc_cnts}

        # ---- 分类/检测指标 ----
        det_metrics_str = ""
        if is_detect:
            # ious_voc  = [0.5]
            # ious_coco = [0.5 + 0.05 * i for i in range(10)]
            num_classes = int(getattr(model_train, "num_classes", 4))
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
            cls_metrics_str = ""  # 检测模式下不输出分类 mAP    
        else:
            test_scores_cat = torch.cat(test_scores, dim=0) if len(test_scores) else torch.empty(0, device='cpu')
            test_labels_cat = torch.cat(test_labels, dim=0) if len(test_labels) else torch.empty(0, dtype=torch.long, device='cpu')
            test_mAP, test_ap_per_class = _compute_map(test_scores_cat, test_labels_cat) if test_scores_cat.numel() else (0.0, {})
            def _ap_str(per_class_ap): return ", ".join([f"c{c}:{ap:.3f}" for c, ap in per_class_ap.items()])
            cls_metrics_str = f" | mAP-test: {test_mAP:.3f} ({_ap_str(test_ap_per_class)})"
            det_metrics_str = ""
        
        # 新增角度map评估
        angle_metrics_str = ""
        if bool(getattr(model_train, "detect", False)) and len(preds_accum) > 0:
            angle_map_dict = eval_angle_map_by_thresholds_det(
                preds=preds_accum,
                gts=gts_accum,
                num_classes=int(getattr(model_train, "num_classes", 4)),
                angle_ok_by_img=angle_ok_by_img,
                thresholds=acc_thresholds
            )
            angle_metrics_str = " | " + " | ".join(
                [f"{k}: {angle_map_dict[k]:.3f}" for k in sorted(angle_map_dict.keys(), key=lambda x:int(x.split('angle')[1]))]
            )

        det_metrics_str = det_metrics_str + angle_metrics_str    

        # ---- 打印 ----
        msg = (
            f"Epoch: {epoch+1}/{Epoch} | "
            f"A-MAAE: {test_maae_ang:.3f} | "
            f"A-RMSE: {test_rmse_ang:.3f} | "
            f"D-MAE: {test_mae_dist:.3f} | "
            f"D-RMSE: {test_rmse_dist:.3f}"
            + cls_metrics_str + det_metrics_str + " | "
            + " | ".join([f"{k}: {test_accs[k]*100:.1f}%" for k in sorted(test_accs.keys(), key=lambda x:int(x.split('@')[1]))])
        )
        print(msg)
        
