import copy
import csv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 在导入 torch 前设置可见 GPU

import warnings
from argparse import ArgumentParser
import math
import datetime
from torch.utils.tensorboard import SummaryWriter


import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset
from datasets.multitask_racket import RacketMultiTaskDataset, detection_collate, build_loader

from utils.crop_utils import _expand_double_box, _read_dets_from_txt, _read_dets_from_json, _find_source_image, crop_from_detections

import glob
import json
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import numpy as np

warnings.filterwarnings("ignore")

# data_dir = '../Dataset/COCO'



# ---------- angles & distance helpers ----------
def _deg_to_norm(deg):     # [-180,180] -> [0,1]
    return (deg + 180.0) / 360.0

def _norm_to_deg(x):       # [0,1] -> [-180,180]
    return x * 360.0 - 180.0

def _wrap_diff_deg(pred_deg, tgt_deg):
    # wrap to [-180,180]
    return (pred_deg - tgt_deg + 180.0).remainder(360.0) - 180.0

def _smooth_huber(x, y, delta):
    d = (x - y).abs()
    return torch.where(d < delta, 0.5 * (d**2) / delta, d - 0.5 * delta).mean()

def _circular_huber_on_norm(pred_norm, tgt_norm, delta=0.05):
    # pred_norm/tgt_norm in [0,1] over the circle
    d = (pred_norm - tgt_norm).abs()
    d = torch.minimum(d, 1.0 - d)
    return torch.where(d < delta, 0.5 * (d**2) / delta, d - 0.5 * delta).mean()

def _expm1_clamp(x):       # log(1+dist) -> dist
    return torch.expm1(x).clamp_min(0.0)

def _acc_thresholds_angles(abs_err_deg, thresholds=(1,2,5,10)):
    tot = abs_err_deg.numel()
    return {f"acc@{int(t)}": float((abs_err_deg <= t).sum().item()) / max(1, tot) for t in thresholds}


def _acc_thresholds_ang_and_dist(abs_err_deg: torch.Tensor,
                                 dist_abs_err_m: torch.Tensor,
                                 thresholds=(1, 2, 5, 10),
                                 dist_thr: float = 0.05):
    """
    同时满足：
      - 角度绝对误差 <= t（度）
      - 距离绝对误差 <= dist_thr（米，默认 0.05m=5cm）

    abs_err_deg: [B, 3] 或任意形状的角度误差（度）
    dist_abs_err_m: [B] 距离绝对误差（米）
    返回: dict，例如 {"acc@1": 0.23, "acc@2": 0.41, ...}
    """
    # 把距离误差 [B] 扩展成与角度误差相同的形状（逐角度分量统计）
    dist_mask = (dist_abs_err_m <= dist_thr).unsqueeze(-1).expand_as(abs_err_deg)  # True/False

    tot = abs_err_deg.numel()
    out = {}
    for t in thresholds:
        ang_mask = (abs_err_deg <= t)
        both_ok = ang_mask & dist_mask                          # 同时满足
        out[f"acc@{int(t)}"] = float(both_ok.sum().item()) / max(1, tot)
    return out

def _compose_total_from_xyz_torch(ang_deg_3):
    """
    ang_deg_3: [B,3]，每行为 (angle_x, angle_y, angle_z)，单位度，范围约 [-180,180]
    返回：theta_deg [B]，合成后的总夹角（度，0..180）
    公式：tan(theta) = sqrt( tan^2(ax)+tan^2(ay)+tan^2(az) )
         若任一 |angle_a|>90° 则 theta = 180 - base
    """
    rad = torch.deg2rad(ang_deg_3)                      # [B,3]
    t   = torch.tan(rad)                                # [B,3]
    R   = torch.sqrt(torch.clamp((t * t).sum(dim=1), min=0.0))     # [B]
    base= torch.rad2deg(torch.atan2(R, torch.ones_like(R)))        # [B]
    back= (ang_deg_3.abs() > 90.0).any(dim=1)                       # [B] bool
    theta = torch.where(back, 180.0 - base, base)                   # [B]
    return torch.clamp(theta, 0.0, 180.0)

def _acc_thresholds_total_and_dist(angle_total_abs_err_deg: torch.Tensor,
                                   dist_abs_err_m: torch.Tensor,
                                   thresholds=(1, 2, 5, 10),
                                   dist_thr: float = 0.05):
    """
    基于“合角度差 + 距离差”的准确率：
      同时满足：
        - |theta_pred - theta_true| <= t（度）
        - |dist_pred - dist_true|  <= dist_thr（米，默认 0.05）
    输入：
      angle_total_abs_err_deg: [B]
      dist_abs_err_m:          [B]
    返回：
      dict: acc@t => (batch 内满足数/样本数) 的比例
    """
    assert angle_total_abs_err_deg.dim() == 1 and dist_abs_err_m.dim() == 1
    B = angle_total_abs_err_deg.numel()
    if B == 0:
        return {f"acc@{int(t)}": 0.0 for t in thresholds}
    dist_ok = (dist_abs_err_m <= dist_thr)
    out = {}
    for t in thresholds:
        ang_ok = (angle_total_abs_err_deg <= float(t))
        both_ok = ang_ok & dist_ok
        out[f"acc@{int(t)}"] = both_ok.float().mean().item()
    return out



# ---------- boxes helpers (把 xyxy(0..1) 转成 cxcywh(0..1) 并组装 YOLO targets) ----------
def xyxy_to_cxcywh(boxes):
    if boxes.numel() == 0:
        return boxes
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    w = (x2 - x1).clamp_min(1e-6)
    h = (y2 - y1).clamp_min(1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return torch.stack([cx, cy, w, h], dim=1)

def lists_to_targets_for_yolo(batch_boxes, batch_labels):
    """
    输入:
        batch_boxes: List[Tensor[Ni,4]] (xyxy, 0..1)
        batch_labels: List[Tensor[Ni]]
    输出:
        targets: dict
          - 'idx': [M,1]  batch 索引 (float)
          - 'cls': [M,1]  类别 id (float)
          - 'box': [M,4]  cxcywh in 0..1 (float)
    """
    idx_all, cls_all, box_all = [], [], []
    for b, (boxes, labels) in enumerate(zip(batch_boxes, batch_labels)):
        if boxes.numel() == 0:
            continue
        cxcywh = xyxy_to_cxcywh(boxes)
        M = cxcywh.size(0)
        device = boxes.device
        idx_all.append(torch.full((M,1), b, dtype=torch.float32, device=device))
        cls_all.append(labels.view(-1,1).to(torch.float32))
        box_all.append(cxcywh.to(torch.float32))
    if len(box_all) == 0:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return {
            'idx': torch.zeros((0,1), dtype=torch.float32, device=device),
            'cls': torch.zeros((0,1), dtype=torch.float32, device=device),
            'box': torch.zeros((0,4), dtype=torch.float32, device=device),
        }
    return {
        'idx': torch.cat(idx_all, dim=0),
        'cls': torch.cat(cls_all, dim=0),
        'box': torch.cat(box_all, dim=0),
    }



def train(args, params):
    # if args.local_rank == 0:
    #     print(f"Training on device(s): {args.device} | world_size={args.world_size}")
   
    # Model
    model = nn.yolo_v11_x(len(params['names']))
    model = torch.load("/root/autodl-tmp/yolov11/runs/yolov11_x/best.pt", map_location='cuda', weights_only=False)  
    model = model["model"]
    for p in model.parameters():
        p.requires_grad_(True)
    model.train().cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(util.set_params(model, params['weight_decay']),
                                params['min_lr'], params['momentum'], nesterov=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.tb_dir) if args.local_rank == 0 else None

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None
    
    # Data
    loader, sampler = build_loader('train', args, params, shuffle=True)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)

    with open(os.path.join(args.save_dir, 'step.csv'), 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            
            # 回归指标累计
            ang_abs_sum = 0.0
            ang_sq_sum  = 0.0
            ang_cnt     = 0
            dist_abs_sum= 0.0
            dist_sq_sum = 0.0
            dist_cnt    = 0
            
            for i, (samples, regs, clss, boxes_list, labels_list) in p_bar:

                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                # 已经 Normalize(IMAGENET)，不要再 /255.
                samples = samples.cuda(non_blocking=True).float()
                regs    = regs.cuda(non_blocking=True).float()
                
                # 组装 YOLO targets（cxcywh, 0..1）
                targets = lists_to_targets_for_yolo(boxes_list, labels_list)
                
                dist_t = regs[:, 0]
                ang_t  = regs[:, 1:4]

                # Forward
                with torch.amp.autocast('cuda'):
                    out_all = model(samples)
                    if isinstance(out_all, (tuple, list)) and len(out_all) == 2 and isinstance(out_all[1], dict):
                        outputs, aux = out_all
                        angle_norm = aux.get("angle_norm", None)  # [B,3] in [0,1]
                        dist_log   = aux.get("dist_log", None)    # [B,1]
                    else:
                        outputs = out_all
                        angle_norm = dist_log = None

                    # YOLO 原生损失
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)
                    
                    # 回归损失（若无多任务头则为 0）
                    loss_ang = loss_dist = torch.tensor(0.0, device=samples.device)
                    if (angle_norm is not None) and (dist_log is not None):
                        loss_ang  = _circular_huber_on_norm(angle_norm, _deg_to_norm(ang_t), delta=0.05)
                        loss_dist = _smooth_huber(dist_log.squeeze(-1),
                                                torch.log1p(dist_t.clamp_min(0.0)), delta=0.2)

                    loss = loss_box + loss_cls + loss_dfl + loss_ang + loss_dist

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode
                loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss).backward()
                
                # tensorboard 记录
                if writer is not None:
                    global_step = step
                    writer.add_scalar("train/loss_box", float(loss_box.detach()), global_step)
                    writer.add_scalar("train/loss_cls", float(loss_cls.detach()), global_step)
                    writer.add_scalar("train/loss_dfl", float(loss_dfl.detach()), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)

                    # 如果带角度/距离分支，也记录
                    if (angle_norm is not None) and (dist_log is not None):
                        writer.add_scalar("train/loss_angle", float(loss_ang.detach()), global_step)
                        writer.add_scalar("train/loss_dist",  float(loss_dist.detach()), global_step)

                        # 训练时的 A/D 简单 MSE 监控
                        # writer.add_scalar("train/angle_MSE_deg",
                        #                   (ang_abs_sum / max(1, ang_cnt)) if ang_cnt else 0.0, global_step)
                        # writer.add_scalar("train/dist_MSE",
                        #                   (dist_abs_sum / max(1, dist_cnt)) if dist_cnt else 0.0, global_step)

                # Optimize
                if step % accumulate == 0:
                    # amp_scale.unscale_(optimizer)  # unscale gradients
                    # util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()
                
                # 角度/距离训练指标
                with torch.no_grad():
                    if (angle_norm is not None) and (dist_log is not None):
                        ang_pred = _norm_to_deg(angle_norm.float())
                        ang_err  = _wrap_diff_deg(ang_pred, ang_t).abs()
                        ang_abs_sum += ang_err.sum().item()
                        ang_sq_sum  += (ang_err ** 2).sum().item()
                        ang_cnt     += ang_err.numel()

                        dist_pred   = _expm1_clamp(dist_log.squeeze(-1).float())
                        d_err       = (dist_pred - dist_t).abs()
                        dist_abs_sum+= d_err.sum().item()
                        dist_sq_sum += ((dist_pred - dist_t) ** 2).sum().item()
                        dist_cnt    += d_err.numel()

                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'
                    a_maae = (ang_abs_sum / max(1, ang_cnt)) if ang_cnt else 0.0
                    d_mae  = (dist_abs_sum / max(1, dist_cnt)) if dist_cnt else 0.0
                    s = ('%10s' * 2 + '%10.3g' * 6) % (
                        f'{epoch + 1}/{args.epochs}', memory,
                        avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg,
                        a_maae, d_mae, optimizer.param_groups[0]['lr'])
                    p_bar.set_description(s)

            # 每个 epoch 结束做一次验证
            if args.local_rank == 0:
                last = test(args, params, ema.ema if ema else model)
                # last: (mAP, mAP@50, Recall, Precision, val_A_MSE, val_D_MSE, acc_dict)
                map_all, map50, rec, pre, val_A_MSE, val_D_MSE, acc_dict = last

                # 取 acc@10（若没有该键则回退为 0.0）
                acc10 = float(acc_dict.get("acc@10", 0.0))
                
                # tensorboard 记录
                if writer is not None:
                    # mAP/Precision/Recall
                    writer.add_scalar("val/mAP",    map_all, epoch)
                    writer.add_scalar("val/mAP50",  map50,   epoch)
                    writer.add_scalar("val/Recall", rec,     epoch)
                    writer.add_scalar("val/Precision", pre,  epoch)

                    # Angle/Distance
                    writer.add_scalar("val/angle_MSE",  val_A_MSE, epoch)
                    writer.add_scalar("val/distance_MSE", val_D_MSE, epoch)
                    writer.add_scalar("val/acc@10", acc10, epoch)

                    # acc@k
                    for k, v in acc_dict.items():
                        writer.add_scalar(f"val/{k}", float(v), epoch)

                print(
                    f"Val | mAP: {map_all:.3f} | mAP@50: {map50:.3f} | R: {rec:.3f} | P: {pre:.3f} | "
                    f"A-MSE: {val_A_MSE:.3f} | D-MSE: {val_D_MSE:.3f} | "
                    + " ".join([f"{k}:{v*100:.1f}%" for k, v in acc_dict.items()])
                )

                # 保存 last
                save = {'epoch': epoch + 1, 'model': copy.deepcopy(ema.ema if ema else model)}
                torch.save(save, os.path.join(args.save_dir, 'last.pt'))

                # -------- 以 acc@10 作为 best 判据 --------
                # 用函数属性记录历史最佳，首次默认 -inf
                best_acc10 = getattr(train, "_best_acc10", float("-inf"))
                if acc10 > best_acc10:
                    torch.save(save, os.path.join(args.save_dir, 'best.pt'))
                    train._best_acc10 = acc10
                    print(f"[Best] acc@10 提升为 {acc10*100:.2f}% ，已保存为 best.pt")
                del save

                # （可选）顺便按 epoch 命名额外留档
                ckpt_name = (
                    f'./weights/ep{epoch+1:03d}-mAP{map_all:.3f}-A-MSE{val_A_MSE:.3f}-D-MSE{val_D_MSE:.3f}-acc10{acc10:.4f}.pt'
                )
                torch.save({'epoch': epoch + 1, 'model': copy.deepcopy(ema.ema if ema else model)}, ckpt_name)
                
    if writer is not None:
        writer.close()            


            

@torch.no_grad()
def test(args, params, model=None):
    loader, _ = build_loader("test", args, params, shuffle=False)

    plot = False
    if not model:
        plot = True
        # model = torch.load(f='./weights/best.pt', map_location='cuda', weights_only=False)  # 替换成对应模型的权重
        model = torch.load(f='/mnt/data/pxy/YOLOv11-pt-master/runs/train-2025_10_23-070318/weights/best.pt', map_location='cuda', weights_only=False)
        model = model['model'].float().fuse()
    model.half().eval()

    # mAP 配置
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()
    n_iou = iou_v.numel()

    metrics = []
    m_pre = m_rec = map50 = mean_ap = 0.0

    # 角度/距离 验证指标
    va_ang_abs_sum = va_ang_sq_sum = 0.0
    va_ang_cnt     = 0
    va_dist_abs_sum= va_dist_sq_sum= 0.0
    va_dist_cnt    = 0
    acc_thrs = (1,2,5,10)
    acc_cnt  = {f"acc@{t}":0 for t in acc_thrs}
    total_samples = 0   # <<< 新增：累计样本数用于 acc 汇总

    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, regs, clss, boxes_list, labels_list in p_bar:
        samples = samples.cuda().half()          # 已经 Normalize 了，不要再 /255.
        _, _, h, w = samples.shape
        scale = torch.tensor((w, h, w, h), device=samples.device)

        out_all = model(samples)
        outputs = out_all[0] if (isinstance(out_all, (tuple, list)) and len(out_all) == 2) else out_all

        # NMS（保持原 util 流程：输入是 [B, 4+nc+A] 的张量）
        outputs = util.non_max_suppression(outputs)

        # YOLO mAP metrics
        for i, output in enumerate(outputs):
            # 组装 target（和原 util.compute_metric 对齐）：cls + xyxy(像素)
            if boxes_list[i].numel() == 0:
                # 没 GT
                metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=samples.device)
                metrics.append((metric, *torch.zeros((2, 0), device=samples.device), torch.zeros((0,), device=samples.device)))
                continue

            cls = labels_list[i].view(-1,1).to(samples.device)
            box = boxes_list[i].to(samples.device)      # xyxy, 0..1
            target = torch.cat([cls, (box * scale)], dim=1)  # [N, 5]

            metric = util.compute_metric(output[:, :6], target, iou_v)
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

        # 角度/距离指标（如果模型带 aux）
        regs = regs.cuda().float()
        if isinstance(out_all, (tuple, list)) and len(out_all) == 2 and isinstance(out_all[1], dict):
            aux = out_all[1]
            angle_norm = aux.get("angle_norm", None)  # [B,3] in [0,1]
            dist_log   = aux.get("dist_log", None)    # [B,1]
            if (angle_norm is not None) and (dist_log is not None):
                ang_t  = regs[:, 1:4]
                dist_t = regs[:, 0]
                ang_pred = _norm_to_deg(angle_norm.float())
                ang_err  = _wrap_diff_deg(ang_pred, ang_t).abs()   # [B,3]

                va_ang_abs_sum += ang_err.sum().item()
                va_ang_sq_sum  += (ang_err**2).sum().item()
                va_ang_cnt     += ang_err.numel()
                
                
                dist_pred = _expm1_clamp(dist_log.squeeze(-1).float())
                d_err = (dist_pred - dist_t).abs()
                va_dist_abs_sum += d_err.sum().item()
                va_dist_sq_sum  += (d_err**2).sum().item()
                va_dist_cnt     += d_err.numel()
                
                # accs = _acc_thresholds_angles(ang_err, acc_thrs)
                # 5cm5°
                # accs = _acc_thresholds_ang_and_dist(ang_err, d_err, acc_thrs, dist_thr=0.05)
                # for k,v in accs.items():
                #     acc_cnt[k] += v * ang_err.numel()
                # 1) 预测三轴角 -> 合角度
                ang_pred_deg = _norm_to_deg(angle_norm.float())      # [B,3]
                theta_pred   = _compose_total_from_xyz_torch(ang_pred_deg)  # [B]
                theta_true   = _compose_total_from_xyz_torch(ang_t)         # [B]

                # 2) 角度与距离的绝对误差
                angle_total_err = (theta_pred - theta_true).abs()           # [B]
                dist_pred       = _expm1_clamp(dist_log.squeeze(-1).float())# [B]  <<< 关键：squeeze(-1)
                d_err           = (dist_pred - dist_t).abs()                 # [B]


                # 3) ACC（按 batch 比例），我们按样本数聚合
                accs = _acc_thresholds_total_and_dist(angle_total_err, d_err, acc_thrs, dist_thr=0.05)
                B = angle_total_err.numel()
                total_samples += B   # <<< 新增累计样本数
                for k, v in accs.items():
                    # v 是该 batch 的比例 -> 折算为“满足样本数”，最终再除以总样本数
                    acc_cnt[k] += v * B
                
    # 计算 mAP
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)] if len(metrics) else None
    if metrics and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics, plot=plot, names=params["names"])

    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))

    # 角度/距离聚合
    val_A_MAAE = va_ang_abs_sum / max(1, va_ang_cnt)
    val_D_MAE  = va_dist_abs_sum / max(1, va_dist_cnt)
    val_A_MSE = math.sqrt(va_ang_sq_sum/max(1,va_ang_cnt))
    val_D_MSE = math.sqrt(va_dist_sq_sum/max(1,va_dist_cnt))
    # 也可算 RMSE：math.sqrt(va_ang_sq_sum/max(1,va_ang_cnt)) / math.sqrt(va_dist_sq_sum/max(1,va_dist_cnt))
    # acc_dict = {k: (acc_cnt[k] / max(1, va_ang_cnt)) for k in acc_cnt}
    # <<< 修复：改为按样本数聚合，避免角度分量数放大
    acc_dict = {k: (acc_cnt[k] / max(1, total_samples)) for k in acc_cnt}

    model.float()  # for possible further training
    
    print(f"Angle MSE: {val_A_MSE:.3f} | Distance MSE: {val_D_MSE:.3f} | ") 
    print(" | ".join([f"{k}:{v*100:.1f}%" for k, v in acc_dict.items()]))
    
    return float(mean_ap), float(map50), float(m_rec), float(m_pre), float(val_A_MSE), float(val_D_MSE), acc_dict


def profile(args, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    model = nn.yolo_v11_n(len(params['names'])).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


# ======= 新增：letterbox 预处理 & 还原工具 =======
def _letterbox(im: Image.Image, new_size: int = 640, stride: int = 32, pad_val: int = 114):
    """
    返回:
      im_resized: (H',W',3) uint8
      ratio: 缩放比例 (sx, sy)
      pad:   (dw, dh) 左右/上下实际 pad 的像素
      (w0,h0): 原图尺寸
    """
    w0, h0 = im.size
    r = min(new_size / w0, new_size / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    im_resized = im.resize(new_unpad, Image.BILINEAR)
    dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]
    dw //= 2
    dh //= 2
    canvas = Image.new('RGB', (new_size, new_size), color=(pad_val, pad_val, pad_val))
    canvas.paste(im_resized, (dw, dh))
    return np.asarray(canvas), (r, r), (dw, dh), (w0, h0)

def _de_letterbox_xyxy(xyxy: torch.Tensor, ratio: Tuple[float,float], pad: Tuple[int,int], orig_wh: Tuple[int,int]):
    """
    将 letterbox 空间的 xyxy（像素）还原到原图像素坐标。
    """
    if xyxy.numel() == 0:
        return xyxy
    dw, dh = pad
    sx, sy = ratio
    # 去 pad
    xyxy[:, [0,2]] -= dw
    xyxy[:, [1,3]] -= dh
    # 除缩放
    xyxy[:, [0,2]] = xyxy[:, [0,2]] / sx
    xyxy[:, [1,3]] = xyxy[:, [1,3]] / sy
    # 裁边界
    W, H = orig_wh
    xyxy[:, [0,2]] = xyxy[:, [0,2]].clamp(0, W - 1)
    xyxy[:, [1,3]] = xyxy[:, [1,3]].clamp(0, H - 1)
    return xyxy

# ======= 新增：单图推理工具 =======
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def _infer_one_image(model, img_path: Path, input_size: int, conf_thres: float, iou_thres: float):
    im = Image.open(img_path).convert('RGB')
    letter, ratio, pad, orig_wh = _letterbox(im, new_size=input_size)

    arr = torch.from_numpy(letter).permute(2, 0, 1).float() / 255.0  # [3,H,W]
    arr = arr.unsqueeze(0)                                           # [1,3,H,W] 先加 batch 维
    arr = (arr - _IMAGENET_MEAN.to(arr.device)) / _IMAGENET_STD.to(arr.device)  # 与 (1,3,1,1) 对齐
    arr = arr.cuda().half()

    with torch.no_grad(), torch.amp.autocast('cuda'):
        out_all = model(arr)
        outputs = out_all[0] if (isinstance(out_all, (tuple, list)) and len(out_all) == 2) else out_all
        det = util.non_max_suppression(outputs, confidence_threshold=conf_thres, iou_threshold=iou_thres)[0]
        if det is None or det.numel() == 0:
            return torch.zeros((0,6), device='cuda')

    det = det.clone()
    det[:, :4] = _de_letterbox_xyxy(det[:, :4], ratio, pad, orig_wh)
    return det


# ======= 新增：保存检测结果 =======
def _save_det_txt_json(det: torch.Tensor, save_txt: Path, save_json: Path):
    """
    TXT 格式：每行 "cls conf x1 y1 x2 y2"
    JSON 格式：{"detections":[{"cls":int,"conf":float,"bbox":[x1,y1,x2,y2]}, ...]}
    """
    save_txt.parent.mkdir(parents=True, exist_ok=True)
    save_json.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    items = []
    for i in range(det.size(0)):
        x1,y1,x2,y2,conf,cls = det[i].tolist()
        line = f"{int(cls)} {conf:.6f} {int(round(x1))} {int(round(y1))} {int(round(x2))} {int(round(y2))}"
        lines.append(line)
        items.append({"cls": int(cls), "conf": float(conf),
                      "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]})
    with open(save_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump({"detections": items}, f, ensure_ascii=False, indent=2)

# ======= 新增：detect 分支 =======
@torch.no_grad()
def detect(args, params):
    """
    批量对目录 (--detect-src) 下图片推理，保存相对路径一致的 det 结果到 --save-det-dir
    """
    assert args.detect_src is not None, "--detect 需要提供 --detect-src"
    det_root = Path(args.save_det_dir) if args.save_det_dir else Path(args.run_root) / "detect"
    det_root.mkdir(parents=True, exist_ok=True)

    # 加载权重
    if args.weights and os.path.isfile(args.weights):
        ckpt = torch.load(args.weights, map_location="cuda", weights_only=False)
        model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    else:
        # 默认用当前 run 的 best.pt
        default_best = Path(args.save_dir) / "best.pt"
        if default_best.exists():
            ckpt = torch.load(default_best, map_location="cuda", weights_only=False)
            model = ckpt['model']
        else:
            raise FileNotFoundError("未找到权重，请通过 --weights 指定，或先训练生成 best.pt")
    model = model.float().fuse().half().eval().cuda()

    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    src_root = Path(args.detect_src)
    files = []
    for ext in exts:
        files += list(src_root.rglob(ext))
    files = sorted(files)
    if len(files) == 0:
        print(f"[detect] 在 {src_root} 下未找到图片")
        return

    pbar = tqdm.tqdm(files, desc=f"[detect] saving to {str(det_root)}")
    for p in pbar:
        det = _infer_one_image(model, p, args.input_size, args.conf_thres, args.iou_thres)
        # 相对路径
        rel = p.relative_to(src_root)
        # 保存为 .txt / .json
        save_txt = det_root / rel.with_suffix(".txt")
        save_json = det_root / rel.with_suffix(".json")
        _save_det_txt_json(det, save_txt, save_json)



def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--data-root', default="/root/autodl-tmp/Dataset/racketpose/data", type=str)   # <<< 新增
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    # parser.add_argument('--device', default='0', type=str, help="GPU ids to use, e.g. '0' or '0,1,2,3'")
    
    # ======= 新增：detect / crop 相关参数 =======
    parser.add_argument('--detect', action='store_true', help='运行检测分支')
    parser.add_argument('--detect-src', type=str, default="/root/autodl-tmp/Dataset/racketpose/data/imgs", help='需要检测的图片目录（递归）')
    parser.add_argument('--save-det-dir', type=str, default="/root/autodl-tmp/Dataset/racketpose/detect/imgs", help='检测结果输出根目录（默认 runs/当前时间/detect）')
    parser.add_argument('--weights', type=str, default="/root/autodl-tmp/yolov11/runs/yolov11_x/best.pt", help='推理时加载的权重路径（.pt），默认用当前 run 的 best.pt')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='NMS 置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU 阈值')

    # parser.add_argument('--crop-from-dets', type=str, default="/root/autodl-tmp/Dataset/racketpose/detect/imgs", help='读取该目录下的 det 结果（detect 输出）进行裁剪')
    # parser.add_argument('--crop-src-root', type=str, default="/root/autodl-tmp/Dataset/racketpose/data/imgs", help='原图根目录（与 detect-src 一致）')
    # parser.add_argument('--crop-dst-root', type=str, default="/root/autodl-tmp/Dataset/racketpose/cropx2/imgs", help='裁剪输出根目录（将按原相对路径创建子目录）')
    # parser.add_argument('--crop-mode', type=str, default='double', choices=['bbox','double'], help='裁剪模式')
    # parser.add_argument('--crop-conf-thres', type=float, default=0.2, help='裁剪时的最小置信度阈值（过滤框）')

    
    args = parser.parse_args()
    
    # # ---------- 设置 CUDA 设备 ----------
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # device_count = torch.cuda.device_count()

    # # 解析 GPU id 列表
    # device_ids = [int(i) for i in args.device.split(',')]
    # args.num_gpus = len(device_ids)
    # print(f"[INFO] Using GPU device(s): {device_ids} (Total visible: {device_count})")

    # if not torch.cuda.is_available():
    #     raise RuntimeError("CUDA is not available. Please check your environment.")

    # # 设置默认 GPU
    # torch.cuda.set_device(device_ids[0])

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0 and not os.path.exists('weights'):
        os.makedirs('weights')

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)
     
    # ==== 禁用 warmup，恒定学习率 ====
    # 1) 保证 warmup_epochs 为 0
    params['warmup_epochs'] = 0

    # 2) 让线性调度器的 max_lr == min_lr -> 恒定 LR
    #   （不想改 args.yaml 的情况下，这里兜底）
    params['min_lr'] = 1e-3
    params['max_lr'] = float(params['min_lr'])  # 关键：强制相等

    # （可选）打印一下，方便确认
    if args.local_rank == 0:
        print(f"[LR] epochs={args.epochs} | warmup_epochs={params['warmup_epochs']} "
            f"| min_lr=max_lr={params['min_lr']}") 
       

    util.setup_seed()
    util.setup_multi_processes()
    
    # —— 运行目录（按时间戳）& TensorBoard —— 
    ts = datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S")
    run_root = os.path.join("runs", f"train-{ts}")
    if args.local_rank == 0:
        os.makedirs(run_root, exist_ok=True)
        os.makedirs(os.path.join(run_root, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_root, "tb"), exist_ok=True)
    args.run_root = run_root                    # 例如 runs/train-2025_10_20-143256
    args.save_dir = os.path.join(run_root, "weights")
    args.tb_dir   = os.path.join(run_root, "tb")

    # profile 可选
    # profile(args, params)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)
    # if args.detect:
    #     detect(args, params)
    # if args.crop_from_dets:
    #     crop_from_detections(args)
    

    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
