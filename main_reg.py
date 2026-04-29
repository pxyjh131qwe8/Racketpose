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

from nets import nn_reg
from utils import util
from datasets.reg_racket import build_loader  

warnings.filterwarnings("ignore")


# ---------- angles & distance helpers ----------
def _deg_to_norm(deg):     # [-180,180] -> [0,1]
    return (deg + 180.0) / 360.0

def _norm_to_deg(x):       # [0,1] -> [-180,180]
    return x * 360.0 - 180.0

def _wrap_diff_deg(pred_deg, tgt_deg):
    # wrap 到 [-180,180]
    return (pred_deg - tgt_deg + 180.0).remainder(360.0) - 180.0

def _smooth_huber(x, y, delta):
    d = (x - y).abs()
    return torch.where(d < delta, 0.5 * (d**2) / delta, d - 0.5 * delta).mean()

def _circular_huber_on_norm(pred_norm, tgt_norm, delta=0.05):
    # pred_norm/tgt_norm ∈ [0,1] 圆周距离
    d = (pred_norm - tgt_norm).abs()
    d = torch.minimum(d, 1.0 - d)
    return torch.where(d < delta, 0.5 * (d**2) / delta, d - 0.5 * delta).mean()

def _expm1_clamp(x):       # log(1+dist) -> dist
    return torch.expm1(x).clamp_min(0.0)

def _acc_thresholds_ang_and_dist(abs_err_deg: torch.Tensor,
                                 dist_abs_err_m: torch.Tensor,
                                 thresholds=(1, 2, 5, 10),
                                 dist_thr: float = 0.05):
    """
    同时满足：
      - 角度绝对误差 <= t（度）
      - 距离绝对误差 <= dist_thr（米，默认 0.05m=5cm）
    逐角度分量统计（x/y/z 一起计数）
    """
    dist_mask = (dist_abs_err_m <= dist_thr).unsqueeze(-1).expand_as(abs_err_deg)
    tot = abs_err_deg.numel()
    out = {}
    for t in thresholds:
        ang_mask = (abs_err_deg <= t)
        both_ok = ang_mask & dist_mask
        out[f"acc@{int(t)}"] = float(both_ok.sum().item()) / max(1, tot)
    return out

import torch

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



def train(args, params):
    # Model（已实现为“无检测”的 yolo_v11_x -> YOLOAngleDist）
    model = nn_reg.yolo_v11_x()
    model.cuda()

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
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    best = 0.0
    amp_scale = torch.amp.GradScaler()

    with open(os.path.join(args.save_dir, 'step.csv'), 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=[
                'epoch', 'angle_loss', 'dist_loss', 'angle_MSE', 'dist_MSE', 'acc@1', 'acc@2', 'acc@5', 'acc@10'
            ])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)

            p_bar = enumerate(loader)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 6) % ('epoch', 'memory', 'loss_ang', 'loss_dist', 'A-MAE', 'D-MAE'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_ang_loss = util.AverageMeter()
            avg_dis_loss = util.AverageMeter()

            # 回归指标累计
            ang_abs_sum = 0.0
            ang_sq_sum  = 0.0
            ang_cnt     = 0
            dist_abs_sum= 0.0
            dist_sq_sum = 0.0
            dist_cnt    = 0

            for i, (samples, regs, clss, _boxes_list, _labels_list) in p_bar:
                # 这里只保留回归，不再使用 boxes/labels
                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda(non_blocking=True).float()  # 已 Normalize
                regs    = regs.cuda(non_blocking=True).float()
                dist_t  = regs[:, 0]       # 真实距离（米，>0）
                ang_t   = regs[:, 1:4]     # 真实角度（度，[-180,180]）

                # Forward
                with torch.amp.autocast('cuda'):
                    out = model(samples)              # dict: {"angle_norm": [B,3], "dist_log": [B,1]}
                    angle_norm = out["angle_norm"]
                    dist_log   = out["dist_log"].squeeze(-1)

                    # 损失
                    loss_ang  = _circular_huber_on_norm(angle_norm, _deg_to_norm(ang_t), delta=0.05)
                    loss_dist = _smooth_huber(dist_log, torch.log1p(dist_t.clamp_min(0.0)), delta=0.2)
                    loss = loss_ang + loss_dist

                avg_ang_loss.update(loss_ang.item(), samples.size(0))
                avg_dis_loss.update(loss_dist.item(), samples.size(0))

                # Backward & Optimize
                amp_scale.scale(loss).backward()
                if step % accumulate == 0:
                    amp_scale.step(optimizer)
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # 训练期指标（用于屏显）
                with torch.no_grad():
                    ang_pred = _norm_to_deg(angle_norm.float())
                    ang_err  = _wrap_diff_deg(ang_pred, ang_t).abs()
                    ang_abs_sum += ang_err.sum().item()
                    ang_sq_sum  += (ang_err ** 2).sum().item()
                    ang_cnt     += ang_err.numel()

                    dist_pred   = _expm1_clamp(dist_log.float())
                    d_err       = (dist_pred - dist_t).abs()
                    dist_abs_sum+= d_err.sum().item()
                    dist_sq_sum += (d_err ** 2).sum().item()
                    dist_cnt    += d_err.numel()

                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'
                    a_maae = (ang_abs_sum / max(1, ang_cnt)) if ang_cnt else 0.0
                    d_mae  = (dist_abs_sum / max(1, dist_cnt)) if dist_cnt else 0.0
                    s = ('%10s' * 2 + '%10.3g' * 4) % (
                        f'{epoch + 1}/{args.epochs}', memory,
                        avg_ang_loss.avg, avg_dis_loss.avg, a_maae, d_mae
                    )
                    p_bar.set_description(s)

                    # TensorBoard
                    if writer is not None:
                        writer.add_scalar("train/loss_angle", float(loss_ang.detach()), step)
                        writer.add_scalar("train/loss_dist",  float(loss_dist.detach()), step)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], step)

            # ===== 验证 =====
            if args.local_rank == 0:
                val_A_MSE, val_D_MSE, acc_dict = test(args, params, ema.ema if ema else model)

                if writer is not None:
                    writer.add_scalar("val/angle_MSE",  val_A_MSE, epoch)
                    writer.add_scalar("val/distance_MSE", val_D_MSE, epoch)
                    for k, v in acc_dict.items():
                        writer.add_scalar(f"val/{k}", float(v), epoch)

                print(
                    f"Val | A-MSE: {val_A_MSE:.3f} | D-MSE: {val_D_MSE:.3f} | "
                    + " ".join([f"{k}:{v*100:.1f}%" for k, v in acc_dict.items()])
                )

                # 保存 last
                save = {'epoch': epoch + 1, 'model': copy.deepcopy(ema.ema if ema else model)}
                torch.save(save, os.path.join(args.save_dir, 'last.pt'))

                # 以 acc@10 作为 best 判据
                acc10 = float(acc_dict.get("acc@10", 0.0))
                best_acc10 = getattr(train, "_best_acc10", float("-inf"))
                if acc10 > best_acc10:
                    torch.save(save, os.path.join(args.save_dir, 'best.pt'))
                    train._best_acc10 = acc10
                    print(f"[Best] acc@10 提升为 {acc10*100:.2f}% ，已保存为 best.pt")
                del save

                # （可选）按 epoch 命名额外留档
                ckpt_name = (
                    f'./weights/ep{epoch+1:03d}-A-MSE{val_A_MSE:.3f}-D-MSE{val_D_MSE:.3f}-acc10{acc10:.4f}.pt'
                )
                torch.save({'epoch': epoch + 1, 'model': copy.deepcopy(ema.ema if ema else model)}, ckpt_name)

    if writer is not None:
        writer.close()


@torch.no_grad()
def test(args, params, model=None):
    loader, _ = build_loader("test", args, params, shuffle=False)

    if not model:
        # 兜底：从默认 best 加载
        # model = torch.load(os.path.join(args.save_dir, 'best.pt'),
        #                    map_location='cuda', weights_only=False)['model'] 
        # model = torch.load("/root/autodl-tmp/yolov11/runs/train-2025_10_29-152031/weights/best.pt", map_location='cuda', weights_only=False)['model']
        # 原图regression测试
        # model = torch.load("/root/autodl-tmp/yolov11/runs/train-2025_11_04-134142/weights/best.pt", map_location='cuda', weights_only=False)['model']
        # crop1.0regression测试
        # model = torch.load("/root/autodl-tmp/yolov11/crop_weights/crop1.0/best.pt", map_location='cuda', weights_only=False)['model']
        # crop1.5regression测试
        # model = torch.load("/root/autodl-tmp/yolov11/crop_weights/crop1.5/best.pt", map_location='cuda', weights_only=False)['model']
        # crop2.0regression测试
        # model = torch.load("/root/autodl-tmp/yolov11/crop_weights/crop2.0/best.pt", map_location='cuda', weights_only=False)['model']
        # crop2.5regression测试
        # model = torch.load("/root/autodl-tmp/yolov11/crop_weights/crop2.5/best.pt", map_location='cuda', weights_only=False)['model'] 
        # anncrop1.5 
        model = torch.load("/root/autodl-tmp/yolov11/runs/train-2025_11_07-210621/weights/best.pt", map_location='cuda', weights_only=False)['model']
        model = model.float().fuse()

    model.half().eval()

    # 聚合角度/距离指标
    va_ang_abs_sum = va_ang_sq_sum = 0.0
    va_ang_cnt     = 0
    va_dist_abs_sum= va_dist_sq_sum= 0.0
    va_dist_cnt    = 0
    acc_thrs = (1,2,5,10)
    acc_cnt  = {f"acc@{t}":0.0 for t in acc_thrs}
    total_samples = 0   # <<< 新增：累计样本数用于 acc 汇总

    p_bar = tqdm.tqdm(loader, desc=('%10s' * 4) % ('', 'A-MSE', 'D-MSE', 'acc@10'))
    for samples, regs, _clss, _boxes_list, _labels_list in p_bar:
        samples = samples.cuda().half()
        regs    = regs.cuda().float()
        dist_t  = regs[:, 0]
        ang_t   = regs[:, 1:4]

        out = model(samples)
        angle_norm = out["angle_norm"]
        dist_log   = out["dist_log"].squeeze(-1)

        ang_pred = _norm_to_deg(angle_norm.float())
        ang_err  = _wrap_diff_deg(ang_pred, ang_t).abs()   # [B,3]
        va_ang_abs_sum += ang_err.sum().item()
        va_ang_sq_sum  += (ang_err**2).sum().item()
        va_ang_cnt     += ang_err.numel()

        dist_pred = _expm1_clamp(dist_log.float())
        d_err = (dist_pred - dist_t).abs()
        va_dist_abs_sum += d_err.sum().item()
        va_dist_sq_sum  += (d_err**2).sum().item()
        va_dist_cnt     += d_err.numel()

        # 同时满足（5cm & k°）的准确率
        # accs = _acc_thresholds_ang_and_dist(ang_err, d_err, acc_thrs, dist_thr=0.05)
        # for k,v in accs.items():
        #     acc_cnt[k] += v * ang_err.numel()  # 累计加权计数
        # === 新：基于“合角度差”的 ACC 判定 ===
        # 1) 三轴角（预测/标签）→ 合角度
        ang_pred_deg = _norm_to_deg(angle_norm.float())      # [B,3]  预测三轴角（度）
        theta_pred   = _compose_total_from_xyz_torch(ang_pred_deg)   # [B]
        theta_true   = _compose_total_from_xyz_torch(ang_t)          # [B]

        # 2) 角度与距离的绝对误差
        angle_total_err = (theta_pred - theta_true).abs()            # [B]
        dist_pred       = _expm1_clamp(dist_log.float())             # [B]
        d_err           = (dist_pred - dist_t).abs()                 # [B]

        # 3) ACC（按 batch 比例），我们按样本数聚合
        accs = _acc_thresholds_total_and_dist(angle_total_err, d_err, acc_thrs, dist_thr=0.05)
        B = angle_total_err.numel()
        total_samples += B   # <<< 新增累计样本数
        for k, v in accs.items():
            # v 是该 batch 的比例 -> 折算为“满足样本数”，最终再除以总样本数
            acc_cnt[k] += v * B

        # === 其余原有的 A/D 误差统计维持不变（前面已有 per-axis ang_err 和 d_err 的 MSE 统计）===


    # 聚合均值
    val_A_MSE = math.sqrt(va_ang_sq_sum / max(1, va_ang_cnt))
    val_D_MSE = math.sqrt(va_dist_sq_sum / max(1, va_dist_cnt))
    # acc_dict  = {k: (acc_cnt[k] / max(1, va_ang_cnt)) for k in acc_cnt}
    # <<< 修复：改为按样本数聚合，避免角度分量数放大
    acc_dict  = {k: (acc_cnt[k] / max(1, total_samples)) for k in acc_cnt}
    
    model.float()  # 以便继续训练

    print(f"Angle MSE: {val_A_MSE:.3f} | Distance MSE: {val_D_MSE:.3f}")
    print(" | ".join([f"{k}:{v*100:.1f}%" for k, v in acc_dict.items()]))

    return float(val_A_MSE), float(val_D_MSE), acc_dict


def profile(args, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    model = nn_reg.yolo_v11_x().fuse()
    model.eval()
    model(torch.zeros(shape))
    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")
    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--data-root', default="/root/autodl-tmp/Dataset/racketpose/cropannx1.5", type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    # —— 去掉 detect/crop 相关全部参数 ——

    args = parser.parse_args()

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

    util.setup_seed()
    util.setup_multi_processes()

    # —— 运行目录（按时间戳）& TensorBoard ——
    ts = datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S")
    run_root = os.path.join("runs", f"train-{ts}")
    if args.local_rank == 0:
        os.makedirs(run_root, exist_ok=True)
        os.makedirs(os.path.join(run_root, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_root, "tb"), exist_ok=True)
    args.run_root = run_root
    args.save_dir = os.path.join(run_root, "weights")
    args.tb_dir   = os.path.join(run_root, "tb")

    # profile(args, params)  # 可选

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)

    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
