# ( global-only)
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

from nets import nn_global as pose_nn   
from utils import util
from datasets.racketpose2 import build_loader  # pose dataset build_loader

warnings.filterwarnings("ignore")


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
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


# ---------------- Pose Loss  ----------------
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


# ---------------- evaluation (GLOBAL-ONLY) ----------------
@torch.no_grad()
def eval_pose_global(
    model,
    loader,
    device="cuda",
    pose_thresholds=((0.05, 5.0), (0.05, 10.0)),
):
    """
    model.eval() ：
      - center_m: [B,3]
      - normal:   [B,3] (unit)
      - cls_prob: [B,nc]
    targets:
      - label [B]
      - center_m [B,3]
      - normal [B,3]
    """
    model.eval()

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

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(imgs)  # global-only

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
    nc = len(params["names"])

    # global-only model
    model = pose_nn.global_pose_v11_x(
        num_classes=nc,
        img_size=args.input_size,
        roi_ch=args.roi_ch,
        global_from=args.global_from,
    ).to(device)

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

    # loaders (train computes stats; val/test reuse train stats)
    train_loader, train_sampler, train_set = build_loader("train", args, params, shuffle=True, center_stats=None)
    train_stats = (train_set.center_mean, train_set.center_std)

    eval_split = "test" if getattr(args, "eval_split", "test") == "test" else "val"
    eval_loader, _, _ = build_loader(eval_split, args, params, shuffle=False, center_stats=train_stats)

    _unwrap(model).set_center_stats(train_set.center_mean.tolist(), train_set.center_std.tolist(), denorm_inference=True)

    # resume
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

    # ddp
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
                k5, k10
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

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pred = model(samples)   # global-only
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

            # -------- eval & save --------
            if args.local_rank == 0:
                eval_model = ema.ema if ema else model

                metrics = eval_pose_global(
                    _unwrap(eval_model),
                    eval_loader,
                    device=device,
                    pose_thresholds=pose_thresholds
                )

                epoch_loss = float(avg_loss.avg)

                k5 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr_small)}deg"
                k10 = f"pose+cls@{int(args.pose_center_thr*100)}cm_{int(args.pose_angle_thr)}deg"
                print(f"[{eval_split.upper()}] epoch_loss={epoch_loss:.6f} | best_loss={best_loss:.6f} | "
                      f"acc={metrics['acc']:.3f} | center_l2={metrics['center_l2_m']:.4f}m | "
                      f"normal={metrics['normal_deg']:.2f}° | {k5}={metrics.get(k5, float('nan')):.3f} | "
                      f"{k10}={metrics.get(k10, float('nan')):.3f}")

                # save last
                model_to_save = copy.deepcopy(_unwrap(eval_model).float())
                save = {
                    "epoch": epoch + 1,
                    "best_loss": float(best_loss),
                    "model": model_to_save,
                    "optimizer": optimizer.state_dict(),
                    "scaler": amp_scale.state_dict(),
                }
                torch.save(save, os.path.join(args.save_dir, "last.pt"))

                # save best by lowest train epoch loss
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
    ckpt = torch.load(args.weight, map_location="cuda", weights_only=False)
    model = ckpt["model"].float().fuse().to(device).eval()

    stats = None
    if hasattr(model, "center_mean") and hasattr(model, "center_std"):
        stats = (model.center_mean.detach().cpu().view(3), model.center_std.detach().cpu().view(3))

    test_loader, _, _ = build_loader("test", args, params, shuffle=False, center_stats=stats)

    pose_thresholds = (
        (args.pose_center_thr, args.pose_angle_thr_small),  # 5cm5deg
        (args.pose_center_thr, args.pose_angle_thr),        # 5cm10deg
    )

    metrics = eval_pose_global(model, test_loader, device=device, pose_thresholds=pose_thresholds)

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

    #  global-only model args
    parser.add_argument("--roi-ch", type=int, default=256) 
    parser.add_argument("--global-from", type=str, default="p5", choices=["p5", "p345"])

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
    run_root = os.path.join("runs", f"pose-global-{ts}")
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
        assert args.weight is not None, "--test need --weight  ckpt"
        test(args, params)

    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
