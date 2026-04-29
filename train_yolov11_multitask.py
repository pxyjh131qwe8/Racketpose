#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, random, argparse
from types import SimpleNamespace as NS
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import runpy
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nets.nn import build_model
from datasets.multitask_racket import RacketMultiTaskDataset, build_loaders
from train_utils import multi_task_fit_one_epoch


# config 
def load_cfg_py(cfg_path: str) -> dict:
    """执行 python 配置，支持 _base_ 继承（相对路径），返回合并后的字典。"""
    cfg_path = str(Path(cfg_path).resolve())

    def _load_one(path: str, acc: dict) -> dict:
        d = runpy.run_path(path)
        bases = d.get("_base_")
        if bases:
            if isinstance(bases, (str, Path)):
                bases = [bases]
            for base in bases:
                acc = _load_one(str((Path(path).parent / str(base)).resolve()), acc)
        for k, v in d.items():
            if not k.startswith("_"):
                acc[k] = v
        return acc

    return _load_one(cfg_path, {})

def to_ns(d: dict) -> NS:
    return NS(**d)

# utils
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了速度，这里不强制确定性
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# losshistory 适配 tensorboard
class TBHistory:
    """
    适配你的 fit_one_epoch_angle 所需接口：
      - append_loss(epoch, train_loss, val_loss)
      - val_loss: list
      - best_val_maae: 任意属性（由训练函数在首次保存 best 时挂载/更新）
    仅把 loss 写入 TensorBoard。你函数内其他指标维持原样打印与保存。
    """
    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.val_loss = []
        self.train_loss = []
        self.best_val_maae = float('inf')  # 供你的函数首次比较使用

    def append_loss(self, epoch: int, train_loss: float, val_loss: float):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("loss/val",   val_loss,   epoch)


# main 
def parse_args():
    ap = argparse.ArgumentParser()
    # 模型配置
    ap.add_argument("--cfg", type=str, default="configs/yolov11_multitask_n.py",
                    help="ViT MultiTask 的 Python 配置路径，例如 configs/vit_multitask_racket.py")
    # 新增：指定 GPU / 设备
    ap.add_argument("--gpus", type=str, default=None,
                    help="逗号分隔的 GPU 索引，例如 '0' 或 '0,1'。会设置 CUDA_VISIBLE_DEVICES，并在>1时启用DataParallel。")
    ap.add_argument("--device", type=str, default="cuda",
                    help="显式设备，如 'cuda'、'cuda:1' 或 'cpu'。若与 --gpus 同时给，优先采用 --gpus。")
    return ap.parse_args()

def main():
    args = parse_args()
    
    # 在任何 torch.cuda 调用之前优先处理 --gpus
    if args.gpus is not None:
        # 例如 "0" 或 "0,1"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()
    import torch     

    cfg = to_ns(load_cfg_py(args.cfg))

    # 计算最终的设备字符串：CLI 优先，其次 cfg.device，默认 'cuda'
    cfg_device = getattr(cfg, "device", "cuda")
    selected_device_str = args.device if args.device is not None else cfg_device

    # 若指定了 --gpus，且 selected_device_str 以 cuda 开头，则统一用 'cuda'
    # （此时 'cuda' 会映射到可见设备中的第 0 张卡）
    if args.gpus is not None and str(selected_device_str).startswith("cuda"):
        selected_device_str = "cuda"

    # 构造 torch.device
    if str(selected_device_str).startswith("cuda") and torch.cuda.is_available():
        device = torch.device(selected_device_str)
    else:
        device = torch.device("cpu")

    # 打印设备信息（可选）
    print(f"[Device] selected_device_str={selected_device_str}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
          f"cuda_available={torch.cuda.is_available()}")
    
    
    cfg = to_ns(load_cfg_py(args.cfg))
    
    # 读取顶层训练超参数
    seed         = getattr(cfg, "seed", 42)
    # device_str   = getattr(cfg, "device", "cuda")
    # device       = torch.device(device_str if torch.cuda.is_available() else "cpu")
    # ---- save_dir 加时间戳 ----
    base_dir  = Path(getattr(cfg, "save_dir", "runs/yolov11_multitask"))
    exp_name  = getattr(cfg, "exp_name", "yolov11")  # 可在 cfg 里自定义实验名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir  = (base_dir / f"{exp_name}_{timestamp}").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard 日志目录（如未指定则默认在 save_dir/tb）
    log_dir_cfg = getattr(cfg, "log_dir", None)
    log_dir     = Path(log_dir_cfg) if log_dir_cfg else (save_dir / "tb")
    log_dir.mkdir(parents=True, exist_ok=True)
    # save_dir     = Path(getattr(cfg, "save_dir", "runs/vit_multitask")); save_dir.mkdir(parents=True, exist_ok=True)
    # log_dir_cfg  = getattr(cfg, "log_dir", None)
    # log_dir      = Path(log_dir_cfg) if log_dir_cfg else (save_dir / "tb"); log_dir.mkdir(parents=True, exist_ok=True)

    epochs       = getattr(cfg, "epochs", 100)
    batch_size   = getattr(cfg, "batch_size", 32)
    num_workers  = getattr(cfg, "num_workers", 4)
    img_size     = getattr(cfg, "img_size", 512)
    mm_to_m      = bool(getattr(cfg, "mm_to_m", False))
    fp16         = bool(getattr(cfg, "fp16", True)) or bool(getattr(cfg, "amp", False))
    lr           = getattr(cfg, "lr", 1e-4)
    weight_decay = getattr(cfg, "weight_decay", 0.05)
    save_period  = getattr(cfg, "save_checkpoint_interval", 10)
    acc_thresholds = tuple(getattr(cfg, "acc_thresholds", (1.0, 2.0, 5.0, 10.0)))
    normalize_boxes = bool(getattr(cfg, "normalize_boxes", True))
    class_to_idx = getattr(cfg, "class_to_idx", None)

    set_seed(seed)
    writer = SummaryWriter(log_dir=str(log_dir))
    loss_history = TBHistory(writer)
    
    
    # data 
    train_loader, val_loader, test_loader = build_loaders(
        data_root=cfg.data_root, 
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        mm_to_m=mm_to_m,
        normalize_boxes=normalize_boxes,
        class_to_idx=class_to_idx,
    )
    
    # model 模型配置在cfg.model下 
    model_cfg = getattr(cfg, "model", cfg) 
    model = build_model(
        model_cfg,
    )
    model = model.to(device)
    
    # 多卡
    model_train = model
    # if device.type == "cuda" and torch.cuda.device_count() > 1:
    #     model_train = nn.DataParallel(model)
        
    # Optim / Scaler 
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=(fp16 and device.type == "cuda")) 
    
    # Epoch Loop
    epoch_step     = len(train_loader)
    epoch_step_val = len(val_loader)
    Epoch          = epochs
    cuda           = (device.type == "cuda")
    local_rank     = 0 
    
    for epoch in range(epochs):
        multi_task_fit_one_epoch(
            model_train=model_train,
            model=model,
            loss_history=loss_history,
            optimizer=optimizer,
            epoch=epoch,
            epoch_step=epoch_step,
            epoch_step_val=epoch_step_val,
            gen=train_loader,
            gen_val=val_loader,
            Epoch=Epoch,
            cuda=cuda,
            fp16=fp16,
            scaler=scaler,
            save_period=save_period,
            save_dir=str(save_dir),
            local_rank=local_rank,
            acc_thresholds=acc_thresholds
        ) 
    
    writer.flush()
    writer.close()    
    
    
if __name__ == "__main__":
    main()    
















