#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本（ViTMultiTask）
- 读取 cfg（python 配置），构建并加载模型（用你已经实现的 build_vit_multitask_from_cfg）
- 使用 datasets.multitask_racket.build_loaders_for_vit_multitask 构建 data loader
- 只评估测试集：mAP、角度 MAAE/RMSE（度）、距离 MAE/RMSE（米）
"""

import os, math, runpy
import argparse
from types import SimpleNamespace as NS
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F

from nets.nn import build_model
from datasets.multitask_racket import RacketMultiTaskDataset, build_loaders
from test_utils import multi_task_test


def _to_ns(d: dict) -> NS:
    return NS(**d)

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

@torch.no_grad()
def average_precision_binary(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    # y_true: [N] ∈ {0,1}, y_score: [N]∈R
    if y_true.numel() == 0 or y_true.sum() == 0:
        return float('nan')
    order = torch.argsort(y_score, descending=True)
    y = y_true[order]
    tp = y.cumsum(0)
    pos = y_true.sum()
    idx = (y == 1).nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return float('nan')
    prec_at_k = tp[idx].float() / (idx + 1).float()
    return float(prec_at_k.mean().item())

@torch.no_grad()
def compute_map_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Tuple[float, Dict[int, float]]:
    probs = logits.float().softmax(dim=-1)
    ap_dict, aps = {}, []
    for c in range(num_classes):
        y_true = (labels == c).long()
        ap = average_precision_binary(y_true, probs[:, c])
        ap_dict[c] = ap
        if not math.isnan(ap):
            aps.append(ap)
    mAP = float(sum(aps) / max(1, len(aps))) if aps else 0.0
    return mAP, ap_dict

def norm_to_deg(norm: torch.Tensor) -> torch.Tensor:
    return norm * 360.0 - 180.0

def wrap_diff_deg(pred_deg: torch.Tensor, tgt_deg: torch.Tensor) -> torch.Tensor:
    return (pred_deg - tgt_deg + 180.0).remainder(360.0) - 180.0

def expm1_clamp(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x).clamp_min(0.0)


# ============ 测试主流程 ============

def evaluate_testset():
    
    # CFG_PATH = "configs/vit_multitask_racket.py"
    args = parse_args()
    
    # 在任何 torch.cuda 调用之前优先处理 --gpus
    if args.gpus is not None:
        # 例如 "0" 或 "0,1"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()
    import torch     

    cfg = _to_ns(load_cfg_py(args.cfg))

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
    
    # # 1) 读取配置（纯配置驱动）
    # cfg = load_cfg_py(cfg_path)

    # 必要字段（若缺省则给默认）
    data_root    = getattr(cfg, "data_root", "./data")
    img_size     = getattr(cfg, "img_size", 224)
    batch_size   = getattr(cfg, "batch_size_test", getattr(cfg, "batch_size", 64))
    num_workers  = getattr(cfg, "num_workers_test", getattr(cfg, "num_workers", 4))
    mm_to_m      = getattr(cfg, "mm_to_m", False)
    num_classes  = getattr(cfg, "num_classes", 4)
    normalize_boxes = getattr(cfg, "normalize_boxes", True)
    class_to_idx = getattr(cfg, "class_to_idx", None)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) 构建/加载模型（权重路径从 cfg.checkpoint 读取）
    print("=> Building model & loading checkpoint from cfg ...")
    model_cfg = getattr(cfg, "model", cfg) 
    model = build_model(
        model_cfg,
    )
    model = model.to(device)
    model.eval()

    # 3) DataLoader（只用 test）
    print("=> Building dataloaders ...")
    _, _, test_loader = build_loaders(
        data_root=data_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        mm_to_m=mm_to_m,
        normalize_boxes=normalize_boxes,
        class_to_idx=class_to_idx,
    )

    # 4) 测试循环
    multi_task_test(
        model_train=model,
        epoch=0,
        epoch_step_test=len(test_loader),
        gen_test=test_loader,
        Epoch=1,
        cuda=torch.cuda.is_available(),
        local_rank=0,
        acc_thresholds=getattr(cfg, "acc_thresholds", (1.0, 2.0, 5.0, 10.0)),
    )
    

# main 
def parse_args():
    ap = argparse.ArgumentParser()
    # 模型配置
    ap.add_argument("--cfg", type=str, default="configs/yolov11_multitask_n_test.py",
                    help="ViT MultiTask 的 Python 配置路径，例如 configs/vit_multitask_racket.py")
    # 新增：指定 GPU / 设备
    ap.add_argument("--gpus", type=str, default=None,
                    help="逗号分隔的 GPU 索引，例如 '0' 或 '0,1'。会设置 CUDA_VISIBLE_DEVICES，并在>1时启用DataParallel。")
    ap.add_argument("--device", type=str, default="cuda",
                    help="显式设备，如 'cuda'、'cuda:1' 或 'cpu'。若与 --gpus 同时给，优先采用 --gpus。")
    return ap.parse_args()

if __name__ == "__main__":
    evaluate_testset()
