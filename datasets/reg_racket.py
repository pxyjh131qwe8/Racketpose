from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _train_tfms(img_size: int = 512):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _val_tfms(img_size: int = 512):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class RacketMultiTaskDataset(Dataset):
    """
    仅加载图像 + 回归值 + 类别，不加载/解析任何 2D 标注框。
    读取 labels/<split>/*.csv
    必需列：filename, dist, angle_x, angle_y, angle_z, label

    输出 (保持与原训练循环的解包兼容)：
      img: Tensor[3,H,W]
      reg: Tensor[4]  -> [dist(m), angle_x(°), angle_y(°), angle_z(°)]
      cls: LongTensor  (标量)
      boxes: Tensor[0,4]  空
      box_labels: LongTensor[0]  空
    """
    def __init__(self,
                 data_root,
                 split,
                 img_size,
                 mm_to_m: bool = False,
                 drop_missing: bool = True):
        self.root = Path(data_root)
        self.labels_dir = self.root / "labels" / split
        assert self.labels_dir.exists(), f"不存在：{self.labels_dir}"
        self.img_size = int(img_size)
        self.tfms = _train_tfms(self.img_size) if split == "train" else _val_tfms(self.img_size)

        self.items: List[Tuple[Path, float, float, float, float, int]] = []
        csv_files = sorted(self.labels_dir.glob("*.csv"))
        assert csv_files, f"{self.labels_dir} 下未找到 csv"

        for csv_path in csv_files:
            with open(csv_path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                need = ["filename", "dist", "angle_x", "angle_y", "angle_z", "label"]
                if not r.fieldnames or any(k not in r.fieldnames for k in need):
                    raise ValueError(f"{csv_path} 缺列（需要 {need}），实际 {r.fieldnames}")
                for row in r:
                    p = row["filename"].replace("\\", "/")
                    img_path = (self.root / p) if not p.startswith("/") else Path(p)
                    try:
                        dist  = float(row["dist"]) if row["dist"] != "" else float("nan")
                        ax    = float(row["angle_x"])
                        ay    = float(row["angle_y"])
                        az    = float(row["angle_z"])
                        label = int(float(row["label"]))
                    except Exception:
                        continue
                    if mm_to_m and not (isinstance(dist, float) and (dist != dist)):  # 非 NaN
                        dist = dist / 1000.0
                    self.items.append((img_path, dist, ax, ay, az, label))

        if drop_missing:
            before = len(self.items)
            self.items = [it for it in self.items if it[0].exists()]
            after = len(self.items)
            if after < before:
                print(f"[Dataset] 丢弃缺图样本 {before-after} 条。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, dist, ax, ay, az, label = self.items[idx]
        img_pil = Image.open(img_path).convert("RGB")
        img = self.tfms(img_pil)  # -> [3, img_size, img_size]

        reg = torch.tensor([dist, ax, ay, az], dtype=torch.float32)
        cls = torch.tensor(label, dtype=torch.long)

        # 为保持训练/验证循环的解包兼容，这里返回空框
        empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
        empty_labels = torch.zeros((0,), dtype=torch.long)
        return img, reg, cls, empty_boxes, empty_labels


# --------- 简单 collate（不做可变长框拼接，直接返回空列表） ----------
def collate_no_boxes(batch):
    """
    batch: List of (img, reg, cls, empty_boxes, empty_labels)

    返回：
      imgs: Tensor[B, 3, H, W]
      regs: Tensor[B, 4]
      clss: LongTensor[B]
      boxes_list: List[Tensor[0,4]]  全为空
      labels_list: List[LongTensor[0]]  全为空
    """
    imgs, regs, clss, boxes, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    regs = torch.stack(regs, dim=0)
    clss = torch.stack(clss, dim=0)
    # 直接把每样本的空张量打包成列表，满足主训练循环的解包需要
    return imgs, regs, clss, list(boxes), list(labels)


# ---------------- loader 构建 ----------------
def build_loaders(
    data_root: str | Path,
    img_size: int = 512,
    batch_size: int = 16,
    num_workers: int = 4,
    mm_to_m: bool = False,
):
    train_set = RacketMultiTaskDataset(data_root, "train", img_size, mm_to_m)
    val_set   = RacketMultiTaskDataset(data_root, "val",   img_size, mm_to_m)
    test_set  = RacketMultiTaskDataset(data_root, "test",  img_size, mm_to_m)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_no_boxes,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_no_boxes,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_no_boxes,
    )
    return train_loader, val_loader, test_loader


def build_loader(split, args, params, shuffle):
    dataset = RacketMultiTaskDataset(
        data_root=args.data_root,
        split=split,
        img_size=args.input_size,
        mm_to_m=False,
    )
    sampler = None
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_no_boxes,
    )
    return loader, sampler
