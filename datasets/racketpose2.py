# datasets/detect_racket.py  (pose version)
from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch.utils import data
from PIL import Image
import torchvision.transforms as T


def _tfms(img_size: int = 640):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),  # 0..1
    ])


class RacketPoseDataset(Dataset):
    """
    每张图一个姿态标签（center + normal + class label）
    CSV 列必须包含：
      filename, center_x,center_y,center_z, normal_x,normal_y,normal_z, label

    返回：
      img: Tensor[3,S,S]
      target: dict{
        "label": LongTensor[],  shape []
        "center_m": FloatTensor[3]
        "center_norm": FloatTensor[3]   # (center_m - mean) / std
        "normal": FloatTensor[3]        # 单位化
      }
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        img_size: int = 640,
        labels_dirname: str = "labels",
        drop_missing: bool = True,
        center_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (mean[3], std[3])
        compute_stats_if_missing: bool = True,
        eps_std: float = 1e-6,
    ):
        self.root = Path(data_root)
        self.split = split
        self.img_size = int(img_size)
        self.tfms = _tfms(self.img_size)
        self.eps_std = float(eps_std)

        labels_path = self.root / labels_dirname / split
        if labels_path.is_dir():
            csv_files = sorted(labels_path.glob("*.csv"))
        elif labels_path.is_file() and labels_path.suffix.lower() == ".csv":
            csv_files = [labels_path]
        else:
            raise FileNotFoundError(f"找不到标签路径：{labels_path}（应为文件夹或csv文件）")
        assert csv_files, f"{labels_path} 下未找到 csv"

        self.items: List[Dict[str, Any]] = []
        required = {"filename", "center_x", "center_y", "center_z", "normal_x", "normal_y", "normal_z", "label"}

        for csv_path in csv_files:
            with open(csv_path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                if not r.fieldnames or not required.issubset(set(r.fieldnames)):
                    raise ValueError(f"{csv_path} 缺列，实际列：{r.fieldnames}，需要：{sorted(required)}")

                for row in r:
                    rel = row["filename"].replace("\\", "/").strip()
                    img_path = (self.root / rel) if not rel.startswith("/") else Path(rel)

                    try:
                        cx = float(row["center_x"])
                        cy = float(row["center_y"])
                        cz = float(row["center_z"])
                        nx = float(row["normal_x"])
                        ny = float(row["normal_y"])
                        nz = float(row["normal_z"])
                        lab = int(float(row["label"]))  # 允许 "0.0"
                    except Exception:
                        continue

                    self.items.append({
                        "img_path": img_path,
                        "center_m": (cx, cy, cz),
                        "normal": (nx, ny, nz),
                        "label": lab,
                    })

        if drop_missing:
            before = len(self.items)
            self.items = [it for it in self.items if Path(it["img_path"]).exists()]
            after = len(self.items)
            if after < before:
                print(f"[PoseDataset:{split}] 丢弃缺图样本 {before-after} 条。")

        assert len(self.items) > 0, f"[PoseDataset:{split}] 没有可用样本（items=0）"

        # --- center 标准化统计量 ---
        if center_stats is not None:
            mean, std = center_stats
            self.center_mean = mean.detach().cpu().float().view(3)
            self.center_std = std.detach().cpu().float().view(3).clamp_min(self.eps_std)
        else:
            if compute_stats_if_missing:
                self.center_mean, self.center_std = self._compute_center_stats()
            else:
                self.center_mean = torch.zeros(3, dtype=torch.float32)
                self.center_std = torch.ones(3, dtype=torch.float32)

        print(f"[PoseDataset:{split}] center_mean(m)={self.center_mean.tolist()} | center_std(m)={self.center_std.tolist()}")

    def _compute_center_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        s = torch.zeros(3, dtype=torch.float64)
        ss = torch.zeros(3, dtype=torch.float64)
        n = 0
        for it in self.items:
            c = torch.tensor(it["center_m"], dtype=torch.float64)
            s += c
            ss += c * c
            n += 1
        mean = (s / max(n, 1)).to(torch.float32)
        var = (ss / max(n, 1) - mean.double() * mean.double()).clamp_min(0.0).to(torch.float32)
        std = torch.sqrt(var).clamp_min(self.eps_std)
        return mean, std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img_path: Path = it["img_path"]

        img_pil = Image.open(img_path).convert("RGB")
        img = self.tfms(img_pil)  # [3,S,S]

        center_m = torch.tensor(it["center_m"], dtype=torch.float32)  # meters
        normal = torch.tensor(it["normal"], dtype=torch.float32)
        normal = normal / (normal.norm(p=2) + 1e-12)

        center_norm = (center_m - self.center_mean) / self.center_std

        target = {
            "label": torch.tensor(it["label"], dtype=torch.long),  # []
            "center_m": center_m,                                  # [3]
            "center_norm": center_norm,                            # [3]
            "normal": normal,                                      # [3]
        }
        return img, target


def pose_collate_to_dict(batch):
    """
    batch: list[(img, target)]
    返回:
      imgs: Tensor[B,3,H,W]
      targets: dict{
        label: LongTensor[B]
        center_m: FloatTensor[B,3]
        center_norm: FloatTensor[B,3]
        normal: FloatTensor[B,3]
      }
    """
    imgs, tlist = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    labels = torch.stack([t["label"] for t in tlist], dim=0).long()
    center_m = torch.stack([t["center_m"] for t in tlist], dim=0).float()
    center_norm = torch.stack([t["center_norm"] for t in tlist], dim=0).float()
    normal = torch.stack([t["normal"] for t in tlist], dim=0).float()

    targets = {
        "label": labels,
        "center_m": center_m,
        "center_norm": center_norm,
        "normal": normal,
    }
    return imgs, targets


def build_loader(
    split: str,
    args,
    params,
    shuffle: bool,
    center_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    labels_dirname: str = "labels",
    drop_missing: bool = True,
):
    """
    兼容你 main 里的调用：
      loader, sampler, dataset = build_loader(...)

    center_stats:
      - train: None  -> dataset 自己算 mean/std
      - val/test: (train_mean, train_std) -> 复用 train 统计量，避免泄漏
    """
    dataset = RacketPoseDataset(
        data_root=args.data_root,
        split=split,
        img_size=getattr(args, "input_size", 640),
        labels_dirname=labels_dirname,
        drop_missing=drop_missing,
        center_stats=center_stats,
        compute_stats_if_missing=(center_stats is None),
    )

    sampler = None
    if getattr(args, "distributed", False):
        sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    loader = data.DataLoader(
        dataset,
        batch_size=getattr(args, "batch_size", 16),
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=getattr(args, "num_workers", 4),
        pin_memory=True,
        collate_fn=pose_collate_to_dict,
        drop_last=False,
    )
    return loader, sampler, dataset
