from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch.utils import data
from PIL import Image
import torchvision.transforms as T
import xml.etree.ElementTree as ET


def _det_tfms(img_size: int = 640):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),  # 0..1
    ])


def voc_parse_boxes(xml_path: Path, class_to_idx=None):
    boxes, labels = [], []
    if (xml_path is None) or (not xml_path.exists()):
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    for obj in root.findall("object"):
        name = obj.findtext("name", default="0")
        if class_to_idx is not None and not name.isdigit():
            lab = int(class_to_idx[name])
        else:
            lab = int(name)

        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin", "0"))
        ymin = float(bb.findtext("ymin", "0"))
        xmax = float(bb.findtext("xmax", "0"))
        ymax = float(bb.findtext("ymax", "0"))
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(lab)

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def resize_boxes_xyxy(boxes: torch.Tensor, orig_hw: Tuple[int, int], new_hw: Tuple[int, int]) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    oh, ow = orig_hw
    nh, nw = new_hw
    sx = nw / ow
    sy = nh / oh
    out = boxes.clone()
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    return out


def xyxy_to_cxcywhn(boxes_xyxy: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros((0, 4))
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    w = (x2 - x1).clamp_min(1e-6)
    h = (y2 - y1).clamp_min(1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    # normalize
    cx = cx / float(img_w)
    cy = cy / float(img_h)
    w = w / float(img_w)
    h = h / float(img_h)
    return torch.stack([cx, cy, w, h], dim=1)


class RacketDetDataset(Dataset):
    """
    纯检测：从 labels/<split>/*.csv 读取 filename
    从 boxes/<video>/<stem>.xml 读取 VOC 框
    返回：
      img: Tensor[3,H,W]
      cls_xywhn: Tensor[N,5] = [cls, cx, cy, w, h] (normalized)
    """
    def __init__(self, data_root, split, img_size=640, drop_missing=True, boxes_dirname="boxes", class_to_idx=None):
        self.root = Path(data_root)
        self.labels_dir = self.root / "labels" / split
        assert self.labels_dir.exists(), f"不存在：{self.labels_dir}"

        self.img_size = int(img_size)
        self.tfms = _det_tfms(self.img_size)

        self.boxes_root = self.root / boxes_dirname
        self.class_to_idx = class_to_idx

        self.items: List[Path] = []
        csv_files = sorted(self.labels_dir.glob("*.csv"))
        assert csv_files, f"{self.labels_dir} 下未找到 csv"

        for csv_path in csv_files:
            with open(csv_path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                if not r.fieldnames or "filename" not in r.fieldnames:
                    raise ValueError(f"{csv_path} 缺列 filename，实际 {r.fieldnames}")
                for row in r:
                    p = row["filename"].replace("\\", "/")
                    img_path = (self.root / p) if not p.startswith("/") else Path(p)
                    self.items.append(img_path)

        if drop_missing:
            before = len(self.items)
            self.items = [p for p in self.items if p.exists()]
            after = len(self.items)
            if after < before:
                print(f"[DetDataset] 丢弃缺图样本 {before-after} 条。")

    def __len__(self):
        return len(self.items)

    def _guess_xml_path(self, img_path: Path) -> Path:
        video_name = img_path.parent.name
        xml_name = img_path.stem + ".xml"
        cand = self.boxes_root / video_name / xml_name
        if cand.exists():
            return cand
        hits = list(self.boxes_root.rglob(xml_name))
        return hits[0] if hits else cand

    def __getitem__(self, idx: int):
        img_path = self.items[idx]
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        xml_path = self._guess_xml_path(img_path)
        boxes_xyxy, labels = voc_parse_boxes(xml_path, self.class_to_idx)  # pixel in orig

        img = self.tfms(img_pil)  # [3,S,S]
        S = self.img_size
        boxes_xyxy = resize_boxes_xyxy(boxes_xyxy, (orig_h, orig_w), (S, S))
        boxes_cxcywhn = xyxy_to_cxcywhn(boxes_xyxy, img_w=S, img_h=S)  # [N,4] normalized

        if boxes_cxcywhn.numel() == 0:
            cls_xywhn = torch.zeros((0, 5), dtype=torch.float32)
        else:
            cls_xywhn = torch.cat([labels.float().unsqueeze(1), boxes_cxcywhn], dim=1).float()  # [N,5]

        return img, cls_xywhn


def det_collate_to_dict(batch):
    """
    返回 ComputeLoss 需要的 targets dict:
      idx: [M,1] float
      cls: [M,1] float
      box: [M,4] float (cxcywh normalized)
    """
    imgs, tlist = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    idx_all, cls_all, box_all = [], [], []
    for b, t in enumerate(tlist):
        if t.numel() == 0:
            continue
        M = t.shape[0]
        idx_all.append(torch.full((M, 1), float(b), dtype=torch.float32))
        cls_all.append(t[:, 0:1].to(torch.float32))
        box_all.append(t[:, 1:5].to(torch.float32))

    if len(box_all) == 0:
        device = imgs.device
        targets = {
            "idx": torch.zeros((0, 1), dtype=torch.float32, device=device),
            "cls": torch.zeros((0, 1), dtype=torch.float32, device=device),
            "box": torch.zeros((0, 4), dtype=torch.float32, device=device),
        }
    else:
        targets = {
            "idx": torch.cat(idx_all, dim=0),
            "cls": torch.cat(cls_all, dim=0),
            "box": torch.cat(box_all, dim=0),
        }
    return imgs, targets


def build_loader(split, args, params, shuffle):
    dataset = RacketDetDataset(
        data_root=args.data_root,
        split=split,
        img_size=args.input_size,
        drop_missing=True,
        boxes_dirname="boxes",
        class_to_idx=None,
    )
    sampler = None
    if getattr(args, "distributed", False):
        sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=det_collate_to_dict,
    )
    return loader, sampler
