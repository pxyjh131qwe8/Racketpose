from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import xml.etree.ElementTree as ET

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)



def _train_tfms(img_size: int=512):
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


# tools voc解析与box缩放

def voc_parse_boxes(xml_path, class_to_idx=None):
    """
    解析 VOC xml，返回 (boxes[N,4], labels[N])
    boxes 为 float，格式 [xmin, ymin, xmax, ymax]，未归一化，像素坐标。
    """
    boxes: List[List[float]] = [] 
    labels: List[int] = [] 
    if not xml_path.exists():
        return torch.zeros((0,4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    
    tree = ET.parse(str(xml_path))
    root = tree.getroot() 
    for obj in root.findall("object"):
        name = obj.findtext("name", default="0")
        if class_to_idx is not None and not name.isdigit():
            lab = int(class_to_idx[name])
        else:
            # 若name是数字字符串"0"、"1"等，直接转为int
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
    
    if len(boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)     



def resize_boxes(boxes: torch.Tensor, orig_hw: Tuple[int, int], new_hw: Tuple[int, int]) -> torch.Tensor:
    """
    将 boxes 从原图尺寸缩放到新尺寸；boxes: [N,4] xyxy
    orig_hw = (H, W), new_hw = (H', W')
    """
    if boxes.numel() == 0:
        return boxes
    oh, ow = orig_hw
    nh, nw = new_hw
    sx = nw / ow
    sy = nh / oh
    out = boxes.clone()
    out[:, [0, 2]] = out[:, [0, 2]] * sx
    out[:, [1, 3]] = out[:, [1, 3]] * sy
    return out


class RacketMultiTaskDataset(Dataset):
    """
    读取 labels/<split>/*.csv
    必需列：filename, dist, angle_x, angle_y, angle_z, label
    额外：若 data_root/boxes/<video_name>/<stem>.xml 存在，则读取 VOC 2D 框。

    输出：(img: Tensor[3,H,W], reg: Tensor[4], cls: LongTensor[], boxes: Tensor[N,4], box_labels: LongTensor[N])
      - reg: [dist(m), angle_x(°), angle_y(°), angle_z(°)]
      - cls: scalar long
      - boxes 已根据 Resize 缩放到与 img 同尺寸；若 normalize_boxes=True，则再 /img_size 归一化
    """
    def __init__(self,
                 data_root,
                 split,
                 img_size,
                 mm_to_m=False,
                 drop_missing=True,
                 with_boxes=True,
                 boxes_dirname="boxes",
                 normalize_boxes=True,
                 class_to_idx=None
                 ):
        
        self.root = Path(data_root)
        self.labels_dir = self.root / "labels" / split
        assert self.labels_dir.exists(), f"不存在：{self.labels_dir}"
        self.img_size = int(img_size)
        self.tfms = _train_tfms(self.img_size) if split == "train" else _val_tfms(self.img_size)

        self.with_boxes = with_boxes
        self.boxes_root = (self.root / boxes_dirname) if with_boxes else None
        self.normalize_boxes = normalize_boxes
        self.class_to_idx = class_to_idx

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
    
    def _guess_xml_path(self, img_path: Path):
        """
        从图片路径推断 VOC xml 路径：
        boxes/<video_name>/<stem>.xml
        其中 video_name = 图片父文件夹名（例如 B-BB-BLUE），stem 与图片同名（不含扩展名）。
        """
        assert self.boxes_root is not None 
        video_name = img_path.parent.name 
        xml_name = img_path.stem + ".xml" 
        cand = self.boxes_root / video_name / xml_name
        if cand.exists():
            return cand
        # 兜底：在 boxes 里递归搜索同名 xml（较慢，仅当上面失败时）
        hits = list(self.boxes_root.rglob(xml_name))
        return hits[0] if hits else cand  # 若找不到，返回期望路径（后续会当作不存在处理）
                        
    def __getitem__(self, idx):
        img_path, dist, ax, ay, az, label = self.items[idx] 
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size   
        
        # 读取boxes(像素坐标，未缩放) 
        if self.with_boxes: 
            xml_path = self._guess_xml_path(img_path) 
            boxes, box_labels = voc_parse_boxes(xml_path, self.class_to_idx)  # boxes: [N,4] xyxy
        else:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            box_labels = torch.zeros((0,), dtype=torch.long) 
        
        # 先记住 orig，再做同尺寸 Resize
        img = self.tfms(img_pil)  # -> [3, H', W'] 且 H'=W'=img_size
        new_h = new_w = self.img_size

        # 将 boxes 缩放到新尺寸
        boxes = resize_boxes(boxes, (orig_h, orig_w), (new_h, new_w))

        # 需要归一化则除以尺寸
        if self.normalize_boxes:
            if boxes.numel() > 0:
                boxes[:, [0, 2]] /= float(new_w)
                boxes[:, [1, 3]] /= float(new_h)

        reg = torch.tensor([dist, ax, ay, az], dtype=torch.float32)
        cls = torch.tensor(label, dtype=torch.long)
        return img, reg, cls, boxes, box_labels    

# --------- detection collate：支持可变长框 ----------
def detection_collate(batch):
    """
    batch: List of (img, reg, cls, boxes, box_labels)
    返回：
        imgs: Tensor[B, 3, H, W]
        regs: Tensor[B, 4]
        clss: LongTensor[B]
        boxes: List[Tensor[Ni,4]]
        labels: List[LongTensor[Ni]]
    """
    imgs, regs, clss, boxes, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    regs = torch.stack(regs, dim=0)
    clss = torch.stack(clss, dim=0)
    return imgs, regs, clss, list(boxes), list(labels)


 



def build_loader(split, args, params, shuffle):
    dataset = RacketMultiTaskDataset(
        data_root=args.data_root,  # 新增 CLI 参数
        split=split,
        img_size=args.input_size,
        mm_to_m=False,
        with_boxes=True,
        normalize_boxes=True,
        class_to_idx=None,
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
        collate_fn=detection_collate
    )
    return loader, sampler