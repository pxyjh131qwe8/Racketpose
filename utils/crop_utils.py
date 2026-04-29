# utils/crop_utils.py
import json
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import tqdm
import xml.etree.ElementTree as ET

def _expand_box(x1, y1, x2, y2, W, H, scale):
    """
    以框中心为原点，按 scale 对宽高等比扩展，并裁剪到图像边界。
    scale=1.0 表示不变；>1 放大；(可支持 <1 缩小)
    """
    # 合法化
    if scale <= 0:
        scale = 1.0

    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    new_w = min(float(W), w * scale)
    new_h = min(float(H), h * scale)

    nx1 = max(0, int(round(cx - new_w / 2)))
    ny1 = max(0, int(round(cy - new_h / 2)))
    nx2 = min(W - 1, int(round(cx + new_w / 2)))
    ny2 = min(H - 1, int(round(cy + new_h / 2)))

    # 保证至少 1px 面积
    if nx2 <= nx1:
        nx2 = min(W - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(H - 1, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _expand_double_box(x1, y1, x2, y2, W, H):
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    new_w = min(W, 2.0 * w)
    new_h = min(H, 2.0 * h)
    nx1 = max(0, int(round(cx - new_w/2)))
    ny1 = max(0, int(round(cy - new_h/2)))
    nx2 = min(W-1, int(round(cx + new_w/2)))
    ny2 = min(H-1, int(round(cy + new_h/2)))
    return nx1, ny1, nx2, ny2

def _read_dets_from_txt(txt_path):
    """每行: cls conf x1 y1 x2 y2"""
    dets = []
    with open(txt_path, "r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            cls, conf, x1, y1, x2, y2 = line.split()
            dets.append((int(cls), float(conf), int(x1), int(y1), int(x2), int(y2)))
    return dets

def _read_dets_from_json(json_path):
    """JSON: {"detections":[{"cls":int,"conf":float,"bbox":[x1,y1,x2,y2]}, ...]}"""
    dets = []
    with open(json_path, "r", encoding="utf-8") as rf:
        data = json.load(rf)
    for d in data.get("detections", []):
        x1, y1, x2, y2 = d["bbox"]
        dets.append((int(d["cls"]), float(d["conf"]), int(x1), int(y1), int(x2), int(y2)))
    return dets

def _find_source_image(src_root, rel_no_ext) :
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        cand = (src_root / rel_no_ext).with_suffix(ext)
        if cand.exists():
            return cand
    return None

def crop_from_detections(args):
    """
    使用 --crop-from-dets 指向的检测结果目录（TXT/JSON均可），
    从 --crop-src-root 读取原图，输出到 --crop-dst-root，保留原相对目录。

    需要的 args 字段：
      - crop_from_dets   (str)
      - crop_src_root    (str)
      - crop_dst_root    (str)
      - crop_mode        ('bbox' | 'more')
      - crop_conf_thres  (float)
    """
    assert args.crop_from_dets, "缺少 --crop-from-dets"
    assert args.crop_src_root,  "缺少 --crop-src-root"
    assert args.crop_dst_root,  "缺少 --crop-dst-root"
    mode = args.crop_mode.lower()
    assert mode in ("bbox", "more"), "--crop-mode 仅支持 bbox / more"

    det_root = Path(args.crop_from_dets)
    src_root = Path(args.crop_src_root)
    dst_root = Path(args.crop_dst_root)
    det_files = sorted(list(det_root.rglob("*.txt")) + list(det_root.rglob("*.json")))
    if len(det_files) == 0:
        print(f"[crop] {det_root} 下未找到检测结果(.txt/.json)")
        return

    kept = 0
    pbar = tqdm.tqdm(det_files, desc=f"[crop:{mode}] -> {str(dst_root)}")
    for f in pbar:
        rel = f.relative_to(det_root)
        rel_no_ext = rel.with_suffix("")  # 用于匹配任意图像扩展名
        img_path = _find_source_image(src_root, rel_no_ext)
        if img_path is None:
            continue

        # 读取检测
        if f.suffix == ".txt":
            dets = _read_dets_from_txt(f)
        else:
            dets = _read_dets_from_json(f)
        if not dets:
            continue

        im = Image.open(img_path).convert("RGB")
        W, H = im.size

        #  用原图相对 src_root 的路径来构造输出路径（保证“按原目录结构”）
        rel_img = img_path.relative_to(src_root)      # e.g. person/a/b/c.jpg
        out_dir = (dst_root / rel_img.parent)         # e.g. <dst_root>/person/a/b

        out_dir.mkdir(parents=True, exist_ok=True)

        # 逐框裁剪
        for k, (cls, conf, x1, y1, x2, y2) in enumerate(dets):
            if conf < args.crop_conf_thres:
                continue
            if mode == "more":
                print(f"[DEBUG] scale={args.scale} (type={type(args.scale)})")
                x1, y1, x2, y2 = _expand_box(x1, y1, x2, y2, W, H, scale=args.scale)
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue

            crop = im.crop((x1, y1, x2, y2))

            # 文件名基于原图文件名生成（不改变目录）
            base_stem = rel_img.stem                   # 原图不带扩展名
            out_name = f"{base_stem}_c{k}_cls{cls}.jpg"
            out_path = out_dir / out_name

            crop.save(out_path, quality=95)
            kept += 1

    print(f"[crop] 完成，生成裁剪图 {kept} 张。")



def _expand_bbox(x1: int, y1: int, x2: int, y2: int, W: int, H: int, scale: float) -> Tuple[int, int, int, int]:
    """以bbox中心等比放大（scale倍），并裁剪到图像边界。"""
    if not scale or scale <= 0:
        scale = 1.0
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    new_w = min(float(W), w * float(scale))
    new_h = min(float(H), h * float(scale))

    nx1 = int(round(cx - new_w * 0.5))
    ny1 = int(round(cy - new_h * 0.5))
    nx2 = int(round(cx + new_w * 0.5))
    ny2 = int(round(cy + new_h * 0.5))

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(W - 1, nx2)
    ny2 = min(H - 1, ny2)

    if nx2 <= nx1:
        nx2 = min(W - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(H - 1, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _imsave(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        img.save(path)
    else:
        img.save(path.with_suffix(".jpg"))


# ------- VOC 解析 & 路径推断（匹配 Dataset 约定） -------
def _voc_parse_boxes(xml_path):
    """
    解析 VOC XML，返回像素 xyxy 列表（不关心类别，仅裁剪用）。
    忽略无效/反向框。
    """
    if not xml_path.exists():
        return []
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = int(round(float(bb.findtext("xmin", "0"))))
            ymin = int(round(float(bb.findtext("ymin", "0"))))
            xmax = int(round(float(bb.findtext("xmax", "0"))))
            ymax = int(round(float(bb.findtext("ymax", "0"))))
        except Exception:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def _guess_voc_xml_for_image(img_path, boxes_root):
    """
    从图片路径推断 VOC xml：
      boxes/<video_name>/<stem>.xml
    video_name = 图片父文件夹名（如 B-BB-BLUE），stem = 图片文件名去后缀。
    找不到则在 boxes_root 递归同名兜底。
    """
    video_name = img_path.parent.name
    xml_name = img_path.stem + ".xml"
    cand = boxes_root / video_name / xml_name
    if cand.exists():
        return cand
    hits = list(boxes_root.rglob(xml_name))
    return hits[0] if hits else cand


# ------- 主入口：按 VOC 标注裁剪 -------
def crop_from_annotations(args):
    """
    使用 VOC XML 标注框裁剪，不依赖检测输出。
    目录约定：
      - 原图根：args.ann_src_root
      - VOC XML 根：args.voc_boxes_root（默认 data_root/boxes ）
      - 输出根：args.crop_ann_root
    """
    imgs_root = Path(args.ann_src_root)
    out_root  = Path(args.crop_ann_root)
    scale     = float(getattr(args, "ann_scale", 1.0))

    # 默认 VOC boxes 根：data_root/boxes
    voc_boxes_root = Path(args.voc_boxes_root) if args.voc_boxes_root else (Path(args.data_root) / "boxes")

    # 遍历所有图片（递归）
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    img_files = [p for p in imgs_root.rglob("*") if p.suffix.lower() in exts]
    if not img_files:
        print(f"[crop-ann|voc] 在 {imgs_root} 下未找到图片")
        return

    total_crops = 0
    miss_xml = 0
    empty_xml = 0

    pbar = tqdm.tqdm(img_files, desc="[crop-ann|voc]")
    for img_path in pbar:
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = im.size

        xml_path = _guess_voc_xml_for_image(img_path, voc_boxes_root)
        if not xml_path.exists():
            print(f"[WARN] 找不到对应 VOC XML: {str(xml_path)}")
            miss_xml += 1
            pbar.set_postfix_str(f"missing xml: {miss_xml}")
            continue

        boxes = _voc_parse_boxes(xml_path)
        if not boxes:
            empty_xml += 1
            pbar.set_postfix_str(f"empty xml: {empty_xml}")
            continue

        # 输出相对路径保持与图片一致
        rel = img_path.relative_to(imgs_root).with_suffix("")
        base_out = out_root / rel

        kept = 0
        for (x1, y1, x2, y2) in boxes:
            nx1, ny1, nx2, ny2 = _expand_bbox(x1, y1, x2, y2, W, H, scale)
            crop = im.crop((nx1, ny1, nx2, ny2))
            out_path = base_out.with_name(base_out.name).with_suffix(img_path.suffix)
            _imsave(crop, out_path)
            kept += 1

        total_crops += kept

    print(f"[crop-ann|voc] 完成：裁出 {total_crops} 张，缺失XML {miss_xml} 个，空XML {empty_xml} 个。")