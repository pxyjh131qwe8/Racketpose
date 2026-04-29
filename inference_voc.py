# import sys
# from pathlib import Path
# ROOT = Path(__file__).resolve().parent   # yolov11-detect/
# sys.path.insert(0, str(ROOT))


# import os
# import copy
# from argparse import ArgumentParser
# from pathlib import Path
# from xml.etree.ElementTree import Element, SubElement, ElementTree

# import cv2
# import torch
# import tqdm
# import yaml

# from nets import nn

# import nets, nets.nn
# print("[DEBUG] nets.__file__   =", nets.__file__)
# print("[DEBUG] nets.nn.__file__=", nets.nn.__file__)
# print("[DEBUG] has yolo_v11_x? =", hasattr(nets.nn, "yolo_v11_x"))
# print("[DEBUG] yolo-like names =", [k for k in dir(nets.nn) if "yolo" in k.lower()])


# from utils import util


# import math






# # VOC writer
# def _indent(elem, level=0):
#     i = "\n" + level * " "
#     if len(elem):
#         if not elem.text or not elem.text.strip():
#             elem.text = i + " "
#         for e in elem:
#             _indent(e, level + 1)
#         if not elem.tail or not elem.tail.strip():
#             elem.tail = i
#     else:
#         if level and (not elem.tail or not elem.tail.strip()):
#             elem.tail = i


# def write_voc_xml(xml_path: str, image_path: str, width: int, height: int, objects: list, folder: str = "imgs"):
#     ann = Element("annotation")
#     SubElement(ann, "folder").text = folder
#     SubElement(ann, "filename").text = os.path.basename(image_path)
#     SubElement(ann, "path").text = image_path

#     source = SubElement(ann, "source")
#     SubElement(source, "database").text = "Unknown"

#     size = SubElement(ann, "size")
#     SubElement(size, "width").text = str(int(width))
#     SubElement(size, "height").text = str(int(height))
#     SubElement(size, "depth").text = "3"
#     SubElement(ann, "segmented").text = "0"

#     for obj in objects:
#         o = SubElement(ann, "object")
#         SubElement(o, "name").text = str(obj["name"])
#         SubElement(o, "pose").text = "Unspecified"
#         SubElement(o, "truncated").text = "0"
#         SubElement(o, "difficult").text = str(int(obj.get("difficult", 0)))

#         bnd = SubElement(o, "bndbox")
#         SubElement(bnd, "xmin").text = str(int(obj["xmin"]))
#         SubElement(bnd, "ymin").text = str(int(obj["ymin"]))
#         SubElement(bnd, "xmax").text = str(int(obj["xmax"]))
#         SubElement(bnd, "ymax").text = str(int(obj["ymax"]))

#         # 如果你原始 VOC 没有 score 字段，把这行删掉即可
#         # if "score" in obj:
#         #     SubElement(o, "score").text = f'{float(obj["score"]):.6f}'

#     _indent(ann)
#     os.makedirs(os.path.dirname(xml_path), exist_ok=True)
#     ElementTree(ann).write(xml_path, encoding="utf-8", xml_declaration=False)


# # image preprocess
# def letterbox(im, new_size=640, color=(114, 114, 114)):
#     """
#     YOLO-style letterbox.
#     return: im_padded, ratio, pad (left, top)
#     """
#     h0, w0 = im.shape[:2]
#     r = min(new_size / h0, new_size / w0)
#     new_unpad = (int(round(w0 * r)), int(round(h0 * r)))  # (w, h)

#     im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

#     dw = new_size - new_unpad[0]
#     dh = new_size - new_unpad[1]
#     dw /= 2
#     dh /= 2

#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

#     im_padded = cv2.copyMakeBorder(
#         im_resized, top, bottom, left, right,
#         cv2.BORDER_CONSTANT, value=color
#     )
#     return im_padded, r, (left, top)


# def xyxy_scale_back(xyxy, r, pad, w0, h0):
#     """
#     xyxy is in letterbox image coords (new_size x new_size).
#     Map back to original image coords.
#     """
#     x1, y1, x2, y2 = xyxy
#     pad_x, pad_y = pad

#     x1 = (x1 - pad_x) / r
#     y1 = (y1 - pad_y) / r
#     x2 = (x2 - pad_x) / r
#     y2 = (y2 - pad_y) / r

#     # clamp
#     x1 = max(0, min(int(round(x1)), w0 - 1))
#     y1 = max(0, min(int(round(y1)), h0 - 1))
#     x2 = max(0, min(int(round(x2)), w0 - 1))
#     y2 = max(0, min(int(round(y2)), h0 - 1))
#     if x2 < x1:
#         x1, x2 = x2, x1
#     if y2 < y1:
#         y1, y2 = y2, y1
#     return x1, y1, x2, y2


# def is_edge_box(x1, y1, x2, y2, w, h, edge_px=0):
#     """
#     edge_px=0: 只要触边(==0 或 ==w-1/h-1) 就算 bad
#     edge_px>0: 距离边界 <= edge_px 也算 bad（更严格）
#     """
#     left_hit = x1 <= edge_px
#     top_hit = y1 <= edge_px
#     right_hit = x2 >= (w - 1 - edge_px)
#     bottom_hit = y2 >= (h - 1 - edge_px)
#     return left_hit or top_hit or right_hit or bottom_hit


# # ---------------- model load ----------------
# # def load_model(ckpt_path: str, num_classes: int, device: str):
# #     ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
# #     if isinstance(ckpt, dict) and "model" in ckpt:
# #         model = ckpt["model"]
# #     else:
# #         model = nn.yolo_v11_x(num_classes)
# #         model.load_state_dict(ckpt, strict=False)

# #     model = copy.deepcopy(model).float().fuse().eval().to(device)
# #     return model

# import copy
# import torch
# from collections import OrderedDict

# def _extract_state_dict(ckpt_obj):
#     """兼容多种 ckpt 结构，返回 state_dict(dict[str, Tensor])"""
#     if isinstance(ckpt_obj, (dict, OrderedDict)):
#         # 常见：{"model": model_obj} 或 {"model": state_dict}
#         if "model" in ckpt_obj:
#             v = ckpt_obj["model"]
#             if hasattr(v, "state_dict"):
#                 return v.state_dict()
#             if isinstance(v, (dict, OrderedDict)):
#                 return v
#         if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], (dict, OrderedDict)):
#             return ckpt_obj["state_dict"]
#         # 可能本身就是 state_dict
#         if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
#             return ckpt_obj
#     # 直接是模型对象
#     if hasattr(ckpt_obj, "state_dict"):
#         return ckpt_obj.state_dict()
#     raise ValueError(f"Cannot extract state_dict from ckpt type: {type(ckpt_obj)}")

# def _strip_prefix(sd, prefixes=("module.", "model.")):
#     out = {}
#     for k, v in sd.items():
#         nk = k
#         for p in prefixes:
#             if nk.startswith(p):
#                 nk = nk[len(p):]
#         out[nk] = v
#     return out

# def _load_shape_matched(model, sd, verbose=True):
#     """只加载 shape 匹配的参数，返回(loaded_keys, skipped_keys, missing_keys)"""
#     msd = model.state_dict()
#     load_sd = {}
#     skipped = []

#     for k, v in sd.items():
#         if k in msd and msd[k].shape == v.shape:
#             load_sd[k] = v
#         else:
#             skipped.append(k)

#     missing, unexpected = model.load_state_dict(load_sd, strict=False)

#     if verbose:
#         print(f"[CKPT] total ckpt keys   : {len(sd)}")
#         print(f"[CKPT] loaded keys      : {len(load_sd)}")
#         print(f"[CKPT] skipped (mismatch): {len(skipped)}")
#         # 头不适配一般在 skipped 里
#         if skipped:
#             print(f"[CKPT] skipped examples : {skipped[:10]}{' ...' if len(skipped) > 10 else ''}")
#         if missing:
#             print(f"[CKPT] missing keys     : {len(missing)}")
#         if unexpected:
#             print(f"[CKPT] unexpected keys  : {len(unexpected)}")

#     return list(load_sd.keys()), skipped, missing

# # ---------------- model load ----------------
# def load_model(ckpt_path: str, num_classes: int, device: str):
#     # 1) 先按当前 num_classes 构建新模型（保证 head 维度正确）
#     model = nn.yolo_v11_x(num_classes)

#     # 2) 读取 ckpt 并抽取 state_dict
#     ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
#     sd = _extract_state_dict(ckpt)
#     sd = _strip_prefix(sd)

#     # 3) 只加载 shape 匹配的参数（head 不适配会自动跳过）
#     _load_shape_matched(model, sd, verbose=True)

#     # 4) fuse / eval / device
#     model = copy.deepcopy(model).float().fuse().eval().to(device)
#     return model


# def find_images(img_dir: str):
#     exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
#     img_dir = Path(img_dir)
#     files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
#     files.sort()
#     return files


# @torch.no_grad()
# def run(args):
#     with open(args.args_yaml, errors="ignore") as f:
#         params = yaml.safe_load(f)
#     names = params["names"]
#     num_classes = len(names)

#     util.setup_seed()
#     util.setup_multi_processes()

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = load_model(args.weights, num_classes=num_classes, device=device)

#     img_files = find_images(args.img_dir)
#     if not img_files:
#         print(f"[Error] No images found under: {args.img_dir}")
#         return

#     img_root = Path(args.img_dir).resolve()
#     if args.out_xml_dir is None:
#         out_root = img_root.parent / "boxes" if img_root.name == "imgs" else img_root.parent / "boxes"
#     else:
#         out_root = Path(args.out_xml_dir).resolve()

#     bad_list = []

#     bs = args.batch_size
#     pbar = tqdm.tqdm(range(0, len(img_files), bs), desc="Infer folder -> VOC")

#     for start in pbar:
#         batch_paths = img_files[start:start + bs]

#         ims = []
#         meta = []  # (path, w0, h0, r, pad)
#         for p in batch_paths:
#             im0 = cv2.imread(str(p))
#             if im0 is None:
#                 bad_list.append(f"{str(p)}\tread_failed")
#                 continue

#             h0, w0 = im0.shape[:2]
#             im_lb, r, pad = letterbox(im0, new_size=args.input_size)

#             im_rgb = cv2.cvtColor(im_lb, cv2.COLOR_BGR2RGB)
#             im_t = torch.from_numpy(im_rgb).permute(2, 0, 1).contiguous().float() / 255.0
#             ims.append(im_t)
#             meta.append((p, w0, h0, r, pad))

#         if not ims:
#             continue

#         x = torch.stack(ims, dim=0).to(device)
#         if args.fp16 and device.startswith("cuda"):
#             x = x.half()

#         out = model(x)
#         dets = util.non_max_suppression(
#             out,
#             confidence_threshold=args.conf_thres,
#             iou_threshold=args.iou_thres
#         )

#         for i, (p, w0, h0, r, pad) in enumerate(meta):
#             det = dets[i]
#             objects = []
#             best_obj = None

#             if det is not None and det.numel():
#                 det = det.detach().cpu()

#                 # det: [N,6] = xyxy conf cls -> 只取最高置信度的一条
#                 best_idx = det[:, 4].argmax().item()
#                 row = det[best_idx]

#                 x1, y1, x2, y2, conf, cls = row.tolist()
#                 conf = float(conf)
#                 cls = int(cls)

#                 name = names[cls] if 0 <= cls < len(names) else str(cls)
#                 xx1, yy1, xx2, yy2 = xyxy_scale_back((x1, y1, x2, y2), r, pad, w0, h0)

#                 # 过滤极小框（可选）
#                 if (xx2 - xx1) >= 2 and (yy2 - yy1) >= 2:
#                     best_obj = {
#                         "name": name,
#                         "xmin": xx1,
#                         "ymin": yy1,
#                         "xmax": xx2,
#                         "ymax": yy2,
#                         "difficult": 0,
#                         "score": conf,
#                     }
#                     objects = [best_obj]  # 只保留一个框



#             # 输出 xml 路径：保持子目录结构
#             rel = p.relative_to(img_root)
#             xml_rel = rel.with_suffix(".xml")
#             xml_path = out_root / xml_rel

#             write_voc_xml(
#                 xml_path=str(xml_path),
#                 image_path=str(p),
#                 width=w0,
#                 height=h0,
#                 objects=objects,
#                 folder=args.voc_folder,
#             )

#             # ---- bad cases 规则：漏检 / 低置信度 / 贴边框 ----
#             # if bad_reason is not None or len(objects) == 0 or max_conf < args.low_conf_thres:
#             #     bad_list.append(f"{str(p)}\t{bad_reason}\tmax_conf={max_conf:.4f}\tcount={len(objects)}")

#             if best_obj is None:
#                 bad_list.append(f"{str(p)}\tno_det")
#             else:
#                 score = float(best_obj["score"])
#                 edge_bad = is_edge_box(
#                     best_obj["xmin"], best_obj["ymin"], best_obj["xmax"], best_obj["ymax"],
#                     w0, h0, edge_px=args.edge_px
#                 )
#                 if score < args.low_conf_thres:
#                     bad_list.append(
#                         f"{str(p)}\tlow_conf={score:.4f}\txyxy=({best_obj['xmin']},{best_obj['ymin']},{best_obj['xmax']},{best_obj['ymax']})"
#                     )
#                 elif edge_bad:
#                     bad_list.append(
#                         f"{str(p)}\tedge_box(edge_px={args.edge_px})\t"
#                         f"xyxy=({best_obj['xmin']},{best_obj['ymin']},{best_obj['xmax']},{best_obj['ymax']})\t"
#                         f"score={score:.4f}"
#                     )

#     bad_txt = Path(args.bad_txt)
#     bad_txt.parent.mkdir(parents=True, exist_ok=True)
#     with open(bad_txt, "w", encoding="utf-8") as f:
#         f.write("\n".join(bad_list))

#     print(f"[Done] XML root: {str(out_root)}")
#     print(f"[Done] Bad cases: {str(bad_txt)} (n={len(bad_list)})")


# def main():
#     parser = ArgumentParser()
#     parser.add_argument("--img-dir", default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0/SP3.0_upper_half/imgs", type=str,
#                         help="any folder containing images (recursive)")
#     parser.add_argument("--weights", default="/root/autodl-tmp/yolov11-detect/runs/det-2026_02_06-084158/weights/best.pt",
#                         type=str, help="best.pt/last.pt")
#     parser.add_argument("--args-yaml", default="utils/args.yaml", type=str, help="need params['names']")
#     parser.add_argument("--out-xml-dir", default=None, type=str,
#                         help="output VOC xml root (default: sibling boxes/)")
#     parser.add_argument("--bad-txt", default="SP3.0_upper_half_bad_cases.txt", type=str)

#     parser.add_argument("--input-size", default=640, type=int)
#     parser.add_argument("--batch-size", default=32, type=int)
#     parser.add_argument("--fp16", action="store_true")

#     parser.add_argument("--conf-thres", default=0.001, type=float)
#     parser.add_argument("--iou-thres", default=0.65, type=float)
#     parser.add_argument("--low-conf-thres", default=0.40, type=float)


#     # 新增：贴边框判定阈值
#     parser.add_argument("--edge-px", default=20, type=int,
#                         help="bbox touches/near image border within edge_px -> bad case (0 means strict touch)")

#     parser.add_argument("--voc-folder", default="imgs", type=str)

#     args = parser.parse_args()
#     run(args)


# if __name__ == "__main__":
#     main()

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent   # yolov11-detect/
sys.path.insert(0, str(ROOT))

import os
import copy
import shutil
from argparse import ArgumentParser
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

import cv2
import torch
import tqdm
import yaml

from nets import nn
import nets, nets.nn
print("[DEBUG] nets.__file__   =", getattr(nets, "__file__", None))
print("[DEBUG] nets.nn.__file__=", getattr(nets.nn, "__file__", None))
print("[DEBUG] has yolo_v11_x? =", hasattr(nets.nn, "yolo_v11_x"))
print("[DEBUG] yolo-like names =", [k for k in dir(nets.nn) if "yolo" in k.lower()])

from utils import util

# VOC writer
def _indent(elem, level=0):
    i = "\n" + level * " "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + " "
        for e in elem:
            _indent(e, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def write_voc_xml(xml_path: str, image_path: str, width: int, height: int, objects: list, folder: str = "imgs"):
    ann = Element("annotation")
    SubElement(ann, "folder").text = folder
    SubElement(ann, "filename").text = os.path.basename(image_path)
    SubElement(ann, "path").text = image_path

    source = SubElement(ann, "source")
    SubElement(source, "database").text = "Unknown"

    size = SubElement(ann, "size")
    SubElement(size, "width").text = str(int(width))
    SubElement(size, "height").text = str(int(height))
    SubElement(size, "depth").text = "3"
    SubElement(ann, "segmented").text = "0"

    for obj in objects:
        o = SubElement(ann, "object")
        SubElement(o, "name").text = str(obj["name"])
        SubElement(o, "pose").text = "Unspecified"
        SubElement(o, "truncated").text = "0"
        SubElement(o, "difficult").text = str(int(obj.get("difficult", 0)))

        bnd = SubElement(o, "bndbox")
        SubElement(bnd, "xmin").text = str(int(obj["xmin"]))
        SubElement(bnd, "ymin").text = str(int(obj["ymin"]))
        SubElement(bnd, "xmax").text = str(int(obj["xmax"]))
        SubElement(bnd, "ymax").text = str(int(obj["ymax"]))

    _indent(ann)
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    ElementTree(ann).write(xml_path, encoding="utf-8", xml_declaration=False)


# image preprocess
def letterbox(im, new_size=640, color=(114, 114, 114)):
    h0, w0 = im.shape[:2]
    r = min(new_size / h0, new_size / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))  # (w, h)

    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw = new_size - new_unpad[0]
    dh = new_size - new_unpad[1]
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im_padded = cv2.copyMakeBorder(
        im_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return im_padded, r, (left, top)


def xyxy_scale_back(xyxy, r, pad, w0, h0):
    x1, y1, x2, y2 = xyxy
    pad_x, pad_y = pad

    x1 = (x1 - pad_x) / r
    y1 = (y1 - pad_y) / r
    x2 = (x2 - pad_x) / r
    y2 = (y2 - pad_y) / r

    x1 = max(0, min(int(round(x1)), w0 - 1))
    y1 = max(0, min(int(round(y1)), h0 - 1))
    x2 = max(0, min(int(round(x2)), w0 - 1))
    y2 = max(0, min(int(round(y2)), h0 - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def is_edge_box(x1, y1, x2, y2, w, h, edge_px=0):
    left_hit = x1 <= edge_px
    top_hit = y1 <= edge_px
    right_hit = x2 >= (w - 1 - edge_px)
    bottom_hit = y2 >= (h - 1 - edge_px)
    return left_hit or top_hit or right_hit or bottom_hit


def is_box_fully_in_upper(x1, y1, x2, y2, h, upper_ratio=0.5):
    """新增规则：整个框都在上半区域 -> ymax <= upper_ratio*H"""
    if h <= 0:
        return False
    thr = upper_ratio * h
    return y2 <= thr


# ---------------- model load (shape matched) ----------------
import torch
from collections import OrderedDict

def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, (dict, OrderedDict)):
        if "model" in ckpt_obj:
            v = ckpt_obj["model"]
            if hasattr(v, "state_dict"):
                return v.state_dict()
            if isinstance(v, (dict, OrderedDict)):
                return v
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], (dict, OrderedDict)):
            return ckpt_obj["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    if hasattr(ckpt_obj, "state_dict"):
        return ckpt_obj.state_dict()
    raise ValueError(f"Cannot extract state_dict from ckpt type: {type(ckpt_obj)}")

def _strip_prefix(sd, prefixes=("module.", "model.")):
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def _load_shape_matched(model, sd, verbose=True):
    msd = model.state_dict()
    load_sd, skipped = {}, []

    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            load_sd[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(load_sd, strict=False)

    if verbose:
        print(f"[CKPT] total ckpt keys   : {len(sd)}")
        print(f"[CKPT] loaded keys      : {len(load_sd)}")
        print(f"[CKPT] skipped (mismatch): {len(skipped)}")
        if skipped:
            print(f"[CKPT] skipped examples : {skipped[:10]}{' ...' if len(skipped) > 10 else ''}")
        if missing:
            print(f"[CKPT] missing keys     : {len(missing)}")
        if unexpected:
            print(f"[CKPT] unexpected keys  : {len(unexpected)}")

    return list(load_sd.keys()), skipped, missing

def load_model(ckpt_path: str, num_classes: int, device: str):
    model = nn.yolo_v11_x(num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _strip_prefix(_extract_state_dict(ckpt))
    _load_shape_matched(model, sd, verbose=True)
    model = copy.deepcopy(model).float().fuse().eval().to(device)
    return model


def find_images(img_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_dir = Path(img_dir)
    files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def _copy_keep_rel(src: Path, dst_root: Path, rel: Path):
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


@torch.no_grad()
def run(args):
    with open(args.args_yaml, errors="ignore") as f:
        params = yaml.safe_load(f)
    names = params["names"]
    num_classes = len(names)

    util.setup_seed()
    util.setup_multi_processes()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, num_classes=num_classes, device=device)

    img_files = find_images(args.img_dir)
    if not img_files:
        print(f"[Error] No images found under: {args.img_dir}")
        return

    img_root = Path(args.img_dir).resolve()
    if args.out_xml_dir is None:
        out_root = img_root.parent / "boxes"
    else:
        out_root = Path(args.out_xml_dir).resolve()

    # bad / fine 统计
    bad_map = {}   # str(path) -> reason
    fine_list = [] # str(path)

    # 可选复制目录
    bad_img_root = Path(args.bad_img_dir).resolve() if args.bad_img_dir else None
    fine_img_root = Path(args.fine_img_dir).resolve() if args.fine_img_dir else None
    if bad_img_root:
        bad_img_root.mkdir(parents=True, exist_ok=True)
    if fine_img_root:
        fine_img_root.mkdir(parents=True, exist_ok=True)

    bs = args.batch_size
    pbar = tqdm.tqdm(range(0, len(img_files), bs), desc="Infer folder -> VOC")

    for start in pbar:
        batch_paths = img_files[start:start + bs]

        ims = []
        meta = []  # (path, w0, h0, r, pad)
        for p in batch_paths:
            im0 = cv2.imread(str(p))
            if im0 is None:
                bad_map[str(p)] = "read_failed"
                continue

            h0, w0 = im0.shape[:2]
            im_lb, r, pad = letterbox(im0, new_size=args.input_size)

            im_rgb = cv2.cvtColor(im_lb, cv2.COLOR_BGR2RGB)
            im_t = torch.from_numpy(im_rgb).permute(2, 0, 1).contiguous().float() / 255.0
            ims.append(im_t)
            meta.append((p, w0, h0, r, pad))

        if not ims:
            continue

        x = torch.stack(ims, dim=0).to(device)
        if args.fp16 and device.startswith("cuda"):
            x = x.half()

        out = model(x)
        dets = util.non_max_suppression(
            out,
            confidence_threshold=args.conf_thres,
            iou_threshold=args.iou_thres
        )

        for i, (p, w0, h0, r, pad) in enumerate(meta):
            det = dets[i]
            objects = []
            best_obj = None

            if det is not None and det.numel():
                det = det.detach().cpu()
                best_idx = det[:, 4].argmax().item()
                row = det[best_idx]

                x1, y1, x2, y2, conf, cls = row.tolist()
                conf = float(conf)
                cls = int(cls)

                name = names[cls] if 0 <= cls < len(names) else str(cls)
                xx1, yy1, xx2, yy2 = xyxy_scale_back((x1, y1, x2, y2), r, pad, w0, h0)

                if (xx2 - xx1) >= 2 and (yy2 - yy1) >= 2:
                    best_obj = {
                        "name": name,
                        "xmin": xx1,
                        "ymin": yy1,
                        "xmax": xx2,
                        "ymax": yy2,
                        "difficult": 0,
                        "score": conf,
                    }
                    objects = [best_obj]

            # 输出 xml 路径：保持子目录结构
            rel = p.relative_to(img_root)
            xml_rel = rel.with_suffix(".xml")
            xml_path = out_root / xml_rel

            write_voc_xml(
                xml_path=str(xml_path),
                image_path=str(p),
                width=w0,
                height=h0,
                objects=objects,
                folder=args.voc_folder,
            )

            # =========================
            # bad_case / fine_case 规则
            # =========================
            reason = None

            if best_obj is None:
                reason = "no_det"
            else:
                score = float(best_obj["score"])
                x1i, y1i, x2i, y2i = best_obj["xmin"], best_obj["ymin"], best_obj["xmax"], best_obj["ymax"]

                if score < args.low_conf_thres:
                    reason = f"low_conf={score:.4f}"
                elif is_edge_box(x1i, y1i, x2i, y2i, w0, h0, edge_px=args.edge_px):
                    reason = f"edge_box(edge_px={args.edge_px})"
                elif args.upper_bad and is_box_fully_in_upper(x1i, y1i, x2i, y2i, h0, upper_ratio=args.upper_ratio):
                    reason = f"upper_half(full, ratio={args.upper_ratio})"

            if reason is not None:
                bad_map[str(p)] = f"{reason}\txyxy=({best_obj['xmin'] if best_obj else 'NA'},{best_obj['ymin'] if best_obj else 'NA'},{best_obj['xmax'] if best_obj else 'NA'},{best_obj['ymax'] if best_obj else 'NA'})"
                if bad_img_root is not None:
                    _copy_keep_rel(p, bad_img_root, rel)
            else:
                fine_list.append(str(p))
                if fine_img_root is not None:
                    _copy_keep_rel(p, fine_img_root, rel)

    # save bad txt
    bad_txt = Path(args.bad_txt)
    bad_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(bad_txt, "w", encoding="utf-8") as f:
        for k, v in bad_map.items():
            f.write(f"{k}\t{v}\n")

    # save fine txt
    fine_txt = Path(args.fine_txt)
    fine_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(fine_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(fine_list))

    print(f"[Done] XML root : {str(out_root)}")
    print(f"[Done] Bad cases: {str(bad_txt)} (n={len(bad_map)})")
    print(f"[Done] Fine     : {str(fine_txt)} (n={len(fine_list)})")
    if bad_img_root:
        print(f"[Done] Copied bad imgs  -> {bad_img_root}")
    if fine_img_root:
        print(f"[Done] Copied fine imgs -> {fine_img_root}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--img-dir", default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0/imgs", type=str,
                        help="any folder containing images (recursive)")
    parser.add_argument("--weights", default="/root/autodl-tmp/yolov11-detect/weights/finetune2/det-2026_02_06-105150/weights/best.pt",
                        type=str, help="best.pt/last.pt")
    parser.add_argument("--args-yaml", default="utils/args.yaml", type=str, help="need params['names']")
    parser.add_argument("--out-xml-dir", default=None, type=str,
                        help="output VOC xml root (default: sibling boxes/)")

    parser.add_argument("--bad-txt", default="finetuneSP3.0bad_cases_epoch2.txt", type=str)
    parser.add_argument("--fine-txt", default="finetuneSP3.0fine_cases_epoch2.txt", type=str)

    # 可选：复制图片到目录（保留相对结构）
    parser.add_argument("--bad-img-dir", default=None, type=str,
                        help="optional: copy bad-case images to this folder")
    parser.add_argument("--fine-img-dir", default=None, type=str,
                        help="optional: copy fine-case images to this folder")

    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--conf-thres", default=0.001, type=float)
    parser.add_argument("--iou-thres", default=0.65, type=float)
    parser.add_argument("--low-conf-thres", default=0.40, type=float)

    # 贴边框判定阈值
    parser.add_argument("--edge-px", default=20, type=int,
                        help="bbox touches/near image border within edge_px -> bad case (0 means strict touch)")

    # 新增：上半区 bad 规则
    parser.add_argument("--upper-bad", action="store_true",
                        help="enable bad-case rule: bbox fully in upper region (likely head)")
    parser.add_argument("--upper-ratio", default=0.5, type=float,
                        help="upper region ratio, default=0.5 (upper half). bbox ymax<=ratio*H => bad")

    parser.add_argument("--voc-folder", default="imgs", type=str)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
