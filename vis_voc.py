#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from argparse import ArgumentParser
import xml.etree.ElementTree as ET

import cv2
import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_voc_xml(xml_path: str):
    """
    Parse VOC-style XML.
    Returns: list of dict: {name, xmin, ymin, xmax, ymax, score(optional)}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="object")
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = int(float(bnd.findtext("xmin", "0")))
        ymin = int(float(bnd.findtext("ymin", "0")))
        xmax = int(float(bnd.findtext("xmax", "0")))
        ymax = int(float(bnd.findtext("ymax", "0")))

        # optional score
        score_node = obj.find("score")
        score = None
        if score_node is not None and score_node.text is not None:
            try:
                score = float(score_node.text.strip())
            except Exception:
                score = None

        objects.append({
            "name": name,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "score": score
        })
    return objects


def draw_boxes(im, objects, show_score=True, score_thres=None, thickness=2):
    """
    Draw bbox + label on image (BGR).
    """
    h, w = im.shape[:2]

    for obj in objects:
        score = obj.get("score", None)
        if score_thres is not None and score is not None and score < score_thres:
            continue

        x1, y1, x2, y2 = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        # box
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        # label
        name = obj.get("name", "object")
        label = name
        if show_score and (score is not None):
            label = f"{name} {score:.3f}"

        # text bg
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        tx1, ty1 = x1, max(0, y1 - th - baseline - 4)
        tx2, ty2 = x1 + tw + 4, y1
        cv2.rectangle(im, (tx1, ty1), (tx2, ty2), (0, 255, 0), -1)
        cv2.putText(im, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return im


def find_images(img_dir: Path):
    files = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def run_single(img_path: str, xml_path: str, out_path: str,
               show_score=True, score_thres=None, thickness=2):
    im = cv2.imread(img_path)
    if im is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    objs = parse_voc_xml(xml_path)
    im = draw_boxes(im, objs, show_score=show_score, score_thres=score_thres, thickness=thickness)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, im)


def run_batch(img_dir: str, xml_dir: str, out_dir: str,
              show_score=True, score_thres=None, thickness=2,
              keep_structure=True):
    img_root = Path(img_dir).resolve()
    xml_root = Path(xml_dir).resolve()
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    imgs = find_images(img_root)
    if not imgs:
        print(f"[Error] No images under: {img_root}")
        return

    missing = []
    pbar = tqdm.tqdm(imgs, desc="Visualizing VOC -> images")
    for img_path in pbar:
        rel = img_path.relative_to(img_root)  # keep subdirs
        xml_path = (xml_root / rel).with_suffix(".xml")

        if not xml_path.exists():
            missing.append(str(rel))
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            missing.append(f"{str(rel)}\tread_img_failed")
            continue

        objs = parse_voc_xml(str(xml_path))
        im = draw_boxes(im, objs, show_score=show_score, score_thres=score_thres, thickness=thickness)

        if keep_structure:
            out_path = out_root / rel
        else:
            out_path = out_root / img_path.name

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), im)

    if missing:
        miss_txt = out_root / "missing_xml.txt"
        with open(miss_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
        print(f"[Warn] Missing XML count={len(missing)}. Saved list: {miss_txt}")

    print(f"[Done] Output dir: {out_root}")


def main():
    parser = ArgumentParser()

    # single
    parser.add_argument("--img", type=str, default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0/imgs/SP-BC-3", help="single image path")
    parser.add_argument("--xml", type=str, default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0/boxes/SP-BC-3", help="single xml path")
    parser.add_argument("--out", type=str, default="/root/autodl-tmp/racketpose2.0/vis/bbox/SP-BC-3", help="single output image path")

    # batch
    parser.add_argument("--img-dir", type=str, default=None, help="image root folder (recursive)")
    parser.add_argument("--xml-dir", type=str, default=None, help="xml root folder (same rel path as images)")
    parser.add_argument("--out-dir", type=str, default="vis", help="output folder for visualized images")
    parser.add_argument("--no-keep-structure", action="store_true", help="do not keep subdir structure in out-dir")

    # draw options
    parser.add_argument("--no-score", action="store_true", help="do not show score text even if score exists")
    parser.add_argument("--score-thres", type=float, default=None, help="only draw boxes with score >= thres (if score exists)")
    parser.add_argument("--thickness", type=int, default=2, help="bbox line thickness")

    args = parser.parse_args()

    show_score = (not args.no_score)

    # single mode
    # if args.img and args.xml:
    #     out_path = args.out
    #     if out_path is None:
    #         out_path = str(Path(args.out_dir) / Path(args.img).name)
    #     run_single(args.img, args.xml, out_path,
    #                show_score=show_score, score_thres=args.score_thres, thickness=args.thickness)
    #     print(f"[Done] Saved: {out_path}")
    #     return

    # batch mode
    if args.img_dir and args.xml_dir:
        run_batch(args.img_dir, args.xml_dir, args.out_dir,
                  show_score=show_score, score_thres=args.score_thres, thickness=args.thickness,
                  keep_structure=(not args.no_keep_structure))
        return

    print("Usage:")
    print("  Single: python vis_voc.py --img xxx.jpg --xml xxx.xml --out vis/xxx.jpg")
    print("  Batch : python vis_voc.py --img-dir imgs --xml-dir boxes --out-dir vis")


if __name__ == "__main__":
    main()
