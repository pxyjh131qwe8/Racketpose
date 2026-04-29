#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import shutil
from pathlib import Path
from argparse import ArgumentParser

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm_path(p: str) -> Path:
    return Path(p.strip()).expanduser()


def load_fine_list(txt_path: Path):
    lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln)
    return lines


def main():
    ap = ArgumentParser()
    ap.add_argument("--fine-txt", required=True, help="fine_cases.txt")
    ap.add_argument("--dataset-root", required=True,
                    help="原始数据集根目录（包含 imgs/ boxes/ labels_with_center_and_normal/）")
    ap.add_argument("--out-root", default="fine_cases",
                    help="输出 fine_cases 目录")
    args = ap.parse_args()

    fine_txt = Path(args.fine_txt).resolve()
    ds_root = Path(args.dataset_root).resolve()
    out_root = Path(args.out_root).resolve()

    imgs_root = ds_root / "imgs"
    boxes_root = ds_root / "boxes"
    labels_root = ds_root / "labels_with_center_and_normal"

    out_imgs = out_root / "imgs"
    out_boxes = out_root / "boxes"
    out_labels = out_root / "labels"
    out_csv = out_root / "fine_cases.csv"

    out_imgs.mkdir(parents=True, exist_ok=True)
    out_boxes.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    fine_list = load_fine_list(fine_txt)
    print(f"[LOAD] fine cases = {len(fine_list)}")

    # ---------- 1. 收集 labels（所有 csv） ----------
    all_csvs = list(labels_root.rglob("*.csv"))
    csv_rows = []
    header = None

    for csv_p in all_csvs:
        with open(csv_p, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                continue
            if header is None:
                header = r.fieldnames
            for row in r:
                csv_rows.append(row)

    # filename -> row
    label_map = {}
    for r in csv_rows:
        fn = r.get("filename", "").replace("\\", "/")
        if fn:
            label_map[fn] = r

    merged_rows = []

    # ---------- 2. 逐条处理 fine_cases ----------
    miss_img = miss_box = miss_label = 0

    for line in fine_list:
        p = norm_path(line)

        # 支持相对路径
        if not p.is_absolute():
            p = imgs_root / p

        if not p.exists():
            print(f"[MISS IMG] {p}")
            miss_img += 1
            continue

        # 相对 imgs_root
        try:
            rel = p.relative_to(imgs_root)
        except ValueError:
            print(f"[SKIP] not under imgs_root: {p}")
            continue

        # ---- copy image ----
        dst_img = out_imgs / rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst_img)

        # ---- copy box ----
        xml_src = (boxes_root / rel).with_suffix(".xml")
        if xml_src.exists():
            dst_xml = out_boxes / rel.with_suffix(".xml")
            dst_xml.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(xml_src, dst_xml)
        else:
            miss_box += 1

        # ---- collect label ----
        key = str(Path("imgs") / rel).replace("\\", "/")
        if key in label_map:
            r = dict(label_map[key])
            r["filename"] = key
            merged_rows.append(r)
        else:
            miss_label += 1

    # ---------- 3. 写出 fine_cases.csv ----------
    if header and merged_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in merged_rows:
                w.writerow(r)

    print("\n[DONE]")
    print(f" fine imgs   : {len(fine_list) - miss_img}")
    print(f" miss imgs   : {miss_img}")
    print(f" miss boxes  : {miss_box}")
    print(f" miss labels : {miss_label}")
    print(f" csv rows    : {len(merged_rows)}")
    print(f" output dir  : {out_root}")
    print(f" output csv  : {out_csv}")


if __name__ == "__main__":
    main()
