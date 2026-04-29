#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
import time
from pathlib import Path
from argparse import ArgumentParser

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm_slash(s: str) -> str:
    return s.replace("\\", "/").lstrip("./")


def read_to_delete_list(to_delete_txt: Path):
    """
    支持每行格式：
      relpath[ \t reason...]
    取第一列作为路径。
    """
    items = []
    for line in to_delete_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        first = line.split("\t", 1)[0].strip()
        if first:
            items.append(norm_slash(first))
    return items


def resolve_relpaths(items, bad_img_root: Path | None):
    """
    items 里可能是：
      - 相对 bad_img_root 的 rel，例如 PP-BB-2/xxx.jpg
      - 绝对路径 /xxx/.../bad_cases_imgs_epoch2/imgs/PP-BB-2/xxx.jpg
      - 或者其它，但我们尽量把它归一为 relpath（相对 bad_img_root）
    """
    rels = []
    for s in items:
        p = Path(s)
        if p.is_absolute():
            if bad_img_root:
                try:
                    rel = p.resolve().relative_to(bad_img_root.resolve())
                    rels.append(norm_slash(str(rel)))
                    continue
                except Exception:
                    # 退化：取最后两级或更多（尽量不误删）
                    rels.append(norm_slash(p.name))
                    continue
            else:
                rels.append(norm_slash(p.name))
        else:
            rels.append(norm_slash(s))
    return rels


def safe_unlink(path: Path, dry_run: bool, logs: list, reason: str):
    if not path.exists():
        logs.append(f"[MISS] {path}  reason={reason}")
        return False
    if dry_run:
        logs.append(f"[DRY-RUN][DEL] {path}  reason={reason}")
        return True
    try:
        path.unlink()
        logs.append(f"[DEL] {path}  reason={reason}")
        return True
    except Exception as e:
        logs.append(f"[DEL-FAIL] {path}  reason={reason}  err={e}")
        return False


def load_csv_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        header = r.fieldnames or []
    return header, rows


def write_csv_rows(csv_path: Path, header, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_match_keys(rel: str):
    """
    你的 labels 里 filename 可能长这样：
      imgs/PP-BB-2/xxx.jpg
      imgs\\PP-BB-2\\xxx.jpg
      PP-BB-2/xxx.jpg
      或绝对路径末尾包含 /imgs/PP-BB-2/xxx.jpg
    所以我们构建多个可匹配 key，做“包含/后缀”匹配。
    """
    rel = norm_slash(rel)
    k1 = rel
    k2 = f"imgs/{rel}"
    # windows 反斜杠版本
    k1w = k1.replace("/", "\\")
    k2w = k2.replace("/", "\\")
    return {k1, k2, k1w, k2w}


def should_drop_filename(fn: str, rel: str) -> bool:
    fn2 = fn.strip()
    if not fn2:
        return False
    fn_n = norm_slash(fn2)
    keys = build_match_keys(rel)

    # 1) 精确等于（最常见）
    if fn_n in keys:
        return True

    # 2) 末尾匹配：/imgs/rel 或 /rel
    if fn_n.endswith("/" + norm_slash(rel)) or fn_n.endswith("/imgs/" + norm_slash(rel)):
        return True

    # 3) Windows 末尾匹配
    if fn2.endswith("\\" + rel.replace("/", "\\")) or fn2.endswith("\\imgs\\" + rel.replace("/", "\\")):
        return True

    return False


def clean_all_labels(labels_root: Path, rels_to_delete: list[str], dry_run: bool, logs: list):
    """
    遍历 labels_root 下所有 csv（包含 train/val/test 和根目录 csv），
    只要有 filename 列，就按 rels_to_delete 删除对应行。
    就地覆盖，并对每个 csv 生成 .bak 时间戳备份（非 dry-run）。
    """
    csv_files = sorted(labels_root.rglob("*.csv"))
    if not csv_files:
        logs.append(f"[WARN] no csv found under labels_root={labels_root}")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    total_removed = 0

    for csv_path in csv_files:
        try:
            header, rows = load_csv_rows(csv_path)
        except Exception as e:
            logs.append(f"[CSV-READ-FAIL] {csv_path} err={e}")
            continue

        if not header or "filename" not in header:
            # 不是标准 labels csv 就跳过
            continue

        kept = []
        removed = []
        for row in rows:
            fn = str(row.get("filename", ""))
            drop = False
            for rel in rels_to_delete:
                if should_drop_filename(fn, rel):
                    drop = True
                    break
            if drop:
                removed.append(row)
            else:
                kept.append(row)

        if not removed:
            continue

        total_removed += len(removed)
        logs.append(f"[CSV] {csv_path} removed={len(removed)} kept={len(kept)}")

        if dry_run:
            continue

        # 备份
        bak = csv_path.with_suffix(csv_path.suffix + f".bak_{ts}")
        try:
            bak.write_text(csv_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        except Exception as e:
            logs.append(f"[CSV-BAK-FAIL] {csv_path} -> {bak} err={e}")

        # 覆盖写回
        try:
            write_csv_rows(csv_path, header, kept)
        except Exception as e:
            logs.append(f"[CSV-WRITE-FAIL] {csv_path} err={e}")

    logs.append(f"[CSV] total removed rows = {total_removed}")


def main():
    ap = ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0", type=str,
                    help="总数据集根目录，例如 /root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0")
    ap.add_argument("--to-delete", default="/root/autodl-tmp/yolov11-detect/SP3.0_upper_to_delete_imgs.txt", type=str,
                    help="to_delete_imgs.txt 路径（每行第一列为图片相对路径或绝对路径）")
    ap.add_argument("--bad-img-root", default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0/SP3.0_bad_manual_upper/imgs", type=str,
                    help="可选：bad_cases_imgs_epoch2/imgs 的路径，用于把绝对路径转回相对 rel（推荐填）")
    ap.add_argument("--dry-run", action="store_true", help="只打印不删除/不改 csv")
    ap.add_argument("--log", default="SP3.0_upper_apply_to_delete_on_root.log", type=str)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    imgs_root = root / "imgs"
    boxes_root = root / "boxes"
    labels_root = root / "labels_with_center_and_normal"

    to_delete_txt = Path(args.to_delete).resolve()
    bad_img_root = Path(args.bad_img_root).resolve() if args.bad_img_root else None

    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")
    if not to_delete_txt.exists():
        raise FileNotFoundError(f"to_delete txt not found: {to_delete_txt}")

    logs = []
    logs.append(f"[INFO] root={root}")
    logs.append(f"[INFO] imgs_root={imgs_root}")
    logs.append(f"[INFO] boxes_root={boxes_root}")
    logs.append(f"[INFO] labels_root={labels_root}")
    logs.append(f"[INFO] to_delete_txt={to_delete_txt}")
    logs.append(f"[INFO] bad_img_root={bad_img_root}")
    logs.append(f"[INFO] dry_run={args.dry_run}")

    raw_items = read_to_delete_list(to_delete_txt)
    rels = resolve_relpaths(raw_items, bad_img_root)
    # 只保留“像图片”的条目（避免误删）
    rels = [r for r in rels if Path(r).suffix.lower() in IMG_EXTS]

    logs.append(f"[INFO] parsed rel count={len(rels)}")
    if len(rels) and len(rels) <= 10:
        logs.append("[INFO] rel examples: " + ", ".join(rels))
    elif len(rels) > 10:
        logs.append("[INFO] rel examples: " + ", ".join(rels[:10]) + " ...")

    # 删除 imgs + boxes
    del_img_cnt = 0
    del_xml_cnt = 0
    for rel in rels:
        img_path = imgs_root / rel
        xml_path = (boxes_root / rel).with_suffix(".xml")
        if safe_unlink(img_path, args.dry_run, logs, "to_delete"):
            del_img_cnt += 1
        if safe_unlink(xml_path, args.dry_run, logs, "to_delete"):
            del_xml_cnt += 1

    logs.append(f"[DONE] deleted imgs={del_img_cnt} xmls={del_xml_cnt}")

    # 清理 labels
    if labels_root.exists():
        clean_all_labels(labels_root, rels, args.dry_run, logs)
    else:
        logs.append(f"[WARN] labels_root not found: {labels_root}")

    log_path = Path(args.log).resolve()
    log_path.write_text("\n".join(logs), encoding="utf-8")
    print(f"[Done] imgs_del={del_img_cnt}, xml_del={del_xml_cnt}")
    print(f"[Log] {log_path}")


if __name__ == "__main__":
    main()
