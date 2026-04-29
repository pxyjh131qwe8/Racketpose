#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path
from argparse import ArgumentParser

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))

def _dedup_name(dst_dir: Path, name: str):
    """
    若 flat 模式下重名，自动加 _1, _2 ...
    """
    base = Path(name).stem
    ext = Path(name).suffix
    cand = dst_dir / name
    if not cand.exists():
        return cand
    k = 1
    while True:
        cand = dst_dir / f"{base}_{k}{ext}"
        if not cand.exists():
            return cand
        k += 1

def main():
    ap = ArgumentParser()
    ap.add_argument("--list-txt", default="/root/autodl-tmp/yolov11-detect/SP3.0_bad_manual_upper.txt", type=str, help="txt file, each line is a relative image path")
    ap.add_argument("--src-root", default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0", type=str, help="image root that the relative paths are based on")
    ap.add_argument("--out-root", default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0/SP3.0_bad_manual_upper", type=str, help="output root; images will be copied into out-root/imgs/")
    ap.add_argument("--flat", action="store_true", help="do NOT keep subdir structure; put all images into one folder")
    ap.add_argument("--skip-missing", action="store_true", help="skip missing images without raising error")
    args = ap.parse_args()

    list_txt = Path(args.list_txt).resolve()
    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_imgs = out_root / "imgs"
    out_imgs.mkdir(parents=True, exist_ok=True)

    if not list_txt.exists():
        raise FileNotFoundError(f"list-txt not found: {list_txt}")
    if not src_root.exists():
        raise FileNotFoundError(f"src-root not found: {src_root}")

    rels = []
    for line in list_txt.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        # 兼容 windows 分隔符
        s = s.replace("\\", "/")
        rels.append(Path(s))

    if not rels:
        print(f"[WARN] empty list: {list_txt}")
        return

    copied = 0
    missing = []
    for rel in rels:
        # 如果 txt 里意外写了开头的 imgs/，你也可以在这里做一次清洗：
        # if rel.parts and rel.parts[0] == "imgs": rel = Path(*rel.parts[1:])

        src = src_root / rel
        if not src.exists():
            missing.append(str(rel))
            if not args.skip_missing:
                continue
            else:
                continue

        if src.suffix.lower() not in IMG_EXTS:
            # 非图片就跳过
            continue

        if args.flat:
            dst = _dedup_name(out_imgs, src.name)
        else:
            dst = out_imgs / rel

        _safe_copy(src, dst)
        copied += 1

    # 写 missing
    miss_path = out_root / "missing.txt"
    if missing:
        miss_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"[WARN] missing = {len(missing)} (saved to {miss_path})")
    else:
        # 若没有缺失，也写空文件方便你确认
        miss_path.write_text("", encoding="utf-8")

    print(f"[DONE] total listed = {len(rels)}")
    print(f"[DONE] copied       = {copied}")
    print(f"[OUT ] imgs dir     = {out_imgs}")
    print(f"[OUT ] missing list = {miss_path}")

if __name__ == "__main__":
    main()
