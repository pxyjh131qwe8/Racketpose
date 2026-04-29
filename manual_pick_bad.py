#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from argparse import ArgumentParser
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HELP = """Keys:
  D / Right Arrow : next
  A / Left Arrow  : prev
  Space           : toggle mark (bad)
  G               : jump to index
  Q / Esc         : save & quit
"""

def find_images(root: Path):
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files

def norm_slash(p: str) -> str:
    return p.replace("\\", "/")

def main():
    ap = ArgumentParser()
    ap.add_argument("--img-root", type=str, required=True,
                    help="图像根目录（递归扫描）。输出相对该目录的路径。")
    ap.add_argument("--out-txt", type=str, default="bad_manual.txt",
                    help="输出txt（每行一个相对路径）")
    ap.add_argument("--start", type=int, default=0, help="从第几张开始")
    ap.add_argument("--window", type=str, default="manual_picker", help="OpenCV窗口名")
    ap.add_argument("--max-w", type=int, default=1600, help="显示时最大宽度（自适应缩放）")
    args = ap.parse_args()

    root = Path(args.img_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"img-root not found: {root}")

    imgs = find_images(root)
    if not imgs:
        print(f"[ERR] no images under: {root}")
        return

    idx = max(0, min(args.start, len(imgs) - 1))
    marked = set()  # store rel paths str

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    def render(p: Path, idx: int):
        im = cv2.imread(str(p))
        if im is None:
            canvas = 255 * (0 * cv2.UMat(200, 800, cv2.CV_8UC3)).get()
            cv2.putText(canvas, "Failed to read image", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            return canvas

        h, w = im.shape[:2]
        scale = 1.0
        if w > args.max_w:
            scale = args.max_w / w
            im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        rel = norm_slash(str(p.relative_to(root)))
        is_marked = rel in marked

        # overlay text
        title = f"[{idx+1}/{len(imgs)}] {rel}"
        status = "MARKED(BAD)=YES" if is_marked else "MARKED(BAD)=NO"
        cv2.rectangle(im, (0, 0), (im.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(im, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(im, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,255) if is_marked else (200,200,200), 2, cv2.LINE_AA)

        # tiny help line
        cv2.putText(im, "A/Left prev | D/Right next | Space toggle | G jump | Q/Esc quit",
                    (10, im.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        return im

    print(HELP)

    while True:
        p = imgs[idx]
        frame = render(p, idx)
        cv2.imshow(args.window, frame)
        key = cv2.waitKey(0) & 0xFF

        rel = norm_slash(str(p.relative_to(root)))

        # Quit: q or ESC
        if key in (ord('q'), 27):
            break

        # Next
        if key in (ord('d'), 83):  # 83 is right arrow on some OpenCV builds; not always stable
            idx = min(len(imgs) - 1, idx + 1)
            continue

        # Prev
        if key in (ord('a'), 81):  # 81 is left arrow on some OpenCV builds; not always stable
            idx = max(0, idx - 1)
            continue

        # Toggle mark
        if key == 32:  # space
            if rel in marked:
                marked.remove(rel)
            else:
                marked.add(rel)
            continue

        # Jump
        if key in (ord('g'), ord('G')):
            print(f"Jump: input index (1..{len(imgs)}), current={idx+1}: ", end="", flush=True)
            try:
                s = input().strip()
                j = int(s)
                j = max(1, min(len(imgs), j))
                idx = j - 1
            except Exception:
                print("Invalid index.")
            continue

        # Arrow keys兼容：有些环境 waitKey 返回 255 或 0；可以用 hjkl 作为替代
        if key in (ord('l'),):  # vim-like
            idx = min(len(imgs) - 1, idx + 1)
            continue
        if key in (ord('h'),):
            idx = max(0, idx - 1)
            continue

    cv2.destroyAllWindows()

    out_path = Path(args.out_txt).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = sorted(marked)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[DONE] marked={len(lines)} saved -> {out_path}")

if __name__ == "__main__":
    main()
