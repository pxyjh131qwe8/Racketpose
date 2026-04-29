#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import shutil
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 常见裁剪后缀模式（支持你的命名：_c0_cls0）
_SUFFIX_PATTERNS = [
    r"(?:[_\-\.])(?:cls\d+)",                               # _cls0
    r"(?:[_\-\.])(?:c\d+)",                                 # _c3
    r"(?:[_\-\.])(?:crop\d+)",                              # _crop1 / -crop23
    r"(?:[_\-\.])(?:part\d+)",                              # _part01
    r"(?:[_\-\.])(?:tile(?:[\-\_]\d+){1,3})",               # _tile-2-4 / _tile_3
    r"(?:[_\-\.])(?:x\d+[_\-]y\d+[_\-]w\d+[_\-]h\d+)",      # _x12_y34_w200_h300
    r"(?:[_\-\.])(?:\d+[_\-]\d+[_\-]\d+[_\-]\d+)",          # _12_34_200_300
    r"(?:[_\-\.])(?:roi\d+)",                               # _roi3
    r"(?:[_\-\.])(?:patch\d+)",                             # _patch15
]
_SUFFIX_RE = re.compile(r"(?:%s)$" % "|".join(_SUFFIX_PATTERNS), re.IGNORECASE)

def strip_crop_suffix(stem: str) -> str:
    """
    反复剥离文件名末尾的裁剪后缀，直到不再匹配。
    例如 B-BB-BLUE_000001_c0_cls0 -> B-BB-BLUE_000001
    """
    prev, curr = None, stem
    while prev != curr:
        prev = curr
        curr = _SUFFIX_RE.sub("", curr)
    return curr

def list_images(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()])

def main():
    ap = argparse.ArgumentParser(description="裁剪图与原图一一对应校验+重命名（剥除裁剪后缀），缺失则回退复制原图")
    ap.add_argument("--orig",  type=Path, default="/root/autodl-tmp/Dataset/racketpose/data/imgs",  help="原图目录")
    ap.add_argument("--crops", type=Path, default="/root/autodl-tmp/Dataset/racketpose/cropx1.5/imgs",  help="裁剪图目录")
    ap.add_argument("--out",   type=Path, default="/root/autodl-tmp/Dataset/racketpose/cropx1.5_out/imgs", help="输出目录")
    ap.add_argument("--apply", action="store_true", help="默认 dry-run；加此参数才实际写文件")
    ap.add_argument("--flat",  action="store_true", help="输出是否扁平化（忽略子目录结构）；默认保持裁剪图相同的子目录层级")
    args = ap.parse_args()

    orig_files = list_images(args.orig)
    crop_files = list_images(args.crops)

    if not orig_files:
        print(f"[ERR] 原图目录无图像: {args.orig}")
        return
    if not crop_files:
        print(f"[WARN] 裁剪图目录无图像: {args.crops}（将全部直接复制原图）")

    # 相对各自根目录
    def rebase(files, root):
        return [(f, f.relative_to(root)) for f in files]

    orig_pairs = rebase(orig_files, args.orig)
    crop_pairs = rebase(crop_files, args.crops)

    # 构建映射：key = (相对父目录, 去裁剪后缀的stem)
    # 原图key不剥后缀；裁剪图key剥后缀
    orig_map = defaultdict(list)
    for f, rel in orig_pairs:
        key = (str(rel.parent), rel.stem)
        orig_map[key].append((f, rel))

    crop_map = defaultdict(list)
    for f, rel in crop_pairs:
        key = (str(rel.parent), strip_crop_suffix(rel.stem))
        crop_map[key].append((f, rel))

    # —— 去重：同一原图若有多张裁剪图，自动按文件名排序取第一张 —— 
    raw_dups = []
    for key, lst in list(crop_map.items()):
        if len(lst) > 1:
            raw_dups.append((key, [str(x[1]) for x in lst]))
            lst_sorted = sorted(lst, key=lambda t: t[1].name)
            print(f"[WARN] 原图 {key} 对应 {len(lst)} 张裁剪图，自动选择第一张：{lst_sorted[0][1].name}")
            crop_map[key] = [lst_sorted[0]]

    # 统计：缺失/多余
    missing = []  # 原图没有对应裁剪（会回退复制原图）
    extras  = []  # 裁剪没有对应原图
    for key in orig_map.keys():
        if key not in crop_map:
            missing.extend([str(rel) for _, rel in orig_map[key]])
    for key, lst in crop_map.items():
        if key not in orig_map:
            extras.extend([str(rel) for _, rel in lst])

    # —— 规划输出 —— 
    # 规则：逐个原图输出一个目标文件：
    #   - 有匹配裁剪图：复制裁剪图，输出名与原图一致（原扩展名）
    #   - 无匹配裁剪图：复制原图，输出名与原图一致
    # 这样确保 planned_copies 数量 == 原图数量
    planned_copies = []  # (src_abs, dst_abs)
    for key, o_lst in orig_map.items():
        for orig_abs, orig_rel in o_lst:
            # 输出相对路径（是否扁平）
            out_rel = Path(orig_rel.name) if args.flat else orig_rel
            out_abs = args.out / out_rel

            if key in crop_map:
                crop_abs, crop_rel = crop_map[key][0]
                # 复制裁剪图，但使用“原图文件名”（保持原扩展名 & 命名完全一致）
                src_abs = crop_abs
            else:
                # 回退：复制原图
                src_abs = orig_abs
            planned_copies.append((src_abs, out_abs))

    # —— 报告 —— 
    print("========== 检查报告 ==========")
    print(f"原图数量:   {len(orig_files)}")
    print(f"裁剪图数量: {len(crop_files)}")
    if raw_dups:
        print(f"[WARN] 发现 {len(raw_dups)} 处裁剪图‘多对一’，已自动选择其中第一张：")
        for key, rels in raw_dups[:20]:
            print("   -", key, ":", rels[:5], "..." if len(rels) > 5 else "")
    if missing:
        print(f"[INFO] 有 {len(missing)} 张原图没有匹配裁剪图：将直接复制原图回填，以保证数量一致。")
        for x in missing[:20]:
            print("   -", x)
        if len(missing) > 20:
            print("   ...")
    if extras:
        print(f"[WARN] 有 {len(extras)} 张裁剪图找不到对应原图（将被忽略）：")
        for x in extras[:20]:
            print("   -", x)
        if len(extras) > 20:
            print("   ...")

    # 一致性断言（计划写出的数量必须等于原图数量）
    if len(planned_copies) != len(orig_files):
        print(f"[FATAL] 计划写出 {len(planned_copies)} 张，但原图有 {len(orig_files)} 张。逻辑应保证二者相等，请检查脚本。")
        return

    # —— 执行 —— 
    print("========== 写出计划 ==========")
    print(f"将写出 {len(planned_copies)} 张到: {args.out}")
    if not args.apply:
        print("[DRY-RUN] 预演结束。如要执行实际写出，请添加 --apply")
        return

    for src, dst in planned_copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)  # 存在则覆盖

    # —— 最终数量检查 —— 
    out_imgs = list_images(args.out)
    print("========== 结果校验 ==========")
    print(f"输出目录图像数: {len(out_imgs)}")
    if len(out_imgs) != len(orig_files):
        print(f"[WARN] 输出数量({len(out_imgs)}) 与原图数量({len(orig_files)})不一致，请检查权限或同名覆盖。")
    else:
        print("[OK] 数量一致，一一对应完成（缺失已用原图回填）。")

if __name__ == "__main__":
    main()
