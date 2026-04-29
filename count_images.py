from pathlib import Path
from argparse import ArgumentParser
from collections import Counter


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--img-dir",
        default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0/imgs",
        type=str,
        help="数据集图片根目录（递归统计）"
    )
    parser.add_argument(
        "--exts",
        default=".jpg,.jpeg,.png,.bmp,.webp",
        type=str,
        help="统计的图片后缀（逗号分隔）"
    )
    args = parser.parse_args()

    img_root = Path(args.img_dir)
    if not img_root.exists():
        raise FileNotFoundError(f"Directory not found: {img_root}")

    exts = {e.lower().strip() for e in args.exts.split(",")}

    files = [p for p in img_root.rglob("*") if p.suffix.lower() in exts]

    print(f"\n[Image Count]")
    print(f"Root: {img_root}")
    print(f"Extensions: {sorted(exts)}")
    print(f"Total images: {len(files)}\n")

    # 可选：按后缀统计
    counter = Counter(p.suffix.lower() for p in files)
    print("Breakdown by extension:")
    for k, v in counter.items():
        print(f"  {k:>6}: {v}")


if __name__ == "__main__":
    main()
