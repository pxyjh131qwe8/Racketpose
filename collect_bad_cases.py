import shutil
from pathlib import Path
from argparse import ArgumentParser


def parse_image_path(line: str) -> str:
    """
    bad_case.txt 每行格式示例：
      /path/to/img.jpg\tlow_conf=0.23
      /path/to/img.png\tedge_box(...)
      /path/to/img.jpg\tno_det

    我们只取 \t 前面的图片路径
    """
    return line.strip().split("\t")[0]


def main():
    parser = ArgumentParser()
    parser.add_argument("--bad-txt", default="/root/autodl-tmp/yolov11-detect/finetuneSP3.0bad_cases_epoch2.txt", type=str, help="bad_cases.txt 路径")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0/bad_cases_imgs_epoch2", type=str, help="复制图片到的目标文件夹")
    parser.add_argument(
        "--keep-structure",
        action="store_true",
        help="是否保持原始子目录结构（推荐开启）"
    )
    parser.add_argument(
        "--img-root",
        default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0/imgs",
        type=str,
        help="图片根目录（只有在 --keep-structure 时需要，用于计算相对路径）"
    )

    args = parser.parse_args()

    bad_txt = Path(args.bad_txt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bad_txt.exists():
        raise FileNotFoundError(f"bad_txt not found: {bad_txt}")

    if args.keep_structure and args.img_root is None:
        raise ValueError("--keep-structure 需要同时指定 --img-root")

    img_root = Path(args.img_root).resolve() if args.img_root else None

    copied = 0
    missing = 0

    with open(bad_txt, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            img_path = Path(parse_image_path(line))

            if not img_path.exists():
                print(f"[WARN] image not found: {img_path}")
                missing += 1
                continue

            if args.keep_structure:
                # 计算相对路径，保持目录结构
                try:
                    rel = img_path.resolve().relative_to(img_root)
                except ValueError:
                    # 如果图片不在 img_root 下，就直接用文件名
                    rel = img_path.name
                dst = out_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
            else:
                # 全部平铺
                dst = out_dir / img_path.name

            shutil.copy2(img_path, dst)
            copied += 1

    print(f"[Done] Copied {copied} images to: {out_dir}")
    if missing > 0:
        print(f"[Warn] {missing} images listed but not found")


if __name__ == "__main__":
    main()
