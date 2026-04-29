import os
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = {"area", "size", "level"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}")
    return df


def summarize_levels(df):
    counts = df["level"].value_counts().to_dict()
    total = len(df)
    summary = {}
    for lvl in [3, 4, 5]:
        c = counts.get(lvl, 0)
        summary[f"p{lvl}"] = c
        summary[f"ratio_p{lvl}"] = c / total if total > 0 else 0.0
    summary["total"] = total
    return summary


def print_summary(name, df):
    s = summarize_levels(df)
    print(f"\n===== {name} =====")
    print(f"total = {s['total']}")
    print(f"p3 = {s['p3']} ({s['ratio_p3']:.4f})")
    print(f"p4 = {s['p4']} ({s['ratio_p4']:.4f})")
    print(f"p5 = {s['p5']} ({s['ratio_p5']:.4f})")
    print(f"size mean = {df['size'].mean():.2f}")
    print(f"size std  = {df['size'].std():.2f}")
    print(f"size min  = {df['size'].min():.2f}")
    print(f"size max  = {df['size'].max():.2f}")


def plot_level_count_bar(df1, df2, name1, name2, save_path):
    s1 = summarize_levels(df1)
    s2 = summarize_levels(df2)

    labels = ["p3", "p4", "p5"]
    y1 = [s1["p3"], s1["p4"], s1["p5"]]
    y2 = [s2["p3"], s2["p4"], s2["p5"]]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar([i - width/2 for i in x], y1, width=width, label=name1)
    plt.bar([i + width/2 for i in x], y2, width=width, label=name2)
    plt.xticks(list(x), labels)
    plt.ylabel("Number of ROIs")
    plt.title("FPN Level Count Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_level_ratio_bar(df1, df2, name1, name2, save_path):
    s1 = summarize_levels(df1)
    s2 = summarize_levels(df2)

    labels = ["p3", "p4", "p5"]
    y1 = [s1["ratio_p3"], s1["ratio_p4"], s1["ratio_p5"]]
    y2 = [s2["ratio_p3"], s2["ratio_p4"], s2["ratio_p5"]]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar([i - width/2 for i in x], y1, width=width, label=name1)
    plt.bar([i + width/2 for i in x], y2, width=width, label=name2)
    plt.xticks(list(x), labels)
    plt.ylabel("Ratio of ROIs")
    plt.title("FPN Level Ratio Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_scatter_size_vs_level(df, name, save_path):
    plt.figure(figsize=(7, 5))
    plt.scatter(df["size"], df["level"], alpha=0.25, s=10)
    plt.yticks([3, 4, 5], ["p3", "p4", "p5"])
    plt.xlabel("sqrt(box area)")
    plt.ylabel("Assigned FPN level")
    plt.title(f"Box Size vs FPN Level ({name})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_scatter_compare(df1, df2, name1, name2, save_path):
    plt.figure(figsize=(8, 5))
    plt.scatter(df1["size"], df1["level"], alpha=0.18, s=10, label=name1)
    plt.scatter(df2["size"], df2["level"], alpha=0.18, s=10, label=name2)
    plt.yticks([3, 4, 5], ["p3", "p4", "p5"])
    plt.xlabel("sqrt(box area)")
    plt.ylabel("Assigned FPN level")
    plt.title("Box Size vs FPN Level Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_boxplot_size_by_level(df1, df2, name1, name2, save_path):
    data = []
    labels = []

    for lvl in [3, 4, 5]:
        v1 = df1[df1["level"] == lvl]["size"].values
        v2 = df2[df2["level"] == lvl]["size"].values
        data.extend([v1, v2])
        labels.extend([f"{name1}-p{lvl}", f"{name2}-p{lvl}"])

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("sqrt(box area)")
    plt.title("Box Size Distribution by FPN Level")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_hist_size(df1, df2, name1, name2, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(df1["size"], bins=50, alpha=0.5, label=name1, density=True)
    plt.hist(df2["size"], bins=50, alpha=0.5, label=name2, density=True)
    plt.xlabel("sqrt(box area)")
    plt.ylabel("Density")
    plt.title("Box Size Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv1", type=str, default="/data1/wangqiurui/pxy/yolov11-detect/weights/gl/fpn_stats_test.csv", help="第一份 CSV，例如 roi1.2/fpn_stats_test.csv")
    parser.add_argument("--csv2", type=str, default="/data1/wangqiurui/pxy/yolov11-detect/weights/gl_roi2.0/fpn_stats_test.csv", help="第二份 CSV，例如 roi2.0/fpn_stats_test.csv")
    parser.add_argument("--name1", type=str, default="ROI=1.2")
    parser.add_argument("--name2", type=str, default="ROI=2.0")
    parser.add_argument("--outdir", type=str, default="fpn_vis")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df1 = load_csv(args.csv1)
    df2 = load_csv(args.csv2)

    print_summary(args.name1, df1)
    print_summary(args.name2, df2)

    plot_level_count_bar(
        df1, df2, args.name1, args.name2,
        os.path.join(args.outdir, "level_count_bar.png")
    )

    plot_level_ratio_bar(
        df1, df2, args.name1, args.name2,
        os.path.join(args.outdir, "level_ratio_bar.png")
    )

    plot_scatter_size_vs_level(
        df1, args.name1,
        os.path.join(args.outdir, f"scatter_{args.name1.replace('=', '').replace('.', '_')}.png")
    )

    plot_scatter_size_vs_level(
        df2, args.name2,
        os.path.join(args.outdir, f"scatter_{args.name2.replace('=', '').replace('.', '_')}.png")
    )

    plot_scatter_compare(
        df1, df2, args.name1, args.name2,
        os.path.join(args.outdir, "scatter_compare.png")
    )

    plot_boxplot_size_by_level(
        df1, df2, args.name1, args.name2,
        os.path.join(args.outdir, "boxplot_size_by_level.png")
    )

    plot_hist_size(
        df1, df2, args.name1, args.name2,
        os.path.join(args.outdir, "hist_size.png")
    )

    print(f"\n图已保存到: {args.outdir}")


if __name__ == "__main__":
    main()