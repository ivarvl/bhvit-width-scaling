import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from profile_model import Counts, profile  # noqa: E402

SWEEP_DIR = os.path.join(REPO, "configs", "sweep")
OUT_DIR = os.path.join(HERE, "output")

FP_BYTES = 4  # FP32 — matches logs/sweep_profile.csv total_bytes
BOP_DIVISOR = 64  # 1 BOP = 1/64 FLOP


def discover_widths():
    # dirs = [d for d in os.listdir(SWEEP_DIR) if re.fullmatch(r"d\d+", d)]
    # return sorted(int(d[1:]) for d in dirs)
    #
    return [64, 128, 192]  # , 224, 256]


def profile_width(d, num_classes, wbits=1, abits=1, some_fp=True, fp_bytes=FP_BYTES):
    with open(os.path.join(SWEEP_DIR, f"d{d}", "config.json")) as f:
        cfg = json.load(f)
    cfg.update(
        {
            "num_classes": num_classes,
            "weight_bits": wbits,
            "input_bits": abits,
            "some_fp": some_fp,
            "shift3": True,
            "shift5": True,
            "disable_layerscale": False,
        }
    )
    c = Counts()
    profile(cfg, c)
    print(d)
    print(c.fp_weight_params + c.bin_weight_params)

    size_mb = (c.bin_weight_params / 8 + c.fp_weight_params * fp_bytes) / 1024 / 1024
    ops_mops = (c.fp_macs + c.bin_macs / BOP_DIVISOR) / 1e6
    return size_mb, ops_mops


def annotate_h(ax, bars, fmt, xmax):
    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            fmt.format(w),
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#222222",
        )


def style_axes(ax, title, xlabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Stage-0 hidden dim $d$", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-classes", type=int, default=37)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    widths = discover_widths()
    sizes_mb, ops_mops = [], []
    for d in widths:
        s, o = profile_width(d, args.num_classes, some_fp=False, fp_bytes=2)
        sizes_mb.append(s)
        ops_mops.append(o)
        print(f"  d={d:<4d}  size={s:6.2f} MB   ops={o:7.2f} MOPs")

    somefp_d = 192
    somefp_size, somefp_ops = profile_width(
        somefp_d, args.num_classes, some_fp=True, fp_bytes=2
    )
    print(
        f"  some-FP (d={somefp_d}):  size={somefp_size:.2f} MB   ops={somefp_ops:.1f} MOPs"
    )

    fp_d = widths[0]
    fp_size, fp_ops = profile_width(
        fp_d,
        args.num_classes,
        wbits=32,
        abits=32,
        fp_bytes=4,
    )
    print(f"\nFP32 baseline (d={fp_d}):  size={fp_size:.2f} MB   ops={fp_ops:.1f} MOPs")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    bin_labels = [str(d) for d in widths]
    y_labels = bin_labels + [
        f"some-FP\n(d={somefp_d})",
        f"FP32\n(d={fp_d})",
    ]
    y_pos = np.arange(len(y_labels))

    bin_color = "#2E86AB"
    bin_color_top = "#1F6E91"
    somefp_color = "#E07B00"
    fp_color = "#9B2D20"
    ref_color = "#444444"

    def plot_panel(ax, bin_vals, somefp_val, fp_val, title, xlabel, fmt):
        colors = [bin_color] * len(bin_vals) + [somefp_color, fp_color]
        values = list(bin_vals) + [somefp_val, fp_val]
        bars = ax.barh(
            y_pos, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        style_axes(ax, title, xlabel)

        xmax = max(values) * 1.18
        ax.set_xlim(0, xmax)
        annotate_h(ax, bars, fmt, xmax)

        # vertical reference line at the FP32 value
        ax.axvline(
            fp_val,
            color=ref_color,
            linestyle="--",
            linewidth=1.4,
            zorder=4,
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    plot_panel(
        ax1,
        sizes_mb,
        somefp_size,
        fp_size,
        "Model weight memory",
        "Weight size (MB)",
        "{:.2f}",
    )
    plot_panel(
        ax2,
        ops_mops,
        somefp_ops,
        fp_ops,
        "Total compute  (FLOPs + BOPs/64)",
        "OPs (MOPs)",
        "{:.1f}",
    )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=bin_color, label="binary (w/a=1/1)"),
        Patch(facecolor=somefp_color, label=f"binary + some FP @ d={somefp_d}"),
        Patch(facecolor=fp_color, label=f"FP32 @ d={fp_d}"),
        Line2D(
            [0],
            [0],
            color=ref_color,
            linestyle="--",
            linewidth=1.4,
            label="FP32 reference",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1.0))

    png = os.path.join(args.out_dir, "sweep_bars.png")
    pdf = os.path.join(args.out_dir, "sweep_bars.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"\nSaved: {png}\n       {pdf}")


if __name__ == "__main__":
    main()
