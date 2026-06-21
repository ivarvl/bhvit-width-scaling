import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOG_DIR = os.path.join(REPO, "logs")
PROFILE_CSV = os.path.join(LOG_DIR, "sweep_profile.csv")
OUT_DIR = os.path.join(HERE, "output")

RUNS = {
    "bin-d64": "sweep-pets-fullbin-d64-1000epoch-warm",
    "bin-d128": "sweep-pets-fullbin-d128-1000epoch-warm",
    "bin-d192": "sweep-pets-fullbin-d192-1000epoch-warm",
    "bin-d192-somefp": "sweep-pets-somefp-d192-1000epoch-warm",
    "fp32-d64": "sweep-pets-fp32-d64-1000epoch-warm",
}
FP_LABEL = "fp32-d64"
SOMEFP_LABEL = "bin-d192-somefp"


def best_top1(run_dir):
    best = float("-inf")
    for f in sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if "val/best_acc1" in ea.Tags().get("scalars", []):
            best = max(best, max(e.value for e in ea.Scalars("val/best_acc1")))
    if best == float("-inf"):
        raise RuntimeError(f"no val/best_acc1 in {run_dir}")
    return best


def load_profile():
    rows = {}
    with open(PROFILE_CSV) as f:
        for r in csv.DictReader(f):
            rows[r["label"]] = r
    return rows


def collect():
    prof = load_profile()
    data = []
    for label, sub in RUNS.items():
        p = prof[label]
        data.append(
            {
                "label": label,
                "d": int(p["d"]),
                "binary": label != FP_LABEL,
                "acc": best_top1(os.path.join(LOG_DIR, sub)),
                "mops": int(p["ops_total"]) / 1e6,
                "mb": (int(p["bin_weight_bytes"]) + int(p["fp_weight_bytes"]))
                / 1024
                / 1024,
            }
        )
    return data


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = collect()

    fp = next(d for d in data if not d["binary"])
    bins = sorted((d for d in data if d["binary"]), key=lambda x: x["d"])
    fullbins = [b for b in bins if b["label"] != SOMEFP_LABEL]
    somefps = [b for b in bins if b["label"] == SOMEFP_LABEL]
    for d in data:
        d["compute_frac"] = d["mops"] / fp["mops"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    bin_color = "#2E86AB"
    somefp_color = "#E07B00"
    fp_color = "#9B2D20"
    ref_color = "#444444"
    ok_color = "#2E7D32"

    fig = plt.figure(figsize=(13, 6.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.0], hspace=0.42, wspace=0.24)
    # left/right swapped: memory on the left (ax2), compute on the right (ax1)
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 0])
    # ax_t = fig.add_subplot(gs[1, :])

    bx = [b["mops"] for b in fullbins]
    by = [b["acc"] for b in fullbins]
    bd = [b["d"] for b in fullbins]

    ax1.axvspan(0, fp["mops"], color=ok_color, alpha=0.07, zorder=0)
    ax1.axvline(
        fp["mops"],
        color=ref_color,
        linestyle="--",
        linewidth=1.4,
        label="FP32 compute budget",
        zorder=2,
    )
    ax1.axhline(
        fp["acc"],
        color=ref_color,
        linestyle=":",
        linewidth=1.4,
        label="FP32 top-1",
        zorder=2,
    )
    ax1.plot(
        bx,
        by,
        "o",
        color=bin_color,
        markersize=7,
        label="binary (w/a = 1/1)",
        zorder=4,
    )
    ax1.scatter(
        [s["mops"] for s in somefps],
        [s["acc"] for s in somefps],
        marker="D",
        s=120,
        color=somefp_color,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="binary + some FP",
    )
    ax1.scatter(
        [fp["mops"]],
        [fp["acc"]],
        marker="*",
        s=320,
        color=fp_color,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label=f"FP32 @ d={fp['d']}",
    )
    for x, y, d in zip(bx, by, bd):
        ax1.annotate(
            f"d={d}",
            (x, y),
            textcoords="offset points",
            xytext=(7, -12),
            fontsize=9,
            color="#222222",
        )
    for s in somefps:
        ax1.annotate(
            f"d={s['d']}",
            (s["mops"], s["acc"]),
            textcoords="offset points",
            xytext=(7, -12),
            fontsize=9,
            color=somefp_color,
        )

    ax1.set_title("Accuracy vs. total compute", fontsize=12, pad=10)
    ax1.set_xlabel("Total compute (MOPs, FLOPs + BOPs/64)", fontsize=11)
    ax1.set_ylabel("Oxford-IIIT Pets top-1 (%)", fontsize=11)
    ax1.grid(linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.legend(fontsize=8.5, loc="lower right", frameon=True, framealpha=0.9)

    bm = [b["mb"] for b in fullbins]
    ax2.axvspan(0, fp["mb"], color=ok_color, alpha=0.07, zorder=0)
    ax2.axvline(
        fp["mb"],
        color=ref_color,
        linestyle="--",
        linewidth=1.4,
        label="FP32 memory budget",
        zorder=2,
    )
    ax2.axhline(
        fp["acc"],
        color=ref_color,
        linestyle=":",
        linewidth=1.4,
        label="FP32 top-1",
        zorder=2,
    )
    ax2.plot(
        bm,
        by,
        "o",
        color=bin_color,
        markersize=7,
        label="binary (w/a = 1/1)",
        zorder=4,
    )
    ax2.scatter(
        [s["mb"] for s in somefps],
        [s["acc"] for s in somefps],
        marker="D",
        s=120,
        color=somefp_color,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="binary + some FP",
    )
    ax2.scatter(
        [fp["mb"]],
        [fp["acc"]],
        marker="*",
        s=320,
        color=fp_color,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label=f"FP32 @ d={fp['d']}",
    )
    for x, y, d in zip(bm, by, bd):
        ax2.annotate(
            f"d={d}",
            (x, y),
            textcoords="offset points",
            xytext=(7, -12),
            fontsize=9,
            color="#222222",
        )
    for s in somefps:
        ax2.annotate(
            f"d={s['d']}",
            (s["mb"], s["acc"]),
            textcoords="offset points",
            xytext=(7, -12),
            fontsize=9,
            color=somefp_color,
        )

    ax2.set_title("Accuracy vs. total memory", fontsize=12, pad=10)
    ax2.set_xlabel("Total weight memory (MB)", fontsize=11)
    ax2.set_ylabel("Oxford-IIIT Pets top-1 (%)", fontsize=11)
    ax2.grid(linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.legend(fontsize=8.5, loc="lower right", frameon=True, framealpha=0.9)

    all_acc = [d["acc"] for d in data]
    lo, hi = min(all_acc), max(all_acc)
    pad = max((hi - lo) * 0.7, 8.0)
    ylim = (lo - (pad / 2), hi + (pad / 3))
    ax1.set_ylim(*ylim)
    ax2.set_ylim(*ylim)

    png = os.path.join(OUT_DIR, "width_vs_precision.png")
    pdf = os.path.join(OUT_DIR, "width_vs_precision.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    print(f"Saved: {png}\n       {pdf}")


if __name__ == "__main__":
    main()
