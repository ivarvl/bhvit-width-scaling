import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from profile_model import Counts, profile  # noqa: E402

REPO = os.path.dirname(HERE)
SWEEP_DIR = os.path.join(REPO, "configs", "sweep")

# (label, d, weight_bits, input_bits, some_fp)
RUNS = [
    ("bin-d64", 64, 1, 1, False),
    ("bin-d80", 80, 1, 1, False),
    ("bin-d96", 96, 1, 1, False),
    ("bin-d112", 112, 1, 1, False),
    ("bin-d128", 128, 1, 1, False),
    ("bin-d144", 144, 1, 1, False),
    ("bin-d160", 160, 1, 1, False),
    ("bin-d176", 176, 1, 1, False),
    ("bin-d192", 192, 1, 1, False),
    ("bin-d208", 208, 1, 1, False),
    ("bin-d224", 224, 1, 1, False),
    ("bin-d240", 240, 1, 1, False),
    ("bin-d256", 256, 1, 1, False),
    ("bin-d192-somefp", 192, 1, 1, True),
    ("fp32-d64", 64, 32, 32, False),
]


def run_one(d, wbits, abits, some_fp, num_classes, extended, bop_div, fp_bytes):
    cfg_path = os.path.join(SWEEP_DIR, f"d{d}", "config.json")
    with open(cfg_path) as f:
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

    flops = c.fp_macs + (c.aux_flops if extended else 0)
    bops = c.bin_macs
    ops_total = flops + bops / bop_div
    bin_bytes = c.bin_weight_params / 8
    fp_bytes = c.fp_weight_params * fp_bytes
    other_bytes = c.other_fp_params * fp_bytes if extended else 0
    total_bytes = bin_bytes + fp_bytes + other_bytes

    # Sum of all `.weight` tensors in the actual nn.Module — matches
    # `sum(p.numel() for n, p in model.named_parameters() if n.endswith('.weight'))`:
    # Conv/Linear .weight (binary or FP) + LayerNorm γ + nn.PReLU.weight inside RPReLU.
    dot_weight_params = c.bin_weight_params + c.fp_weight_params + c.gamma_prelu_params

    return {
        "d": d,
        "wbits": wbits,
        "abits": abits,
        "some_fp": some_fp,
        "fp_macs": c.fp_macs,
        "bin_macs": c.bin_macs,
        "aux_flops": c.aux_flops,
        "ops_total": int(ops_total),
        "bin_weight_params": c.bin_weight_params,
        "fp_weight_params": c.fp_weight_params,
        "gamma_prelu_params": c.gamma_prelu_params,
        "dot_weight_params": dot_weight_params,
        "other_fp_params": c.other_fp_params,
        "bin_weight_bytes": int(bin_bytes),
        "fp_weight_bytes": int(fp_bytes),
        "other_bytes": int(other_bytes),
        "total_bytes": int(total_bytes),
    }


def fmt_ops(x):
    if x >= 1e9:
        return f"{x / 1e9:.2f}G"
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    return f"{x / 1e3:.1f}K"


def fmt_mb(b):
    return f"{b / 1024 / 1024:.2f}MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--num-classes",
        type=int,
        default=37,
        help="PETS has 37 classes; pass 1000 for ImageNet-style.",
    )
    ap.add_argument("--bop-divisor", type=int, default=64)
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "logs", "sweep_profile.csv"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for label, d, wbits, abits, some_fp in RUNS:
        r = run_one(
            d,
            wbits,
            abits,
            some_fp,
            args.num_classes,
            args.extended,
            args.bop_divisor,
            2 if wbits == 1 else 4,
        )
        r["label"] = label
        rows.append(r)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["label"] + [k for k in rows[0] if k != "label"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    mode = "EXTENDED" if args.extended else "PAPER"
    print(
        f"# Mode: {mode}   BOP divisor: 1/{args.bop_divisor}   classes: {args.num_classes}"
    )
    print(f"# CSV: {args.out}")
    print()
    print(
        f"{'label':<10} {'d':>4} {'w/a':>5}  "
        f"{'FP MACs':>9} {'Bin MACs':>10} {'OPs':>10}  "
        f"{'BinMem':>8} {'FPMem':>8} {'Total':>8}"
    )
    print("-" * 84)
    for r in rows:
        print(
            f"{r['label']:<10} {r['d']:>4} "
            f"{r['wbits']:>2}/{r['abits']:<2}  "
            f"{fmt_ops(r['fp_macs']):>9} "
            f"{fmt_ops(r['bin_macs']):>10} "
            f"{fmt_ops(r['ops_total']):>10}  "
            f"{fmt_mb(r['bin_weight_bytes']):>8} "
            f"{fmt_mb(r['fp_weight_bytes']):>8} "
            f"{fmt_mb(r['total_bytes']):>8}"
        )


if __name__ == "__main__":
    main()
