import argparse
import json
from dataclasses import dataclass, field


@dataclass
class Counts:
    fp_macs: int = 0
    bin_macs: int = 0
    # Auxiliary FLOPs from norms, softmax, activations, residual adds, etc.
    aux_flops: int = 0
    # Weight parameters (Conv/Linear weights).
    fp_weight_params: int = 0  # FP weights of FP Conv/Linear
    bin_weight_params: int = 0  # 1-bit weights of binary Conv/Linear
    # All other FP params
    other_fp_params: int = 0
    gamma_prelu_params: int = 0
    breakdown: list = field(default_factory=list)

    def add_fp_macs(self, macs, label):
        self.fp_macs += macs
        self.breakdown.append((label, "FP-MAC", macs))

    def add_bin_macs(self, macs, label):
        self.bin_macs += macs
        self.breakdown.append((label, "BIN-MAC", macs))

    def add_aux(self, n_elem, ops_per_elem, label):
        self.aux_flops += n_elem * ops_per_elem
        self.breakdown.append((label, "AUX", n_elem * ops_per_elem))

    def add_norm(self, C):
        self.other_fp_params += 2 * C
        self.gamma_prelu_params += C

    def add_rprelu(self, C):
        self.other_fp_params += 3 * C
        self.gamma_prelu_params += C


LN_OPS = 8
SOFTMAX_OPS = 4
RPRELU_OPS = 3
GELU_OPS = 8


def linear_macs(in_features, out_features, tokens):
    return tokens * in_features * out_features


def conv2d_macs(c_in, c_out, k, h_out, w_out, groups=1):
    return h_out * w_out * (c_in // groups) * k * k * c_out


def profile(cfg, counts):
    img = cfg["image_size"]
    patch = cfg["patch_size"]
    window = cfg.get("window_size", 7)
    hidden_sizes = cfg["hidden_size"]
    inter_sizes = cfg["intermediate_size"]
    num_heads = cfg["num_attention_heads"]
    depths = cfg["depths"]
    num_classes = cfg["num_classes"]
    weight_bits = cfg["weight_bits"]
    input_bits = cfg["input_bits"]
    some_fp = cfg["some_fp"]
    shift3 = cfg["shift3"]
    shift5 = cfg["shift5"]
    disable_layerscale = cfg["disable_layerscale"]

    binary_path = weight_bits == 1 and input_bits == 1

    def matmul(macs, label, weights):
        if binary_path:
            counts.add_bin_macs(macs, label)
        else:
            counts.add_fp_macs(macs, label)
        if weight_bits == 1:
            counts.bin_weight_params += weights
        else:
            counts.fp_weight_params += weights

    s0 = img // patch
    H_per_stage = [s0 // (2**i) for i in range(4)]
    T_per_stage = [h * h for h in H_per_stage]

    C0 = hidden_sizes[0]
    counts.add_fp_macs(conv2d_macs(3, C0, patch, s0, s0), "stem.proj")
    counts.fp_weight_params += 3 * patch * patch * C0
    counts.add_norm(C0)  # stem.norm γ,β
    counts.other_fp_params += C0 * T_per_stage[0]  # position embed
    counts.add_aux(T_per_stage[0] * C0, LN_OPS, "stem.norm")
    counts.add_aux(T_per_stage[0] * C0, GELU_OPS, "stem.gelu")
    counts.add_aux(T_per_stage[0] * C0, 1, "stem.pos_embed_add")

    for stage_idx in range(4):
        C = hidden_sizes[stage_idx]
        I = inter_sizes[stage_idx]
        H = H_per_stage[stage_idx]
        T = T_per_stage[stage_idx]
        n_blocks = depths[stage_idx]
        is_attn = stage_idx >= 2

        for b in range(n_blocks):
            pfx = f"stage{stage_idx}.block{b}"

            # LayerNorm before
            counts.add_aux(T * C, LN_OPS, f"{pfx}.ln_before")
            counts.add_norm(C)

            if not is_attn:
                counts.add_aux(T * C, 1, f"{pfx}.tm.move")
                counts.other_fp_params += C
                for ci in range(3):
                    macs = conv2d_macs(C, C, 3, H, H, groups=4)
                    w = (C // 4) * 3 * 3 * C
                    matmul(macs, f"{pfx}.tm.cov{ci + 1}", w)
                    counts.add_aux(T * C, 1, f"{pfx}.tm.cov{ci + 1}.bias")
                    counts.other_fp_params += C
                    counts.add_aux(T * C, RPRELU_OPS, f"{pfx}.tm.rprelu{ci + 1}")
                    counts.add_rprelu(C)
                counts.add_aux(T * C, 2, f"{pfx}.tm.sum")
                counts.add_aux(T * C, LN_OPS, f"{pfx}.tm.norm")
                counts.add_norm(C)
                counts.add_aux(T * C, 1, f"{pfx}.tm.residual")

            else:
                Nw = (H // window) ** 2
                Neff = window * window + Nw
                tokens_attn = Nw * Neff
                d_k = C // num_heads[stage_idx]
                heads = num_heads[stage_idx]

                counts.add_aux(T * C, 2, f"{pfx}.attn.token_FA.pool")
                counts.add_aux(Nw * C, 3, f"{pfx}.attn.token_FA.mix")
                counts.other_fp_params += C  # a1
                counts.add_aux(tokens_attn * C, LN_OPS, f"{pfx}.attn.token_FA.ln")
                counts.add_norm(C)

                for proj in ("q", "k", "v"):
                    matmul(linear_macs(C, C, tokens_attn), f"{pfx}.attn.{proj}", C * C)
                    counts.add_aux(tokens_attn * C, LN_OPS, f"{pfx}.attn.{proj}.ln")
                    counts.add_norm(C)
                    counts.add_aux(
                        tokens_attn * C, RPRELU_OPS, f"{pfx}.attn.{proj}.rprelu"
                    )
                    counts.add_rprelu(C)
                    counts.other_fp_params += 2 * C  # move, move2

                qkt_macs = Nw * heads * Neff * Neff * d_k
                av_macs = Nw * heads * Neff * Neff * d_k
                if input_bits == 1:
                    counts.add_bin_macs(qkt_macs, f"{pfx}.attn.QK^T")
                    counts.add_bin_macs(av_macs, f"{pfx}.attn.AV")
                else:
                    counts.add_fp_macs(qkt_macs, f"{pfx}.attn.QK^T")
                    counts.add_fp_macs(av_macs, f"{pfx}.attn.AV")

                counts.add_aux(Nw * heads * Neff * Neff, 1, f"{pfx}.attn.scale")
                counts.add_aux(
                    Nw * heads * Neff * Neff, SOFTMAX_OPS, f"{pfx}.attn.softmax"
                )
                counts.add_aux(tokens_attn * C, LN_OPS, f"{pfx}.attn.norm_context")
                counts.add_norm(C)
                counts.add_aux(tokens_attn * C, 3, f"{pfx}.attn.qkv_residual")
                counts.add_aux(
                    tokens_attn * C, RPRELU_OPS, f"{pfx}.attn.rprelu_context"
                )
                counts.add_rprelu(C)
                counts.other_fp_params += C  # parm

                # SelfOutput.dense
                matmul(linear_macs(C, C, T), f"{pfx}.attn.output.dense", C * C)
                counts.other_fp_params += C  # move
                counts.add_aux(T * C, LN_OPS, f"{pfx}.attn.output.norm")
                counts.add_norm(C)
                counts.add_aux(T * C, 1, f"{pfx}.attn.output.residual")
                counts.add_aux(T * C, RPRELU_OPS, f"{pfx}.attn.output.rprelu")
                counts.add_rprelu(C)
                if not disable_layerscale:
                    counts.add_aux(T * C, 1, f"{pfx}.attn.output.layerscale")
                    counts.other_fp_params += C

            counts.add_aux(T * C, 1, f"{pfx}.residual1")
            counts.add_aux(T * C, LN_OPS, f"{pfx}.ln_after")
            counts.add_norm(C)

            # MLP fc1
            matmul(linear_macs(C, I, T), f"{pfx}.mlp.fc1", C * I)
            counts.other_fp_params += C
            counts.add_aux(T * I, LN_OPS, f"{pfx}.mlp.fc1.norm")
            counts.add_norm(I)
            counts.add_aux(T * I, 1, f"{pfx}.mlp.fc1.residual")
            counts.add_aux(T * I, RPRELU_OPS, f"{pfx}.mlp.fc1.rprelu")
            counts.add_rprelu(I)

            # MLP fc2
            matmul(linear_macs(I, C, T), f"{pfx}.mlp.fc2", I * C)
            counts.other_fp_params += I
            counts.add_aux(T * C, LN_OPS, f"{pfx}.mlp.fc2.norm")
            counts.add_norm(C)
            counts.add_aux(T * I, 1, f"{pfx}.mlp.fc2.pool")
            counts.add_aux(T * C, 1, f"{pfx}.mlp.fc2.residual")
            counts.add_aux(T * C, RPRELU_OPS, f"{pfx}.mlp.fc2.rprelu")
            counts.add_rprelu(C)
            if not disable_layerscale:
                counts.add_aux(T * C, 1, f"{pfx}.mlp.fc2.layerscale")
                counts.other_fp_params += C

            counts.add_aux(T * C, 1, f"{pfx}.residual2")
            for shift_on, name in ((shift3, "shift3"), (shift5, "shift5")):
                if shift_on:
                    counts.add_aux(T * C * 3, 2, f"{pfx}.{name}")
                    counts.other_fp_params += 3 * C

        if stage_idx < 3:
            C_in = hidden_sizes[stage_idx]
            C_out = hidden_sizes[stage_idx + 1]
            H_out = H_per_stage[stage_idx + 1]
            T_in = T_per_stage[stage_idx]
            T_out = T_per_stage[stage_idx + 1]

            counts.add_aux(T_in * C_in, LN_OPS, f"merge{stage_idx}.norm0")
            counts.add_norm(C_in)
            macs = conv2d_macs(C_in, C_out, 2, H_out, H_out)
            weights = C_in * 4 * C_out
            if some_fp:
                counts.add_fp_macs(macs, f"merge{stage_idx}.proj (FP)")
                counts.fp_weight_params += weights
            else:
                matmul(macs, f"merge{stage_idx}.proj (binary)", weights)
            counts.add_aux(T_out * C_out, LN_OPS, f"merge{stage_idx}.norm")
            counts.add_norm(C_out)
            counts.add_aux(T_out * C_out, 1, f"merge{stage_idx}.residual")
            counts.add_aux(T_out * C_out, RPRELU_OPS, f"merge{stage_idx}.rprelu")
            counts.add_rprelu(C_out)
            counts.add_aux(T_out * C_out, 1, f"merge{stage_idx}.pos_embed")
            counts.other_fp_params += T_out * C_out

    C_last = hidden_sizes[-1]
    counts.add_aux(T_per_stage[-1] * C_last, 1, "head.mean_pool")
    counts.add_aux(C_last, LN_OPS, "head.layernorm")
    counts.add_norm(C_last)
    counts.add_fp_macs(linear_macs(C_last, num_classes, 1), "head.classifier")
    counts.fp_weight_params += C_last * num_classes
    counts.other_fp_params += num_classes


def human(n, suffix="OPs"):
    for unit in ("", "K", "M", "G", "T"):
        if abs(n) < 1000:
            return f"{n:.2f} {unit}{suffix}"
        n /= 1000.0
    return f"{n:.2f} P{suffix}"


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Path to model config.json")
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--weight-bits", type=int, default=1)
    ap.add_argument("--input-bits", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument(
        "--some-fp",
        action="store_true",
        help="Keep patch-merge projections at full precision (the † variant)",
    )
    ap.add_argument("--shift3", action="store_true")
    ap.add_argument("--shift5", action="store_true")
    ap.add_argument("--disable-layerscale", action="store_true")
    ap.add_argument(
        "--fp-bytes",
        type=int,
        default=2,
        help="Bytes per FP weight param (paper: 2 = FP16). Use 4 for FP32.",
    )
    ap.add_argument(
        "--bop-divisor",
        type=int,
        default=64,
        help="BOP-to-FLOP divisor (paper: 64). Use 32 for the stricter convention.",
    )
    ap.add_argument(
        "--extended",
        action="store_true",
        help="Include LayerNorm/softmax/RPReLU FLOPs and "
        "position-embedding params in the totals.",
    )
    ap.add_argument("--show-breakdown", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    if args.image_size is not None:
        cfg["image_size"] = args.image_size
    cfg.update(
        {
            "num_classes": args.num_classes,
            "weight_bits": args.weight_bits,
            "input_bits": args.input_bits,
            "some_fp": args.some_fp,
            "shift3": args.shift3,
            "shift5": args.shift5,
            "disable_layerscale": args.disable_layerscale,
        }
    )

    c = Counts()
    profile(cfg, c)

    flops = c.fp_macs + (c.aux_flops if args.extended else 0)
    bops = c.bin_macs
    ops_total = flops + bops / args.bop_divisor

    bin_weight_bytes = c.bin_weight_params / 8
    fp_weight_bytes = c.fp_weight_params * args.fp_bytes
    other_bytes = c.other_fp_params * args.fp_bytes if args.extended else 0
    total_bytes = bin_weight_bytes + fp_weight_bytes + other_bytes

    print(f"Config:                {args.config}")
    print(f"  image_size           {cfg['image_size']}")
    print(f"  hidden_size          {cfg['hidden_size']}")
    print(f"  intermediate_size    {cfg['intermediate_size']}")
    print(f"  depths               {cfg['depths']}")
    print(f"  num_attention_heads  {cfg['num_attention_heads']}")
    print(f"  weight/input bits    {args.weight_bits}/{args.input_bits}")
    print(f"  some_fp (FDL)        {args.some_fp}")
    print(f"  num_classes          {args.num_classes}")
    print(
        f"  counting mode        {'EXTENDED (incl. aux)' if args.extended else 'PAPER (conv/linear MACs only)'}"
    )
    print(f"  BOP divisor          1/{args.bop_divisor}")
    print()
    print("Compute (per single inference, batch=1):")
    print(f"  FP MACs (conv/linear)        {human(c.fp_macs)}")
    print(f"  Binary MACs (conv/linear)    {human(c.bin_macs)}")
    print(
        f"  Aux FLOPs (norms/softmax/…)  {human(c.aux_flops)}   "
        f"[{'included' if args.extended else 'NOT included'} in totals]"
    )
    print(f"  ---")
    print(f"  FLOPs (reported)             {human(flops)}")
    print(f"  BOPs (reported)              {human(bops)}")
    print(f"  OPs = FLOPs + BOPs/{args.bop_divisor:<3d}      {human(ops_total)}")
    print()
    print("Parameters:")
    print(f"  Binary weight params         {human(c.bin_weight_params, suffix='')}")
    print(f"  FP weight params             {human(c.fp_weight_params, suffix='')}")
    print(
        f"  Total weight params          {human(c.fp_weight_params + c.bin_weight_params, suffix='')}"
    )
    print(
        f"  Other FP params (γ/β/pe/…)   {human(c.other_fp_params, suffix='')}   "
        f"[{'included' if args.extended else 'NOT included'} in size]"
    )
    print()
    print("Memory (parameters, packed):")
    print(f"  Binary weight memory   {human_bytes(bin_weight_bytes)}")
    print(f"  FP weight memory       {human_bytes(fp_weight_bytes)}")
    if args.extended:
        print(f"  Other FP memory        {human_bytes(other_bytes)}")
    print(f"  Total reported size    {human_bytes(total_bytes)}")

    if args.show_breakdown:
        print("\nPer-op breakdown:")
        for label, kind, x in c.breakdown:
            print(f"  {kind:8s}  {human(x, suffix=''):>14s}   {label}")


if __name__ == "__main__":
    main()
