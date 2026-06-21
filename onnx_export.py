import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models import get_model
from transformer.utils_quant import QuantizeConv2d, QuantizeConv2d2, QuantizeLinear


def fold_binary_weights_inplace(model):
    count = 0
    with torch.no_grad():
        for mod in model.modules():
            if not isinstance(mod, (QuantizeLinear, QuantizeConv2d, QuantizeConv2d2)):
                continue
            if getattr(mod, "weight_bits", 32) != 1:
                continue
            W = mod.weight.data
            if isinstance(mod, QuantizeLinear):
                scaling = W.abs().mean(dim=1, keepdim=True)
                centered = W - W.mean(dim=-1, keepdim=True)
            else:
                scaling = W.abs().mean(dim=[1, 2, 3], keepdim=True)
                centered = W - W.mean(dim=[1, 2, 3], keepdim=True)
            mod.weight.data.copy_(scaling * torch.sign(centered))
            mod.weight_bits = 32
            count += 1
    return count


def _detect_binary_axis(arr, tol=1e-4):
    for axis in range(arr.ndim):
        if arr.shape[axis] < 2:
            continue
        moved = np.moveaxis(arr, axis, 0)
        flat = moved.reshape(moved.shape[0], -1)
        if flat.shape[1] < 2:
            continue
        abs_vals = np.abs(flat)
        if not (abs_vals > 0).any(axis=1).all():
            continue
        # Reject broadcasted scalars: at least one channel must straddle 0
        # (real binary layers have both +s_c and -s_c per output channel).
        has_pos = (flat > 0).any(axis=1)
        has_neg = (flat < 0).any(axis=1)
        if not (has_pos & has_neg).any():
            continue
        max_abs = abs_vals.max(axis=1)
        min_nonzero = np.where(abs_vals > 0, abs_vals, np.inf).min(axis=1)
        rel_spread = (max_abs - min_nonzero) / np.maximum(max_abs, 1e-12)
        if (rel_spread < tol).all():
            return axis
    return None


def quantize_int8_weights_in_onnx(onnx_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(onnx_path, load_external_data=True)
    rename = {}
    new_dq_nodes = []
    new_inits = []
    drop_inits = set()
    converted = 0
    saved_fp = 0

    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init)
        if arr.ndim < 2:
            continue
        axis = _detect_binary_axis(arr)
        if axis is None:
            continue

        moved = np.moveaxis(arr, axis, 0)
        max_abs = (
            np.abs(moved.reshape(moved.shape[0], -1)).max(axis=1).astype(np.float32)
        )
        safe_scale = np.where(max_abs > 0, max_abs, 1.0)
        shape = (-1,) + (1,) * (moved.ndim - 1)
        int8_moved = np.round(moved / safe_scale.reshape(shape)).astype(np.int8)
        int8_arr = np.moveaxis(int8_moved, 0, axis)

        int8_name = init.name + "_int8"
        scale_name = init.name + "_scale"
        zp_name = init.name + "_zp"
        dq_out = init.name + "_dq"

        new_inits.append(numpy_helper.from_array(int8_arr, name=int8_name))
        new_inits.append(numpy_helper.from_array(max_abs, name=scale_name))
        new_inits.append(
            numpy_helper.from_array(np.zeros_like(max_abs, dtype=np.int8), name=zp_name)
        )
        new_dq_nodes.append(
            helper.make_node(
                "DequantizeLinear",
                inputs=[int8_name, scale_name, zp_name],
                outputs=[dq_out],
                name=init.name + "_DQ",
                axis=axis,
            )
        )
        drop_inits.add(init.name)
        rename[init.name] = dq_out
        saved_fp += arr.nbytes - int8_arr.nbytes - max_abs.nbytes
        converted += 1

    if converted == 0:
        print(
            "[int8] No binary-folded initializers detected — did you call fold_binary_weights_inplace first?"
        )
        return 0

    kept = [i for i in model.graph.initializer if i.name not in drop_inits]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(kept + new_inits)

    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in rename:
                node.input[i] = rename[name]

    existing = list(model.graph.node)
    model.graph.ClearField("node")
    # Prepend DQs — they only depend on initializers, so this respects topo order.
    model.graph.node.extend(new_dq_nodes + existing)

    data_file = Path(onnx_path).name + ".data"
    # Remove any stale external-data file before re-saving, otherwise
    # save_model appends rather than overwriting.
    stale = Path(onnx_path).parent / data_file
    if stale.exists():
        stale.unlink()
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=1024,
    )
    print(
        f"[int8] Converted {converted} weight initializer(s). "
        f"Approx FP32 bytes saved: {saved_fp / 1e6:.2f} MB"
    )
    return converted


def _namespace_with_defaults(ckpt_args, overrides):
    """Build the args object get_model() expects from the checkpoint's args."""
    defaults = dict(
        shift3=False,
        shift5=False,
        replace_ln_bn=False,
        disable_layerscale=False,
        enable_cls_token=False,
        gsb=False,
        recu=False,
        some_fp=False,
        drop_path=0.0,
        nb_classes=1000,
    )
    if isinstance(ckpt_args, Namespace):
        for k, v in vars(ckpt_args).items():
            defaults[k] = v
    elif isinstance(ckpt_args, dict):
        defaults.update(ckpt_args)
    defaults.update({k: v for k, v in overrides.items() if v is not None})
    return Namespace(**defaults)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    ap.add_argument(
        "--config", required=True, help="Path to model config.json dir or file"
    )
    ap.add_argument("--output", required=True, help="Output .onnx path")
    ap.add_argument(
        "--model-type", default=None, help="Override checkpoint's model_type"
    )
    ap.add_argument("--weight-bits", type=int, default=None)
    ap.add_argument("--input-bits", type=int, default=None)
    ap.add_argument(
        "--input-size", type=int, default=None, help="Override input image size"
    )
    ap.add_argument(
        "--num-classes", type=int, default=None, help="Override num_classes"
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--dynamic-batch", action="store_true", help="Export with dynamic batch axis"
    )
    ap.add_argument(
        "--int8-weights",
        action="store_true",
        help="Fold binary weights (sign(W)*scale) and store them as INT8 + "
        "per-channel DequantizeLinear, shrinking the .data file ~4x. "
        "Lossless for binary layers; FP layers (stem, classifier, RPReLU, BN) "
        "stay FP32. Activations are still binarized via Sign in the graph.",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ckpt_args = ckpt.get("args") if isinstance(ckpt, dict) else None
    if ckpt_args is None:
        raise RuntimeError(
            "Checkpoint has no embedded args; pass --model-type/--weight-bits/--input-bits manually."
        )

    overrides = {
        "model_type": args.model_type,
        "weight_bits": args.weight_bits,
        "input_bits": args.input_bits,
        "input_size": args.input_size,
        "nb_classes": args.num_classes,
    }
    model_args = _namespace_with_defaults(ckpt_args, overrides)

    config_path = args.config
    if Path(config_path).is_dir():
        config_path = str(Path(config_path) / "config.json")

    model = get_model(
        model_args,
        config_path,
        model_args.model_type,
        model_args.weight_bits,
        model_args.input_bits,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing key(s); first few: {missing[:5]}")
    if unexpected:
        print(
            f"[warn] {len(unexpected)} unexpected key(s); first few: {unexpected[:5]}"
        )
    model.eval()

    if args.int8_weights:
        n_folded = fold_binary_weights_inplace(model)
        print(f"[int8] Pre-baked binary weights in {n_folded} layer(s).")

    # Wrap so ONNX gets a clean (logits,) output instead of an ImageClassifierOutput
    class _Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values, return_dict=False)[0]

    wrapped = _Wrapper(model).eval()

    img_size = model_args.input_size
    dummy = torch.randn(args.batch_size, 3, img_size, img_size)

    with torch.no_grad():
        logits = wrapped(dummy)
    print(f"Sanity check: input {tuple(dummy.shape)} -> logits {tuple(logits.shape)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"pixel_values": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        wrapped,
        dummy,
        args.output,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=True,
    )
    print(f"Exported ONNX model to {args.output}")

    if args.int8_weights:
        quantize_int8_weights_in_onnx(args.output)
        print(f"Re-saved INT8-weight ONNX to {args.output}")


if __name__ == "__main__":
    main()
