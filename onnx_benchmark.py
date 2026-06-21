import argparse
import time
from argparse import Namespace

import numpy as np
import onnxruntime as ort
import psutil
from torch.utils.data import DataLoader

from datasets import build_dataset

_PROC = psutil.Process()


def current_rss_mb():
    return _PROC.memory_info().rss / (1024 * 1024)


PROVIDER_MAP = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


def pick_providers(name, available):
    if name == "auto":
        for p in ("CUDAExecutionProvider", "CPUExecutionProvider"):
            if p in available:
                return [p]
        return available[:1]
    ep = PROVIDER_MAP[name]
    if ep not in available:
        raise RuntimeError(f"Provider {ep} not available. Available: {available}")
    # CUDA/TensorRT fall back to CPU on unsupported ops.
    return [ep, "CPUExecutionProvider"] if ep != "CPUExecutionProvider" else [ep]


def infer_input_size(session):
    shape = session.get_inputs()[0].shape
    # Expect (N, 3, H, W). H and W may be strings (dynamic).
    h = shape[2] if isinstance(shape[2], int) else 224
    return h


def topk_correct(logits, targets, ks=(1, 5)):
    out = {}
    topk = np.argsort(-logits, axis=1)[:, : max(ks)]
    for k in ks:
        out[k] = int((topk[:, :k] == targets[:, None]).any(axis=1).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to .onnx model")
    ap.add_argument(
        "--data-set",
        required=True,
        choices=["CIFAR", "IMNET", "PETS", "INAT", "INAT19"],
        help="Dataset name (matches datasets.build_dataset)",
    )
    ap.add_argument("--data-path", required=True, help="Path to dataset root")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Override input size (default: read from ONNX or 224)",
    )
    ap.add_argument(
        "--provider", default="auto", choices=["auto", "cpu", "cuda", "tensorrt"]
    )
    ap.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup batches (not timed)"
    )
    ap.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit to N batches (default: full test set)",
    )
    ap.add_argument(
        "--inat-category",
        default="name",
        help="iNat category (only used for INAT/INAT19)",
    )
    ap.add_argument(
        "--intra-op-threads",
        type=int,
        default=0,
        help="ORT intra-op threads (0 = ORT default)",
    )
    args = ap.parse_args()

    so = ort.SessionOptions()
    if args.intra_op_threads > 0:
        so.intra_op_num_threads = args.intra_op_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    providers = pick_providers(args.provider, available)
    print(f"Available providers: {available}")
    print(f"Using providers:     {providers}")

    session = ort.InferenceSession(args.onnx, sess_options=so, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"ONNX input '{input_name}' shape: {input_shape}")
    print(f"ONNX output '{output_name}' shape: {session.get_outputs()[0].shape}")

    onnx_n = input_shape[0]
    if isinstance(onnx_n, int) and onnx_n != args.batch_size:
        print(
            f"[warn] ONNX has fixed batch={onnx_n} but --batch-size={args.batch_size}. "
            "Re-export with --dynamic-batch or change --batch-size."
        )

    img_size = args.input_size or infer_input_size(session)
    ds_args = Namespace(
        data_set=args.data_set,
        data_path=args.data_path,
        input_size=img_size,
        inat_category=args.inat_category,
        # build_transform reads these only on is_train=True, but set to be safe.
        aa="noaug",
        color_jitter=0.0,
        train_interpolation="bicubic",
        reprob=0.0,
        remode="pixel",
        recount=1,
    )
    dataset, nb_classes = build_dataset(args=ds_args, split="test")
    print(
        f"Dataset {args.data_set}: {len(dataset)} samples, {nb_classes} classes, "
        f"input size {img_size}"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    correct = {1: 0, 5: 0}
    total = 0
    latencies = []
    total_inference_time = 0.0
    peak_rss_mb = current_rss_mb()

    ks = (1, 5) if nb_classes >= 5 else (1,)

    for batch_idx, (images, targets) in enumerate(loader):
        x = images.numpy().astype(np.float32, copy=False)
        y = targets.numpy()

        is_warmup = batch_idx < args.warmup
        t0 = time.perf_counter()
        logits = session.run([output_name], {input_name: x})[0]
        _ = float(logits.sum())
        dt = time.perf_counter() - t0

        peak_rss_mb = max(peak_rss_mb, current_rss_mb())

        if not is_warmup:
            latencies.append(dt)
            total_inference_time += dt
            counts = topk_correct(logits, y, ks=ks)
            for k in ks:
                correct[k] += counts[k]
            total += y.shape[0]

        if (batch_idx + 1) % 20 == 0:
            running_top1 = (correct[1] / total * 100) if total else 0.0
            tag = "(warmup)" if is_warmup else ""
            print(
                f"  batch {batch_idx + 1}/{len(loader)}  "
                f"top1={running_top1:.2f}%  last={dt * 1000:.1f} ms  "
                f"rss={current_rss_mb():.0f} MB (peak {peak_rss_mb:.0f}) {tag}"
            )

        if args.max_batches is not None and (batch_idx + 1) >= (
            args.max_batches + args.warmup
        ):
            break

    if total == 0:
        print("No timed batches — increase --max-batches or reduce --warmup.")
        return

    lat = np.array(latencies)
    per_image_ms = lat / args.batch_size * 1000

    print("\n=== Results ===")
    print(f"Samples evaluated:        {total}")
    print(f"Top-1 accuracy:           {correct[1] / total * 100:.2f}%")
    if 5 in ks:
        print(f"Top-5 accuracy:           {correct[5] / total * 100:.2f}%")
    print()
    print(f"Timed batches:            {len(latencies)} (batch_size={args.batch_size})")
    print(f"Throughput:               {total / total_inference_time:.1f} images/sec")
    print(f"Peak process RSS:         {peak_rss_mb:.1f} MB")
    print()
    print("Per-batch latency (ms):")
    print(f"  mean   {lat.mean() * 1000:8.2f}")
    print(f"  p50    {np.percentile(lat, 50) * 1000:8.2f}")
    print(f"  p95    {np.percentile(lat, 95) * 1000:8.2f}")
    print(f"  p99    {np.percentile(lat, 99) * 1000:8.2f}")
    print(f"  min    {lat.min() * 1000:8.2f}")
    print(f"  max    {lat.max() * 1000:8.2f}")
    print()
    print("Per-image latency (ms):")
    print(f"  mean   {per_image_ms.mean():8.3f}")
    print(f"  p50    {np.percentile(per_image_ms, 50):8.3f}")


if __name__ == "__main__":
    main()
