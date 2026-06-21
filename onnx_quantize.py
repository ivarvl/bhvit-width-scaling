import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.utils.data import DataLoader, Subset

from datasets import build_dataset


class DataLoaderCalibrationReader(CalibrationDataReader):
    def __init__(self, loader, input_name):
        self._input_name = input_name
        self._iter = iter(
            {input_name: images.numpy().astype(np.float32, copy=False)}
            for images, _ in loader
        )

    def get_next(self):
        return next(self._iter, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="fp32 ONNX model path")
    ap.add_argument("--output", required=True, help="int8 ONNX model path")
    ap.add_argument(
        "--data-set",
        required=True,
        choices=["CIFAR", "IMNET", "PETS", "INAT", "INAT19"],
    )
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--inat-category",
        default="name",
        help="iNat category (only used for INAT/INAT19)",
    )
    ap.add_argument(
        "--calibration",
        default="minmax",
        choices=["minmax", "entropy", "percentile"],
    )
    ap.add_argument(
        "--per-channel",
        action="store_true",
        default=True,
        help="Per-channel weight quantization (default: on)",
    )
    ap.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    prep = dst.with_suffix(".prep.onnx")

    # Symbolic shape inference + graph cleanup — required for reliable static quant.
    quant_pre_process(str(src), str(prep), skip_symbolic_shape=False)

    # Read input name + size from the preprocessed model.
    sess = ort.InferenceSession(str(prep), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape
    img_size = args.input_size or (shape[2] if isinstance(shape[2], int) else 224)
    del sess

    ds_args = Namespace(
        data_set=args.data_set,
        data_path=args.data_path,
        input_size=img_size,
        inat_category=args.inat_category,
        aa="noaug",
        color_jitter=0.0,
        train_interpolation="bicubic",
        reprob=0.0,
        remode="pixel",
        recount=1,
    )
    # Use the test transform (deterministic) for calibration.
    dataset, _ = build_dataset(args=ds_args, split="test")

    n = min(args.num_samples, len(dataset))
    rng = np.random.default_rng(0)
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    calib_subset = Subset(dataset, indices)

    loader = DataLoader(
        calib_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    reader = DataLoaderCalibrationReader(loader, input_name)

    calib_method = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }[args.calibration]

    print(
        f"Calibrating with {n} samples "
        f"(batch={args.batch_size}, method={args.calibration}, "
        f"per_channel={args.per_channel}, input_size={img_size})"
    )

    quantize_static(
        model_input=str(prep),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=args.per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=calib_method,
    )

    prep.unlink(missing_ok=True)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
