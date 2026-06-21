import argparse
import copy

import torch
import torch.backends.cudnn as cudnn

import utils
from datasets import build_dataset
from engine import evaluate
from main_new import get_args_parser
from models import SyncBatchNormT, get_model

ARCH_FIELDS = (
    "model",
    "model_type",
    "weight_bits",
    "input_bits",
    "drop_path",
    "shift3",
    "shift5",
    "replace_ln_bn",
    "disable_layerscale",
    "enable_cls_token",
    "gsb",
    "recu",
    "some_fp",
    "avg_res3",
    "avg_res5",
)


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    if not args.checkpoint:
        raise ValueError("--checkpoint is required to evaluate on the test set.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Rebuild the model with the exact architecture it was trained with. The
    # training checkpoint stores its full args namespace; fall back to the CLI
    # args for older checkpoints that don't.
    model_args = copy.deepcopy(args)
    ckpt_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if ckpt_args is not None:
        for field in ARCH_FIELDS:
            if hasattr(ckpt_args, field):
                setattr(model_args, field, getattr(ckpt_args, field))
    else:
        print(
            "[warn] checkpoint has no saved args; using CLI flags for the architecture"
        )

    print(args)

    dataset_test, model_args.nb_classes = build_dataset(args=args, split="test")
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(f"Creating model: {model_args.model}")
    model = get_model(
        model_args,
        model_args.model,
        model_args.model_type,
        model_args.weight_bits,
        model_args.input_bits,
    )
    if model_args.replace_ln_bn:
        model = SyncBatchNormT.convert_sync_batchnorm(model)
    model.to(device)

    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    missing, unexpected = model.load_state_dict(
        state_dict, strict=not args.no_strict_load
    )
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)} keys")

    test_stats = evaluate(data_loader_test, model, device)
    print(
        f"Accuracy of the network on the {len(dataset_test)} test images: "
        f"acc@1 {test_stats['acc1']:.3f}%  acc@5 {test_stats['acc5']:.3f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "BHViT test-set evaluation script", parents=[get_args_parser()]
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        required=True,
        help="path to the trained checkpoint to evaluate (e.g. logs/<run>/best.pth)",
    )
    args = parser.parse_args()
    main(args)
