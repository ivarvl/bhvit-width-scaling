import argparse
import math
import os
import sys
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTConfig, ViTForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets import build_dataset


def timm_deit_to_hf_vit(timm_sd, num_layers=12):
    new = {}
    new["vit.embeddings.cls_token"] = timm_sd["cls_token"]
    new["vit.embeddings.position_embeddings"] = timm_sd["pos_embed"]
    new["vit.embeddings.patch_embeddings.projection.weight"] = timm_sd[
        "patch_embed.proj.weight"
    ]
    new["vit.embeddings.patch_embeddings.projection.bias"] = timm_sd[
        "patch_embed.proj.bias"
    ]
    for i in range(num_layers):
        qkv_w = timm_sd[f"blocks.{i}.attn.qkv.weight"]
        qkv_b = timm_sd[f"blocks.{i}.attn.qkv.bias"]
        d = qkv_w.shape[0] // 3
        new[f"vit.encoder.layer.{i}.attention.attention.query.weight"] = qkv_w[:d]
        new[f"vit.encoder.layer.{i}.attention.attention.key.weight"] = qkv_w[d : 2 * d]
        new[f"vit.encoder.layer.{i}.attention.attention.value.weight"] = qkv_w[2 * d :]
        new[f"vit.encoder.layer.{i}.attention.attention.query.bias"] = qkv_b[:d]
        new[f"vit.encoder.layer.{i}.attention.attention.key.bias"] = qkv_b[d : 2 * d]
        new[f"vit.encoder.layer.{i}.attention.attention.value.bias"] = qkv_b[2 * d :]
        new[f"vit.encoder.layer.{i}.attention.output.dense.weight"] = timm_sd[
            f"blocks.{i}.attn.proj.weight"
        ]
        new[f"vit.encoder.layer.{i}.attention.output.dense.bias"] = timm_sd[
            f"blocks.{i}.attn.proj.bias"
        ]
        new[f"vit.encoder.layer.{i}.intermediate.dense.weight"] = timm_sd[
            f"blocks.{i}.mlp.fc1.weight"
        ]
        new[f"vit.encoder.layer.{i}.intermediate.dense.bias"] = timm_sd[
            f"blocks.{i}.mlp.fc1.bias"
        ]
        new[f"vit.encoder.layer.{i}.output.dense.weight"] = timm_sd[
            f"blocks.{i}.mlp.fc2.weight"
        ]
        new[f"vit.encoder.layer.{i}.output.dense.bias"] = timm_sd[
            f"blocks.{i}.mlp.fc2.bias"
        ]
        new[f"vit.encoder.layer.{i}.layernorm_before.weight"] = timm_sd[
            f"blocks.{i}.norm1.weight"
        ]
        new[f"vit.encoder.layer.{i}.layernorm_before.bias"] = timm_sd[
            f"blocks.{i}.norm1.bias"
        ]
        new[f"vit.encoder.layer.{i}.layernorm_after.weight"] = timm_sd[
            f"blocks.{i}.norm2.weight"
        ]
        new[f"vit.encoder.layer.{i}.layernorm_after.bias"] = timm_sd[
            f"blocks.{i}.norm2.bias"
        ]
    new["vit.layernorm.weight"] = timm_sd["norm.weight"]
    new["vit.layernorm.bias"] = timm_sd["norm.bias"]
    return new


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model(x).logits
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="./dataset")
    p.add_argument(
        "--data-set",
        default="CIFAR",
        choices=["CIFAR", "IMNET", "INAT", "INAT19", "PETS"],
    )
    p.add_argument(
        "--inat-category",
        default="name",
        choices=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "supercategory",
            "family",
            "genus",
            "name",
        ],
    )
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--color-jitter", type=float, default=0.4)
    p.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    p.add_argument("--train-interpolation", type=str, default="bicubic")
    p.add_argument("--reprob", type=float, default=0.25)
    p.add_argument("--remode", type=str, default="pixel")
    p.add_argument("--recount", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--mixup", type=float, default=0.8)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument(
        "--config",
        default="configs/deit-tiny-patch16-224-cifar",
        help="HuggingFace ViTConfig directory for the resulting teacher.",
    )
    p.add_argument("--output", default="weights/deit-tiny-cifar10-224.pth")
    p.add_argument("--log-dir", default="logs/teacher-deit-tiny-cifar10-224")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="fraction of the training set held out (uniformly per class) "
        "for validation; the test set is reserved for evaluate_test.py",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    train_ds, nb_classes = build_dataset(
        args=args, split="train", val_fraction=args.val_fraction
    )
    val_ds, _ = build_dataset(args=args, split="val", val_fraction=args.val_fraction)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build HF model from local config; override num_labels to match the dataset.
    cfg = ViTConfig.from_pretrained(args.config)
    cfg.num_labels = nb_classes
    model = ViTForImageClassification(cfg)

    # Load ImageNet-pretrained DeiT-Tiny via timm and remap into HF ViT layout.
    print("Loading ImageNet-pretrained deit_tiny_patch16_224 via timm...")
    timm_model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
    remapped = timm_deit_to_hf_vit(
        timm_model.state_dict(), num_layers=cfg.num_hidden_layers
    )
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # Only the classifier should be missing (freshly initialized for 10 classes).
    expected_missing = {"classifier.weight", "classifier.bias"}
    real_missing = set(missing) - expected_missing
    assert not real_missing, f"Unexpected missing keys: {real_missing}"
    assert not unexpected, f"Unexpected keys in remapped state_dict: {unexpected}"
    del timm_model

    model.to(device)

    mixup_fn = (
        Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.label_smoothing,
            num_classes=nb_classes,
        )
        if args.mixup > 0 or args.cutmix > 0
        else None
    )

    criterion = (
        SoftTargetCrossEntropy()
        if mixup_fn is not None
        else nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    )

    no_decay = ["bias", "LayerNorm.weight", "layernorm"]
    params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    best_acc = 0.0
    global_step = 0
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for i, (x, y) in enumerate(train_loader):
            lr = lr_at(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(x).logits
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            n_batches += 1
            if global_step % 50 == 0:
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
            global_step += 1
        train_loss = running / max(1, n_batches)
        acc = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        print(
            f"epoch {epoch:3d} | loss {train_loss:.4f} | val acc {acc:.4f} | "
            f"lr {lr:.2e} | elapsed {elapsed / 60:.1f}m"
        )
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("val/acc1", acc, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.output)
            print(f"  saved new best to {args.output} (acc={acc:.4f})")
        writer.add_scalar("val/best_acc1", best_acc, epoch)

    print(f"Done. Best val acc: {best_acc:.4f}. Weights at {args.output}.")
    writer.close()


if __name__ == "__main__":
    main()
