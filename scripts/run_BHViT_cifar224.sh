#!/usr/bin/env bash
# Paper-faithful BHViT-Tiny training on CIFAR-10:
#   - input_size = 224 (DeiT-style upscale of CIFAR), so the 4-stage feature
#     pyramid becomes 56 -> 28 -> 14 -> 7 and MSMHA / MSGDC are not degenerate.
#   - configs/BHViT_tiny/config.json (image_size=224, patch_size=4, window_size=7),
#     the same architecture the paper reports 93.30% with (Table 1, NP=13.2M).
#   - DeiT-Tiny teacher fine-tuned on CIFAR-10 at 224
#     (run scripts/finetune_teacher_cifar224.sh first).
#
# LR scaling in main_new.py is Goyal-style: it only scales *up* when the global
# batch exceeds 512. With a smaller batch we keep --lr=5e-4 as-is (matches the
# paper's effective LR at batch=512). If you go above 512 (e.g. multi-GPU), the
# LR will be scaled linearly to maintain consistency.

DATA_DIR=./dataset

torchrun --nproc_per_node=1 --master_port=25641 main_new.py \
    --num-workers=4 \
    --batch-size=128 \
    --epochs=300 \
    --dropout=0.0 \
    --drop-path=0.1 \
    --opt=adamw \
    --sched=cosine \
    --weight-decay=0.05 \
    --lr=5e-4 \
    --warmup-epochs=5 \
    --color-jitter=0.4 \
    --aa=rand-m9-mstd0.5-inc1 \
    --reprob=0.25 \
    --mixup=0.8 \
    --cutmix=1.0 \
    --data-path=${DATA_DIR} \
    --data-set=CIFAR \
    --input-size=224 \
    --output-dir=logs/BHViT-cifar10-224-lrfix \
    --teacher-model-type=deit \
    --teacher-model=configs/deit-tiny-patch16-224-cifar \
    --teacher-model-file=weights/deit-tiny-cifar10-224-new.pth \
    --model=configs/BHViT_tiny/config.json \
    --model-type=dbhvit \
    --replace-ln-bn \
    --weight-bits=1 \
    --input-bits=1 \
    --shift3 \
    --shift5 \
    --some-fp \
    --regularization_loss \
    --recu
