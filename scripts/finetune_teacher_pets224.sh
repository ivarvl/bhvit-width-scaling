#!/usr/bin/env bash
# Fine-tune ImageNet-pretrained DeiT-Tiny on Oxford-IIIT Pet (37 classes) at
# 224x224 to use as a distillation teacher for BHViT. Writes:
#   weights/deit-tiny-pets-224.pth
#   configs/deit-tiny-patch16-224-pets/config.json (already in repo)
set -e

python scripts/finetune_teacher.py \
    --data-path=./dataset \
    --data-set=PETS \
    --input-size=224 \
    --batch-size=256 \
    --num-workers=4 \
    --epochs=50 \
    --lr=1e-4 \
    --min-lr=1e-6 \
    --warmup-epochs=3 \
    --weight-decay=0.05 \
    --mixup=0.8 \
    --cutmix=1.0 \
    --label-smoothing=0.1 \
    --config=configs/deit-tiny-patch16-224-pets \
    --output=weights/deit-tiny-pets-224.pth \
    --log-dir=logs/teacher-deit-tiny-pets-224
