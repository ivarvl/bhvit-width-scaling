#!/usr/bin/env bash
# FP32-weight, FP32-activation BHViT-tiny on PETS at d=64.
# This is the precision baseline the width sweep is compared against.
#
# Same architecture as the binary runs; only differences:
#   - --weight-bits=32 --input-bits=32 (no quantization)
#   - --some-fp dropped (moot when everything is FP)
#   - --recu / --regularization_loss dropped (BNN-only regularizers)
#   - --shift3 / --shift5 kept (architectural choice, not quant-specific)

set -e

D=${1:-64}
CFG=configs/sweep/d${D}/config.json
if [ ! -f "$CFG" ]; then
    echo "config not found: $CFG" >&2; exit 1
fi

DATA_DIR=./dataset
OUT=logs/sweep-pets-fp32-d${D}-1000epoch-newval-warm

torchrun --nproc_per_node=1 --master_port=25641 main_new.py \
    --init-from=logs/sweep-pets-fp32-d64-1000epoch-newval/checkpoint.pth \
    --eval-every=5 \
    --num-workers=4 \
    --batch-size=24 \
    --epochs=1000 \
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
    --data-set=PETS \
    --input-size=224 \
    --output-dir=${OUT} \
    --teacher-model-type=deit \
    --teacher-model=configs/deit-tiny-patch16-224-pets \
    --teacher-model-file=weights/deit-tiny-pets-224.pth \
    --model=${CFG} \
    --model-type=dbhvit \
    --replace-ln-bn \
    --weight-bits=32 \
    --input-bits=32 \
    --shift3 \
    --shift5
