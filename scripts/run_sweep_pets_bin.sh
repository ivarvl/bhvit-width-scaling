#!/usr/bin/env bash
# Width-vs-precision sweep on Oxford-IIIT Pet (37 classes).
# Trains the 1-bit BHViT-tiny variant at a chosen base width d.
#
# Usage:
#   ./scripts/run_sweep_pets_bin.sh 64
#   ./scripts/run_sweep_pets_bin.sh 80
#   ...
# Valid d ∈ {64, 80, 96, 112, 128}. Each d uses configs/sweep/d${d}/config.json.
#
# Architecturally identical to scripts/run_BHViT_pets224.sh; only the model
# config changes, so the binary architecture knobs (--shift3 --shift5
# --some-fp --recu --regularization_loss) are kept on for every binary run.

set -e

D=${1:?"usage: $0 <d>   (e.g. 64, 80, 96, 112, 128)"}

CFG=configs/sweep/d${D}/config.json
if [ ! -f "$CFG" ]; then
    echo "config not found: $CFG" >&2; exit 1
fi

DATA_DIR=./dataset
OUT=logs/sweep-pets-bin-d${D}

torchrun --nproc_per_node=1 --master_port=25641 main_new.py \
    --num-workers=4 \
    --batch-size=96 \
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
    --data-set=PETS \
    --input-size=224 \
    --output-dir=${OUT} \
    --teacher-model-type=deit \
    --teacher-model=configs/deit-tiny-patch16-224-pets \
    --teacher-model-file=weights/deit-tiny-pets-224.pth \
    --model=${CFG} \
    --model-type=dbhvit \
    --replace-ln-bn \
    --weight-bits=1 \
    --input-bits=1 \
    --shift3 \
    --shift5 \
    --some-fp \
    --regularization_loss \
    --recu
