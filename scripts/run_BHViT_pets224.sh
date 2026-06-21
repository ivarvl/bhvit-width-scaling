#!/usr/bin/env bash
# BHViT-Tiny training on Oxford-IIIT Pet (37 classes) at input_size=224.
#   - configs/BHViT_tiny/config.json (image_size=224, patch_size=4, window_size=7).
#   - DeiT-Tiny teacher fine-tuned on Pets at 224
#     (run scripts/finetune_teacher_pets224.sh first).
#
# LR scaling in main_new.py is Goyal-style: it only scales *up* when the global
# batch exceeds 512. With a smaller batch we keep --lr=5e-4 as-is. If you go
# above 512 (e.g. multi-GPU), the LR will be scaled linearly to maintain
# consistency.

DATA_DIR=./dataset

torchrun --nproc_per_node=1 --master_port=25641 main_new.py \
    --num-workers=4 \
    --batch-size=96 \
    --epochs=250 \
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
    --output-dir=logs/BHViT-pets-224-inc1 \
    --teacher-model-type=deit \
    --teacher-model=configs/deit-tiny-patch16-224-pets \
    --teacher-model-file=weights/deit-tiny-pets-224.pth \
    --model=configs/BHViT_tiny_inc1/config.json \
    --model-type=dbhvit \
    --replace-ln-bn \
    --weight-bits=1 \
    --input-bits=1 \
    --shift3 \
    --shift5 \
    --some-fp \
    --regularization_loss \
    --recu
