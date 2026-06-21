DATA_DIR=./dataset

torchrun --nproc_per_node=1 --master_port=25641   main_new.py \
    --num-workers=2 \
    --batch-size=512 \
    --epochs=100 \
    --dropout=0.0 \
    --drop-path=0.0 \
    --opt=adamw \
    --sched=cosine \
    --weight-decay=0.00 \
    --lr=5e-4 \
    --warmup-epochs=0 \
    --color-jitter=0.0 \
    --aa=noaug \
    --reprob=0.0 \
    --mixup=0.0 \
    --cutmix=0.0 \
    --data-path=${DATA_DIR} \
    --data-set=CIFAR \
    --input-size=32 \
    --output-dir=logs/BHViT-cifar10-stride4 \
    --teacher-model-type=deit \
    --teacher-model=configs/deit-tiny-patch4-32 \
    --teacher-model-file=weights/deit-tiny-cifar10.pth \
    --model=configs/BHViT_tiny_cifar/config.json \
    --model-type=dbhvit \
    --replace-ln-bn \
    --weight-bits=1 \
    --input-bits=1 \
    --shift3 \
    --shift5 \
    --some-fp \
    #--resume= \
    #--current-best-model= \
    #--recu \
    #--regularization_loss \
