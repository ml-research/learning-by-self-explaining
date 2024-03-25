#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
SEED=$2
NUM=$3
DATA=$4

#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
SEED=$2
NUM=$3
DATA=$4

#---------------------------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=$DEVICE python train.py --data-dir $DATA --lsx-iters 1 --epochs 100 \
--expl-epochs 300 --name src_rrr --lr 0.001 --lexpl-reg 0 --l1-reg 0 --batch-size 32 --logic-batch-size 16 \
--n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed $SEED --num $NUM --perc 1. --prop-thresh 0.5 --topk 1 \
--num-workers 0 \
--fp-ckpt runs/.../model.pth \
--dataset unconfound --mode test
