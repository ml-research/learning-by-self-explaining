#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
NSFR=$4
MODEL="srl-$NUM"
DATASET=clevr-hans-state

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_slot_att_set_transformer_xil_with_reasoning.py \
--data-dir $DATA --epochs 20 --name $MODEL --lr 0.001 --l2_grads 100 --batch-size 128 \
--n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 \
--nsfr $NSFR --mode train


#CUDA_VISIBLE_DEVICES=$DEVICE python train_clevr_hans_slot_att_set_transformer_xil_with_reasoning.py \
#--data-dir $DATA --epochs 20 --name $MODEL --lr 0.001 --l2_grads 100 --batch-size 64 \
#--n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 \
#--nsfr $NSFR --mode testgrad --fp-ckpt
