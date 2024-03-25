#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
SEED=$2
DATA=$3

#-------------------------------------------------------------------------------#

CUDA_VISIBLE_DEVICES=$DEVICE python experiments.py --training_mode test --random_seed $SEED \
--dataset $DATA  --explanation_loss_weight 1 --learning_rate 0.004 --n_epochs 00 --n_pretraining_epochs 2060 \
--few_shot_train_percent 1.0 --n_critic_batches 68 --model Net1 \
--model_pt runs/.../pretrained_model.pt