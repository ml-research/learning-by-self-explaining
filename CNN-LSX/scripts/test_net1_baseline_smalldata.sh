#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
SEED=$2
DATA=$3

#-------------------------------------------------------------------------------#

CUDA_VISIBLE_DEVICES=$DEVICE python experiments.py --training_mode test --random_seed $SEED \
--dataset $DATA  --explanation_loss_weight 1 --learning_rate 0.001 --n_epochs 0 --n_pretraining_epochs 2060 \
--few_shot_train_percent 0.02 --n_critic_batches 18 --model Net1 --batch_size 32 --logging_disabled \
--model_pt runs/.../pretrained_model.pt