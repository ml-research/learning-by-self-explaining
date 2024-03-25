#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
SEED=$2
DATA=$3
LAMBDAC=$4
LAMBDAE=$5
LAMBDAEF=$6

#-------------------------------------------------------------------------------#

CUDA_VISIBLE_DEVICES=$DEVICE python experiments.py --training_mode pretrain_and_joint_and_finetuning \
--random_seed $SEED --dataset $DATA  --learning_rate 0.01 \
--classification_loss_weight $LAMBDAC --explanation_loss_weight $LAMBDAE --explanation_loss_weight_finetune $LAMBDAEF \
--n_epochs 50 --n_pretraining_epochs 10 --n_finetuning_epochs 2000 \
--few_shot_train_percent 1.0 --n_critic_batches 68 --model Net1 \
--explanation_mode input_x_gradient
