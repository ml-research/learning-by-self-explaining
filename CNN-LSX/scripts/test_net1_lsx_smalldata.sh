#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
SEED=$2
DATA=$3

#-------------------------------------------------------------------------------#

# --> 1 200 train samples
CUDA_VISIBLE_DEVICES=$DEVICE python experiments.py --training_mode test \
--random_seed $SEED --dataset $DATA  --learning_rate 0.001 \
--classification_loss_weight 1 --explanation_loss_weight 50 --explanation_loss_weight_finetune 5 \
--n_epochs 50 --n_pretraining_epochs 10 --n_finetuning_epochs 2000 \
--few_shot_train_percent 0.02 --n_critic_batches 18 --model Net1 \
--explanation_mode input_x_gradient --batch_size 32 \
--model_pt runs/.../finetuned_model.pt
