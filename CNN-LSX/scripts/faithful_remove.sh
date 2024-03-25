#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
DATA=$2

#--model_pt is the lsx trained model_pt
#--vanilla_model_pt the baseline trained model_pt (only classification)

#-------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------#
CUDA_VISIBLE_DEVICES=$DEVICE python faithful_remove.py --random_seed 0 --dataset $DATA --batch_size 64 \
--test_batch_size 64 --logging_disabled \
--model_pt runs/.../finetuned_model.pt \
--vanilla_model_pt runs/.../Net1.pt
