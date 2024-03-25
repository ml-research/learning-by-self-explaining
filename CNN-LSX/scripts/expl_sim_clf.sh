#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
DATA=$2

# vanilla_model is the baseline trained model, i.e. only classification

#-------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------#
CUDA_VISIBLE_DEVICES=$DEVICE python expl_sim_clf.py --random_seed 0 --dataset $DATA --batch_size 64 \
--test_batch_size 64 --logging_disabled \
--model_pt runs/.../finetuned_model.pt \
--vanilla_model_pt runs/.../Net1.pt
