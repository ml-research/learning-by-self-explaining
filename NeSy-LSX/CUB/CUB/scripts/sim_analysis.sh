#!/bin/bash

DEVICE=$1
SEED=$2
NUM=$3
PERC=$4

NCLASSES=10

# For LSX
CUDA_VISIBLE_DEVICES=$DEVICE python3 experiments.py \
cub sim -seed $SEED -log_dir runs/LSX__Seed${SEED}_${NUM}/outputs/ -e 200 -srl-iters 1 -expl-epochs 400 \
-optimizer sgd -use_aux -use_attr -n_imgclasses $NCLASSES \
-weighted_loss -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.005 -scheduler_step 1000 \
-bottleneck -prop-thresh 0.4 -prop-min-thresh 0.125 -min_n_atoms_per_clause 4 -topk 1 \
-perc $PERC \
-data_dir CUB_processed/class_filtered_10_majority_noise \
-fp_ckpt runs/LSX__Seed${SEED}_${NUM}/outputs/last_model_${SEED}_0.5.pth \
-fp_ckpt_sim runs/SequentialModel_WithVal__Seed6_0/outputs/last_model_6.pth

CUDA_VISIBLE_DEVICES=$DEVICE python3 experiments.py \
cub sim -seed $SEED -log_dir runs/SequentialModel_WithVal__Seed${SEED}_${NUM}/outputs/ -e 200 -srl-iters 1 \
-expl-epochs 400 \
-optimizer sgd -use_aux -use_attr -n_imgclasses $NCLASSES \
-weighted_loss -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.005 -scheduler_step 1000 \
-bottleneck -prop-thresh 0.4 -prop-min-thresh 0.125 -min_n_atoms_per_clause 4 -topk 1 \
-perc $PERC \
-data_dir CUB_processed/class_filtered_10_majority_noise \
-fp_ckpt runs/SequentialModel_WithVal__Seed${SEED}_${NUM}/outputs/last_model_${SEED}.pth \
-fp_ckpt_sim runs/SequentialModel_WithVal__Seed6_0/outputs/last_model_6.pth

