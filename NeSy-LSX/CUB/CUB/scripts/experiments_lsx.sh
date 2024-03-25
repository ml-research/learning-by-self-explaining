#!/bin/bash

DEVICE=$1
SEED=$2
NUM=$3
PERC=$4

## ======================= Koh et al. data preprocessing for predicting the concepts via CNN =======================

## Concept Model
CUDA_VISIBLE_DEVICES=$DEVICE python3 experiments.py cub Concept_XtoC -seed $SEED -ckpt \
-log_dir runs/ConceptModel__Seed${SEED}_${NUM}/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr \
-weighted_loss -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 \
-lr 0.01 -scheduler_step 1000 -bottleneck \
-data_dir /workspace/repositories/SelfReflectiveCoder_CUB/CUB_processed/class_filtered_10 \
-image_dir /workspace/datasets_wolf/CUB_200_2011

# generate data from perception model (i.e. inception model)
CUDA_VISIBLE_DEVICES=$DEVICE python3 CUB/generate_new_data.py ExtractConcepts --model_path \
/workspace/repositories/SelfReflectiveCoder_CUB/runs/ConceptModel__Seed${SEED}_${NUM}/outputs/best_model_${SEED}.pth \
--data_dir CUB_processed/class_filtered_10 \
--out_dir CUB_processed/ConceptModel${SEED}_${NUM}__PredConcepts

## ======================= Our additional data preprocessing =======================

python3 CUB/data_processing.py -save_dir CUB_processed/class_filtered_10_majority_noise \
-data_dir /workspace/datasets_wolf/CUB_200_2011 -filter_attributes -post_processing majority_noise

## ======================= Main experiments =======================
# run baseline
CUDA_VISIBLE_DEVICES=$DEVICE python3 experiments.py cub Sequential_CtoY -seed $SEED \
-log_dir runs/SequentialModel_WithVal__Seed${SEED}_${NUM}/outputs/ -e 600 -optimizer sgd \
-use_aux -use_attr -weighted_loss -ckpt -n_imgclasses 10 \
-n_attributes 112 -no_img -b 64 -weight_decay 0.00004 -lr 0.005 -scheduler_step 1000 -perc $PERC \
-data_dir CUB_processed/class_filtered_10_majority_noise \

# Test vanilla model
CUDA_VISIBLE_DEVICES=$DEVICE python3 CUB/inference.py \
-model_dir runs/SequentialModel_WithVal__Seed${SEED}_${NUM}/outputs/last_model_${SEED}.pth \
-eval_data test -use_attr -n_attributes 112 -bottleneck -no_img -n_imgclasses 10 \
-log_dir runs/SequentialModel_WithVal__Seed${SEED}_${NUM}/outputs/ \
-data_dir CUB_processed/class_filtered_10_majority_noise

#-------------------------------------------#
# Run LSX
CUDA_VISIBLE_DEVICES=$DEVICE python3 experiments.py \
cub LSX -seed $SEED -log_dir runs/SEL__Seed${SEED}_${NUM}/outputs/ -e 200 -srl-iters 1 -expl-epochs 400 \
-optimizer sgd -use_aux -use_attr -n_imgclasses 10 \
-weighted_loss -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.005 -scheduler_step 1000 \
-bottleneck -prop-thresh 0.25 -prop-min-thresh 0.125 -min_n_atoms_per_clause 4 -topk 1 \
-perc $PERC \
-data_dir CUB_processed/class_filtered_10_majority_noise \

# Test LSX model
CUDA_VISIBLE_DEVICES=$DEVICE python3 CUB/inference.py \
-model_dir runs/SEL__Seed${SEED}_${NUM}/outputs/last_model_${SEED}_0.5.pth \
-eval_data test -use_attr -n_attributes 112 -bottleneck -no_img -use_sigmoid -n_imgclasses 10 \
-data_dir CUB_processed/class_filtered_10_majority_noise \
-log_dir runs/SEL__Seed${SEED}_${NUM}/outputs/
