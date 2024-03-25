#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1

# fp-ckpt-vanilla represents the path to the baseline model checkpoint

#-------------------------------------------------------------------------------#

python expl_sim.py --data-dir /pathto/CLEVR-Hans3/ \
--no-cuda --seed 0 --batch-size 64 \
--fp-ckpt runs/.../model.pth \
--fp-ckpt-vanilla runs/.../model.pth