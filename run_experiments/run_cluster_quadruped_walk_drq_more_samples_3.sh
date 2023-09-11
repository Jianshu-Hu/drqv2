#!/bin/bash

cd /bigdata/users/jhu/drqv2/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=quadruped_walk_K_2
seed=3

echo "start running $tag with seed $seed"
python train.py task=quadruped_walk aug_K=2 experiment=$tag seed=$seed replay_buffer_num_workers=0
