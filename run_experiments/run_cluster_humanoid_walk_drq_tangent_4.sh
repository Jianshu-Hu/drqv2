#!/bin/bash

cd /bigdata/users/jhu/drqv2/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=humanoid_walk_K_2_add_KL_add_tangent
seed=4

echo "start running $tag with seed $seed"
python train.py task=humanoid_walk aug_K=2 add_KL_loss=true tangent_prop=true experiment=$tag seed=$seed replay_buffer_num_workers=0
