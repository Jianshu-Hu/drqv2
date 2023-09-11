#!/bin/bash

cd /bigdata/users/jhu/drqv2/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=hopper_hop_K_2_add_KL_add_tangent
seed=3

echo "start running $tag with seed $seed"
python train.py task=hopper_hop aug_K=2 add_KL_loss=true tangent_prop=true experiment=$tag seed=$seed replay_buffer_num_workers=4 num_train_frames=1000000
