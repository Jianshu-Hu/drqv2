#!/bin/bash

cd /bigdata/users/ve490-fall23/lyf/drqv2/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=acrobot_swingup
seed=1

echo "start running $tag with seed $seed"
python train.py aug_type=2 task=acrobot_swingup experiment=$tag seed=$seed replay_buffer_num_workers=4 num_train_frames=1000000
seed=2

echo "start running $tag with seed $seed"
python train.py aug_type=2 task=acrobot_swingup experiment=$tag seed=$seed replay_buffer_num_workers=4 num_train_frames=1000000
seed=3

echo "start running $tag with seed $seed"
python train.py aug_type=2 task=acrobot_swingup experiment=$tag seed=$seed replay_buffer_num_workers=4 num_train_frames=1000000
seed=4

echo "start running $tag with seed $seed"
python train.py aug_type=2 task=acrobot_swingup experiment=$tag seed=$seed replay_buffer_num_workers=4 num_train_frames=1000000
seed=5

echo "start running $tag with seed $seed"
python train.py aug_type=2 task=acrobot_swingup experiment=$tag seed=$seed replay_buffer_num_workers=4 num_train_frames=1000000
