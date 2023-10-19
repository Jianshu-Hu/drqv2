#!/bin/bash

cd /bigdata/users/ve490-fall23/lyf/drqv2/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=reacher_hard
seed=1

echo "start running $tag with seed $seed"
python train.py aug_type=3 task=reacher_hard experiment=$tag seed=$seed replay_buffer_num_workers=4 num_train_frames=1000000
