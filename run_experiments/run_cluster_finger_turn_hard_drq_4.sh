#!/bin/bash

cd /bigdata/users/jhu/drqv2/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=finger_turn_hard
seed=4

echo "start running $tag with seed $seed"
python train.py task=finger_turn_hard experiment=$tag seed=$seed replay_buffer_num_workers=4 num_train_frames=1000000
