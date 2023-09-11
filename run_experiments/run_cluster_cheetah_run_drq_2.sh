#!/bin/bash

cd /bigdata/users/jhu/drqv2/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=cheetah_run
seed=2

echo "start running $tag with seed $seed"
python train.py task=cheetah_run experiment=$tag seed=$seed replay_buffer_num_workers=4 num_train_frames=1000000
