#!/bin/bash

cd /bigdata/users/jhu/drqv2/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=quadruped_run
seed=3

echo "start running $tag with seed $seed"
python train.py task=quadruped_run experiment=$tag seed=$seed replay_buffer_num_workers=0
