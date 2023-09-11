#!/bin/bash

cd /bigdata/users/jhu/drqv2/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=reach_duplo
seed=1

echo "start running $tag with seed $seed"
python train.py task=reach_duplo experiment=$tag seed=$seed replay_buffer_num_workers=10
