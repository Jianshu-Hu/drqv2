#!/bin/bash
# source /bigdata/users/jhu/anaconda3/bin/activate
# conda activate equiRL

task=$1
aug_type=$2
seed=$3
identifier=$4
num_train_frames=1000000
replay_buffer_num_workers=1

echo "identifier $identifier: task=$task, aug_type=$aug_type, seed=$seed, num_train_frames=$num_train_frames, replay_buffer_num_workers=$replay_buffer_num_workers"

current_date=$(date +%Y.%m.%d)
source_folder="exp_local/$current_date"

# python train.py task=$task experiment=$task aug_type=$aug_type seed=$seed replay_buffer_num_workers=$replay_buffer_num_workers num_train_frames=$num_train_frames

save_results_foler="saved_exps"
if [ ! -d "$save_results_foler" ]; then
    mkdir $save_results_foler
    echo "Created save results folder: $save_results_foler"
fi

if [ ! -d "$save_results_foler/$task-$seed-$identifier" ]; then
    mkdir "$save_results_foler/$task-$seed-$identifier"
    echo "Created save results folder: $save_results_foler//$task-$seed-$identifier"
fi

mkdir "$save_results_foler/$task-$seed-$identifier/aug$aug_type"
find_aug_type="*aug_type=$aug_type*"
find_seed="*seed=$seed*"
find_task="*task=$task*"
result_folder=$(find $source_folder -type d -name $find_aug_type -name $find_task -name $find_seed | head -n 1)
mv $result_folder $save_results_foler/$task-$seed-$identifier/aug$aug_type/