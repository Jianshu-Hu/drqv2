#!/bin/bash
task=$1
aug_type=$2
seed=$3
identifier=$4
num_train_frames=1000000
replay_buffer_num_workers=1

echo "identifier $identifier: task=$task, aug_type=$aug_type, seed=$seed, num_train_frames=$num_train_frames, replay_buffer_num_workers=$replay_buffer_num_workers"

python train.py task=$task experiment=$task aug_type=$aug_type seed=$seed replay_buffer_num_workers=$replay_buffer_num_workers num_train_frames=$num_train_frames

save_results_foler="saved_exps"
if [ ! -d "$save_results_foler" ]; then
    mkdir $save_results_foler
    echo "Created save results folder: $save_results_foler"
fi

if [ ! -d "$save_results_foler/$task-$seed-$identifier" ]; then
    mkdir "$save_results_foler/$task-$seed-$identifier"
    echo "Created save results folder: $save_results_foler//$task-$seed-$identifier"
fi

current_date=$(date +%Y.%m.%d)
source_folder="exp_local/$current_date"

mkdir "$save_results_foler/$task-$seed-$identifier/aug$aug_type"
mv $source_folder/* $save_results_foler/$task-$seed-$identifier/aug$aug_type/