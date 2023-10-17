#!/bin/bash
task=$1
seed=$2
num_train_frames=1000000
replay_buffer_num_workers=4

echo "-------------------------------------------------"
echo "task=$task, seed=$seed, num_train_frames=$num_train_frames, replay_buffer_num_workers=$replay_buffer_num_workers"

current_date=$(date +%Y.%m.%d)
echo "Current date: $current_date"
result_source_folder="exp_local/$current_date"
echo "Result source folder: $result_source_folder"

save_results_foler="saved_exps"
echo "Save results folder: $save_results_foler"
if [ ! -d "$save_results_foler" ]; then
    mkdir $save_results_foler
    echo "Created save results folder: $save_results_foler"
fi

task_result_folder="$task-$seed-$current_date"
mkdir "$save_results_foler/$task_result_folder"

for aug_type in {1..11}
do
    echo "Start running $task with seed $seed and aug_type $aug_type"
    python train.py task=$task experiment=$task seed=$seed replay_buffer_num_workers=$replay_buffer_num_workers num_train_frames=$num_train_frames aug_type=$aug_type
    mkdir "$save_results_foler/$task_result_folder/aug$aug_type"
    mv $result_source_folder/* $save_results_foler/$task_result_folder/aug$aug_type/
    echo "Done running task $task with seed $seed and aug_type $aug_type, results saved in $save_results_foler/$task_result_folder/aug$aug_type"
done