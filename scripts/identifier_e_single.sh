#!/bin/bash
# prepare tmp environment
identifier=$1
task=$2
aug_type=$3
feat_aug_type=$4
seed=$5
num_train_frames=1000000
replay_buffer_num_workers=1

# prepare target folder
ori_dir=$(pwd)
echo "ori_dir: $ori_dir"
saved_figs_dir="saved_figs"
if [ ! -d "$saved_figs_dir" ]; then
    mkdir $saved_figs_dir
    echo "Created saved_figs: $saved_figs_dir"
fi
figs_target_dir="$saved_figs_dir/$identifier-$task-aug$aug_type-feat_aug$feat_aug_type-seed$seed"
if [ ! -d "$figs_target_dir" ]; then
    mkdir $figs_target_dir
    echo "Created target_dir: $figs_target_dir"
fi

save_exps_dir="saved_exps"
if [ ! -d "$save_exps_dir" ]; then
    mkdir $save_exps_dir
    echo "Created save_exps: $save_exps_dir"
fi

task_id_dir="$save_exps_dir/$identifier-$task"
if [ ! -d "$task_id_dir" ]; then
    mkdir $task_id_dir
    echo "Created task_id_dir: $task_id_dir"
fi

aug_dir="$task_id_dir/aug$aug_type-feat_aug$feat_aug_type"
if [ ! -d "$aug_dir" ]; then
    mkdir $aug_dir
    echo "Created aug_dir: $aug_dir"
fi

# prepare tmp environment
tmp_dir="/tmp/drqv2-xcc-$identifier"
if [ ! -d "$tmp_dir" ]; then
    mkdir $tmp_dir
    echo "Created tmp_dir: $tmp_dir"
    cp -r cfgs $tmp_dir
    cp -r curves $tmp_dir
    cp *.py $tmp_dir
    cp *.yml $tmp_dir
fi

cd $tmp_dir

source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

# remove redundant results
current_date=$(date +%Y.%m.%d)
source_folder="exp_local/$current_date"

find_aug_type="*aug_type=$aug_type*"
find_feat_aug_type="*feat_aug_type=$feat_aug_type*"
find_task="*task=$task*"
result_folders=$(find $source_folder -type d -name $find_aug_type -name $find_feat_aug_type -name $find_task)
for result_folder in $result_folders
do
    echo "rm -rf $result_folder"
    rm -rf $result_folder
done
find_seed="*seed=$seed*"
result_folder=$(find $source_folder -type d -name $find_aug_type -name $find_feat_aug_type -name $find_seed -name $find_task | head -n 1)
if [ -d "$result_folder" ]; then
    echo "rm -rf $result_folder"
    rm -rf $result_folder
fi

# run experiment
python train.py task=$task experiment=$task aug_type=$aug_type feat_aug_type=$feat_aug_type seed=$seed replay_buffer_num_workers=$replay_buffer_num_workers num_train_frames=$num_train_frames

# move results back
cp -r $tmp_dir/$result_folders $ori_dir/$aug_dir

result_pngs=$(find $result_folder -name "*step*.png")
cp $result_pngs $ori_dir/$figs_target_dir

# clean up tmp environment
rm -rf $tmp_dir