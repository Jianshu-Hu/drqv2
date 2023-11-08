#!/bin/bash
# prepare tmp environment
identifier=$1
task=$2
aug_type=$3
seed=$4
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
target_dir="$saved_figs_dir/$identifier-$task-aug$aug_type-seed$seed"
if [ ! -d "$target_dir" ]; then
    mkdir $target_dir
    echo "Created target_dir: $target_dir"
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

# run experiment
current_date=$(date +%Y.%m.%d)
source_folder="exp_local/$current_date"
python train.py task=$task experiment=$task aug_type=$aug_type seed=$seed replay_buffer_num_workers=$replay_buffer_num_workers num_train_frames=$num_train_frames

# move results back
find_aug_type="*aug_type=$aug_type*"
find_task="*task=$task*"
find_seed="*seed=$seed*"
result_folder=$(find $source_folder -type d -name $find_aug_type -name $find_task -name $find_seed | head -n 1)
result_pngs=$(find $result_folder -name "*step*.png")
cp $result_pngs $ori_dir/$target_dir

# clean up tmp environment
rm -rf $tmp_dir