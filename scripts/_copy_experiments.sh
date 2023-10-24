#!/bin/bash
tasks=("acrobot_swingup" "reacher_hard" "walker_run")
src="/bigdata/users/ve490-fall23/xcc/drqv2/saved_exps"
for task in "${tasks[@]}"
do
    rm -r $src/c-$task/aug3
    mkdir $src/c-$task/aug3
    folders=($(ls $src/a-$task/aug2))
    for folder in "${folders[@]}"
    do
        rm -r $src/c-$task/aug3/$folder
        mkdir $src/c-$task/aug3/$folder
        cp $src/a-$task/aug2/$folder/eval.csv $src/c-$task/aug3/$folder
    done
done