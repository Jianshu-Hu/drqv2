#!/bin/bash
bash scripts/activate_env.sh
source scripts/copy_to_tmp.sh

identifier="b"
tasks=("acrobot_swingup" "reacher_hard" "walker_run")
aug_types=(1 2 3 4 5 6 7 8 9 10)
seeds=(42 43 44 45 46)

for task in "${tasks[@]}"
do
    for aug_type in "${aug_types[@]}"
    do
        oarsub -p "host in ('cerwyn','umber','hornwood', 'manderly', 'snow', 'greyjoy', 'tully', 'eyrie', 'daenerys','martell','targaryen','lannister')" -l host=1/gpuset=1/cpu=1,walltime=168:00:00 "bash scripts/run_single.sh $identifier $task $aug_type ${seeds[@]}"
    done
done