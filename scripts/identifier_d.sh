#!/bin/bash
identifier="d"
tasks=("acrobot_swingup" "reacher_hard" "walker_run")
aug_types=(1)
seeds=(1)

for task in "${tasks[@]}"
do
    for aug_type in "${aug_types[@]}"
    do
        for seed in "${seeds[@]}"
        do
            run_task="bash scripts/identifier_d_single.sh $identifier $task $aug_type $seed"
            echo "Run task: $run_task"
            oarsub -p "host in ('cerwyn','umber','hornwood', 'manderly', 'snow', 'greyjoy', 'tully', 'eyrie', 'daenerys','martell','targaryen','lannister')" -l host=1/gpuset=1/cpu=1,walltime=168:00:00 "$run_task"
        done
    done
done