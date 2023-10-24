#!/bin/bash
identifier="c"
tasks=("acrobot_swingup" "reacher_hard")
aug_types=(2 1)
seeds=(1 2 3 4 5)

for seed in "${seeds[@]}"
do
    run_taska="bash scripts/run_single.sh $identifier acrobot_swingup 2 $seed"
    echo "Run task: $run_task"
    oarsub -p "host in ('cerwyn','umber','hornwood', 'manderly', 'snow', 'greyjoy', 'tully', 'eyrie', 'daenerys','martell','targaryen','lannister')" -l host=1/gpuset=1/cpu=1,walltime=168:00:00 "$run_taska"

    run_taskb="bash scripts/run_single.sh $identifier reacher_hard 1 $seed"
    echo "Run task: $run_task"
    oarsub -p "host in ('cerwyn','umber','hornwood', 'manderly', 'snow', 'greyjoy', 'tully', 'eyrie', 'daenerys','martell','targaryen','lannister')" -l host=1/gpuset=1/cpu=1,walltime=168:00:00 "$run_taskb"
done
