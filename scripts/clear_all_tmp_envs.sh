#!/bin/bash
hosts=("cerwyn" "umber" "hornwood" "manderly" "snow" "greyjoy" "tully" "eyrie" "daenerys" "martell" "targaryen" "lannister")
for host in "${hosts[@]}"
do
    echo "Clearing tmp env(s) on host $host"
    oarsub -p "host in ('$host')" -l host=1/gpuset=1/cpu=1,walltime=168:00:00 "bash scripts/clear_tmp_envs.sh"
done