#!/bin/bash
start_id=$1
end_id=$2
for((i=$start_id;i<=$end_id;i++)); do
    echo "Deleting task $i"
    oardel $i
done