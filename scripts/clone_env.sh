#!/bin/bash
tmp_dir=$1
if [ ! -d "$tmp_dir" ]; then
    mkdir $tmp_dir
    echo "Created new env: $tmp_dir"
    cp -r cfgs $tmp_dir
    cp -r curves $tmp_dir
    cp *.py $tmp_dir
    cp *.yml $tmp_dir
    cp -r scripts $tmp_dir
fi

cd $tmp_dir