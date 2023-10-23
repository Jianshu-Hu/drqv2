#!/bin/bash
target_dir="/tmp/drqv2-xcc"
mkdir $target_dir
cp *.py $target_dir
cp -r scripts $target_dir
cp -r cfgs $target_dir
cd $target_dir