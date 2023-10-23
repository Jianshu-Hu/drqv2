#!/bin/bash
target_dir_xcc="/tmp/drqv2-xcc"
mkdir $target_dir_xcc
cp *.py $target_dir_xcc
cp -r scripts $target_dir_xcc
cp -r cfgs $target_dir_xcc
cd $target_dir_xcc