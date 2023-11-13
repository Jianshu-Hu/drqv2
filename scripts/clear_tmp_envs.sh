#!/bin/bash
identifier=$1
tmp_dir="/tmp/drqv2-xcc-$identifier"
if [ -d "$tmp_dir" ]; then
    rm -rf $tmp_dir
    echo "Deleted tmp_dir: $tmp_dir"
fi
