#!/bin/bash
tmp_dir="/tmp/drqv2-xcc"
if [ -d "$tmp_dir" ]; then
    rm -rf $tmp_dir
    echo "Deleted tmp_dir: $tmp_dir"
fi
