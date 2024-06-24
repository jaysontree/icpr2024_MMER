#!/bin/bash

save_dir=./run/$1   

# save_code=$save_dir/code
# if [ ! -d $save_code  ];then
#   mkdir -p $save_dir
#   mkdir -p $save_code
#   echo mkdir $save_code
# else
#   echo dir exist
# fi

# cp ./*.py $save_code
# cp ./*.txt $save_code

# nohup python -u train.py $1 1> $save_dir/A_log.txt 2>&1 &

nohup python -m torch.distributed.launch --nproc_per_node 4 train.py $1 1> $save_dir/A_log.txt 2>&1 &