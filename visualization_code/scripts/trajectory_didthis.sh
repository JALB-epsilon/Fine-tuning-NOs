#!/bin/bash

config=$1
path_to_data=$2

python create_trajectory.py -p ${path_to_data} --complex split --filename "GRF=0-epoch={:0>2d}.ckpt" --dimension 3 -c ${config} --steps `for ((i=0; i<100; i=i+1)); do echo $i; done`
