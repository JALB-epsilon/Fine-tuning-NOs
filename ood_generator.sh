#!/bin/bash
# This script generates the OOD data for the given dataset
#create list of config files
list=$(ls config/acoustic/GRF_15Hz/GeLU*.yaml)
#loop through each config file
for c in $list
    do
        for data in {0..5}
            do
            CUDA_VISIBLE_DEVICES=0 python3 OOD.py --config $c --ood_sample $data
            done
    done