#!/bin/bash

name=$1

python combine_files.py -p ../databases/acoustic/GRF_7Hz/${name} --x -90 20 45 --y -50 70 49 --trajectory ../databases/acoustic/GRF_7Hz/${name}/PCA_weights_save_epoch\=99_complex\=split_dim\=2/directions.h5_proj_cos.h5 --output ${name}_landscape.vtp --show-edges
python view_surface.py -s ${name}_landscape.vtp -t ${name}_landscape_trajectory.vtp --show_isovalues --traj_diameter 0.5 --color_surface --font_size 20 --show_steps --size 1920 1080 --trajectory_palette Oranges
