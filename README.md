# Fine-tuning Neural-Operator architectures for training and generalization.
This is the official implementation of the paper "Fine-tuning Neural-Operator architectures for training and generalization".

## Get started (Libraries)
If you want to reproduce all the results (including the baselines) shown in the paper,

You can then set up a conda environment with all dependencies like so:
```
conda env create -f environment.yaml
conda activate forward-operator
```

## Dataset 
The data set is proportioned upon request. It must be located in a directory with the following structure:
```
databases/acoustic/GRF_{Freq}Hz/data
databases/acoustic/GRF_{Freq}Hz/model
```
In the place holder is $7, 12, 15$ Hz the Frequencies of the experiments. The configuration for the experiments are located in 
```
dataset_time-harmonic-waves_hawen_parameters
```
## Configuration of architectures
All the architectures' parameters are located in */config* directory.
```
config/acoustic/GRF_{Freq}Hz/<Architecture>.yaml
```

## Traning

They can be trained with
```
CUDA_VISIBLE_DEVICES={k} python3 main.py -c <path_to_config_file>
```

### Evaluation
We train multiple times in the code. The evaluation is a function of the amoung of saving files. It can be implemented as follows

```
CUDA_VISIBLE_DEVICES={k} python3 evaluation.py -n <number_training_save_model> -c <path_to_config_file>
```

### Plot (no visualization)
The same structure follows for plotting

```
 python3 reconstruction_plot.py -c <path_to_config_file>
```

### Loss Landscape visualization
```
visualization_code
```



