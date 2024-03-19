# Fine-tuning Neural-Operator Architectures for Training and Generalization

This is the official implementation of the paper 


## Paper 

Benitez, J., Furuya, T., Faucher, F., Kratsios, A., Tricoche, X., & de Hoop, M. V. (2023). Out-of-distributional risk bounds for neural operators with applications to the Helmholtz equation. arXiv preprint arXiv:2301.11509.

> **Source:** [arXiv preprint arXiv:2301.11509](https://arxiv.org/abs/2301.11509)

## Get Started (Libraries)

To reproduce all the results, including the baselines, shown in the paper, follow these steps:

1. Set up a conda environment with all dependencies using the provided `environment.yaml` file:
    ```bash
    conda env create -f environment.yaml
    conda activate forward-operator
    ```
2. Proceed with running the code or experiments as described in the paper.

## Dataset 
The data set is presented here [Official data set](https://rice.app.box.com/s/haczq8oad4b5cvi8pf8cp01sz4f0vfey)  It must be located in a directory with the following structure:
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

### Loss Landscape visualization
```
visualization_code
```

## Remarks about architecture
We updated the sFNO+eps v2 to include layer LayerScale as mentioned in [@touvron2021going]

If we remove LayerScale sFNO+eps v2 and sFNO+eps **long** v1 are equivalent. 
