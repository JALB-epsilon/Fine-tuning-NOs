from models import FNO, sFNO, sFNO_epsilon_v1, sFNO_epsilon_v2, sFNO_epsilon_v2_proj
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import yaml
import argparse 
import utilities
import os
import shutil
import torch

def datasetFactory(config):
    c_data =config["data"]
    gl = utilities.GettingLists(data_for_training=c_data["n_sample"],
                                wave_eq = c_data["PDE_type"],
                                data_base = c_data["process"], 
                                PATH = c_data["PATH"])
    return utilities.MyLoader(GL=gl, do = "train", config=config)


def choosing_model(config):
    c_nn = config["model"]
    c_train = config["train"]
    # 7 Hz data only contains the real part of the field
    if config["Project"]["database"]=='GRF_7Hz':
        if config["Project"]["name"] == "FNO":         
            model =FNO(
                    wavenumber = c_nn["modes_list"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"]
                    )

        elif config["Project"]["name"] == "sFNO":
            model =sFNO(
                    wavenumber = c_nn["modes_list"],
                    drop = c_nn["drop"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"]
                        )

        
        elif config["Project"]["name"] == "sFNO+epsilon_v1":
            model =sFNO_epsilon_v1(
                    wavenumber = c_nn["modes_list"],
                    drop = c_nn["drop"],
                    drop_path = c_nn["drop_path"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"]
                    )

        elif config["Project"]["name"] == "sFNO+epsilon_v2":
            model =sFNO_epsilon_v2( 
                modes = c_nn["modes_list"],
                drop_path_rate = c_nn["drop_path"],
                drop = c_nn["drop"],
                depths = c_nn["depths"], 
                dims = c_nn["dims"],
                learning_rate = c_train["lr"], 
                step_size= c_train["step_size"],
                gamma= c_train["gamma"],
                weight_decay= c_train["weight_decay"]
                )
    # 12/15 Hz data only contains real and imaginary parto f the field
    elif config["Project"]["database"]==('GRF_12Hz') or ('GRF_15Hz'):               
        
        if config["Project"]["name"] == "FNO":         
            Proj = torch.nn.Linear(c_nn["features"], 2, dtype=torch.float)
            model =FNO(
                    wavenumber = c_nn["modes_list"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"],
                    proj = Proj
                    )

        elif config["Project"]["name"] == "sFNO":
            Proj = torch.nn.Linear(c_nn["features"], 2, dtype=torch.float)
            model =sFNO(
                    wavenumber = c_nn["modes_list"],
                    drop = c_nn["drop"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"],
                    proj = Proj
                    )


        elif config["Project"]["name"] == "sFNO+epsilon_v1":
            Proj = torch.nn.Linear(c_nn["features"], 2, dtype=torch.float)
            model =sFNO_epsilon_v1(
                    wavenumber = c_nn["modes_list"],
                    drop = c_nn["drop"],
                    drop_path = c_nn["drop_path"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"],
                    proj = Proj
                        )

        elif config["Project"]["name"] == "sFNO+epsilon_v2":
            #sFNO_epsilon_v2_proj is the same arch as sFNO_epsilon_v2
            #we just allow to have an independent projection layer.
            Proj = torch.nn.Linear(c_nn["dims"][-1], 2, dtype=torch.float)
            model =sFNO_epsilon_v2_proj( 
                modes = c_nn["modes_list"],
                drop_path_rate = c_nn["drop_path"],
                drop = c_nn["drop"],
                depths = c_nn["depths"], 
                dims = c_nn["dims"],
                learning_rate = c_train["lr"], 
                step_size= c_train["step_size"],
                gamma= c_train["gamma"],
                weight_decay= c_train["weight_decay"],
                proj = Proj
                )

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training of the Architectures', add_help=False)
    parser.add_argument('-c','--config_file', type=str, 
                                help='Path to the configuration file',
                                default='config/acoustic/GRF_7Hz/FNO25k.yaml')
    args=parser.parse_args()
    config_file = args.config_file
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    print(config) 
    model = choosing_model(config)
    print(model)
    save_file = os.path.join(config["ckpt"]["PATH"], 
                            #"all_epochs",
                            config["ckpt"]["save_dir"]
                            )

    if os.path.exists(save_file):
        val = input("The model directory %s exists. Overwrite? (y/n)"%save_file)
        if val == 'y':
            shutil.rmtree(save_file)
    '''checkpoint_callback = ModelCheckpoint(
                            dirpath=save_file,
                            every_n_epochs = 1,
                            save_top_k = 1,
                            filename="{GRF}-{epoch:02d}",
                        )'''
    trainer = pl.Trainer(max_epochs=config["train"]["epochs"], 
                        accelerator='gpu', 
                        devices=1,
                        default_root_dir=save_file) 
                        #callbacks=[checkpoint_callback])
    train_dataloader, val_dataloader = datasetFactory(config)
    trainer.fit(model, train_dataloader, val_dataloader)