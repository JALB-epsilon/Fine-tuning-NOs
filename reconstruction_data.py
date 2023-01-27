from main import choosing_model
import yaml
import argparse 
import utilities
import os
import torch 
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from utilities import to_numpy

def saving_files(x, y, out, database, name):
    PATH = "make_graph/data"+'/'+database+'/'+name
    x = to_numpy(x)
    y = to_numpy(y)
    out = to_numpy(out)    
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    os.chdir(PATH)
    np.save('wavespeed.npy', x)
    np.save('data.npy', y)
    np.save(f'data_{name}.npy', out)
        


 

def datasetFactoryTest(config):
    c_data =config["data"]
    gl = utilities.GettingLists(data_for_training=c_data["n_sample"],
                                wave_eq = c_data["PDE_type"],
                                data_base = c_data["process"], 
                                PATH = c_data["PATH"])
    return utilities.MyLoader(GL=gl, do = "test", config=config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Getting data from the test set', add_help=False)
    parser.add_argument('-c','--config_file', type=str, 
                                help='Path to the configuration file',
                                default='config/acoustic/GRF_7Hz/FNO25k.yaml')
    args=parser.parse_args()
    config_file = args.config_file
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    print(config) 

    if config["Project"]["name"]== "sFNO+epsilon_v2":
        file_ckpt = "epoch=199-step=166800.ckpt"
    else: 
        file_ckpt = "epoch=99-step=50000.ckpt"

    c_save = config["ckpt"]
    model = choosing_model(config)
    test_dataloader = datasetFactoryTest(config)
    myloss = utilities.LpLoss(size_average=False)
    PATH = os.path.join(c_save["PATH"], c_save["save_dir"], "lightning_logs", f"version_{0}",\
            "checkpoints", file_ckpt)


    checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()   
    model.eval()   
    k_list =  [k for k in range(10)]
    save = True
    batch_size = 20
    with torch.no_grad():
        for x, y in test_dataloader:
            
            s= x.shape[2]
            x, y = (x[:batch_size,...]).cuda(), (y[:batch_size,...]).cuda()
            out = model(x).reshape(batch_size, s, s, -1)
            break
    saving_files(x, y, out, database=config["data"]["process"], name =config["ckpt"]["alias"])
