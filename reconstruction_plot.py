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

def plotting(in_, NN_out, out, name, database,
            k_list =[1,2,3,4], save=False, vmin=-0.5, vmax =0.5, 
            shrink = 0.8):
  in_ =   to_numpy(in_)[k_list,...]       
  NN_out = to_numpy(NN_out)[k_list,...]
  out = to_numpy(out)[k_list,...]
  for k in k_list: 
    if database == 'GRF_7Hz':
        s = 128 
        in_k = in_[k,...].reshape(s,s)
        out_k = out[k,...].reshape(s,s)
        NN_k =NN_out[k,...].reshape(s,s)
        
        plt.figure(figsize=(10,10))
        plt.subplot(131)
        plt.imshow(in_k, vmin=1., vmax =3., cmap = 'jet')
        plt.colorbar(shrink =shrink)
        plt.title(f'wavespeed: {k}')

        plt.subplot(132)
        plt.imshow(out_k, vmin=vmin, vmax =vmax, cmap = 'seismic')
        plt.colorbar(shrink =shrink)
        plt.title(f'HDG sample: {k}')

        plt.subplot(133)
        plt.imshow(NN_k, vmin=vmin, vmax =vmax, cmap = 'seismic')
        plt.colorbar(shrink =shrink)
        plt.title(f'{name} sample: {k}')
    
    elif database ==('GRF_12Hz') or ('GRF_15Hz'):
            s = 64 
            in_k = in_[k,...].reshape(s,s)
            out_k = out[k,...].reshape(s,s,-1)
            NN_k =NN_out[k,...].reshape(s,s,-1)
            
            plt.figure(figsize=(20,10))
            plt.subplot(231)
            plt.imshow(in_k, vmin=1., vmax =5., cmap = 'jet')
            plt.colorbar(shrink =shrink)
            plt.title(f'wavespeed: {k}')

            plt.subplot(232)
            plt.imshow(out_k[:,:,0].reshape(s,s), vmin=vmin, vmax =vmax, cmap = 'seismic')
            plt.colorbar(shrink =shrink)
            plt.title(f'HDG (real) sample: {k}')

            plt.subplot(233)
            plt.imshow(NN_k[:,:,0].reshape(s,s), vmin=vmin, vmax =vmax, cmap = 'seismic')
            plt.colorbar(shrink =shrink)
            plt.title(f'{name} (real) sample: {k}')

            plt.subplot(235)
            plt.imshow(out_k[:,:,1].reshape(s,s), vmin=vmin, vmax =vmax, cmap = 'seismic')
            plt.colorbar(shrink =shrink)
            plt.title(f'HDG (imaginary) sample: {k}')

            plt.subplot(236)
            plt.imshow(NN_k[:,:,1].reshape(s,s), vmin=vmin, vmax =vmax, cmap = 'seismic')
            plt.colorbar(shrink =shrink)
            plt.title(f'{name} (imaginary) sample: {k}')

    if save== True:
        saving_dir = f'make_graph/figures/{database}/'+f'{name}'
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        plt.savefig(f"{saving_dir}/ex_{k}.png")
 

def datasetFactoryTest(config):
    c_data =config["data"]
    gl = utilities.GettingLists(data_for_training=c_data["n_sample"],
                                wave_eq = c_data["PDE_type"],
                                data_base = c_data["process"], 
                                PATH = c_data["PATH"])
    return utilities.MyLoader(GL=gl, do = "test", config=config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plotting from test data', add_help=False)
    parser.add_argument('-c','--config_file', type=str, 
                                help='Path to the configuration file',
                                default='config/acoustic/GRF_7Hz/FNO25k.yaml')
    parser.add_argument('-s','--shrink', type=float, 
                                help='shrink bar value',
                                default=0.8)
    args=parser.parse_args()
    config_file = args.config_file
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    print(config) 
    c_save = config["ckpt"]
    model = choosing_model(config)
    
    if config["Project"]["name"]== "sFNO+epsilon_v2":
        file_ckpt = "epoch=199-step=166800.ckpt"
    else: 
        file_ckpt = "epoch=99-step=50000.ckpt"

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
    with torch.no_grad():
        for x, y in test_dataloader:
            batch_size, s= x.shape[0:2]
            x, y = x.cuda(), y.cuda()
            out = model(x).reshape(batch_size, s, s,-1)
            break
    plotting(
            in_ = x,
            NN_out =out, 
            out= y, 
            name=config["ckpt"]["alias"], 
            database = config["data"]["process"],
            k_list= k_list, 
            save = save,  
            shrink= args.shrink,
            vmin=-0.2, 
            vmax =0.2)