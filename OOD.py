import yaml
from evaluation import saving_files
import argparse 
import utilities
from utilities import to_numpy
import os
import torch 
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

def load_ood(arg, size = 64, dir_skeleton= None):
    if dir_skeleton is None: 
        dir_skeleton = 'set_{:02d}'.format(args.ood_sample)        
    dir_ood = os.path.join('OOD', "OOD_files", dir_skeleton)
    data = np.load(os.path.join(dir_ood, f'data_set{args.ood_sample}_freq{args.freq}.npy'))
    model = np.load(os.path.join(dir_ood, f'model_set{args.ood_sample}.npy'))
    model = torch.tensor(model*1e-3, dtype=torch.float).view(-1,size,size,1)
    data =torch.tensor(data, dtype=torch.float).view(-1, size,size,2)
    print(f'vp= {model.shape}, data={data.shape}')
    return model, data

 
def test_ood(config, args, name =None, dir_skeleton= None, realization_k = 0, x=None, y=None):
    if dir_skeleton is None: 
        dir_skeleton = 'set_{:02d}'.format(args.ood_sample)+f'_freq{args.freq}'
    if name is None:
        name= config["ckpt"]["alias"]
    model = utilities.choosing_model(config)
    if x is None or y is None:
        x, y =load_ood(args)
    
    myloss = utilities.LpLoss(size_average=False)
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()   
    loss_dict = {
                'test_loss_ood': 0.0
                }

    x, y = x.cuda(), y.cuda()
    batch_size, s, s, _ = x.shape
    out = model(x).reshape(batch_size, s, s, -1)
    loss_test = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
    loss_dict['test_loss_ood']+= loss_test.item()/batch_size
    print(f"test test_loss_ood: {loss_dict['test_loss_ood']}")
    if args.save_graph:
        print("generating and saving graph")
        utilities.plotting(in_ = x, NN_out = out, out = y,
                          name = name, database=dir_skeleton, PATH='OOD', ksample=realization_k)
    if args.save_npy:
        print("saving npy files")
        utilities.saving_files(in_files = to_numpy(x), out_files = to_numpy(y), 
                               NN_out_files = to_numpy(out), NN_name = name,
                               database=dir_skeleton, PATH='OOD', realization_k = realization_k)
    
    return loss_dict['test_loss_ood']

if __name__ == '__main__':
    parser = argparse.ArgumentParser('out of distribution check', add_help=False)
    parser.add_argument('-c','--config_file', type=str, 
                                help='Path to the configuration file',
                                default='config/acoustic/GRF_7Hz/FNO25k.yaml')
    parser.add_argument('-ckpt', '--checkpoint', type=str, 
                                help='Path to the checkpoint file',
                                default=None)

    parser.add_argument('-ood','--ood_sample', type=int, 
                                help='out of distribution set',
                                default=0)
    parser.add_argument('-sg','--save-graph', type=bool, 
                                help='Saving Image',
                                default=True)
    parser.add_argument('-snpy','--save-npy', type=bool,
                                help='Saving NPY',  
                                default=True)
    parser.add_argument('-f','--freq', type=int,
                                help='frequency of the OOD',    
                                default=None)
    parser.add_argument('-vmax','--vamax', type=float,
                                help='vmax of the OOD',    
                                default=0.5)
    parser.add_argument('-vmin','--vmin', type=float,
                                help='vmin of the OOD',    
                                default=-0.5)
    parser.add_argument('-s','--shrink', type=float, 
                                help='shrink bar value',
                                default=0.8)

    
    args=parser.parse_args()
    config_file = args.config_file

    assert args.ood_sample in [0,1,2,3,4,5], "out of distribution sample should be in [0,1,2,3,4,5]"
    assert args.freq in [None, 12, 15], "frequency should be in [12,15] Hz"

    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.freq is None:
        args.freq = config['data']['frequency']

    # getting the name of the dataset
     
    dir_skeleton = 'set_{:02d}'.format(args.ood_sample)+f'_freq{args.freq}'

    if args.checkpoint is None:
        c_save = config["ckpt"]
        if config["ckpt"]["alias"]== "sFNO+epsilon_v2": 
            ckpt = "epoch=199-step=166800.ckpt"
        else: 
            ckpt = "epoch=99-step=50000.ckpt"
        x, y =load_ood(args)
        list_test = []
        for k in range(0,3):
            args.checkpoint = os.path.join(c_save["PATH"], 
                                            c_save["save_dir"], 
                                            "lightning_logs",
                                             f"version_{k}",\
                                             "checkpoints", ckpt) 
            list_test.append(test_ood(config,args=args, realization_k = k, x=x, y=y))
            print(list_test)
        saving_files(list_test, database=dir_skeleton, name =config["ckpt"]["alias"], dir_= "OOD")


    print(f"Load from checkpoint {args.checkpoint}")  