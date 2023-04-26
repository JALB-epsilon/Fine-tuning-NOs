import yaml
import argparse 
import utilities
import os
import torch 
import numpy as np
from main import datasetFactory
import pytorch_lightning as pl


def saving_files(data, database, name, dir_= "make_graph"):
    if len(data) != 1:
        PATH = os.path.join(dir_, "test_loss", database)
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        np.savetxt(os.path.join(PATH, f'{name}.csv'), data, delimiter=",")
    else:
        PATH = os.path.join(dir_, "test_loss", database,f"{name}.csv")
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        #add a new row in th csv file 
        with open(PATH, "a") as f:
            f.write(str(data[0]))


def test(config, args):
    model = utilities.choosing_model(config)
    test_dataloader = datasetFactory(config, do = args.do, args=None)
    myloss = utilities.LpLoss(size_average=False)

    print(f"Load from checkpoint {args.checkpoint}")  
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()   
    loss_dict = {
                'test_loss': 0.0
                }

    with torch.no_grad():
        for x, y in test_dataloader:
            batch_size, s= x.shape[0:2]
            x, y = x.cuda(), y.cuda()
            out = model(x).reshape(batch_size, s, s, -1)
            loss_test = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            loss_dict['test_loss']+= loss_test.item()
    return loss_dict['test_loss'] / len(test_dataloader.dataset)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Testing losses', add_help=False)
    parser.add_argument('-c','--config_file', type=str, 
                                help='Path to the configuration file',
                                default='config/acoustic/GRF_7Hz/FNO25k.yaml')
    parser.add_argument('-do', '--do', type=str,
                                help='do',  
                                default="test")
    parser.add_argument('-n', '--numb_samples', type= int, default = 3)
    parser.add_argument('-ckpt', '--checkpoint', type=str, 
                                help='Path to the checkpoint file',
                                default=None)
    
    args=parser.parse_args()
    config_file = args.config_file
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if config["model"]["activ"] is None:
        activ = "Identity"
    else: 
        activ = config["model"]["activ"]
    database= activ+"_"+config["data"]["process"]
    name= config["ckpt"]["alias"]
    if args.checkpoint is None:
        c_save = config["ckpt"]
        if config["ckpt"]["alias"]== "sFNO+epsilon_v2": 
            ckpt = "epoch=199-step=166800.ckpt"
        else: 
            ckpt = "epoch=99-step=50000.ckpt"
        
        list_test = []
        for k in range(0,args.numb_samples):
            args.checkpoint = os.path.join(c_save["PATH"], 
                                            c_save["save_dir"], "lightning_logs", f"version_{k}",\
                                             "checkpoints", ckpt) 
            list_test.append(test(config,args=args))
        print(list_test)
        saving_files(list_test, database=database, name =name)
    elif args.checkpoint is not None: 
        list_test= test(config,args=args)
        print(list_test)
        saving_files([list_test], database=database, name =name)

