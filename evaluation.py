from main import choosing_model
import yaml
import argparse 
import utilities
import os
import torch 
import pytorch_lightning as pl
import numpy as np


def datasetFactoryTest(config):
    c_data =config["data"]
    gl = utilities.GettingLists(data_for_training=c_data["n_sample"],
                                wave_eq = c_data["PDE_type"],
                                data_base = c_data["process"], 
                                PATH = c_data["PATH"])
    return utilities.MyLoader(GL=gl, do = "test", config=config)


def saving_files(data, database, name):
    PATH = "make_graph/test_loss"+'/'+database
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    os.chdir(PATH)
    np.savetxt(f'{name}.csv',data, delimiter=",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Testing losses', add_help=False)
    parser.add_argument('-n', '--numb_samples', type= int, default = 6)
    parser.add_argument('-c','--config_file', type=str, 
                                help='Path to the configuration file',
                                default='config/acoustic/GRF_7Hz/FNO25k.yaml')
    args=parser.parse_args()
    config_file = args.config_file
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    if config["Project"]["name"]== "sFNO+epsilon_v2": 
        file_ckpt = "epoch=199-step=166800.ckpt"
    else: 
        file_ckpt = "epoch=99-step=50000.ckpt"

    print(config) 
    c_save = config["ckpt"]
    model = choosing_model(config)
    test_dataloader = datasetFactoryTest(config)
    myloss = utilities.LpLoss(size_average=False)

    list_test = []
    for k in range(0,args.numb_samples+1):
        PATH = os.path.join(c_save["PATH"], c_save["save_dir"], "lightning_logs", f"version_{k}",\
            "checkpoints", file_ckpt) 
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
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
                loss_dict[f'test_loss']+= loss_test.item()
    

        test_loss = loss_dict['test_loss'] / len(test_dataloader.dataset)
        print(test_loss)
        list_test.append(test_loss)

    saving_files(list_test, database=config["data"]["process"], name =config["ckpt"]["alias"])

