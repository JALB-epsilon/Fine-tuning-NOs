import numpy as np
import torch
import copy
import math
import h5py
import os
import argparse
import sys
import json
import tqdm

'''
Code adapted from Tom Goldstein's implementation of the 2018 NeurIPS paper:

Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein.
Visualizing the Loss Landscape of Neural Nets. NIPS, 2018.

Github: https://github.com/tomgoldstein/loss-landscape

Given a series of models corresponding to learning steps, compute the PCA of
the models' weights, treated as giant parameter vectors. The first few
principal components, along with the final model, can be used to create a 2D
reference frame upon which the steps can be projected.
'''


sys.path.append('../')

from main import choosing_model, datasetFactory
import yaml
import utilities

from loss_landscape import net_plotter, plot_2D

from projection import setup_PCA_directions, project_trajectory
from scatterplotmatrix import scatterplot_matrix as splom

def evaluate(model, dataloader, loss, train=False, cuda=False, verbose=False):
    the_loss = 0.
    if train:
        lossname = 'training'
    else:
        lossname = 'testing'
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, ncols=100, desc=f'Computing {lossname} loss')
        for x, y in pbar:
            batch_size, s= x.shape[0:2]
            if cuda:
                x, y = x.cuda(), y.cuda()
            out = model(x).reshape(batch_size, s, s)
            loss_test = loss(out.view(batch_size,-1), y.view(batch_size,-1))
            the_loss += loss_test.item()

    the_loss = the_loss / len(dataloader.dataset)
    if verbose:
        print(f'loss = {the_loss}')
    return the_loss, 0

def to_path(checkpath, config):
    return os.path.join(checkpath, config['ckpt']['save_dir'])

def step_to_filename(checkpath, config, basename, step):
    return os.path.join(to_path(checkpath, config), basename.format(step))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot optimization trajectory',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c','--config_file', type=str, required=True,
                        help='Path to the model configuration file')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to checkpoint files')
    parser.add_argument('--dir_type', default='weights',
        help="""direction type: weights (all weights except bias and BN paras) |
                                states (include BN.running_mean/var)""")
    parser.add_argument('--ignore', action='store_true', help='ignore bias and BN paras: biasbn (no bias or bn)')
    parser.add_argument('--complex', type=str, default='split', help='Method to handle imaginary part of complex weights (split/both, ignore/real, keep/same, imaginary)')
    parser.add_argument('--filename', help='Regex filename for checkpoint modes_list')
    parser.add_argument('--steps', type=int, nargs='+', help='list of all available step ids')
    parser.add_argument('--dir_file', help='load/save the direction file for projection')
    parser.add_argument('--proj_method', type=str, default='cos', help='Projection method onto PCA coordinates')
    parser.add_argument('--dimension', type=int, default=2, help='Spatial dimensions in which to draw curve')
    parser.add_argument('--debug', action='store_true', help='Run verification code for PCA projection forward and backward')
    parser.add_argument('--verbose', action='store_true', help='Select verbose output')

    args = parser.parse_args()
    config_file = args.config_file
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    c_save = config["ckpt"]
    model = choosing_model(config)
    test_dataloader = datasetFactory(config, train=False, prefix='../')
    myloss = utilities.LpLoss(size_average=False)

    #--------------------------------------------------------------------------
    # load the final model
    #--------------------------------------------------------------------------
    last_id = args.steps[-1]
    final_model = step_to_filename(args.path, config, args.filename, last_id)

    checkpoint = torch.load(final_model, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    w = net_plotter.get_weights(model)
    s = model.state_dict()

    #--------------------------------------------------------------------------
    # collect models to be projected
    #--------------------------------------------------------------------------
    model_files = {}
    for epoch in args.steps:
        model_file = step_to_filename(args.path, config, args.filename, epoch)
        if not os.path.exists(model_file):
            print('model %s does not exist' % model_file)
            exit(-1)
        else:
            model_files[epoch] = model_file

    def callback(step):
        name = model_files[step]
        model2 = choosing_model(config)
        checkpoint = torch.load(name, map_location=lambda storage, loc: storage)
        model2.load_state_dict(checkpoint['state_dict'])
        # model2.cuda()
        model2.eval()
        return model2

    #--------------------------------------------------------------------------
    # load or create projection directions
    #--------------------------------------------------------------------------
    args.path = to_path(args.path, config)
    if not args.dir_file:
        print('computing PCA directions for {} models'.format(len(args.steps)))
        args.dir_file = setup_PCA_directions(args, callback, w, s, verbose=args.verbose)

    print(f'dir_file={args.dir_file}')

    #--------------------------------------------------------------------------
    # projection trajectory to given directions
    #--------------------------------------------------------------------------
    proj_file = project_trajectory(args, w, s, callback)
