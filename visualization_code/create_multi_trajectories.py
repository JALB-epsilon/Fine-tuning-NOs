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
import re
import copy

'''
Code adapted from Tom Goldstein's implementation of the 2018 NeurIPS paper:

Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein.
Visualizing the Loss Landscape of Neural Nets. NIPS, 2018.

Github: https://github.com/tomgoldstein/loss-landscape

Given a series of models corresponding to learning steps or multiple 
training runs, compute the PCA of the models' weights, treated as giant
parameter vectors. The first few principal components, along with the 
final models, can be used to create a 2D reference frame upon 
which the steps can be projected.

Note: this is the multi-run extension of the method proposed in the 2018
NeurIPS paper cited above.
'''

import yaml

from projection import setup_PCA_directions, project_trajectory, tensorlist_to_tensor
from scatterplotmatrix import scatterplot_matrix as splom
sys.path.append('../')
import utilities

sys.path.append('../../')
sys.path.append('../../loss_landscape')
import loss_landscape
from loss_landscape import net_plotter, plot_2D, h5_util

def normalized_distance(model1, model2, ref_model, verbose=False):
    s1 = model1.state_dict()
    s2 = model2.state_dict()
    s = ref_model.state_dict()
    delta = net_plotter.get_diff_states(s1, s2)
    net_plotter.normalize_directions_for_states(delta, s, norm='weight')
    delta = tensorlist_to_tensor(delta)
    return torch.linalg.norm(delta)

def distance(model1, model2, verbose=False, norm='l2'):
    s1 = model1.state_dict()
    s2 = model2.state_dict()
    print('state dict 1 =\n')
    if verbose:
        for k in s1.keys():
            n1 = torch.linalg.norm(s1[k])
            n2 = torch.linalg.norm(s2[k])
            d = torch.linalg.norm(s1[k]-s2[k])
            nn1 = d/n1
            nn2 = d/n2
            nn3 = d/(n1+n2)
            nn4 = 2*d/(n1+n2)
            print(f'tensor {k} with size {s1[k].size()} has norm {n1} in model 1, norm {n2} in model 2, difference value {d}, normalized {nn1}, {nn2}, {nn3}, {nn4}')
    delta = net_plotter.get_diff_states(s1, s2)
    if verbose:
        for t in delta:
            n = torch.linalg.norm(t)
            print(f'difference in tensor has norm {n}')
    delta = tensorlist_to_tensor(delta)
    dist = torch.linalg.norm(delta)
    if verbose:
        print('overall norm of difference: {}'.format(dist))
    return dist


def evaluate(model, dataloader, loss, train=False, cuda=False, verbose=False, rank=0):
    the_loss = 0.
    print(f'dataloader={dataloader}')
    if train:
        lossname = 'training'
    else:
        lossname = 'testing'
    with torch.no_grad():
        '''
        pbar = tqdm.tqdm(dataloader, ncols=100, desc=f'Computing {lossname} loss (rank={rank})')
        print(f'pbar[0]={type(list(pbar)[0])}')
        for x, y in pbar:
        '''
        for stuff in dataloader:
            print(f'len(stuff)={len(stuff)}')
            print('stuff contains: {}'.format([type(a) for a in stuff]))
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
    print('config={}'.format(config))
    return os.path.join(checkpath, config['ckpt']['save_dir'])

def step_to_filename(checkpath, config, basename, step):
    return os.path.join(to_path(checkpath, config), basename.format(step))

def collect_filenames(path, template):
    p = re.compile(template)
    p1 = re.compile('epoch=(\d*)')
    p2 = re.compile('loss=(\d\.\d*)')
    names = os.listdir(path)
    files = []
    for name in names:
        match = re.search(p, name)
        if match is not None:
            epoch_match = re.search(p1, name)
            loss_match = re.search(p2, name)
            if epoch_match is not None and loss_match is not None:
                epoch = epoch_match.group(1)
                loss = loss_match.group(1)
                print(f'file {name} has epoch {epoch} and loss {loss}')
                files.append({'name': os.path.join(path, name), 'epoch': epoch, 'loss': loss})
            else:
                print('unrecognized checkpoint filename: {}'.format(name))
    print('before sorting files=\n{}'.format(files))
    files.sort(key=lambda val: int(val['epoch']))
    print('after sorting files=\n{}'.format(files))
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot optimization trajectory',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c','--config_file', type=str, required=True,
                        help='Path to the model configuration file')
    parser.add_argument('-p', '--path', type=str, required=True, action='append', help='Path to checkpoint files')
    parser.add_argument('--dir_type', default='weights',
        help="""direction type: weights (all weights except bias and BN paras) |
                                states (include BN.running_mean/var)""")
    parser.add_argument('--ignore', action='store_true', help='ignore bias and BN paras: biasbn (no bias or bn)')
    parser.add_argument('--complex', type=str, default='split', help='Method to handle imaginary part of complex weights (split/both, ignore/real, keep/same, imaginary)')
    parser.add_argument('--filename', type=str, help='Regex filename for checkpoint modes_list')
    # parser.add_argument('--steps', type=int, nargs='+', help='list of all available step ids')
    parser.add_argument('--dir_file', help='load/save the direction file for projection')
    parser.add_argument('--output_path', type=str, help='Directory in which to export direction and projection files')
    parser.add_argument('--proj_method', type=str, default='cos', help='Projection method onto PCA coordinates')
    parser.add_argument('--stride', type=int, default=1, help='Stride across checkpoints to construct low-dimensional coordinates')
    parser.add_argument('--dimension', type=int, default=2, help='Spatial dimensions in which to draw curve')
    parser.add_argument('--debug', action='store_true', help='Run verification code for PCA projection forward and backward')
    parser.add_argument('--verbose', action='store_true', help='Select verbose output')

    args = parser.parse_args()
    config_file = args.config_file
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    c_save = config["ckpt"]

    #--------------------------------------------------------------------------
    # Acquire all filenames to be used to determine low dimensional embedding
    #--------------------------------------------------------------------------
    all_trajectories = []
    all_files = []
    for i, apath in enumerate(args.path):
        file_info = collect_filenames(apath, args.filename)
        curve_name = os.path.normpath(apath).split(os.sep)[-1]
        for f in file_info:
            f['trajectory ID'] = i
            f['curve name'] = curve_name
            all_files.append(f)
        all_trajectories.append(file_info)
    
    ntraj = len(all_trajectories)
    if ntraj > 1:
        # --------------------------------------------------------------------------
        # Select barycenter of converged models as origin for PCA dim reduction
        # --------------------------------------------------------------------------
        all_trained_models = []
        trained_names = []
        seed_names = []
        all_state_dicts = []
        all_init_state_dicts = []
        for trajectory in all_trajectories:
            filename = trajectory[-1]['name']
            other_filename = trajectory[0]['name']
            seed_names.append(other_filename)
            trained_names.append(filename)
            ckpt = torch.load(filename, map_location=lambda storage, loc: storage)
            ckpt2 = torch.load(other_filename, map_location=lambda storage, loc: storage)
            all_state_dicts.append(ckpt['state_dict'])
            all_init_state_dicts.append(ckpt2['state_dict'])
        ntraj = len(all_trajectories)
        nckpts = np.sum([len(traj) for traj in all_trajectories])

        # compute distance between seed point and final points across all curves
        for i in range(ntraj-1):
            init_model = utilities.choosing_model(config)
            init_model.load_state_dict(all_init_state_dicts[i])
            init_model.eval()
            train_model = utilities.choosing_model(config)
            train_model.load_state_dict(all_state_dicts[i])
            train_model.eval()
            for j in range(i+1, ntraj):
                another_init_model = utilities.choosing_model(config)
                another_init_model.load_state_dict(all_init_state_dicts[j])
                another_init_model.eval()
                another_train_model = utilities.choosing_model(config)
                another_train_model.load_state_dict(all_state_dicts[j])
                another_train_model.eval()
                dij_init = distance(init_model, another_init_model)
                dij_init_normalized = normalized_distance(init_model, another_init_model, train_model)
                dij_train = distance(train_model, another_train_model)
                dij_train_normalized = normalized_distance(train_model, another_train_model, train_model)
                di_init_train = distance(init_model, train_model)
                dj_init_train = distance(another_init_model, another_train_model)
                print(f'traj {i}.init.name = {seed_names[i]}')
                print(f'traj {j}.init.name = {seed_names[j]}')
                print(f'traj {i}.train.name = {trained_names[i]}')
                print(f'traj {j}.train.name = {trained_names[j]}')
                print(f'd[{i},{j}].init={dij_init} / normalized = {dij_init_normalized}')
                print(f'd[{i},{j}].train = {dij_train} / normalized = {dij_train_normalized}')
                print(f'd[{i}].init_to_train = {di_init_train}')
                print(f'd[{j}].init_train = {dj_init_train}')

        # compute average
        reference_state_dict = copy.deepcopy(all_state_dicts[0])
        all_keys = all_state_dicts[0].keys()
        print(f'all keys are: {all_keys}')
        for state_dict in all_state_dicts[1:]:
            for key in all_keys:
                reference_state_dict[key] += state_dict[key]
        for key in all_keys:
            reference_state_dict[key] /= float(ntraj)
            for i in range(ntraj):
                d = torch.linalg.norm(reference_state_dict[key]-all_state_dicts[i][key])
                print(f'model[{i}][{key}] is at distance {d} from average')
        reference_weights = [ reference_state_dict[k] for k in reference_state_dict.keys() ]
    elif ntraj==1:
        reference_model = utilities.choosing_model(config)
        trajectory = all_trajectories[0]
        nckpts = len(trajectory)
        ntraj = 1
        filename = trajectory[-1]['name']
        ckpt = torch.load(filename, map_location=lambda storage, loc: storage)
        reference_state_dict = ckpt['state_dict']
        # reference_model.load_state_dict(ckpt['state_dict'])
        # reference_model.eval()
        reference_weights = [ reference_state_dict[k] for k in reference_state_dict.keys() ]
    else:
        print('ERROR: no trajectory file in input')
        sys.exit(1)

    def callback(stepid):
        step = all_files[stepid] 
        name = step['name']
        print(f'loading step {name}')
        amodel = utilities.choosing_model(config)
        checkpoint = torch.load(name, map_location=lambda storage, loc: storage)
        amodel.load_state_dict(checkpoint['state_dict'])
        # model2.cuda()
        amodel.eval()
        return amodel

    #--------------------------------------------------------------------------
    # load or create projection directions
    #--------------------------------------------------------------------------
    if args.output_path is None:
        print('automatically selecting direction output path from config file')
        args.path = to_path(args.path[0], config)
    else:
        args.path = args.output_path
    print(f'path argument is {args.path}')
    args.steps = np.arange(0, len(all_files), args.stride, dtype=int) #list(range(len(all_files))) 
    if args.steps[-1] != len(all_files)-1: 
        np.append(args.steps, len(all_files)-1)
    if not args.dir_file:
        print('computing PCA directions for {} models'.format(nckpts))
        if ntraj == 1:
            fname = all_files[0]['curve name']
        else:
            fname = None           
        args.dir_file = setup_PCA_directions(args, callback, reference_weights, reference_state_dict, verbose=args.verbose, filename=fname)

    print(f'dir_file={args.dir_file}')

    #--------------------------------------------------------------------------
    # projection trajectory to given directions
    #--------------------------------------------------------------------------
    total_steps = 0
    for traj in all_trajectories:
        curve_name = traj[0]['curve name']
        args.steps = list(range(total_steps, total_steps+len(traj)))
        total_steps += len(traj)
        print(f'curve {curve_name} will be saved in {args.path}')
        proj_file = project_trajectory(args, reference_weights, reference_state_dict, callback, model_name=curve_name)

