"""
    Calculate the loss surface in parallel.

    Code adapted from Tom Goldstein's implementation of the 2018 NeurIPS paper:

    Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein.
    Visualizing the Loss Landscape of Neural Nets. NIPS, 2018.

    Github: https://github.com/tomgoldstein/loss-landscape

    Given PCA directions, the code samples the loss associated with models
    whose weights lie in the corresponding two-dimensional weight
    parameterization plane.
"""
import numpy as np
import torch
import copy
import math
import h5py
import os
import argparse
import sys
import json
import csv
import yaml

from projection import setup_PCA_directions, project_trajectory, tensorlist_to_tensor, pca_coords_to_weights
from scatterplotmatrix import scatterplot_matrix as splom
sys.path.append('../')
import utilities
from main import datasetFactory

sys.path.append('../../')
sys.path.append('../../loss_landscape')
import loss_landscape
from loss_landscape import net_plotter, plot_2D, h5_util

from create_multi_trajectories import evaluate

import plot_2D
import time
import socket
import sys
import numpy as np
import torchvision
import torch.nn as nn
import tqdm

import dataloader
import evaluation
import projection as proj
from projection import shapeof, sizeof
# import plotter_helper as plotter
import plot_2D
import plot_1D
import model_loader
import scheduler
import mpi4pytorch as mpi

def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the prefix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    return surf_file + ".h5"

def load_these_directions(dir_file, dir_names):
    """ Load direction(s) from the direction file."""

    print('This direction loader can import more than 2 dimensions')

    directions = []
    f = h5py.File(dir_file, 'r')
    for name in dir_names:
        if name in f.keys():
            directions.append(h5_util.read_list(f, name))
        else:
            break

    print(f'directions contain {len(directions)} vectors')
    return directions


def setup_surface_file(args, surf_file, dir_file):
    print('-------------------------------------------------------------------')
    print('setup_surface_file')
    print('-------------------------------------------------------------------')

    print('surf_file is {}'.format(surf_file))
    print('dir_file is {}'.format(dir_file))

    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, 'a' if os.path.exists(surf_file) else 'w')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.ynum))
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file

def to_path(checkpath, config):
    return os.path.join(checkpath, config['ckpt']['save_dir'])

def to_filename(checkpath, config, filename):
    return os.path.join(to_path(checkpath, config), filename)

def crunch(surf_file, net, w, s, d, loss_key, comm, rank, args, samples, loss_func):
    """
        Calculate the loss values of modified models in parallel
        using MPI. Each individual rank saves its results in a separate
        csv file that can then be consolidated into a surface geometry using
        'combine.py'
    """

    coords = samples[rank]

    print('Computing %d values for rank %d'% (len(coords), rank))
    start_time = time.time()
    total_sync = 0.0

    fname = surf_file + f'_rank={rank}.csv'
    if os.path.exists(fname):
        print(f'Creating new filename since {fname} already exists')
        tstr = time.asctime(time.gmtime(time.time())).replace(' ', '_')
        fname = surf_file + f'{tstr}_rank={rank}.csv'

    # Note: the CSV file cannot stay open otherwise changes will only be
    # recorded upon completion of the loop. Given the odds that a MPI
    # job is cut short on an HPC architecture, we elect instead to save
    # each loss value in a csv file as soon as it is computed.
    with open(fname, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['x', 'y', 'loss', 'time'])
        writer.writeheader()

    # Loop over all uncalculated loss values
    pbar = tqdm.tqdm(coords, total=len(coords), ncols=100, desc=f'Sampling loss surface for rank {rank}')
    for c in pbar:

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            #net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, c)
            net = pca_coords_to_weights(c, d, w, what='split') # what = how to handle complex coefficients
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, c)

        # Record the time to compute the loss value
        loss_start = time.time()
        loss, mse = loss_func(net, rank)
        loss_compute_time = time.time() - loss_start
        with open(fname, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['x', 'y', 'loss', 'time'])
            writer.writerow({'x': c[0], 'y': c[1], 'loss': loss, 'time': loss_compute_time})
    total_time = time.time() - start_time
    print(f'Rank {rank} done!  Total time: {total_time}')

###############################################################
#                          MAIN
###############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')

    # data parameters

    # model parameters
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to checkpoint files')
    parser.add_argument('-c','--config_file', type=str, required=True,
                        help='Path to the model configuration file')
    parser.add_argument('--filename', help='Filename of final model')
    parser.add_argument('--model_name', default='dummy model', help='model name')
    parser.add_argument('--loss_name', '-l', default='mse', help='loss functions: crossentropy | mse')
    parser.add_argument('--skip', default=None, type=int, help='Were to resume computation on all ranks')
    parser.add_argument('--samples', default=None, type=str, help='File containing the explicit list of surface locations to sample')

    # direction parameters
    parser.add_argument('--dir_file', required=True, help='specify the name of direction file, or the path to an existing direction file')
    parser.add_argument('--dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', type=float, nargs=3, default='-1 1 51', help='xmin xmax xnum')
    parser.add_argument('--y', type=float, nargs=3, default='-1 1 51', help='ymin ymax ynum')
    parser.add_argument('--testing', action='store_true', help='Sample testing loss (default: training loss)')
    parser.add_argument('--xnorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')

    args = parser.parse_args()

    args.raw_data = False
    args.data_split = 0

    # reproducibility is already available by default in data setup
    # torch.manual_seed(10)
    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------
    torch.set_num_threads(4)
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        device = torch.device('cuda')
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' %
              (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))
    else:
        device = torch.device('cpu')

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    try:
        if args.x is not None and args.y is not None:
            args.xmin, args.xmax, args.xnum = args.x
            args.xnum = int(args.xnum)
            args.ymin, args.ymax, args.ynum = args.y # (None, None, None)
            args.ynum = int(args.ynum)
            print(f'Surface sampling bounds: [{args.xmin}, {args.xmax}] x [{args.ymin}, {args.ymax}]')
            print(f'Sampling density: {args.xnum} x {args.ynum} = {args.xnum *args.ynum} samples')
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1 1 51') #-1:1:51')

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    data = None
    if args.testing:
        loss_label = 'test_loss'
    else:
        loss_label = 'train_loss'

    config_file = args.config_file
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    c_save = config["ckpt"]
    args.path = os.path.join(args.path, config['ckpt']['save_dir'])
    model = utilities.choosing_model(config)
    if args.testing:
        dataloader = datasetFactory(config, do = "test", args=None)
    else:
        dataloader, _ = datasetFactory(config, do = "train", args=None)
    myloss = utilities.LpLoss(size_average=False)

    #--------------------------------------------------------------------------
    # load the final model
    #--------------------------------------------------------------------------
    final_model = os.path.join(args.path, args.filename)

    checkpoint = torch.load(final_model, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    if args.cuda:
        model.cuda()
    model.eval()

    w = net_plotter.get_weights(model) # initial parameters
    s = copy.deepcopy(model.state_dict()) # deepcopy since state_dict are references
    if args.cuda and args.ngpu > 1:
        # data parallel with multiple GPUs on a single node
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    if args.samples is not None:
         print('Importing sampling locations from file')
         with open(args.samples, 'r') as fp:
             coords = json.load(fp)
         #print(f'{len(coords)} coordinates will be sampled')
         #print(f'type(coords) is {type(coords)}')
         #print(f'type(coords[0]) is {type(coords[0])}')
         #print(f'type(coords[0][0]) is {type(coords[0][0])}')
    elif args.x is not None and args.y is not None:
         coords = [ (x, y) for x in np.linspace(args.xmin, args.xmax, args.xnum) for y in np.linspace(args.ymin, args.ymax, args.ynum)]
         #print(f'type(coords) is {type(coords)}')
         #print(f'type(coords[0]) is {type(coords[0])}')
         #print(f'type(coords[0][0]) is {type(coords[0][0])}')
    else:
        raise ValueError('Missing information to determine sampling locations')

    n_per_rank = len(coords) // nproc
    rem = len(coords) - n_per_rank*nproc
    n_per_rank_0 = n_per_rank + rem
    print(f'each rank will sample {n_per_rank} positions')


    print('Assigning samples to ranks')
    counter = 0
    samples = [ [] for i in range(nproc) ]
    samples[0] = coords[0:n_per_rank_0]
    for r in range(1,nproc):
        samples[r] = coords[n_per_rank_0 + (r-1)*n_per_rank : n_per_rank_0 + r*n_per_rank]
    print('done')

    if args.samples is None and args.skip is not None:
         for r in range(0, nproc):
             samples[r] = samples[r][args.skip:]

    # for i in range(n_per_rank+n_per_rank_extra):
    #     samples[0].append(coords[i])
    #     counter += 1
    # for r in range(1, nproc):
    #     for i in range(n_per_rank):
    #         samples[r].append(coords[counter])
    #         counter += 1

    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    dir_file = os.path.join(args.path, args.dir_file)

    if not args.surf_file:
        args.surf_file = f'{args.dir_file}_surface_[{args.xmin}-{args.xmax}]x[{args.ymin}-{args.ymax}]_{args.xnum}x{args.ynum}.h5'
    surf_file = os.path.join(args.path, args.surf_file)
    #if rank == 0:
    #    setup_surface_file(args, surf_file, dir_file)

    # wait until master has setup the direction file and surface file
    #mpi.barrier(comm)

    # load directions
    directions = load_these_directions(dir_file, ['direction_0', 'direction_1'])
    print(f'type(directions) is {type(directions)}')
    print(f'type(directions[0]) is {type(directions[0])}')
    print(f'type(directions[0][0]) is {type(directions[0][0])}')
    # calculate the cosine similarity of the two directions
    if False and len(directions) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(directions[0]), proj.nplist_to_tensor(directions[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    class loss_callback:
        def __init__(self, dataset, loss, train, cuda):
            self.dataset = dataset
            self.loss = loss
            self.train = train
            self.cuda = cuda
        def __call__(self, model, rank=0, verbose=False):
            return evaluate(model, self.dataset, self.loss, train=self.train, verbose=verbose, cuda=self.cuda, rank=rank)

    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    crunch(surf_file, model, w, s, directions, loss_label, comm, rank, args, samples, loss_func=loss_callback(dataloader, myloss, not args.testing, args.cuda))

    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    if args.plot and rank == 0:
        if args.y and args.proj_file:
            plot_2D.plot_contour_trajectory(surf_file, dir_file,
                args.proj_file, loss_label, vmin=args.vmin, vmax=args.vmax,
                vlevel=args.vlevel, show=args.show)
        elif args.y:
            plot_2D.plot_2d_contour(surf_file, loss_label, args.vmin,
                args.vmax, args.vlevel, args.show)
        else:
            plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax,
                args.loss_max, args.log, args.show)
