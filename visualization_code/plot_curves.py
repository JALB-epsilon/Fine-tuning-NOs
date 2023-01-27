from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib import cm
import h5py
import argparse
import numpy as np
from os.path import exists
import seaborn as sns
import yaml
import os
from scatterplotmatrix import scatterplot_matrix as splom

def plot_trajectories(traj_files, image_name='somecurves.png', show=False):
    """ Plot optimization trajectory on the plane spanned by given directions."""

    curves = []

    print(f'There are {len(traj_files)} filenames in input')

    # all3d = True
    for traj in traj_files:
        fname = traj['filename']
        label = traj['label']
        complex = traj['complex_handling']
        print(f'filename = {fname}')
        assert exists(fname), f'Projection file {fname} does not exist.'
        f = h5py.File(fname, 'r')
        allcoords = []
        last_dim = 0
        print(f'f.keys are {f.keys()}')
        while True:
            name = 'proj_{:0>2d}coord'.format(last_dim)
            if name in f.keys():
                allcoords.append(list(f[name]))
                last_dim += 1
            else:
                break

        # color
        if fname.find('sFNO_eps') != -1:
            color = 'gold'
        elif fname.find('sFNO') != -1:
            color = 'blue'
        else:
            color = 'red'

        # marker and linewidth and dashes
        linewidth = 1
        dashes = (1,0)
        if fname.find('debug') != -1 or label.find('debug') != -1:
            marker = 'd'
            linewidth = 3
            dashes = (2,1)
        elif fname.find('split') != -1:
            marker = '.'
        elif fname.find('real') != -1:
            marker = 's'
        else:
            marker = '^'

        if complex != 'split':
            continue

        curves.append({ 'coords': allcoords, 'label': label, 'complex': complex, 'color': color, 'marker': marker, 'linewidth': linewidth, 'dashes': dashes })
        f.close()
        print(f'lastdim={last_dim}')

    print(f'There are {len(curves)} curves afterwards')
    ndim = len(curves[0]['coords'])

    fig = None
    if ndim == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i, c in enumerate(curves):
            color = c['color']
            coords = c['coords']
            marker = c['marker']
            ax.plot3D(coords[0], coords[1], coords[2], color=color, label=c['label'] + '/' + c['complex'], marker=marker)
            # ax.scatter(c['x'], c['y'], c['z'], color='black')
            ax.legend(bbox_to_anchor=(0.95, 1), loc='upper left', borderaxespad=0)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    elif ndim == 2:
        fig = plt.figure(figsize=(12, 6))
        for i, c in enumerate(curves):
            color = c['color']
            coords = c['coords']
            marker = c['marker']
            plt.plot(coords[0], coords[1], marker=marker, color=color, label=c['label'] + '/' + c['complex'], linewidth=c['linewidth'], dashes=c['dashes'])
            plt.xlabel('X')
            plt.ylabel('Y')
        plt.legend(loc='upper left', bbox_to_anchor=(0.95, 1), borderaxespad=0)
    elif ndim > 3:
        print('plotting splom')
        data = [ c['coords'] for c in curves ]
        names = [ 'PCA dim {}'.format(i) for i in range(len(data[0])) ]
        colors = [ c['color'] for c in curves ]
        markers = [ c['marker'] for c in curves]
        labels = [ c['label'] + '/' + c['complex'] for c in curves ]
        fig = splom(data, names=names, labels=labels, colors=colors, markers=markers)

    fig.savefig('somecurves.png', dpi=300, bbox_inches='tight', format='png')
    plt.show()

def to_path(checkpath, config):
    return os.path.join(checkpath, config['ckpt']['save_dir'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot learning trajectories')
    parser.add_argument('-c','--config_file', type=str, required=True, nargs='+', action='append',
                        help='Path to the model configuration file')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to checkpoint files')
    parser.add_argument('-d', '--dimension', type=int, default=2, help='Number of PCA dimensions')
    parser.add_argument('--show', action='store_true', default=True, help='show plots')

    args = parser.parse_args()
    print(args.config_file)

    traj_files = []
    for config_file in args.config_file:
        config_file = config_file[0]
        print(f'config_file is {config_file}')
        with open(config_file, 'r') as stream:
            config = yaml.load(stream, yaml.FullLoader)

        for complex in ['split', 'real', 'imaginary']:
            path = os.path.join(to_path(args.path, config), 'PCA_weights_save_epoch=99_complex='+ complex + '_dim=' + str(args.dimension))
            fname = os.path.join(path, 'directions.h5_proj_cos.h5')
            fname_debug = fname + '-debug.h5'
            traj_files.append({ 'label': config['Project']['name']+config['Project']['experiment'], 'filename': fname, 'complex_handling': complex })
            if exists(fname_debug):
                traj_files.append({ 'label': 'debug-'+ config['Project']['name']+config['Project']['experiment'], 'filename': fname, 'complex_handling': complex })


    plot_trajectories(traj_files, args.show)
