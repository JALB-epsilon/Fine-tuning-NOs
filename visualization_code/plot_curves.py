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
import sys
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
        dashes = (1, 0)
        if fname.find('debug') != -1 or label.find('debug') != -1:
            marker = 'd'
            linewidth = 3
            dashes = (2, 1)
        elif fname.find('split') != -1:
            marker = '.'
        elif fname.find('real') != -1:
            marker = 's'
        else:
            marker = '^'

        if complex != 'split':
            continue
        curves.append({'coords': allcoords, 'label': label, 'complex': complex,
                      'color': color, 'marker': marker, 'linewidth': linewidth, 'dashes': dashes})
    f.close()
    print(f'lastdim={last_dim}')

def select_kwargs(kwargs, i, ncurves, npts):
    newkwa = {}
    for k in kwargs.keys():
        arg = kwargs[k]
        if isinstance(arg, list):
            if len(arg) == ncurves:
                newkwa[k] = arg[i]
            elif len(arg) != ncurves and len(arg) != npts:
                newkwa[k] = arg[0]
            else:
                newka[k] = arg
        else:
            newkwa[k] = arg
    return newkwa

def do_plot(curves, filename='somecurves.png', stride=1, dim=2, **kwargs):
    print(f'There are {len(curves)} curves afterwards')
    ndim = len(curves[0]) # curves[i] = [ [x0, x1, x2, ...], [y0, y1, y2, ...], ... ]
    if ndim > dim:
        ndim = dim
    print(f'ndim={ndim}')
    # check that list arguments have size matching number of curves
    ncurves = len(curves)

    '''
    curves = [ curve1, curve2, curve3, ... ]
    curve1 = [ [x...], [y...], [z...], [t...], ...]
    subcurve = curve1[:ndim, :]
    subcurves = curves[ :. :ndim, :]
    '''

    fig = None
    if ndim == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        save_kwargs = kwargs
        for i, c in enumerate(curves):
            c = np.array(c)
            print(c.shape)
            print(f'c=[{list(c[0])}, {list(c[1])}, {list(c[2])}]')
            kwargs = select_kwargs(save_kwargs, i, ncurves, len(c[0]))
            print(f'c.shape={c.shape}')
            a = np.array(range(c.shape[1]))
            print(f'a is {a}')
            if stride > 1:
                subcurve = c[:, ::stride]
                ranks = a[::stride]
                print(f'ranks is {ranks}')
                print(f'subcurve has shape {subcurve.shape}')
            else:
                subcurve = c.view()
                ranks = np.ndarray(range(c.shape[1]))
            ax.plot3D(c[0], c[1], c[2], zorder=1, **kwargs)
            scat = ax.scatter(subcurve[0], subcurve[1], subcurve[2], zorder=2, c=ranks, vmin=0, vmax=300, cmap=cm.get_cmap('viridis'))
            # ax.scatter(c['x'], c['y'], c['z'], color='black')
        ax.legend(bbox_to_anchor=(0.1, 1), loc='upper right', borderaxespad=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        cax = plt.axes([0.87, 0.1, 0.025, 0.8])
        plt.colorbar(mappable=scat, cax=cax, label='training steps')
    elif ndim == 2:
        fig = plt.figure(figsize=(12, 6))
        print('kwargs = {}'.format(kwargs))
        save_kwargs = kwargs
        # all_colors = create_colorscale(curves, stride)
        ax = plt.axes([0.05, 0.05, 0.75, 0.9])
        for i, c in enumerate(curves):
            c = np.array(c)
            kwargs = select_kwargs(save_kwargs, i, ncurves, len(c[0]))
            print(f'kwargs = {kwargs}')
            ax.plot(c[0], c[1], zorder=1, **kwargs)
            a = np.array(range(c.shape[1]))
            if stride > 1:
                subcurve = c[:, ::stride]
                ranks = a[::stride]
            else:
                subcurve = c.view()
                ranks = a
            # print(f'c={c}\nsubcurve={subcurve}\nall_colors={all_colors}')
            # colors = all_colors[:len(subcurve[0])]
            # print(f'shape of subcurve={subcurve.shape}, {len(colors)} colors')
            plt.scatter(subcurve[0], subcurve[1], c=ranks, vmin=0, vmax=300, zorder=2, cmap=cm.get_cmap('viridis'))
            plt.xlabel('X')
            plt.ylabel('Y')
        cax = plt.axes([0.87, 0.1, 0.025, 0.8])
        plt.colorbar(cax=cax, label='training steps')
        plt.legend(loc='upper left', bbox_to_anchor=(0.95, 1), borderaxespad=0)
    elif ndim > 3:
        print('plotting splom')
        data = np.array(curves)[:, :ndim, :]
        names = [ 'PCA dim {}'.format(i) for i in range(len(data[0])) ]
        # colors = [ c['color'] for c in curves ]
        # markers = [ c['marker'] for c in curves]
        # labels = [ c['label'] + '/' + c['complex'] for c in curves ]
        fig = splom(data, names=names, stride=stride, colorby='rank', **kwargs)

    fig.savefig(filename, dpi=300, bbox_inches='tight', format='png')
    plt.show()

def to_path(checkpath, config):
    return os.path.join(checkpath, config['ckpt']['save_dir'])

def fix_list(alist, default, n):
    if alist is None or len(alist) == 0:
        return [default] * n
    elif len(alist) == 1 and n > 1:
        return [ alist[0] ] * len(args.filename)
    elif len(alist) != n:
        print('ERROR: mismatch between attributes and filenames')
        sys.exit(0)
    else:
        return alist

def to_single_list(something):
    if something is None:
        return None
    output = []
    if isinstance(something, list):
        for an_item in something:
            output.extend(to_single_list(an_item))
    else:
        output.append(something)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot learning trajectories')
    parser.add_argument('-f', '--filename', type=str, nargs='+', action='append', help='Curves filenames')
    parser.add_argument('-p', '--path', type=str, nargs='+', action='append', help='Path to projected curve files')
    parser.add_argument('-d', '--dimension', type=int, default=2, help='Number of PCA dimensions')
    parser.add_argument('-c', '--color', type=str, nargs='+', action='append', help='Colors to assign to curves')
    parser.add_argument('-m', '--marker', type=str, nargs='+', action='append')
    parser.add_argument('-l', '--label', type=str, nargs='+', action='append', help='Labels for curves')
    parser.add_argument('-w', '--width', type=int, nargs='+', action='append', help='Curve colors')
    parser.add_argument('-s', '--stride', type=int, default=1, help='Stride to subsample curve for plotting')
    parser.add_argument('-o', '--output', type=str, default='somecurve.png', help='Output filename for created visualization')
    parser.add_argument('--show', action='store_true', default=True, help='show plots')

    args = parser.parse_args()
    # print(args.config_file)
    print(args)

    if args.filename is None or len(args.filename) == 0:
        print('Nothing to display')
        sys.exit(0)
    args.filename = to_single_list(args.filename)
    args.path = to_single_list(args.path)
    args.marker = to_single_list(args.marker)
    print(args.color)
    args.color = to_single_list(args.color)
    print(args.color)
    args.label = to_single_list(args.label)
    args.width = to_single_list(args.width)

    ncurves = len(args.filename)

    print(args.path)
    args.path = fix_list(args.path, '.', ncurves)
    print(args.path)
    args.color = fix_list(args.color, 'black', ncurves)
    print(args.color)
    args.marker = fix_list(args.marker, 'none', ncurves)
    print(args.marker)
    if args.label is None or len(args.label)==0:
        args.label = args.filename
    args.label = fix_list(args.label, 'train curve', ncurves)
    print(args.label)
    args.width = fix_list(args.width, 1, ncurves)
    print(args.width)

    
    all_curves = []
    print(f'args.path={args.path}, args.filename={args.filename}')
    args.path = to_single_list(args.path)
    args.filename = to_single_list(args.filename)
    print(f'args.path={args.path}, args.filename={args.filename}')
    print(f'args.marker={args.marker}')
    print(f'args.width={args.width}')
    print(f'args.label={args.label}')
    for path, filename in zip(args.path, args.filename):
        print(f'path={path}, filename={filename}')
        fname = os.path.join(path, filename)
        assert os.path.exists(fname), f'Projection file {fname} does not exist.'
        f = h5py.File(fname, 'r')
        print(f'f.keys are {f.keys()}')
        coords = []
        for k in f.keys():
            coords.append(list(f[k]))
        all_curves.append(coords)
    do_plot(all_curves, filename=args.output, stride=args.stride, dim=args.dimension, color=args.color, marker='none', linewidth=args.width, label=args.label)
