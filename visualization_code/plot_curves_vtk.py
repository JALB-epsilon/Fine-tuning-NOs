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
import vtk
from vtk.util import numpy_support
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

def do_plot(curves, filename='somecurves.png', colors=None, markers=None, linewidths=None, labels=None):
    print(f'There are {len(curves)} curves afterwards')
    ndim = len(curves[0]) # curves[i] = [ [x0, x1, x2, ...], [y0, y1, y2, ...], ... ]
    print(f'ndim={ndim}')
    ncurves = len(curves)
    if colors is None:
        colors = ['black'] * ncurves
    if markers is None:
        markers = '-' * ncurves
    if linewidths is None:
        linwidths = [2] * ncurves
    if labels is None:
        labels = [''] * ncurves

    fig = None
    if ndim == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i, c in enumerate(curves):
            color = colors[i]
            coords = np.array(c)
            marker = marker[i]
            label = labels[i]
            width = linewidths[i]
            ax.plot3D(coords[0], coords[1], coords[2], color=color, label=label, marker=marker, linewidth=width)
            # ax.scatter(c['x'], c['y'], c['z'], color='black')
            ax.legend(bbox_to_anchor=(0.95, 1), loc='upper left', borderaxespad=0)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    elif ndim == 2:
        fig = plt.figure(figsize=(12, 6))
        for i, c in enumerate(curves):
            color = colors[i]
            coords = np.array(c)
            marker = markers[i]
            label = labels[i]
            width = linewidths[i]
            plt.plot(coords[0], coords[1], marker=marker, color=color, label=label, linewidth=width)
            plt.xlabel('X')
            plt.ylabel('Y')
        plt.legend(loc='upper left', bbox_to_anchor=(0.95, 1), borderaxespad=0)
    elif ndim > 3:
        print('plotting splom')
        data = curves
        names = [ 'PCA dim {}'.format(i) for i in range(len(data[0])) ]
        # colors = [ c['color'] for c in curves ]
        # markers = [ c['marker'] for c in curves]
        # labels = [ c['label'] + '/' + c['complex'] for c in curves ]
        fig = splom(data, names=names, label=labels, color=colors, marker=markers)

    fig.savefig(filename, dpi=300, bbox_inches='tight', format='png')
    plt.show()

def size(something):
    if something is None:
        return 0
    elif isinstance(something, list):
        return len(something)
    elif isinstance(something, np.ndarray):
        return something.shape[0]
    elif isinstance(something, torch.Tensor):
        return something.size()[0]
    else:
        print(f'ERROR: unrecognized sizeable object: {something}')
        return 0

def do_plot_vtk(curves, losses=None, filename='somecurves.png', colors=None, linewidths=None, labels=None):
    ndim = len(curves[0]) # curves[i] = [ [x0, x1, x2, ...], [y0, y1, y2, ...], ... ]
    print(f'ndim={ndim}')
    ncurves = len(curves)

    named_colors = vtk.vtkNamedColors()

    ren=vtk.vtkRenderer()
    win = vtk.vtkRenderWindow()
    win.AddRenderer(ren)

    if ndim == 3:
        for i, c in enumerate(curves):
            npts = len(c)
            coords = numpy_support.numpy_to_vtk(np.array(c).transpose())
            pts = vtk.vtkPoints()
            pts.SetData(coords)
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            line_cells = vtk.vtkCellArray()
            line_cells.InsertNextCell(npts)
            pt_cells = vtk.vtkCellArray()
            for i in range(npts):
                line_cells.InsertNextPoint(i)
                pt_cells.InsertNextCell(1)
                pt_cells.InsertCellPoint(i)
            pd.SetLines(line_cells)
            pd.SetVerts(pt_cells)
            if size(losses) == ncurves:
                curve_losses = losses[i]
                if size(curve_losses) == npts:
                    radii = np.array(curve_losses)
                else:
                    radii = np.ones((npts))
            else:
                radii = np.ones((npts))
            pd.GetPointData().SetScalars(numpy_support.numpy_to_vtk(radii))
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pd)
            mapper.ScalarVisibilityOff()
            if colors is not None:
                if len(colors) == ncurves:
                    color = colors[i]
                else:
                    color = colors[0]
            else:
                color = 'red'
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(named_colors.GetColor3d(color).GetData())
            if linewidths is None:
                linewidths = 1
            actor.GetProperty().SetLineWidth(linewidths)
            actor.GetProperty().RenderLinesAsTubesOn()
            ren.AddActor(actor)
            glyphs = vtk.vtkGlyph3D()
            glyphs.SetInputData(pd)
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(10)
            glyphs.SetSourceConnect(sphere.GetOutputPort())
            gl_mapper = vtk.vtkPolyDataMapper()
            gl_mapper.SetInputConnection(glyphs.GetOutputPort())
            gl_actor = vtk.vtkActor()
            gl_actor.SetMapper(gl_mapper)
            ren.AddActor(gl_actor)
        win.SetSize(1024, 1024)
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(win)
        interactor.Initialize()
        win.Render()
        interactor.Start()
    elif ndim == 2:
        for i, c in enumerate(curves):
            npts = len(c)
            if size(losses) == ncurves and size(losses[i]) == npts:
                coords = numpy_support.numpy_to_vtk(np.array([c[0], c[1], losses[i]]).transpose())
            else:
                coords = numpy_support.numpy_to_vtk(np.array(c).transpose())
            pts = vtk.vtkPoints()
            pts.SetData(coords)
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            line_cells = vtk.vtkCellArray()
            line_cells.InsertNextCell(npts)
            pt_cells = vtk.vtkCellArray()
            for i in range(npts):
                line_cells.InsertNextPoint(i)
                pt_cells.InsertNextCell(1)
                pt_cells.InsertCellPoint(i)
            pd.SetLines(line_cells)
            pd.SetVerts(pt_cells)
            radii = np.ones((npts))
            pd.GetPointData().SetScalars(numpy_support.numpy_to_vtk(radii))
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pd)
            mapper.ScalarVisibilityOff()
            if colors is not None:
                if len(colors) == ncurves:
                    color = colors[i]
                else:
                    color = colors[0]
            else:
                color = 'red'
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(named_colors.GetColor3d(color).GetData())
            if linewidths is None:
                linewidths = 1
            actor.GetProperty().SetLineWidth(linewidths)
            actor.GetProperty().RenderLinesAsTubesOn()
            ren.AddActor(actor)
            glyphs = vtk.vtkGlyph3D()
            glyphs.SetInputData(pd)
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(10)
            glyphs.SetSourceConnect(sphere.GetOutputPort())
            gl_mapper = vtk.vtkPolyDataMapper()
            gl_mapper.SetInputConnection(glyphs.GetOutputPort())
            gl_actor = vtk.vtkActor()
            gl_actor.SetMapper(gl_mapper)
            ren.AddActor(gl_actor)
        win.SetSize(1024, 1024)
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(win)
        interactor.Initialize()
        win.Render()
        interactor.Start()
    elif ndim > 3:
        print('plotting splom')
        data = curves
        names = [ 'PCA dim {}'.format(i) for i in range(len(data[0])) ]
        # colors = [ c['color'] for c in curves ]
        # markers = [ c['marker'] for c in curves]
        # labels = [ c['label'] + '/' + c['complex'] for c in curves ]
        fig = splom(data, names=names, labels=labels, colors=colors, markers=markers)

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
    do_plot(all_curves, filename=args.output, colors=args.color, markers=args.marker, linewidths=args.width, labels=args.label)
