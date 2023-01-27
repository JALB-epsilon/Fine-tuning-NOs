import csv
import numpy as np
import argparse
import pandas as pd
import json
import vtk
import os
import h5py
import vtk_colors as colors
import vtk_io_helper as io_helper
import sys

'''
    Collect all surface samples spread across multiple csv files (with arbitrary
    structure, or total lack thereof) and form a surface by triangulation. The
    surface and its underlying mesh can both be visually inspected afterwards.
    A better visualization of the loss landscape surface can then be created with
    'view_surface.py'
'''

from vtk.util.numpy_support import numpy_to_vtk

def do_warp(input, factor=10):
    warp = vtk.vtkWarpScalar()
    if isinstance(input, vtk.vtkAlgorithm):
        warp.SetInputConnection(input.GetOutputPort())
    else:
        warp.SetInputData(input)
    warp.SetScaleFactor(factor)
    return warp

def do_wrap(input, radius):
    tubes = vtk.vtkTubeFilter()
    if isinstance(input, vtk.vtkAlgorithm):
        tubes.SetInputConnection(input.GetOutputPort())
    else:
        tubes.SetInputData(input)
    tubes.SetNumberOfSides(12)
    tubes.SetRadius(radius)
    return tubes

# convert an array of 2D positions into a polydata that can be warped
# using vtkWarpScalar
def interpolate_trajectory(trajectory, triangulation, factor):
    append = vtk.vtkAppendFilter()
    append.AddInputData(triangulation)
    append.Update()
    tris = vtk.vtkUnstructuredGrid()
    tris.ShallowCopy(append.GetOutput())
    losses = tris.GetPointData().GetScalars()
    values = []
    coords = []
    xs = []
    ys = []
    for p in trajectory:
        xs.append(p[0])
        ys.append(p[1])
        subid = 0
        pcoord = [0,0,0]
        weights = [0 for i in range(8)]
        subid = 0
        cellid = tris.FindCell([p[0], p[1], 0], None, 0, 1.0e-6, vtk.reference(subid), pcoord, weights)
        print(f'cellid = {cellid}')
        if cellid < 0:
            print(f'position {[p[0], p[1], 0]} is outside surface domain')
            continue

        cell = tris.GetCell(cellid)
        ids = [ cell.GetPointId(i) for i in range(0, cell.GetNumberOfPoints())]
        v = 0
        print(f'ids={ids}')
        for i, id in enumerate(ids):
            print(f'loss = {losses.GetTuple(id)}')
            v += weights[i]*losses.GetTuple(id)[0]
        values.append(args.factor*v)
        coords.append([p[0], p[1], values[-1]])

    print(f'x bounds of trajectory: {np.min(xs)} - {np.max(xs)}')
    print(f'y bounds of trajectory: {np.min(ys)} - {np.max(ys)}')
    values = np.array(values)
    scalars = numpy_to_vtk(values)
    coords = np.array(coords)
    coords = numpy_to_vtk(coords)
    pts = vtk.vtkPoints()
    pts.SetData(coords)
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.GetPointData().SetScalars(scalars)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(values))
    for i in range(len(values)):
        lines.InsertCellPoint(i)
    poly.SetLines(lines)
    return poly

def compute(args):
    df = pd.DataFrame()

    if args.skip and os.path.exists(args.output):
        print(f'{args.output} found. Nothing to be done')
        sys.exit(0)

    total_size = 0
    if args.input is None and args.path is not None:
        args.input = []
        filenames = os.listdir(args.path)
        for name in filenames:
            if os.path.splitext(name)[1].lower() == '.csv':
                args.input.append(name)

    for fname in args.input:
        if args.path:
            fname = os.path.join(args.path, fname)
        print(f'importing {fname}')
        df1 = pd.read_csv(fname)
        total_size += df1.shape[0]
        if df.shape[0] == 0:
            df = df1
        else:
            df = pd.concat([df, df1], ignore_index=True)

    if args.restrict and not args.trim:
        args.trim = True
    if args.trim and args.x is not None and args.y is not None:
        xmin, xmax, xnum = args.x
        ymin, ymax, ynum = args.y
        xnum = int(xnum)
        ynum = int(ynum)
        n = df.shape[0]
        if args.restrict:
            xs = np.linspace(xmin, xmax, xnum)
            ys = np.linspace(ymin, ymax, ynum)
            df = df[(df['x'].isin(xs)) & (df['y'].isin(ys))]
        else:
            df = df[(df['x']>=xmin) & (df['x']<=xmax) & (df['y']>=ymin) & (df['y']<=ymax) ]

        n1 = df.shape[0]
        print(f'after trimming, number of samples went from {n} to {n1}')

    if len(args.input) == 0:
        print('No data file available. Done')
        sys.exit(0)

    points = np.array([[x, y, 0] for x, y in zip(df['x'], df['y'])])
    print(points)

    coords = numpy_to_vtk(points)
    dataset = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(coords)
    dataset.SetPoints(pts)
    loss = numpy_to_vtk(np.array(df['loss']))
    minloss = np.min(loss)
    maxloss = np.max(loss)
    meanloss = np.mean(loss)
    print(f'loss stats: min: {minloss}, mean: {meanloss}, max: {maxloss}')
    dataset.GetPointData().SetScalars(loss)

    tri = vtk.vtkDelaunay2D()
    tri.SetInputData(dataset)
    tri.Update()
    triangulation = tri.GetOutput()
    bounds = triangulation.GetBounds()

    if args.trajectory:
        f = h5py.File(args.trajectory, 'r')
        xs = list(f['proj_00coord'])
        ys = list(f['proj_01coord'])
        curve = [ (x,y) for x, y in zip(xs, ys)]
        f.close()
        curve_dataset = interpolate_trajectory(curve, triangulation, args.factor)
        if args.output:
            base, ext = os.path.splitext(args.output)
            io_helper.saveVTK_XML(curve_dataset, base + '_trajectory' + ext)
        wrapped_curve = do_wrap(curve_dataset, args.radius)
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(wrapped_curve.GetOutputPort())
        curve_actor = vtk.vtkActor()
        curve_actor.SetMapper(mapper)
        curve_actor.GetProperty().SetColor(args.color)

    warp = vtk.vtkWarpScalar()
    warp.SetInputConnection(tri.GetOutputPort())
    warp.SetScaleFactor(args.factor)
    warp.Update()
    surface = warp.GetOutput()
    if args.output:
        print(f'saving surface in {args.output}')
        io_helper.saveVTK_XML(surface, args.output)

    plane = vtk.vtkPlaneSource()
    plane.SetXResolution(10)
    plane.SetYResolution(10)
    plane.SetOrigin(bounds[0], bounds[2], 0)
    plane.SetPoint1(bounds[1], bounds[2], 0)
    plane.SetPoint2(bounds[0], bounds[3], 0)
    pmapper = vtk.vtkDataSetMapper()
    pmapper.SetInputConnection(plane.GetOutputPort())
    pactor = vtk.vtkActor()
    pactor.SetMapper(pmapper)
    pactor.GetProperty().SetColor(1,1,1)

    tactor = None
    if args.show_edges:
        edges = vtk.vtkExtractEdges()
        edges.SetInputConnection(warp.GetOutputPort())
        tubes = vtk.vtkTubeFilter()
        tubes.SetInputConnection(edges.GetOutputPort())
        tubes.SetNumberOfSides(10)
        tubes.SetRadius(0.1)

        tmapper = vtk.vtkPolyDataMapper()
        tmapper.SetInputConnection(tubes.GetOutputPort())
        tactor = vtk.vtkActor()
        tactor.SetMapper(tmapper)
        tactor.GetProperty().SetColor(1,0,0)

    cmap = colors.make_colormap('viridis', [minloss, maxloss])

    # cmap = vtk.vtkColorTransferFunction()
    # cmap.AddRGBPoint(minloss, 1, 1, 0)
    # cmap.AddRGBPoint(maxloss, 0, 0, 1)
    # cmap.AddRGBPoint(meanloss, 0.5, 0.5, 0.5)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(warp.GetOutputPort())
    mapper.ScalarVisibilityOn()
    mapper.SetLookupTable(cmap)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1,1,0)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    if tactor is not None:
        renderer.AddActor(tactor)
    renderer.AddActor(pactor)
    if args.trajectory:
        renderer.AddActor(curve_actor)
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1024, 1024)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)


    df.sort_values('x', ascending=True, inplace=True)
    xs = df['x'].unique()
    print(f'xs={xs}')

    df.sort_values('y', ascending=True, inplace=True)
    ys = df['y'].unique()
    print(f'ys={ys}')

    if args.x is not None and args.y is not None:
        goal_xs = np.linspace(args.x[0], args.x[1], int(args.x[2]))
        goal_ys = np.linspace(args.y[0], args.y[1], int(args.y[2]))
        missing = []
        for x in goal_xs:
            if not x in xs:
                # print(f'column {x} is missing ({len(goal_ys)} missing values)')
                missing.extend([[x,y] for y in goal_ys])
            else:
                subdf = df[df['x']==x]
                ys = subdf['y'].unique()
                for y in goal_ys:
                    if not y in ys:
                        # print(f'value at ({x}, {y}) is missing')
                        missing.append([x,y])

        n = len(missing)
        total = len(goal_xs) * len(goal_ys)
        print(f'There are a total of {n} missing values ({float(n)/float(total)*100.}%)')
        done = total-n
        print(f'total unique computed values: {done}')
        print(f'total computed values: {total_size}')
        print(f'redundant values: {total_size-done}')

        print('missing values are:\n{}'.format(missing))
        if args.missing:
            with open(args.missing, 'w') as fp:
                json.dump(missing, fp)

            with open(args.missing, 'r') as fp:
                all_c = json.load(fp)
                print('\n\n\n\n{}'.format(all_c))

    interactor.Initialize()
    window.Render()
    interactor.Start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine information contained in csv file to reconstruct a data lattice')
    parser.add_argument('-p', '--path', type=str, default='', help='Path to append to filenames')
    parser.add_argument('-i', '--input', type=str, action='append', help='CSV filename contening a fraction of the data')
    parser.add_argument('--x', type=float, nargs=3, required=False, help='X sampling: xmin, xmax, xnum')
    parser.add_argument('--y', type=float, nargs=3, required=False, help='Y sampling: ymin, ymax, ynum')
    parser.add_argument('--fieldnames', type=str, default='', help='Column names')
    parser.add_argument('-o', '--output', type=str, default='', help='Filename to export reconstucted surface')
    parser.add_argument('--missing', type=str, default='', help='Filename to use to export coordinates of missing samples')
    parser.add_argument('--trajectory', type=str, default='', help='Filename of projected training trajectory')
    parser.add_argument('--radius', type=float, default=1, help='Radius of tubes depicting trajectory')
    parser.add_argument('--factor', type=float, default=10, help='Loss magnification')
    parser.add_argument('--color', type=float, nargs=3, default=[1,1,1], help='Color for trajectory representation')
    parser.add_argument('--show-edges', action='store_true', help='Display surface edges')
    parser.add_argument('--trim', action='store_true', help='Restrict surface to prescribed x,y domain')
    parser.add_argument('--restrict', action='store_true', help='Restrict the mesh to the prescribed samples')
    parser.add_argument('--skip', action='store_true', help='Indicate whether to do anything if output file exists already')

    args = parser.parse_args()
    compute(args)
