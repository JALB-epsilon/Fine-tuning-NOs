import vtk
import argparse
from scipy import interpolate
import math
from matplotlib import pyplot as plt
import json
import sys
import os
import numpy as np
import vtk_camera
import vtk_colorbar
import vtk_colors
import vtk_io_helper
from vtk_colors import make_colormap
from vtk.util.numpy_support import *

'''
Program to visualize loss surface and training trajectory with a few additional
features.
'''

frame_counter=0

parameters = {
    'surf_name': [ '', 'name of surface'],
    'traj_diameter': [0.1, 'trajectory diameter' ],
    'step_diameter': [0.4, 'Diameter of step spherical representation'],
    'size': [ [1024, 1024], 'image resolution' ],
    'ncontours': [20, 'Number of isocontours'],
    'curve_color': [ [1,1,1], 'Default color of trajectory' ],
    'traj_offset': [ 0.01, 'Vertical offset for trajectory' ],
    'iso_diameter': [ 0.05, 'Isocontour tubes diameters' ],
    'font_size': [ 12, 'Font size for color legend' ],
    'log_offset': [ 0.1, 'Logarithm offset' ],
    'surface_palette': [ 'viridis', 'Name of color palette for surface' ],
    'trajectory_palette': ['Oranges', 'Name of color palette for trajectory' ],
    'show_isovalues': [ False, 'Display isovalues on isolines' ],
    'show_steps': [ False, 'Display individual learning steps along trajectory'],
    'color_surface': [ True, 'Color map loss on surface' ],
    'frame_basename': [ 'frame', 'Basename of frame snapshots to be saved' ],
    'camera_basename': [ 'camera', 'Basename for exported camera settings' ],
    'camera_file': [ '', 'Name of file containing camera information' ],
    'print_text': [ '', 'Text to be displayed' ],
    'save_frame': [ False, 'Save frame and exit'],
    'flatten': [False, 'Flatten into 2D visualization'],
    'range': [ [0.0,0.0], 'Value range to consider for color mapping'],
    'show_colorbars': [True, 'Display colorbars of used color mappings'],
    'background': [ [0.,0.,0.], 'Background color']
}

def save_frame(window, verbose=False):
    global frame_counter
    global frame_basename

    # ---------------------------------------------------------------
    # Save current contents of render window to PNG file
    # ---------------------------------------------------------------
    if frame_counter >= 0:
        file_name = frame_basename + str(frame_counter).zfill(5) + ".png"
    else:
        file_name = frame_basename + '.png'
    image = vtk.vtkWindowToImageFilter()
    image.SetInput(window)
    png_writer = vtk.vtkPNGWriter()
    png_writer.SetInputConnection(image.GetOutputPort())
    png_writer.SetFileName(file_name)
    window.Render()
    png_writer.Write()
    frame_counter += 1
    if verbose:
        print(file_name + " has been successfully exported")

def key_pressed_callback(inter, event):
    global camera_basename
    # ---------------------------------------------------------------
    # Attach actions to specific keys
    # ---------------------------------------------------------------
    key = inter.GetKeySym()
    window = inter.GetRenderWindow()
    cam = window.GetRenderers().GetFirstRenderer().GetActiveCamera()
    if key == "s":
        save_frame(window=window)
    elif key == "c":
        print('about to save camera setting')
        vtk_camera.save_camera(camera=cam, filename=camera_basename)
        vtk_camera.print_camera(cam)
    elif key == "q":
        if args.verbose:
            print("User requested exit.")
        sys.exit()

def log_xform(data, args):
    coords = data.GetPoints().GetData()
    newcoords = vtk.vtkFloatArray()
    newcoords.DeepCopy(coords)
    for i in range(newcoords.GetNumberOfTuples()):
        p = newcoords.GetTuple3(i)
        z = math.log(p[2]+args.log_offset)
        print(f'{p[2]} -> {z}')
        newcoords.SetTuple3(i, p[0], p[1], z)
    newpts = vtk.vtkPoints()
    newpts.SetData(newcoords)
    data.SetPoints(newpts)
    return data

def flat_xform(data):
    coords = data.GetPoints().GetData()
    newcoords = vtk.vtkFloatArray()
    newcoords.DeepCopy(coords)
    for i in range(newcoords.GetNumberOfTuples()):
        p = newcoords.GetTuple3(i)
        newcoords.SetTuple3(i, p[0], p[1], 0)
    newpts = vtk.vtkPoints()
    newpts.SetData(newcoords)
    data.SetPoints(newpts)
    return data

def shift(data, offset):
    xform = vtk.vtkTransform()
    xform.Identity()
    xform.Translate(0, 0, offset)
    _shift = vtk.vtkTransformPolyDataFilter()
    _shift.SetTransform(xform)
    _shift.SetInputData(data)
    _shift.Update()
    shifted = vtk.vtkPolyData()
    shifted.DeepCopy(_shift.GetOutput())
    return shifted

def view(args):
    if args.path is not None and args.path:
        args.surface = args.path + '/' + args.surface
        args.trajectory = args.path + '/' + args.trajectory
        if args.info is not None:
            args.info = args.path + '/' + args.info

    global frame_basename
    frame_basename = args.frame_basename
    global camera_basename
    camera_basename = args.camera_basename

    if not os.path.exists(args.surface):
        print('{} does not exist. Nothing to visualize!'.format(args.surface))
        sys.exit(0)
    surf_reader = vtk_io_helper.readVTK(args.surface)
    surf_reader.Update()

    if True or not args.flatten:
        geom = vtk.vtkGeometryFilter()
        geom.SetInputConnection(surf_reader.GetOutputPort())
        geom.Update()

        # 2. create a mapper to its geometry
        surface = geom.GetOutput()
        if args.flatten:
            surface = flat_xform(surface)

        info = None
        if args.info is not None:
            print('importing training information')
            with open(args.info, 'r') as json_file:
                info = json.load(json_file)

        if args.do_log:
            surface = log_xform(surface, args)
            is_log = True

        normals_algo = vtk.vtkPolyDataNormals()
        normals_algo.SetInputData(surface)
        normals_algo.Update()
        surface = normals_algo.GetOutput()

        surf_mapper = vtk.vtkPolyDataMapper()
        surf_actor = vtk.vtkActor()
        surf_mapper.SetInputData(surface)
        surf_actor.SetMapper(surf_mapper)
        surf_actor.GetProperty().SetSpecular(0.25)
        surf_actor.GetProperty().SetDiffuse(0.9)

    # 3. create a color map for the value range
    if args.surf_name:
        value_bounds = surface.GetPointData().GetArray(args.surf_name).GetRange()
        surface.GetPointData().SetActiveScalars(args.surf_name)
    else:
        value_bounds = surface.GetPointData().GetScalars().GetRange()
    if args.range != [0,0]:
        value_bounds = args.range
    cmap = make_colormap(args.surface_palette, value_bounds)
    if args.color_surface:
        surf_mapper.ScalarVisibilityOn()
        surf_mapper.SetLookupTable(cmap)
    else:
        surf_mapper.ScalarVisibilityOff()
        surf_actor.GetProperty().SetColor(1,1,1)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(surf_actor)

    # Create a text actor
    if len(args.print_text) > 1:
        txt = vtk.vtkTextActor()
        if info is not None:
            loss = info['training']['loss']
            loss_as_str = '{:0.8f}'.format(loss)
            steps = info['training']['steps'][-1]
            args.print_text += '/' + str(steps) + 'it./loss=' + loss_as_str
        txt.SetInput(args.print_text)
        txtprop = txt.GetTextProperty()
        txtprop.SetFontFamilyToArial();
        txtprop.BoldOn();
        txtprop.SetFontSize(28);
        txtprop.SetColor(1,1,1);
        txt.SetDisplayPosition(20, 30);
        renderer.AddActor(txt)

    if (args.color_surface or not args.show_isovalues) and args.show_colorbars:
        surf_bar_param = vtk_colorbar.colorbar_param(title='Training\nloss', title_font_size=40, title_col=[0,0,0], label_col=[0,0,0], title_offset=10, nlabels=10, font_size=30, width=150, height=600, pos=[0.9, 0.3])
        surf_bar = vtk_colorbar.colorbar(ctf=cmap, param=surf_bar_param)
        renderer.AddActor2D(surf_bar.get())

    contour = vtk.vtkContourFilter()
    contour.SetInputData(surface)
    # fix set of loss values that covers a wide range of training scenarios
    vals = [10.0, 5.0, 3.0, 2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.001, 0.0005, 0.0001, 0.00005]
    for i, v in enumerate(vals):
        contour.SetValue(i, v)

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(contour.GetOutputPort())
    stripper.Update()

    if args.ncontours > 0:
        isolines = stripper.GetOutput()

    if args.is_log:
        npts = isolines.GetNumberOfPoints()
        values = vtk.vtkDoubleArray()
        values.SetNumberOfTuples(npts)
        values.SetNumberOfComponents(1)
        values.SetName('Actual loss values')
        log_values = isolines.GetPointData().GetScalars()
        for i in range(npts):
            logv = log_values.GetTuple1(i)
            values.SetTuple1(i, math.exp(logv)-args.log_offset)
        if args.ncontours > 0:
            isolines.GetPointData().AddArray(values)
            isolines.GetPointData().SetActiveScalars('Actual loss values')

    if args.ncontours > 0:
        isolines = shift(isolines, args.traj_offset)

    if args.show_isovalues:
        textprop = vtk.vtkTextProperty()
        textprop.FrameOff()
        if not args.color_surface:
            textprop.SetColor(0,0,0)
        else:
            textprop.SetColor(1,1,1)
        textprop.SetFontSize(args.font_size)
        textprop.BoldOn()

        contour_mapper = vtk.vtkLabeledContourMapper()
        contour_mapper.LabelVisibilityOn()
        contour_mapper.SetSkipDistance(100)
        contour_mapper.SetTextProperty(textprop)
        contour_mapper.SetInputData(isolines)
        contour_mapper.ScalarVisibilityOff()
    else:
        contour_mapper = vtk.vtkPolyDataMapper()
        contour_mapper.SetInputData(isolines)
        if False and not args.color_surface:
            contour_mapper.ScalarVisibilityOn()
            contour_mapper.SetLookupTable(cmap)
        else:
            contour_mapper.ScalarVisibilityOn()

    contour_actor = vtk.vtkActor()
    contour_actor.GetProperty().SetLineWidth(args.iso_diameter)
    contour_actor.SetMapper(contour_mapper)
    contour_actor.GetProperty().SetColor(1,1,1)
    contour_actor.AddPosition([0,0,1])

    renderer.AddActor(contour_actor)
    if args.trajectory is not None:
        curve_reader = vtk.vtkXMLPolyDataReader()
        curve_reader.SetFileName(args.trajectory)
        curve_reader.Update()
        curve = curve_reader.GetOutput()

        nsteps = curve.GetNumberOfPoints()
        cmap_traj = make_colormap(args.trajectory_palette, [0, nsteps-1])

        if args.flatten:
            curve = flat_xform(curve)

        if args.do_log:
            curve = log_xform(curve, args)

        curve = shift(curve, args.traj_offset)

        if False and args.show_colorbars:
            traj_bar_param = vtk_colorbar.colorbar_param(title='Training steps', title_offset=10, nlabels=10, font_size=18, width=80, height=500, pos=[0.9, 0.1])
            traj_bar = vtk_colorbar.colorbar(ctf=cmap_traj, param=traj_bar_param)
            renderer.AddActor2D(traj_bar.get())

        stops = vtk.vtkPolyData()

        actual_steps = None
        if args.show_steps:
            verts = vtk.vtkCellArray()
            scalars = numpy_to_vtk(np.array([i for i in range(nsteps)]))
            for i in range(curve.GetNumberOfPoints()):
                verts.InsertNextCell(1)
                verts.InsertCellPoint(i)
            curve.SetVerts(verts)
            curve.GetPointData().SetScalars(scalars)
            sphere = vtk.vtkSphereSource()
            sphere.SetThetaResolution(12)
            sphere.SetPhiResolution(12)
            sphere.SetRadius(args.step_diameter)
            glyphs = vtk.vtkGlyph3D()
            glyphs.SetSourceConnection(sphere.GetOutputPort())
            glyphs.SetInputData(curve)
            glyphs.ScalingOff()

            gmapper = vtk.vtkPolyDataMapper()
            gmapper.SetInputConnection(glyphs.GetOutputPort())
            gmapper.ScalarVisibilityOn()
            gmapper.SetLookupTable(cmap_traj)
            gactor = vtk.vtkActor()
            gactor.SetMapper(gmapper)
            gactor.AddPosition(0,0,args.traj_offset)
            gactor.GetProperty().SetColor(1,0,0)
            renderer.AddActor(gactor)
        elif info is not None:
            steps = info['training']['steps']
            narrays = curve.GetPointData().GetNumberOfArrays()
            print('there are {} arrays'.format(narrays))
            iter = None
            for i in range(narrays):
                name = curve.GetPointData().GetArray(i).GetName()
                if name.lower() == 'iterations' or name.lower() == 'iteration':
                    iter = curve.GetPointData().GetArray(i)

            if iter is not None:
                # find recorded steps within list of iterations
                iter = np.asarray(iter)
                actual_steps = [(np.abs(iter-s)).argmin() for s in steps]
            elif curve.GetNumberOfPoints() == len(steps):
                actual_steps = steps
            else:
                print('Missing iteration information in trajectory')

            if actual_steps is not None:
                steps_array = vtk.vtkIntArray()
                steps_array.SetNumberOfTuples(len(steps))
                steps_array.SetNumberOfComponents(1)
                coords = vtk.vtkFloatArray()
                coords.SetNumberOfComponents(3)
                coords.SetNumberOfTuples(len(steps))
                for i in range(len(steps)):
                    steps_array.SetTuple1(i, steps[i])
                    p = curve.GetPoints().GetPoint(actual_steps[i])
                    coords.SetTuple3(i, p[0], p[1], p[2])
                steps_array.SetName('Iteration')
                pts = vtk.vtkPoints()
                pts.SetData(coords)
                stops.SetPoints(pts)
                stops.GetPointData().SetScalars(steps_array)
                sphere = vtk.vtkSphereSource()
                sphere.SetThetaResolution(20)
                sphere.SetPhiResolution(20)
                sphere.SetRadius(args.step_diameter)
                glyphs = vtk.vtkGlyph3D()
                glyphs.SetSourceConnection(sphere.GetOutputPort())
                glyphs.SetInputData(stops)
                glyphs.ScalingOff()
                gmapper = vtk.vtkPolyDataMapper()
                gmapper.SetInputConnection(glyphs.GetOutputPort())
                gmapper.ScalarVisibilityOn()
                cmap2 = make_colormap(args.trajectory_palette, [0, steps[-1]])
                gmapper.SetLookupTable(cmap2)
                gactor = vtk.vtkActor()
                gactor.SetMapper(gmapper)
                if args.show_colorbars:
                    traj_bar_param = colorbar.colorbar_param(title='Iterations', title_offset=10, nlabels=11, font_size=args.font_size, width=80, height=300, pos=[0.9, 0.1])
                    traj_bar = colorbar.colorbar(ctf=cmap2, param=traj_bar_param, is_float=False)
                    renderer.AddActor2D(traj_bar.get())
                renderer.AddActor(gactor)

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(curve)
        tube.SetRadius(args.traj_diameter/2)
        tube.SetNumberOfSides(20)
        traj_mapper = vtk.vtkPolyDataMapper()
        traj_mapper.SetInputConnection(tube.GetOutputPort())
        # curve_mapper.SetInputData(curve)
        traj_mapper.ScalarVisibilityOff()
        traj_actor = vtk.vtkActor()
        traj_actor.SetMapper(traj_mapper)
        # curve_actor.GetProperty().SetLineWidth(args.traj_diameter)
        traj_actor.GetProperty().SetColor(args.curve_color)
        renderer.AddActor(traj_actor)

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(args.size[0], args.size[1])
    renderer.SetBackground(args.background)
    window.StencilCapableOn() # for proper display of log loss values
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    if args.camera_file:
        renderer.SetActiveCamera(vtk_camera.load_camera(args.camera_file))
    elif args.flatten:
        renderer.GetActiveCamera().ParallelProjectionOn()
        vtk_camera.print_camera(renderer.GetActiveCamera())
    if args.light_file is not None and len(args.light_file) > 0:
        newlc = vtk_camera.load_lights(args.light_file)
        renderer.SetLightCollection(newlc)

    if not args.save_frame:
        interactor.AddObserver("KeyPressEvent", key_pressed_callback)
        interactor.Initialize()
    window.Render()
    if args.save_frame:
        global frame_counter
        frame_counter = -1
        save_frame(window)
        sys.exit(0)
    else:
        interactor.Start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize loss landscape', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--path', type=str, help='Path to surface information')
    parser.add_argument('-s', '--surface', type=str, required=True, help='File containing loss surface geometry')
    parser.add_argument('-t', '--trajectory', type=str, required=True, help='File containing training trajectory')
    parser.add_argument('--info', type=str, help='File containing training information')
    parser.add_argument('--light_file', type=str, nargs='+', action='append', help='Name of file(s) containing light information')

    for item in parameters.keys():
        default, help = parameters[item]
        # print(f'default={default}, help={help}')
        if isinstance(default, list):
            a = default[0]
            parser.add_argument('--' + item, type=type(a), nargs=len(default), default=default, help=help)
        elif isinstance(default, bool):
            parser.add_argument('--' + item, action='store_true', help=help)
        else:
            parser.add_argument('--' + item, type=type(default), default=default, help=help)

    parser.add_argument('--is_log', action='store_true', help='Indicate that the dataset has been log converted beforehand')
    parser.add_argument('--do_log', action='store_true', help='Log convert the dataset before visualizing it')

    args = parser.parse_args()

    view(args)
