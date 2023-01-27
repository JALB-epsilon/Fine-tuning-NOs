import vtk
import argparse
from scipy import interpolate
import math
from matplotlib import pyplot as plt
import json
import sys
import os
import numpy as np
import random

from vtk.util.numpy_support import *

'''
Helper functions to create color palettes and color maps 
'''

# Colorful axis orientation cube
def make_cube_axis_actor(dims):
    colors = vtk.vtkNamedColors()

    # Annotated Cube setup
    annotated_cube = vtk.vtkAnnotatedCubeActor()
    annotated_cube.SetFaceTextScale(0.366667)

    # Cartesian labeling
    annotated_cube.SetXPlusFaceText('{}+'.format(dims[0]))
    annotated_cube.SetXMinusFaceText('{}-'.format(dims[0]))
    annotated_cube.SetYPlusFaceText('{}+'.format(dims[1]))
    annotated_cube.SetYMinusFaceText('{}-'.format(dims[1]))
    annotated_cube.SetZPlusFaceText('{}+'.format(dims[2]))
    annotated_cube.SetZMinusFaceText('{}-'.format(dims[2]))

    # Change the vector text colors
    annotated_cube.GetTextEdgesProperty().SetColor(
    colors.GetColor3d('Black'))
    annotated_cube.GetTextEdgesProperty().SetLineWidth(1)

    annotated_cube.GetXPlusFaceProperty().SetColor(
    colors.GetColor3d('Green'))
    annotated_cube.GetXMinusFaceProperty().SetColor(
    colors.GetColor3d('Green'))
    annotated_cube.GetYPlusFaceProperty().SetColor(
    colors.GetColor3d('Red'))
    annotated_cube.GetYMinusFaceProperty().SetColor(
    colors.GetColor3d('Red'))
    annotated_cube.GetZPlusFaceProperty().SetColor(
    colors.GetColor3d('Yellow'))
    annotated_cube.GetZMinusFaceProperty().SetColor(
    colors.GetColor3d('Yellow'))
    annotated_cube.SetXFaceTextRotation(90)
    annotated_cube.SetYFaceTextRotation(180)
    annotated_cube.SetZFaceTextRotation(-90)

    annotated_cube.GetCubeProperty().SetOpacity(0)
    #return annotated_cube

    # Colored faces cube setup
    cube_source = vtk.vtkCubeSource()
    cube_source.Update()
    face_colors = vtk.vtkUnsignedCharArray()
    face_colors.SetNumberOfComponents(3)
    # x
    face_colors.InsertNextTypedTuple(colors.GetColor3ub('Red'))
    face_colors.InsertNextTypedTuple(colors.GetColor3ub('Red'))
    # y
    face_colors.InsertNextTypedTuple(colors.GetColor3ub('Green'))
    face_colors.InsertNextTypedTuple(colors.GetColor3ub('Green'))
    # z
    face_colors.InsertNextTypedTuple(colors.GetColor3ub('Blue'))
    face_colors.InsertNextTypedTuple(colors.GetColor3ub('Blue'))

    cube_source.GetOutput().GetCellData().SetScalars(face_colors)
    cube_source.Update()
    m = vtk.vtkPolyDataMapper()
    m.SetInputData(cube_source.GetOutput())
    m.Update()
    a = vtk.vtkActor()
    a.SetMapper(m)
    # Assemble the colored cube and annotated cube texts into a composite prop.
    assembly = vtk.vtkPropAssembly()
    assembly.AddPart(annotated_cube)
    assembly.AddPart(a)
    return assembly


def create_vtk_colors(values):
    unique_vals = np.unique(values)
    nvals = unique_vals.shape[0]
    random.seed(a=13081975)
    unique_colors = np.array([ random.randrange(0,255) for i in range(3*nvals) ])

    ids = np.searchsorted(unique_vals, values, side='left')
    colors = []
    for id in ids:
        c = [ unique_colors[3*id-3], unique_colors[3*id-2], unique_colors[3*id-1] ]

    colors = np.array([ [ unique_colors[3*id-3], unique_colors[3*id-2], unique_colors[3*id-1] ] for id in ids ])

    return numpy_to_vtk(colors, array_type=vtk.VTK_UNSIGNED_CHAR)

def import_palette(palette_name='viridis', N=16):
    try:
        cmap = plt.get_cmap(palette_name, N)
    except:
        print('invalid palette name or number of colors')
        raise ValueError('invalid palette name or number of colors')
    lut = cmap(X=range(N))
    colors = vtk.vtkColorSeries()
    colors.ClearColors()
    colors.SetNumberOfColors(N)
    # print('import color map with {} colors'.format(N))
    for i in range(N):
        rgba = lut[i]
        rgb = [int(rgba[0]*255.), int(rgba[1]*255.), int(rgba[2]*255.)]
        colors.SetColor(i, vtk.vtkColor3ub(rgb))
    colors.SetColorSchemeName('{} ({})'.format(palette_name, N))
    # print('color map created:\n {}'.format(colors))
    return colors

def make_colormap(scheme_name, ctrl_pts):
    colors = vtk.vtkColorSeries()
    # colors = newcolors
    m = colors.GetNumberOfColorSchemes()
    # print(f'There are {m} color schemes')
    g = colors.SetColorSchemeByName(scheme_name)
    if g == m:
        # print('Requested color scheme was not found in VTK list')
        try:
            colors = import_palette(scheme_name)
        except:
            print('unable to find requested color map: {}'.format(scheme_name))
            raise
    else:
        print(f'Requested color scheme {scheme_name} has index {g}')
    n = colors.GetNumberOfColors()
    # print(f'{n} colors')
    if len(ctrl_pts) == 2:
        f = interpolate.interp1d(x=[0, n-1], y=ctrl_pts)
        ctrl_pts = f(range(n))
    elif len(ctrl_pts) != n:
        raise ValueError('Numbers of colors and control points don\'t match')
    cmap = vtk.vtkColorTransferFunction()
    for i in range(n):
        c = colors.GetColor(i)
        d=[0,0,0]
        for j in range(3):
            # print(c[j])
            d[j] = float(c[j])/255.
        cmap.AddRGBPoint(ctrl_pts[i], d[0], d[1], d[2])
        # print(f'{i}: {ctrl_pts[i]} . {c} / {d}')
    return cmap
