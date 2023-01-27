import vtk
import json
import os
import time
import math
import numpy as np

'''
Helper functions to import/export and print out camera and light settings
'''

def make_2d_camera(dataset, window):
    xmin, xmax, ymin, ymax, zmin, zmax = dataset.GetBounds()
    camera = vtk.vtkCamera()
    center = [(xmin+xmax)/2., (ymin+ymax)/2.]
    print(f'center={center}')
    camera.SetFocalPoint(center[0], center[1], zmax)
    alpha = camera.GetViewAngle()/180.*np.pi
    print(f'alpha in radians is {alpha}')
    window_width = window.GetSize()[0]
    print(f'window width = {window_width}')
    camera.SetPosition(center[0], center[1], window_width/alpha)
    print(f'camera position set to {[center[0], center[1], window_width/alpha]}')
    return camera


# for fully reproducible results, the window size is needed
def save_camera(camera=None, renderer=None, filename='camera.json'):
    if camera is None:
        if renderer is not None:
            camera = renderer.GetActiveCamera()
        else:
            raise ValueError('Missing camera input')
    pos =  camera.GetPosition()
    foc =  camera.GetFocalPoint()
    up =   camera.GetViewUp()
    clip = camera.GetClippingRange()
    angle = camera.GetViewAngle()
    cam = { 'position': pos, 'focal_point': foc, 'view_up': up, 'clipping_range': clip, 'angle': angle}

    if os.path.exists(filename):
        t = time.asctime(time.gmtime(time.time())).replace(' ', '_')
        basename, ext = os.path.splitext(filename)
        if not ext:
            ext = '.json'
        filename = f'{basename}_{t}{ext}'
    with open(filename, 'w') as output:
        json.dump(cam, output)
    print(f'saved camera in {filename}')

def load_camera(filename='camera.json'):
    with open(filename, 'r') as json_file:
        cam = json.load(json_file)
        camera = vtk.vtkCamera()
        camera.SetPosition(cam['position'])
        camera.SetFocalPoint(cam['focal_point'])
        camera.SetViewUp(cam['view_up'])
        camera.SetClippingRange(cam['clipping_range'])
        if 'angle' in cam.keys():
            camera.SetViewAngle(cam['angle'])
        return camera


def save_light(light=None, renderer=None, filename='light.json'):
    if light is None and renderer is None:
        raise ValueError('No light information provided')
    elif light is None:
        lc = renderer.GetLights()
        it = lc.NewIterator()
        if not it.IsDoneWithTraversal():
            light = it.GetNextItem()

    pos = light.GetPosition()
    foc = light.GetFocalPoint()
    angle = light.GetConeAngle()
    cola = light.GetAmbientColor()
    cold = light.GetDiffuseColor()
    cols = light.GetSpecularColor()
    intens = light.GetIntensity()
    lightdic = { 'position': pos, 'focal_point': foc, 'angle': angle, 'ambient_color': cola,
            'diffuse_color': cold, 'specular_color': cols, 'intensity': intens }
    if os.path.exists(filename):
        t = time.asctime(time.gmtime(time.time())).replace(' ', '_')
        basename, ext = os.path.splitext(filename)
        if not ext:
            ext = '.json'
        filename = f'{basename}_{t}{ext}'
    with open(filename, 'w') as output:
        json.dump(lightdic, output)
    print(f'saved light in {filename}')


def load_one_light(filename):
    with open(filename, 'r') as json_file:
        light_data = json.load(json_file)
        light = vtk.vtkLight()
        light.SetPosition(light_data['position'])
        light.SetFocalPoint(light_data['focal_point'])
        light.SetConeAngle(light_data['angle'])
        light.SetAmbientColor(light_data['ambient_color'])
        light.SetDiffuseColor(light_data['diffuse_color'])
        light.SetSpecularColor(light_data['specular_color'])
        light.SetIntensity(light_data['intensity'])
        light.PositionalOn()
        return light

def load_lights(filename='light.json'):
    collection = vtk.vtkLightCollection()
    if isinstance(filename, list):
        for name in filename:
            collection.AddItem(load_one_light(name))
    else:
        collection.AddItem(load_one_light(filename))
    return collection


def print_camera(camera=None, renderer=None):
    if camera is None:
        if renderer is not None:
            camera = renderer.GetActiveCamera()
        else:
            raise ValueError('Missing camera input')
    # ---------------------------------------------------------------
    # Print out the current settings of the camera
    # ---------------------------------------------------------------
    print('Camera settings:')
    print(f' * position:        {camera.GetPosition()}')
    print(f' * focal point:     {camera.GetFocalPoint()}')
    print(f' * up vector:       {camera.GetViewUp()}')
    print(f' * clipping range:  {camera.GetClippingRange()}')
    print(f' * view angle:      {camera.GetViewAngle()}')
