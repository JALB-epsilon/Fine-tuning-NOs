import vtk
import os

'''
Misc helper functions
'''

def is_algorithm(object):
    return isinstance(object, vtk.vtkAlgorithm)

def is_dataset(object):
    return isinstance(object, vtk.vtkDataSet)

def connect(input, output):
    if is_algorithm(input) and is_algorithm(output):
        output.SetInputConnection(input.GetOutputPort())
    elif is_dataset(input) and is_algorithm(output):
        output.SetInputData(input)
    else:
        raise TypeError(f'Invalid types {type(input)} / {type(output)} in connect')
