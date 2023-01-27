import torch
import h5py
import sys
import os

sys.path.append('../')
import utilities

def sizeof(t):
    n = 0
    if isinstance(t, list):
        for w in t:
            n += w.numel()
    elif isinstance(t, torch.Tensor):
        n = t.numel()
    elif isinstance(t, h5py.Dataset):
        n = t.size
    else:
        assert False
        print(f'Unrecognized object of type {type(t)}')
    return n

def shapeof(t):
    sh = []
    if isinstance(t, list):
        for w in t:
            sh.append(shapeof(w))
    else:
        sh.append([t.shape, sizeof(t), t.dtype])
    return sh

def get_loader(config, train=False, prefix=''):
    if train:
        what = 'train'
    else:
        what = 'test'
    c_data =config["data"]
    if prefix:
        path = os.path.join(prefix, c_data['PATH'])
    else:
        path = c_data['PATH']
    gl = utilities.GettingLists(data_for_training=c_data["n_sample"],
                                wave_eq = c_data["PDE_type"],
                                data_base = c_data["process"],
                                PATH = path)
    return utilities.MyLoader(GL=gl, do = what, config=config)
