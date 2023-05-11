"""
    Project a model or multiple models to a plane spaned by given directions.
"""

import numpy as np
import torch
import os
import copy
import h5py
import sys
import random

from projection_helper import sizeof, shapeof

sys.path.append('/Users/xmt/code/github/loss_landscape')
import net_plotter
import h5_util
import tqdm

from sklearn.decomposition import PCA, TruncatedSVD
from scipy.linalg import svd


def boolean_query(prompt):
    asw = input(prompt)
    asw = asw.lower()
    if asw=='y' or asw=='yes' or asw=='1' or asw=='t' or asw=='true':
        return True
    elif asw=='n' or asw=='no' or asw=='0' or asw=='f' or asw=='false':
        return False
    else:
        print('Warning: unrecognized answer. Assuming no.')
        return True


def tensorlist_to_tensor(weights):
    """ Concatenate a list of tensors into one tensor.

        Args:
            weights: a list of parameter tensors, e.g. net_plotter.get_weights(net).

        Returns:
            concatnated 1D tensor
    """
    return torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.FloatTensor(w) for w in weights])


def nplist_to_tensor(nplist):
    """ Concatenate a list of numpy vectors into one tensor.

        Args:
            nplist: a list of numpy vectors, e.g., direction loaded from h5 file.

        Returns:
            concatnated 1D tensor
    """
    v = []
    for d in nplist:
        w = torch.tensor(d*np.float64(1.0))
        # Ignoreing the scalar values (w.dim() = 0).
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)


def npvec_to_tensorlist(direction, params):
    """ Convert a numpy vector to a list of tensors with the same shape as "params".

        Args:
            direction: a list of numpy vectors, e.g., a direction loaded from h5 file.
            base: a list of parameter tensors from net

        Returns:
            a list of tensors with the same shape as base
    """
    if isinstance(params, list):
        w2 = copy.deepcopy(params)
        idx = 0
        n = 0
        for w in w2:
            n = n+w.numel()
            w.copy_(torch.tensor(direction[idx:idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert(idx == len(direction))
        return w2
    else:
        s2 = []
        idx = 0
        for (k, w) in params.items():
            s2.append(torch.Tensor(direction[idx:idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert(idx == len(direction))
        return s2


def cal_angle(vec1, vec2):
    """ Calculate cosine similarities between two torch tensors or two ndarraies
        Args:
            vec1, vec2: two tensors or numpy ndarraies
    """
    if isinstance(vec1, torch.Tensor) and isinstance(vec1, torch.Tensor):
        return torch.dot(vec1, vec2)/(vec1.norm()*vec2.norm()).item()
    elif isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
        return np.ndarray.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def project_1D(w, d):
    """ Project vector w to vector d and get the length of the projection.

        Args:
            w: vectorized weights
            d: vectorized direction

        Returns:
            the projection scalar
    """
    assert len(w) == len(d), 'dimension does not match for w (' + str(len(w)) + ') and d (' + str(len(d)) + ')'
    scale = torch.dot(w, d)/d.norm()
    return scale.item()


def lift_1D(x, d):
    return x*d/d.norm()


def project_2D(d, dx, dy, proj_method):
    """ Project vector d to the plane spanned by dx and dy.

        Args:
            d: vectorized weights
            dx: vectorized direction
            dy: vectorized direction
            proj_method: projection method
        Returns:
            x, y: the projection coordinates
    """
    if proj_method == 'cos':
        # when dx and dy are orthorgonal
        x = project_1D(d, dx)
        y = project_1D(d, dy)
    elif proj_method == 'lstsq':
        # solve the least squre problem: Ax = d
        A = np.vstack([dx.numpy(), dy.numpy()]).T
        [x, y] = np.linalg.lstsq(A, d.numpy())[0]

    return x, y


def project_3D(d, dx, dy, dz, proj_method):
    """ Project vector d to the 3D space spanned by dx, dy, and dz.

        Args:
            d: vectorized weights
            dx: vectorized direction
            dy: vectorized direction
            dz: vectorized direction
            proj_method: projection method
        Returns:
            x, y, z: the projection coordinates
    """
    if proj_method == 'cos':
        # when dx and dy are orthorgonal
        x = project_1D(d, dx)
        y = project_1D(d, dy)
        z = project_1D(d, dz)
    elif proj_method == 'lstsq':
        # solve the least squre problem: Ax = d
        A = np.vstack([dx.numpy(), dy.numpy(), dz.numpy()]).T
        [x, y, z] = np.linalg.lstsq(A, d.numpy())[0]

    return x, y, z


def project_ND(d, axes, proj_method):
    """ Project vector d to the space spanned by all axes.

        Args:
            d: vectorized weights
            axes[0, ...]: vectorized direction
            proj_method: projection method
        Returns:
            [x, y, z, ...]: the projection coordinates
    """
    ndim = len(axes)
    coords = []
    if proj_method == 'cos':
        # when dx and dy are orthorgonal
        for n, axis in enumerate(axes):
            coords.append(project_1D(d, axis))
    elif proj_method == 'lstsq':
        # solve the least squre problem: Ax = d
        A = np.vstack([axis.numpy() for axis in axes]).T
        coords = np.linalg.lstsq(A, d.numpy())[0]

    return coords


def lift_ND(coords, axes, proj_method):
    """ Lift coordinates to the embedding space.

        Args:
            coords: PCA coordinates
            axes[0, ...]: basis vectors of PCA space
            proj_method: projection method
        Returns:
            t: vectorized weight difference
    """
    ndim = len(axes)
    assert (ndim == len(coords))
    t = torch.zeros_like(axes[0])
    for i, x in enumerate(coords):
        t += lift_1D(x, axes[i])
    return t

def load_all_directions(dir_file):
    """ Load direction(s) from the direction file."""
    directions = []
    f = h5py.File(dir_file, 'r')
    lastdim = 0
    while True:
        label = 'direction_{}'.format(lastdim)
        if label in f.keys():
            directions.append(h5_util.read_list(f, label))
            lastdim += 1
        else:
            break

    print(f'directions contain {len(directions)} vectors')
    return directions

def project_trajectory(args, w, s, callback, model_name=None):
    """
        Project the optimization trajectory onto the given two directions.

        Args:
          args.dir_file: the h5 file that contains the directions
          w: weights of the final model
          s: states of the final model
          model_name: the name of the model
          args.steps: the list of available checkpoint indices
          args.dir_type: the type of the direction, weights or states
          args.proj_method: cosine projection
          args.dimension: 2, 3 or higher dimensional plot
          callback: method to obtain model from step index

        Returns:
          proj_file: the projection filename
    """
    if model_name is not None:
        proj_file = args.dir_file + '_' + model_name
    proj_file += '_proj_' + args.proj_method + '.h5'
    if os.path.exists(proj_file):
        replace = input('The projection file exists! Replace?')
        if replace:
            os.remove(proj_file)
        else:
            return proj_file

    # read directions and convert them to vectors
    directions = load_all_directions(args.dir_file)
    axes = []
    for d in directions:
        axes.append(nplist_to_tensor(d))

    print(f'directions contains {len(directions)} axes')
    ndim = len(directions)

    refw = w
    w = transform_tensors(w, args.complex)
    if args.complex == 'imaginary' or args.complex == 'real':
        debug = False

    allcoords = [ [] for i in range(ndim) ]
    other_coords = [ [] for i in range(ndim) ]
    errors = []

    pbar = tqdm.tqdm(args.steps, desc='Projecting learning steps', ncols=100)
    for step in pbar:
        net2 = callback(step)
        if args.dir_type == 'weights':
            w2 = net_plotter.get_weights(net2)
            w2 = transform_tensors(w2, args.complex)
            d = net_plotter.get_diff_weights(w, w2)
        elif args.dir_type == 'states':
            s2 = net2.state_dict()
            d = net_plotter.get_diff_states(s, s2)
        d = tensorlist_to_tensor(d)

        coords = project_ND(d, axes, args.proj_method)

        for i in range(ndim):
            allcoords[i].append(coords[i])

    skip = False
    if os.path.exists(proj_file):
        skip = boolean_query(f'{proj_file} exists already. Replace? ')
        if not skip:
            os.remove(proj_file)

    if not skip:
        f = h5py.File(proj_file, 'w')
        for i in range(ndim):
            label = 'proj_{:0>2d}coord'.format(i)
            f[label] = np.array(allcoords[i])
        f.close()

    return proj_file


def real_type(w):
    if w.dtype is torch.complex64:
        return torch.float32
    elif w.dtype is torch.complex128:
        return torch.float64
    else:
        return w.dtype


def from_values(t, start, length, wref):
    return t[start:start+length].view(wref.shape).view(real_type(wref))


def untransform_tensors(w, refw, what):
    weights = []
    if what.lower() == 'split':
        with torch.no_grad():
            for wi, refwi in zip(w, refw):
                nrows = wi.shape[0]
                realnrows = int(nrows/2)
                nrows = realnrows
                if refwi.dtype is torch.float32 or refwi.dtype is torch.float64:
                    # real tensor was padded with as many zeros to signify
                    weights.append(wi[0:nrows].view(refwi.dtype))
                elif refwi.dtype is torch.complex64 or refwi.dtype is torch.complex128:
                    # complex tensor was converted to real followed by imaginary values:
                    re = wi[0:nrows]
                    im = wi[nrows:]
                    weights.append(torch.complex(re, im))
                else:
                    raise ValueError('Unrecognized data type for this weight: {}'.format(w.dtype))
    elif what.lower() == 'real' or what.lower() == 'skip' or what.lower() == 'ignore':
        with torch.no_grad():
            for wi, refwi in zip(w, refw):
                if refwi.dtype is torch.float32 or refwi.dtype is torch.float64:
                    weights.append(wi)
                elif refwi.dtype is torch.complex64 or refwi.dtype is torch.complex128:
                    re = wi
                    im = torch.zeros_like(re)
                    weights.append(torch.complex(re, im))
                else:
                    raise ValueError('Unrecognized data type for this weight: {}'.format(w.dtype))
    elif what.lower() == 'imaginary':
        at = 0
        with torch.no_grad():
            for wi, refwi in zip(w, refw):
                if refwi.dtype is torch.float32 or refwi.dtype is torch.float64:
                    weights.append(torch.zeros_like(refwi)) # real values were discarded
                elif refwi.dtype is torch.complex64 or refwi.dtype is torch.complex128:
                    im = wi
                    re = torch.zeros_like(im)
                    weights.append(torch.complex(re, im))
                else:
                    raise ValueError('Unrecognized data type for this weight: {}'.format(w.dtype))
    else:
        raise ValueError('Unrecognized complex flattening name')
    return weights


def pca_coords_to_weights(coords, axes, refw, what):
    '''
    Transform coordinates in PCA space to weights of model

    coords: coordinates in PCA space
    axes: PCA axes (as vectors)
    refw: weights of the final model used as origin of the reference frame
    '''
    assert(len(coords) == len(axes))
    t = torch.zeros_like(axes[0])
    # t: lifted version of coords in transformed and flattened space
    for c,a in zip(coords, axes):
        t += c*a
    # w0: list of transformed reference weights
    w0 = transform_tensors(refw, what)
    # w: list of weight differences between transformed model and transformed ref model
    w = npvec_to_tensorlist(t, w0)
    # w1: list of transformed model weights
    w1 = []
    for wi, w0i in zip(w,w0):
        w1.append(wi + w0i)
    # w2: list of untransformed weights
    w2 = untransform_tensors(w1, refw, what)
    return w2

def transform_tensor(t, what, verbose=False):
    if not isinstance(t, torch.Tensor):
        print(f'WARNING: not a tensor type in transform_tensor ({type(t)})')
        print(f'size of list: {len(t)}, shape = {shapeof(t)}')
        assert False
        return

    if what.lower() == 'imaginary':
        if not torch.is_complex(t):
            return None
        else:
            return t.imag
    elif what.lower() == 'split':
        if not torch.is_complex(t):
            return torch.cat((t, torch.zeros_like(t)), dim=0)
        else:
            return torch.cat((t.real, t.imag), dim=0)
    elif what.lower() == 'ignore' or what.lower() == 'real':
        if not torch.is_complex(t):
            return t
        else:
            return t.real


def transform_tensors(t, what, verbose=False):
    if verbose:
        print(f'entering transform_tensor: what={what}')
    if what.lower() == 'keep':
        if verbose:
            print('leaving tensor (list) unchanged')
        return t
    elif isinstance(t, list):
        t1 = []
        for w in t:
            w2 = transform_tensor(w, what, verbose)
            if w2 is not None:
                if verbose:
                    print('w2 is not None, size={}'.format(w2.numel()))
                t1.append(w2)
        return t1
    else:
        return transform_tensor(t, what, verbose)


def setup_PCA_directions(args, callback, w, s, verbose=False, filename=None):
    """
        Find PCA directions for the optimization path from the initial model
        to the final trained model.

        Returns:
            dir_name: the h5 file that stores the directions.
    """

    if verbose:
        print(f'input tensor w contains {sizeof(w)} values and has shape {shapeof(w)}')

    actual_dim = np.min([args.dimension, len(args.steps)])
    if actual_dim != args.dimension:
        print(f'WARNING: unable to compute {args.dimension} PCA dimensions. Only {actual_dim} will be computed')
        args.dimension = actual_dim

    # Name the .h5 file that stores the PCA directions.
    folder_name = args.path + '/PCA_' + args.dir_type
    if args.ignore:
        folder_name += '_ignore=' + args.ignore
    folder_name += '_save_epoch=' + str(args.steps[-1])
    folder_name += '_complex=' + str(args.complex)
    folder_name += '_dim=' + str(args.dimension)
    os.system('mkdir ' + folder_name)
    if filename is not None:
        prefix = filename + '_'
    else:
        prefix = ''
    dir_name = os.path.join(folder_name, prefix + 'directions.h5')
    if verbose:
        print(f'PCA directions computed from learning path will be stored in {dir_name}')

    # skip if the direction file exists
    if os.path.exists(dir_name):
        f = h5py.File(dir_name, 'a')
        if 'explained_variance_' in f.keys():
            f.close()
            return dir_name

    # load models and prepare the optimization path matrix
    matrix = []
    wsave = w
    # we will work with the transformed (real) version of the models
    w = transform_tensors(w, args.complex)

    pbar = tqdm.tqdm(args.steps, ncols=100, desc='Loading training steps')
    for step in pbar:
        pbar.set_description('step #{}'.format(step))
        net2 = callback(step)
        if args.dir_type == 'weights':
            w2 = net_plotter.get_weights(net2)
            display = random.random() < 0.1
            if verbose:
                print('transforming tensor {}'.format(shapeof(w2)))
            # compute real version of the weights
            w2 = transform_tensors(w2, args.complex)
            if verbose:
                print('into tensor {}'.format(shapeof(w2)))
            d = net_plotter.get_diff_weights(w, w2)
        elif args.dir_type == 'states':
            s2 = net2.state_dict()
            d = net_plotter.get_diff_states(s, s2)
        if args.ignore == 'biasbn':
            net_plotter.ignore_biasbn(d)
        d = tensorlist_to_tensor(d)
        if verbose:
            print('converting that tensor into {}'.format(shapeof(d)))
        if d is not None:
            matrix.append(d.numpy())

    # Perform PCA on the optimization path matrix
    if verbose:
        print ("Perform PCA on the models")
    matrix = np.array(matrix)
    if verbose:
        print(matrix.shape)

    A = torch.from_numpy(matrix)
    _U, _S, _V = torch.pca_lowrank(A, q=args.dimension, center=True)
    covar = torch.square(_S)/(len(args.steps)-1)
    principal_directions = _V.numpy()
    pcs = []
    for i in range(principal_directions.shape[1]):
        pcs.append(np.array(principal_directions[:,i]))
    if verbose:
        print(f'there are {len(pcs)} principal components')


    # convert vectorized directions to the same shape as models to save in h5 file.
    if verbose:
        print(f'type of w is {type(w)}')
    xi_directions = []
    if args.dir_type == 'weights':
        for pc in pcs:
            xi_directions.append(npvec_to_tensorlist(pc, w))
    elif args.dir_type == 'states':
        for pc in pcs:
            xi_directions.append(npvec_to_tensorlist(pc, s))

    if args.ignore == 'biasbn':
        for xd in xi_directions:
            net_plotter.ignore_biasbn(xd)

    if verbose:
        print(f'dir_name={dir_name}')
    if os.path.exists(dir_name):
        replace = boolean_query(f'{dir_name} exists already. Replace? ')
        if replace:
            os.remove(dir_name)
        else:
            return dir_name

    f = h5py.File(dir_name, 'w')
    for i, xd in enumerate(xi_directions):
        label = 'direction_{}'.format(i)
        h5_util.write_list(f, label, xd)
    f['singular_values_'] = _S
    f['covariance_values'] = covar
    f.close()
    if verbose:
        print ('transformed PCA directions saved in: %s' % dir_name)

    complexdir_name = dir_name[:-4] + '_complex.h5'
    f = h5py.File(complexdir_name, 'w')
    for i, xd in enumerate(xi_directions):
        label = 'direction_{}'.format(i)
        x = untransform_tensors(xd, wsave, args.complex)
        if verbose:
            print(f'after untransformation:\n\tx={shapeof(x)}\n\treference={shapeof(wsave)}')
        h5_util.write_list(f, label, x)
    f.close()

    return dir_name
