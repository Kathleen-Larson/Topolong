import os
import time
import sys

import numpy as np
import surfa as sf

import torch
import torch.nn as nn

from itertools import permutations

# Math
def _dot(x, y, dim=-1):
    """
    Calculates dot product of two tensors (because torch.dot only works with 1D tensors)
    """
    return torch.sum(x * y, dim=dim).unsqueeze(dim)



def _expand(x, unsqueeze_dims, repeats=None):
    """
    Custom function to expand the shape of a tensor (useful for matrix operations with large
    tensors without using any for loops). Mostly helps keep the code clean.
    """
    if repeats is not None:
        assert (
            len(repeats) == len(x.shape) + len(unsqueeze_dims),
            "In _expand(), len(repeats) must equal len(x.shape) + len(unsqueeze_dims)"
        )
    for dim in unsqueeze_dims:
        x = x.unsqueeze(dim)
    return x if repeats is None else x.repeat(repeats)


def _norm(x, dim):
    """
    Custom function to normalize a tensor without reducing its number of dimensions. Also mostly
    just helps to keep the code cleaner.
    """
    return torch.norm(x, dim=dim).unsqueeze(dim)


# --------------------------------------------------------------------------------------------------
"""
Filters
"""

def init_convolution(in_shape, out_shape, conv_weight_data, kernel_size=3, stride=1, padding=1,
                     dilation=1, device=None):
    in_channels = in_shape[1]
    out_channels = out_shape[1]
    device = 'cpu' if device is None else device

    while len(conv_weight_data.shape) < len(in_shape):
        conv_weight_data = conv_weight_data.unsqueeze(dim=0)

    conv_fn = nn.Conv3d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        dilation=dilation, bias=False, device=device,
    )
    conv_fn.weight.data = conv_weight_data.to(torch.float).to(device)
    conv_fn.weight.requires_grad = False
    return conv_fn


def apply_convolution(data, conv):
    """
    Applies convolution-based function to image data
    """
    dtype = data.dtype
    shape = data.shape
    
    while len(data.shape) < len(conv.weight.data.shape):
        data = data.unsqueeze(dim=0)

    data = conv(data.to(conv.weight.data.dtype)).to(dtype)

    while len(data.shape) > len(shape):
        data = data.squeeze(dim=0)

    return data


def write_image(data, fname):
    img = sf.Volume(data.detach().cpu().numpy())
    img.save(fname)
    

# --------------------------------------------------------------------------------------------------
"""
Misc
"""

def GLM(X, y, compute_residuals=False):
    Xt = X.transpose(-2, -1)
    B = torch.linalg.inv(Xt @ X) @ Xt @ y
    if compute_residuals:
        yhat = X @ B
        res = y - yhat
        return yhat, res
    else:
        
        return torch.linalg.inv(Xt @ X) @ Xt @ y

def search_lut(lut, names):
    if names is None:
        return []
    
    names = [names] if isinstance(names, str) else names
    return [key for key, val in lut.items() for string in names if string in val.name]


# --------------------------------------------------------------------------------------------------
"""
System
"""

def check_tensor(x, dtype=None, ndim=None, shape=None):
    """
    Corrects input to specific criteria or throws expection if impossible (adapted from 
    surfa.array.check_array)
    """
    def list_string(lst):
        if len(lst) == 1:
            return str(lst[0])
        elif len(lst) == 2:
            return ' or '.join(map(str, lst))
        else:
            return ', '.join(map(str, lst[:-1])) + ', or ' + str(lst[-1])

    # Check number of dimensions:
    if ndim is not None:
        if x.ndim != ndim and x.squeeze().ndim == ndim:
            x = x.squeeze()

    # Check shape (e.g. each dimension)
    if shape is not None:
        if np.isscalar(shape):
            shape = [shape] * ndim
        if len(shape) != ndim:
            fatal('Error: if specifying shape in check_tensor, length of shape must equal ndim')

        shape = tuple(shape)
        if tuple(x.shape) != shape:
            for perm in permutations(reversed(range(ndim))):
                is_match = [True] * ndim
                for n, (d1, d2) in enumerate(zip(x.permute(perm).shape, shape)):
                    if d2 > -1 and d1 != d2:
                        is_match[n] = False
                if np.all(is_match):
                    x = x.permute(perm)
                    break
            if not np.all(is_match):
                fatal(f'Error: mismatch between required shape ({shape}) and input shape ' +
                      f'({x.shape})')

    # Check data type
    if dtype is not None:
        try:
            x = x.to(dtype)
        except RuntimeError:
            fatal(f'Error casting tensor to desired datatype (requested {dtype})')
    return x

    
def fatal(message):
    print(message)
    sys.exit(1)
