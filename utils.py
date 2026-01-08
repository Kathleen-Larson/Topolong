import os, time
import numpy as np
import surfa as sf
import torch



def _dot(x, y, d=-1):
    """
    Calculates dot product of two tensors (because torch.dot only works with 1D tensors)
    """
    return torch.sum(x * y, dim=d).unsqueeze(d)



def _expand(x, unsq, rpts=None):
    """
    Custom function to expand the shape of a tensor (useful for matrix operations with large
    tensors without using any for loops). Mostly helps keep the code clean.
    """
    if rpts is not None: assert len(rpts) == len(x.shape) + len(unsq), \
       "In _expand(), len(rpts) must equal len(x.shape) + len(unsq)"
    for d in unsq: x = x.unsqueeze(d)
    return x if rpts is None else x.repeat(rpts)


def _norm(x, d):
    """
    Custom function to normalize a tensor without reducing its number of dimensions. Also mostly
    just helps to keep the code cleaner.
    """
    return torch.norm(x, dim=d).unsqueeze(d)
