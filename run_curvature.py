import os, torch
import numpy as np
import surfa as sf

from curvature import MeanCurvature

###
fname = 'data/lh.pial'
mc = MeanCurvature(fname)

mc.input_meshdata = _mean_curvature(mc.input_meshdata)
