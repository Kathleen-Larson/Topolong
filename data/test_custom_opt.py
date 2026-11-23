import os, time
import numpy as np
import surfa as sf
import torch
import time

import utils

from tensor_mesh import TensorMesh
from tensor_image import TensorImage
from vertex_optimizer import MRISPlaceSurface

# --------------------------------------------------------------------------------------------------
"""
Right now this is attempting to replicate lh.orig-->lh.white.preaparc (mris_place_surface)
"""

device=torch.device('cuda')

# Optimizer input params
lr = 1.
n_iters = 50
max_disp = 0.3
momentum = 0.5

L_crv = 1.0
L_int = 0.2
L_spr = 0.3
L_wsf = 5.0 #5.0
L_bsf = 5.0 #5.0

# Load data
data_dir = 'data/'

image_fname = 'data/brain.finalsurfs.mgz'
aseg_fname = 'data/aseg.presurf.mgz'
wm_fname = 'data/wm.mgz'
lut_fname = 'data/ASegStatsLUT.txt'
mesh_moving_fname = 'data/lh.orig'

# Run optimizer
opt = MRISPlaceSurface(
    insurf=mesh_moving_fname,
    image=image_fname,
    mesh_repulse=mesh_repulse_fname,
    seg=aseg_fname,
    lut=lut_fname,
    surf='pial',
    l_curvature=L_crv,
    l_intensity=L_int,
    l_nspring=L_spr,
    l_tspring=L_spr,
    l_within_surf=L_wsf,
    l_between_surf=L_bsf,
    use_ggvf=True,
    device=device,
    verbose=True
)

print('Initialization complete')
t = time.time()

#opt.lbfgs_optimize(max_steps=n_iters, max_iters=10, line_search_fn='strong_wolfe')
opt.sgd_optimize(max_steps=n_iters, lr=lr, max_disp=max_disp, momentum=momentum)

print(f'optimization time: {(time.time() - t):>.0f} s')

mesh_opt = opt.mesh

# Save results
output_fname = (
    f'data/lh.pial.n_iters_{n_iters}.sgd' +
    f'int_{L_int:.0e}.curv_{L_crv:.0e}.spring_{L_spr:.0e}.vrep_{L_wsf:.0e}.srep_{L_wsf:.0e}'
)

mesh_opt.save(output_fname)

print(f'image: {image_fname}')
print(f'input mesh: {mesh_moving_fname}')
print(f'output mesh: {output_fname}')
