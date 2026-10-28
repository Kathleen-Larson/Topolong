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
n_iters = 200

L_crv = 1.0
L_int = 0.2
L_spr = 0.3
L_srp = 5.0
L_vrp = 5.0

v_prox = 0.05
s_prox = 0.05 #0.05

borders_dict = {
    'border_hi': 67.4082,
    'border_lo': 44.2246,
    'outside_hi': 61.6123,
    'outside_lo': 10,
    'inside_hi': 104.144,
}

# Load data
data_dir = 'data/'

image_fname = 'data/brain.finalsurfs.mgz'
image = sf.load_volume(image_fname)
timage = TensorImage(image)

aseg_fname = 'data/aseg.presurf.mgz'
aseg = sf.load_volume(aseg_fname)

mesh_moving_fname = 'data/lh.orig'
mesh_moving = sf.load_mesh(mesh_moving_fname)

mesh_repulse = mesh_moving.copy()

lut_fname = 'data/ASegStatsLUT.txt'
lut = sf.load_label_lookup(lut_fname)
breakpoint()
# Run optimizer
opt = MRISPlaceSurface(
    in_pial=mesh_moving,
    image=image,
    seg=aseg,
    wm=wm,
    mesh_repulse=mesh_repulse,
    lut=lut,
    surf='pial',
    l_curvature=L_crv,
    l_intensity=L_int,
    l_nspring=L_spr,
    l_tspring=L_spr,
    l_vert_repulse=L_vrp,
    l_surf_repulse=L_srp,    
    borders_dict=borders_dict,
    use_ggvf=True,
    vert_prox_ratio=v_prox,
    surf_prox_ratio=s_prox,
    separate_loss_types=True,
    adaptive=False,
    verbose=True
)

print('Initialization complete')
t = time.time()

opt.lbfgs_optimize(max_steps=n_iters, max_iters=10, line_search_fn='strong_wolfe')
print(f'optimization time: {(time.time() - t):>.0f} s')

mesh_opt = opt.mesh

valid = ~opt.rip_verts_flag.cpu()
disp = np.sqrt(np.sum((mesh_moving.vertices - mesh_opt.vertices) ** 2, axis=1))[valid]

print(f'mean displacement: {disp.mean():.3f} +- {disp.std():.3f} mm')
print(f'max displacement: {disp.max():.3f} mm')

# Save results
output_fname = (
    f'data/lh.pial.n_iters_{n_iters}.lr_{lr:.0e}.' +
    f'int_{L_int:.0e}.curv_{L_crv:.0e}.spring_{L_spr:.0e}.vrep_{L_vrp:.0e}.srep_{L_srp:.0e}.' +
    f'vert_prox_ratio_{v_prox:.0e}.surf_prox_ratio_{s_prox:.0e}.old_opt_test'
)

mesh_opt.save(output_fname)

print(f'image: {image_fname}')
print(f'input mesh: {mesh_moving_fname}')
print(f'output mesh: {output_fname}')
