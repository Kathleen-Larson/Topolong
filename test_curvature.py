import os, time
import numpy as np
import surfa as sf
import torch

from trianglemesh import TriangleMesh as tmesh
from curvature import MeanCurvature as mc
from optimizer import MeshLBFGS
from utils import _expand, _norm


# --------------------------------------------------------------------------------------------------

# Initialize mesh classes
mesh_targ = tmesh(inpath='data/lh.sphere.ico2.reg.clean')
mesh_noisy = tmesh(inpath='data/lh.sphere.ico2.reg.noisy')

mc_targ = mc(mesh_targ, device='gpu')
mc_noisy = mc(mesh_noisy, device='gpu')

breakpoint()
# Run optimizer
opt = MeshLBFGS(moving=mc_noisy, target=mc_targ, device='gpu')
coords_noisy = mesh_noisy.verts.clone()
coords_smoothed = opt.optimize(coords_noisy)

breakpoint()
# Print results
rad_new = _norm(coords_smoothed, -1)
rad_targ = _norm(mesh_targ.verts, -1)

print('Optimization results:')
print(f'  Average radius: {rad_new.mean():.4f} (target: {rad_targ.mean()})')
print(f'  Radius std dev: {rad_new.std():.4f}')
print(f'  Min radius: {rad_new.min():.4f}')
print(f'  Max radius: {rad_new.max():.4f}')

mesh_noisy.verts = coords_smoothed
mesh_noisy._write(outpath='data/lh.sphere.ico2.reg.noisy.smoothed')
