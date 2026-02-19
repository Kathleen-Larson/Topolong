import os, time
import numpy as np
import surfa as sf
import torch

from tensor_mesh import TensorMesh
from tensor_image import TensorImage
from optimizer import LBFGS
from utils import _expand, _norm


def curvature_loss(xyz, weight=1.):
    sq = (tmesh_moving._mean_curvature(xyz) - f_targ) ** 2
    return weight * sq.sum()

def intensity_loss(xyz, weight=1.):
    target_intensity = 64
    sq = 0
    for coord in xyz:
        sq += (timage._interp(coord) - target_intensity) ** 2
    return weight * sq.sum()

def optimize(X, lambda_curv=1., lambda_int=1., history_size=10, max_iter=10, max_steps=100):
    def closure():
        opt.zero_grad()
        obj = curvature_loss(X)
        #obj = curvature_loss(X, lambda_curv) + intensity_loss(X, lambda_int)
        #print(f'Loss = {obj.item()}')
        obj.backward()
        return obj

    X.requires_grad = True

    opt = torch.optim.LBFGS(
        [X], history_size=history_size, max_iter=max_iter, line_search_fn="strong_wolfe"
    )
    for i in range(max_steps):
        print(f'Iteration {i}')
        opt.step(closure)
    return X


# TO ADD:  intensity, then self intersection constraints

# how it actually works:
# compute normal, compute intensity profile along normal, take max negative gradient along that

# --------------------------------------------------------------------------------------------------

device=torch.device('cuda')

# Load data
image = sf.load_volume('data/sphere.ico4.image.noise_free.mgz')
timage = TensorImage(image, device=device)

mesh_moving = sf.load_mesh('data/sphere.ico4.outer.noisy')
tmesh_moving = TensorMesh(mesh_moving, device=device)

mesh_target = sf.load_mesh('data/sphere.ico4.outer')
tmesh_target = TensorMesh(mesh_target, device='gpu')
f_targ = tmesh_target._mean_curvature()

X_opt = optimize(tmesh_moving.verts, history_size=10, max_iter=10, max_steps=100)

# Print results
mesh_smoothed = mesh_moving.copy()
mesh_smoothed.vertices = X_opt.detach().cpu()

rad_old = np.linalg.norm(mesh_moving.vertices, axis=-1)
rad_new = np.linalg.norm(mesh_smoothed.vertices, axis=-1)

print('Optimization results:')
print(f'  Initial radius: {rad_old.mean():.4f} +- {rad_old.std():.4f}')
print(f'  Optimized radius: {rad_new.mean():.4f} +- {rad_new.std():.4f}')

mesh_smoothed.save('data/sphere.ico4.outer.noisy.smoothed_2')

breakpoint()
