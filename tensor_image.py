import os
import surfa as sf
import numpy as np
import time

import torch
import torch.nn.functional as F

from scipy.spatial import cKDTree

import utils


#---------------------------------------------------------------------------------------------------

class TensorImage:
    def __init__(self,
                 image,
                 ref_image=None,
                 dtype='float',
                 device=None,
                 is_talairach=True,
                 compute_gradient=False,
                 do_smoothing=False,
                 smoothing_sigma=1.
    ):
        self.device = torch.device(
            'cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu'
        ) if not isinstance(device, torch.device) else device
        
        # Image data
        if isinstance(image, sf.Volume):
            self.image = image.copy()
            self.data = torch.tensor(
                self.image.data.astype(np.int32 if dtype=='int' else np.float32),
                dtype=(torch.int if dtype=='int' else torch.float)
            ).to(self.device)

            ref_image = self.image

        elif torch.is_tensor(image):
            self.data = image.to(torch.int if dtype=='int' else torch.float)
            if ref_image is None:
                utils.fatal('Error in TensorImage: ref_image required if input is tensor')

            ref_image = ref_image.image if isinstance(ref_image, TensorImage) else ref_image.copy()
            self.image = ref_image.copy()
            self.image.data = image.cpu().numpy()
                
        self.ndims = 3
        self.dtype = self.data.dtype
        
        # Spatial info
        self.voxsize = torch.tensor(ref_image.geom.voxsize, dtype=torch.float, device=self.device)
        self.shape = torch.tensor(ref_image.geom.shape, dtype=torch.float, device=self.device)

        self.transform_dict = {}
        self.transform_dict['vox2world'] = {
            'start': torch.tensor(ref_image.geom.vox2world[:3, -1]).to(self.device),
            'rotation': torch.tensor(ref_image.geom.vox2world[:3, :3]).to(self.device)
        }
        self.transform_dict['world2vox'] = {
            'start': torch.tensor(ref_image.geom.vox2world[:3, -1]).to(self.device),
            'rotation': torch.tensor(ref_image.geom.world2vox[:3, :3]).to(self.device)
        }
        self.transform_dict['vox2surf'] = {
            'start': torch.tensor(ref_image.geom.vox2surf[:3, -1]).to(self.device),
            'rotation': torch.tensor(ref_image.geom.vox2surf[:3, :3]).to(self.device)
        }
        self.transform_dict['surf2vox'] = {
            'start': torch.tensor(ref_image.geom.vox2surf[:3, -1]).to(self.device),
            'rotation': torch.tensor(ref_image.geom.surf2vox[:3, :3]).to(self.device)
        }

        for geom, dct in self.transform_dict.items():
            for key, val in dct.items():
                self.transform_dict[geom][key] = val.to(self.device)

    def _index2sub(self, idxs):
        """
        Convert linear index to subscript
        """
        H, W, D = self.geom.shape
        i = idxs // (D * W)
        j = (idxs // D) % W
        k = idxs % D
        return i, j, k

    def _sub2idx(self, i, j, k):
        """
        Convert subscript to linear index
        """
        H, W, D = self.geom.shape
        return i * (D * W) + j * D + k

    def interpolate(self, coords, data_type='data', mode='bilinear', geom='surf2vox'):
        """
        Interpolate image value at real-valued coordinates. Coords should be a single coordinate or 
        a [N x 3] tensor. "mode" specifies the interpolation mode for grid_sample().
        """
        if len(coords.shape) == 1:
            coords = coords.unsqueeze(0)
        N = coords.shape[-2]

        # Convert coordinates to image space and normalize for grid sampling
        grid = self.transform(coords.reshape(-1, 3), geom=geom)
        grid = _unsqueeze(((2 * grid / (self.shape - 1)) - 1), dim=0, N=3).flip(dims=(-1,))
        
        # Sample at normalized coords
        data = _unsqueeze(self.data, dim=0, N=(5 - len(self.data.shape)))
        values = F.grid_sample(
            data if self.data.is_floating_point() else data.float(),
            grid.float(), mode=mode, padding_mode='border', align_corners=True
        ).reshape(-1, N).to(self.dtype)
        
        return values

    def _interp_derivative(self, coords, dirs, sigma=None, mode='bilinear', flip_sign=False,
                           geom='surf2vox'):
        """
        Doing the same thing that MRIsampleVolumeDerivativeScale does in utils/mri.cpp
        """        
        if len(coords.shape) == 1:
            coords = coords.unsqueeze(0)
        N = coords.shape[-2]

        # Create vectors of coordinates to smooth over
        step_sz = 0.25 if sigma is None else max(0.25, float(sigma) / 5.) # surface units, NOT voxel
        max_dist = step_sz if sigma is None else max(2. * float(sigma), step_sz)
        K = int(max_dist / step_sz)

        arr = step_sz * torch.arange(1, K + 1, dtype=coords.dtype, device=coords.device)
        pts = coords.unsqueeze(dim=2) + (
            torch.cat([-arr.flip(dims=(0,)), arr]).view(1, 1, -1, 1) * dirs.view(-1, 1, 1, 3)
        )

        # Convert coordinates to image space and normalize for grid sampling
        grid = self.transform(pts.reshape(-1, 3), geom=geom)
        grid = ((2 * grid / (self.shape - 1)) - 1).view(1, 1, 1, -1, 3).flip(dims=(-1,))

        # Interpolate
        data = _unsqueeze(self.data, dim=0, N=(5 - len(self.data.shape)))
        values = F.grid_sample(
            data if data.is_floating_point() else data.float(),
            grid.float(), mode=mode, padding_mode='border', align_corners=True
        ).reshape(-1, N, 2 * K)

        # Combine
        k = (
            torch.ones_like(arr) if sigma is None
            else torch.exp(-arr.pow(2) / (2. * sigma * sigma))
        )
        grad = (
            (k * values[..., K:]).sum(dim=-1) - (k * values[..., :K].flip(dims=(-1,))).sum(dim=-1)
        ) / (2.0 * arr.sum())

        if flip_sign:
            grad = -grad

        return grad

    def gradient_image(self, img=None, step_sz=None, order=1):
        """
        Computes the gradient vector image w/ central differences
        """
        img = F.pad(
            (self.data if img is None else img)[None, None].float(), (1,) * 6, mode='replicate'
        )[0, 0]

        grad = torch.stack([
            0.5 * (img[2:, 1:-1, 1:-1] - img[:-2, 1:-1, 1:-1]),
            0.5 * (img[1:-1, 2:, 1:-1] - img[1:-1, :-2, 1:-1]),
            0.5 * (img[1:-1, 1:-1, 2:] - img[1:-1, 1:-1, :-2]),
        ], dim=-1)
        return grad
        
    def smooth(self, img=None, sigma=1.):
        """
        Applies Gaussian smoothing to image data w/ specified std
        """
        img = self.data if img is None else img
        
        # Initialize smoothing kernel
        window = np.round(sigma) * 2 + 1
        center = (window - 1) / 2
        mesh = [(-0.5 * pow(torch.arange(window) - center, 2))] * 3
        mesh = torch.stack(torch.meshgrid(*mesh, indexing='ij'), dim=-1)
        kernel = (
            (1 / pow(2 * torch.pi * sigma**2, 1.5))
            * torch.exp(-(pow(mesh, 2).sum(dim=-1)) / (2 * sigma ** 2))
        )
        kernel /= kernel.sum()

        # Generate smoothing filter
        smooth_filter = utils.init_convolution(
            in_shape=img.unsqueeze(dim=0).unsqueeze(dim=0).shape,
            out_shape=img.unsqueeze(dim=0).unsqueeze(dim=0).shape,
            conv_weight_data=kernel,
            kernel_size=kernel.shape,
            padding=[(k - 1) // 2 for k in kernel.shape],
            stride=1,
            dilation=1,
            device=self.data.device
        )

        # Apply
        vol = smooth_filter(self.data.unsqueeze(dim=0).unsqueeze(dim=0)).squeeze()
        return vol

    def transform(self, coords, geom):
        """
        Transform a set of coordinates to image space based on the specified geometry
        (e.g., vox2world)
        """
        if geom is None:
            utils.fatal('Error: must provide valid geometry identifier for TensorImage.transform')

        T = self.transform_dict.get(geom)
        if T is None:
            utils.fatal(f'Error: {geom} is not a valid key for TensorImage.transform_dict')

        coords = (
            (coords - T['start']) @ T['rotation'].T if geom.endswith('vox')
            else (coords @ T['rotation'].T) + T['start']
        )
        return coords


def _unsqueeze(X, dim, N):
    N += len(X.shape)
    while len(X.shape) < N:
        X = X.unsqueeze(dim=dim)
    return X


## GGVF of a TensorImage
class GGVF(TensorImage):
    def __init__(self, image, K=0.05, dt=None, n_iters=30, device=None):
        """
        Computes the generalized gradient vector flow (GGVF) field image (Xu and Prince, 1998)
        """
        super().__init__(image=image, device=device)

        self.K = K
        self.n_iters = n_iters

        self._compute_ggvf()

    def _compute_ggvf(self):
        """
        Compute the GGVF field (and edge map)
        """
        # Gradient image
        grad = self.gradient_image(img=self.smooth()).movedim(-1, 0).float()
        mag = grad.norm(dim=0, keepdim=True)
        g = torch.exp(-1. * (mag / mag.max()) / self.K)
        h = 1.0 - g
        dt = 1. / (6. * g.max().item())

        # Constant terms w/in ggvf calculation
        g_dt = dt * g
        hg_dt = dt * h * grad
        h_dt_1 = 1. - dt * h

        # Iterate w/ Laplacian
        V = grad.clone()
        lapV = torch.empty_like(V)

        for it in range(self.n_iters):
            Vp = F.pad(V.unsqueeze(dim=1), (1,) * 6, mode='replicate').squeeze(dim=1)
            torch.add(Vp[:, 2:, 1:-1, 1:-1], Vp[:, :-2, 1:-1, 1:-1], out=lapV)
            lapV += Vp[:, 1:-1, 2:, 1:-1]
            lapV += Vp[:, 1:-1, :-2, 1:-1]
            lapV += Vp[:, 1:-1, 1:-1, 2:]
            lapV += Vp[:, 1:-1, 1:-1, :-2]
            lapV.add_(V, alpha=-6.).mul_(g_dt)
            V.mul_(h_dt_1).add_(lapV).add_(hg_dt)

        self.data = V
