import os
import surfa as sf
import numpy as np

import torch
import torch.nn as nn

from scipy.spatial import cKDTree


class TensorImage:
    def __init__(self, image, device):
        # Store data
        self.device = device
        self.geom = image.geom
        self.data = torch.tensor(image.data.astype(np.float32), dtype=torch.float).to(self.device)
        
        # Store image coordinates system
        x, y, z = torch.meshgrid(
            [torch.linspace(-(X - 1) / 2, (X - 1) / 2, X, dtype=torch.float) * dX
             for X, dX in zip(self.geom.shape, self.geom.voxsize)], indexing = 'ij'
        )
        self.voxel_centers = (
            torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
            #@ torch.tensor(self.geom.vox2world[:3, :3], dtype=torch.float)
        ).to(self.device)

        # Build search tree
        self.kdtree = cKDTree(self.voxel_centers.cpu())
        self.max_search_dist = np.sqrt(self.geom.voxsize.sum()) * 2
        
    def _index2sub(self, idxs):
        """
        Convert linear index to subscript
        """
        H, W, D = self.geom.shape
        i = idxs // (D * W)
        j = (idxs // D) % W
        k = idxs % D
        return i, j, k
    
    def _interp(self, coord):
        """
        Interpolate image value at real-values coordinate (currently only works for 1 coord at a 
        time)
        """
        # Get closest voxels w/ kd tree (cpu)
        _, idxs = self.kdtree.query(coord.detach().cpu(), k=8, distance_upper_bound=self.max_search_dist)
        i, j, k = self._index2sub(idxs)

        # Calculate distances/interpolation (gpu)
        dists = torch.sqrt(((self.voxel_centers[idxs] - coord) ** 2).sum(dim=1))
        data_values = self.data[i, j, k]
        weights = (1 / dists) / (1 / dists).sum()
        value = (weights * data_values).sum()

        """
        # Method 1:
        with torch.no_grad():
            _, idxs = torch.topk(
                ((self.voxel_centers - coord) ** 2).sum(dim=-1), k=8, dim=0, largest=False
            )
            i, j, k = self._index2sub(idxs)

        dists = torch.sqrt(((self.voxel_centers[idxs] - coord) ** 2).sum())        
        data_values = self.data[i, j, k]
        weights = (1 / dists) / (1 / dists).sum()
        value = (weights * data_values).sum()
        """
        return value
    """
        # Batch processing
        N = coords.shape[0] if len(coords.shape) == 2 else 1
        batch_sz = min(N, 50) # number of verts to process at a time

        values = torch.zeros((N,), dtype=self.data.dtype, device=self.device)

        for m in range(0, N, batch_sz):
            n = min(m + batch_sz, N)
            sz = n - m

            diffs = self.voxel_centers.unsqueeze(0) - coords[m:n].unsqueeze(1)
            breakpoint()
            dists2, idxs = torch.topk(
                ((self.voxel_centers.unsqueeze(1) - coords[m:n].unsqueeze(0)) ** 2).sum(dim=-1),
                k=8, dim=0, largest=False
            )
            i, j, k = self._index2sub(idxs.flatten())
            
            data_values = self.data[i.view(sz, 8), j.view(sz, 8), k.view(sz, 8)]
            dists = torch.sqrt(dists2)
            weights = (1 / dists) / (dists).sum(dim=1, keepdim=True)
            values[m:n] = (weights * data_values.transpose(0, 1)).sum(dim=0)
      
        return values
    """
