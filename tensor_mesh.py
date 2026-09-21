import os
import surfa as sf
import numpy as np
import functools
import time

#from scipy.sparse import coo_matrix

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import utils

eps = 1e-8

#---------------------------------------------------------------------------------------------------

class TensorMesh:
    def __init__(
            self,
            mesh,
            device=None,
            compute_proximity_tree=True,
            verts_requires_grad=False,
            proximity_ratio=0.1
    ):
        """
        Triangular mesh topology represented by tensors of vertices and faces. This class is based
        almost entirely on surfa.mesh, but with tensors instead of numpy arrays for gpu support.
        """
        self.device = torch.device(
            'cuda' if device == 'gpu' or device == 'cuda' and torch.cuda.is_available() else 'cpu'
        ) if not isinstance(device, torch.device) else device
        
        # Store data from mesh
        self.mesh = mesh
        self.verts = torch.tensor(mesh.vertices, dtype=torch.double).to(self.device)
        self.tris = torch.tensor(mesh.faces).to(self.device)

        if verts_requires_grad:
            self.verts.requires_grad = True

        self.nverts = len(mesh.vertices)
        self.ntris = len(mesh.faces)
        self.ndims = self.verts.shape[1]
        self.geom = mesh.geom
                
        # Get triangles containing each vertex
        self._compute_vert_tris()
        
        # Get one-/two-hop neighbors for each vertex
        self._compute_neighbors()

        # Get LUT for each vertex containing non-neighbors within a min distance
        if compute_proximity_tree:
            self.proximity_ratio = proximity_ratio
            self._compute_proximity_trees()
            
        # Set everything to correct device and compute vertex properties
        self._update_vert_properties()


    #-----------------------------------------------------------------------------------------------
    """
    Functions that probably only need to be run at initialization (unless the mesh structure changes
    significantly for some reason).
    """

    def _compute_vert_tris(self):
        """
        Get all triangles containing each vertex (and its indices w/in the tri data)
        """
        # Map vertices within triangles
        vert_tris, vert_tri_idxs = list(
            map(list, zip(*[torch.where(self.tris == v) for v in range(self.nverts)]))
        )
        self.vert_tris = pad_sequence(vert_tris, batch_first=True, padding_value=-1)
        self.vert_tri_idxs = pad_sequence(vert_tri_idxs, batch_first=True, padding_value=-1)

        # Set device
        self.vert_tris = self.vert_tris.to(self.device)
        self.vert_tri_idxs = self.vert_tri_idxs.to(self.device)
        
    def _compute_neighbors(self):
        """
        Get lists of 1 and 2 hop neighbors
        """
        # One-hop neighbors
        vert_neighbors_1hop = [
            torch.unique(self.tris[self.vert_tris[v][self.vert_tris[v] != -1]].flatten())
            for v in range(self.nverts)
        ]
        self.vert_neighbors_1hop = pad_sequence(
            [torch.cat([
                torch.tensor([v]).to(self.device),
                vert_neighbors_1hop[v][torch.where(vert_neighbors_1hop[v] != v)]
            ]) for v in range(self.nverts)],
            batch_first=True, padding_value=-1
        ).to(self.device)
        self.vert_nneighbors_1hop = (self.vert_neighbors_1hop != -1).sum(dim=1)

        # Two-hop neighbors
        vert_neighbors_2hop = [
            torch.unique(torch.cat([vert_neighbors_1hop[u] for u in vert_neighbors_1hop[v]]))
            for v in range(self.nverts)
        ]
        self.vert_neighbors_2hop = pad_sequence(
            [torch.cat([
                torch.tensor([v]).to(self.device),
                vert_neighbors_2hop[v][torch.where(vert_neighbors_2hop[v] != v)]
            ]) for v in range(self.nverts)],
            batch_first=True, padding_value=-1
        ).to(self.device)
        self.vert_nneighbors_2hop = (self.vert_neighbors_2hop != -1).sum(dim=1)

    def _compute_proximity_trees(self):
        """
        Get all vertices within a certain distance (defined by a percentage of the mesh bounding 
        box) from center vertex, excluding 1 hop neighbors (QUESTION: should it be 2 hop???)
        """
        # Use the kdtree of the original surfa mesh to find all vertices within range
        max_dist = np.min(np.diff(np.stack(self.mesh.bbox()), axis=0)).item() * self.proximity_ratio
        close_verts = self.mesh.kdtree.query_ball_point(
            self.verts.detach().cpu(), max_dist, return_sorted=False
        )
        
        self.vert_proximity_tree = pad_sequence(
            [torch.tensor(
                [x for x in close_verts[v] if x not in self.vert_neighbors_2hop[v]],
                dtype=self.vert_neighbors_2hop[v].dtype
            ) for v in range(self.nverts)],
            batch_first=True, padding_value=-1
        ).to(self.device)

        self.vert_nproximity = (self.vert_proximity_tree != -1).sum(dim=1)
        self.max_vert_nproximity = self.vert_nproximity.max().item()

    def _freeze_verts(self, freeze_idxs=None):
        """
        Takes an array of indices and stores them as frozen/ripped verts. These vertex indices will
        be avoided in spring energy computations.
        """
        self.rip_verts_flag = torch.zeros((self.nverts,), dtype=bool, device=self.verts.device)
        if freeze_idxs is not None:
            self.rip_verts_flag[freeze_idxs] = True
        
    #-----------------------------------------------------------------------------------------------
    """
    Functions to compute properties that should be recalculated each time the mesh vertices are 
    updated. To change the mesh vertices, use the TensorMesh._update_verts() function (do NOT use 
    "tmesh.verts = <new_verts>", as this will not recompute everything.
    """
    
    def _compute_edge_lengths(self):
        """
        Lengths of edges of triangle faces
        """
        edges = self.tris[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 3, 2)
        self.edge_lengths = (
            (self.verts[edges[..., 1]] - self.verts[edges[..., 0]]) ** 2
        ).sum(dim=2).sqrt()

    def _compute_vert_norms(self):
        """
        Compute vertex normals
        """
        # Get face edges (consistent w/ surfa notation)
        e1 = self.verts[self.tris][:, 1, :] - self.verts[self.tris][:, 0, :]
        e2 = self.verts[self.tris][:, 2, :] - self.verts[self.tris][:, 1, :]
        e3 = self.verts[self.tris][:, 2, :] - self.verts[self.tris][:, 0, :]

        e1 = e1 / utils._norm(e1, dim=1)
        e2 = e2 / utils._norm(e2, dim=1)
        e3 = e3 / utils._norm(e3, dim=1)

        # Face normals and angles corresponding to each vertex
        tri_norms = torch.cross(e1, e2, dim=-1)
        self.tri_norms = tri_norms / utils._norm(tri_norms, dim=1)

        M1 = torch.stack([e3, -e1, e2], dim=1)
        M2 = torch.stack([e1, e2, e3], dim=1)
        self.tri_angles = torch.arccos(
            utils._dot(M1, M2, dim=-1) / (utils._norm(M1, dim=-1) * utils._norm(M2, dim=-1))
        ).squeeze()

        # Reshape normals/angles to be vertex encoded
        tri_norms_vert = torch.where(
            utils._expand(self.vert_tris, unsqueeze_dims=(-1,), repeats=(1, 1, self.ndims)) > -1,
            self.tri_norms[self.vert_tris, ...], 0
        )
        
        # Vertex normals
        face_mask = (self.vert_tris > -1).unsqueeze(dim=-1)
        vert_norms = torch.where(
            (self.vert_tris > -1).unsqueeze(dim=-1), tri_norms_vert, 0
        ).sum(dim=1)
        self.vert_norms = (vert_norms / utils._norm(vert_norms, dim=1)).squeeze()
        
    def _compute_vert_tangents(self):
        """
        Compute the tangent vectors for each vertex (used in curvature and spring energy 
        calculations)
        """
        # Permute/cross normal vector
        N = self.vert_norms
        vec1 = torch.linalg.cross(N, torch.stack([N[..., 1], N[..., 2], N[..., 0]], dim=-1))
        vec2  = torch.linalg.cross(N, torch.stack([N[..., 1], -N[..., 2], N[..., 0]], dim=-1))

        # Tangents
        e1 = torch.where(torch.linalg.norm(vec1) < 0.001, vec2, vec1)
        e2 = torch.linalg.cross(N, e1)

        # Normalize
        self.vert_tangents0 = e1 / utils._norm(e1, dim=1)
        self.vert_tangents1 = e2 / utils._norm(e2, dim=1)

    def _update_vert_properties(self, X=None):
        """
        Update the mesh vertices and associated properties
        """
        self.verts.data = X.data if X is not None else self.verts
        self.mesh.vertices = X.data.detach().cpu() if X is not None else self.verts.detach().cpu()
        
        self._compute_vert_norms()
        self._compute_vert_tangents()
        self._compute_edge_lengths()

    #-----------------------------------------------------------------------------------------------
    """
    Other functions
    """
    
    def compute_points_along_normals(self, N=5, stepsize=0.5, direction='both', offset=False):
        """
        Computes a series of points extending along vertex normals
        """
        arr = torch.linspace(
            start=(-stepsize * N if direction in ['both', 'in'] else 0),
            end=(stepsize * N if direction in ['both', 'out'] else 0),
            steps=(N * 2 + 1 if direction == 'both' else N + 1)
        ).to(self.device)
        arr = torch.cat(
            [arr - (stepsize / 2), (arr[-1] + (stepsize / 2)).unsqueeze(dim=0)]
        ) if offset else arr

        vert_norm_points = self.verts.unsqueeze(-1) + (
            self.vert_norms.detach().unsqueeze(-1) * arr.unsqueeze(0)
        )
        return vert_norm_points.transpose(1, 2)

    def neighbor_displacements(self, pair_type, exclude_ripped=False, return_mask=True):
        """
        Compute the distance vectors between each vertex and the set of either:
        1. 1-hop neigbors (pair_type == '1hop')
        2. 2-hop neighbors (pair_type == '2hop')
        3. Proximity tree adjacency (pair_type == 'prox')
        TO-DO: double check if exclude_center is actually necessary
        """
        # Get correct neighbor adjacency
        if pair_type == '1hop':
            neighbors = self.vert_neighbors_1hop
        elif pair_type == '2hop':
            neighbors = self.vert_neighbors_2hop
        elif pair_type == 'prox':
            neighbors = self.vert_proximity_tree

        nneighbors = neighbors.shape[1]
        valid_neighbors = neighbors > -1
        
        # Exclude ripped (if flagged)
        if exclude_ripped:
            valid_neighbors &= (~self.rip_verts_flag)[neighbors]
            valid_neighbors[self.rip_verts_flag] = False

        # Distance calculation
        V0 = utils._expand(self.verts, unsqueeze_dims=(1,), repeats=(1, nneighbors, 1))
        Vn = torch.where(
            utils._expand(valid_neighbors, unsqueeze_dims=(-1,), repeats=(1, 1, self.ndims)),
            self.verts[neighbors], V0
        )
        
        if return_mask:
            return (Vn - V0), valid_neighbors
        else:
            return (Vn - V0)

    def smooth_over_neighbors(
            self, vec, mask, N_avgs=1, n_hops=1, signed=True, start_idx=0, remove_outliers=False
    ):
        """
        Smooth an input array over neighbors
        """
        if vec.shape[0] != self.nverts:
            utils.fatal('smooth_over_neighbors requires input with vec.shape[0] == nverts')
        if len(vec.shape) < 2:
            vec = vec.unsqueeze(dim=-1)

        # Remove outliers?
        if remove_outliers:
            is_outlier = (vec > (vec.mean() + 2 * vec.std())) | (vec < (vec.mean() - 2 * vec.std()))
            mask &= (~is_outlier).squeeze()

        # Parse valid neighbors
        neighbors = self.vert_neighbors_2hop if n_hops == 2 else self.vert_neighbors_1hop
        valid_neighbors = torch.where(
            (neighbors > -1) & mask.unsqueeze(dim=-1), mask[neighbors], False
        ).unsqueeze(dim=-1)
        
        if start_idx > 0:
            valid_neighbors[:, :start_idx] = False
            
        # Iterative smoothing
        for _ in range(N_avgs):
            signed_mask = (
                valid_neighbors & (
                    (vec[neighbors] * vec.unsqueeze(dim=1)).sum(dim=-1, keepdims=True) >= 0
                ) if signed
                else valid_neighbors
            )
            n_valid = signed_mask.sum(dim=1).clamp(1)
            vec = torch.where(signed_mask, vec[neighbors], 0).sum(dim=1) / n_valid
        
        if remove_outliers:
            vec[is_outlier] = vec[mask].mean()
        return vec
