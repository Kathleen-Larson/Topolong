import os, time
import numpy as np
import surfa as sf
import torch

import torch.linalg as tla
from torch.nn.utils.rnn import pad_sequence as pad_seq


###############
class TriangleMesh:
    """
    Custom mesh class where relevant attributes are converted to tensors for GPU processing. In all 
    cases, the first index corresponds to the vertex v0 (or triangle t0). A TriangleMesh will have 
    the following stored quantities:
    -- nverts (nv):      total number of vertices
    -- ntris (nt):       total number of triangle faces
    -- nvert_tris (nvt): list of the number of triangle faces that have v0 as a corner
    -- nnbrs_1hop (nn1): list of the number of one-hop neighbors (aka other vertices that share 
                         triangle faces with v0)
    -- nnbrs_2nop (nn2): list of the number of two-hop neighbors (aka other vertices that share 
                         triangle faces with vertices in the list of one-hop neighbors)

    A TriangleMesh will have the following attributes (with dimensions listed, where nd=ndims=3). 
    For tensors containing lists of neighbors (not list as in datatype), if the vertex does not 
    have the maximal number of neighbors, the remainder of the row is padded with -1.
    -- verts:         [nv x nd]       - xyz coordinates of vertices
    -- tris:          [nt x 3]        - vertex ids associated with each triangle face
    -- vert_tris:     [nv x max(nvt)] - list of triangles containing each vertex
    -- vert_tri_inds: [nv x max(nvt)] - index of vertex within tris (is always 0, 1, or 2)
    -- nbrs_1hop:     [nv x nn1]      - list of one-hop neighbors for each vertex
    -- nbrs_2hop:     [nv x nn2]      - list of two-hop neighbors for each vertex
    """
        
    def __init__(self, inpath:str):
        # Load w/ surfa
        self.mesh = sf.load_mesh(inpath)
        self.ndims = 3
        
        # Convert mesh data to tensor
        self.verts = torch.tensor(self.mesh.vertices)
        self.tris = torch.tensor(self.mesh.faces)
        self.nverts = len(self.verts)
        self.ntris = len(self.tris)

        # Compile list of triangles for each vertex
        tri_nbrs_1hop, vert_tri_inds = list(
            map(list, zip(*[torch.where(self.tris==p) for p in range(self.nverts)]))
        )
        self.vert_tris = pad_seq(tri_nbrs_1hop, batch_first=True, padding_value=-1)
        self.vert_tri_inds = pad_seq(vert_tri_inds, batch_first=True, padding_value=-1)
        self.nvert_tris = (self.vert_tris != -1).sum(dim=1)

        # Compile list of one-hop neighbors for each vertex
        nbrs_1hop = [
            torch.unique(self.tris[tri_nbrs_1hop[p]].flatten()) for p in range(self.nverts)
        ]
        self.nbrs_1hop = pad_seq([
            torch.cat([torch.tensor([p]), nbrs_1hop[p][torch.where(nbrs_1hop[p]!=p)]])
            for p in range(self.nverts)
        ], batch_first=True, padding_value=-1)
        self.nnbrs_1hop = (self.nbrs_1hop!=-1).sum(dim=1)

        # Now do the two-hop neighbors
        nbrs_2hop = [
            torch.unique(torch.cat([nbrs_1hop[p1] for p1 in nbrs_1hop[p]]))
            for p in range(self.nverts)
        ]
        self.nbrs_2hop = pad_seq([
            torch.cat([torch.tensor([p]), nbrs_2hop[p][torch.where(nbrs_2hop[p]!=p)]])
            for p in range(self.nverts)
        ], batch_first=True, padding_value=-1)
        self.nnbrs_2hop = (self.nbrs_2hop!=-1).sum(dim=1)
        
    def _reset_verts(self, mesh):
        """
        Resets all updated properties back to initial state (useful for debugging)
        """
        self.verts = torch.tensor(mesh.vertices)

    def _write(self, outpath):
        """
        Writes mesh to file
        """
        self.mesh.save(outpath)


