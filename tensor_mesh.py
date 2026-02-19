import os
import surfa as sf
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence as pad_seq


class TensorMesh:
    def __init__(self, mesh, device=None):
        # Store data from mesh
        self.verts = torch.tensor(mesh.vertices)
        self.tris = torch.tensor(mesh.faces)
        self.vnorms = torch.tensor(mesh.vertex_normals)

        self.nverts = len(mesh.vertices)
        self.ntris = len(mesh.faces)
        self.ndims = self.vnorms.shape[1]
        self.geom = mesh.geom
    
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
        """
        self.nbrs_1hop = pad_seq([
            torch.cat([torch.tensor([p]), nbrs_1hop[p][torch.where(nbrs_1hop[p]!=p)]])
            for p in range(self.nverts)
        ], batch_first=True, padding_value=-1)
        self.nnbrs_1hop = (self.nbrs_1hop!=-1).sum(dim=1)
        """
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

        # Lastly, set everything to correct device
        self.device = torch.device(
            'cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu'
        ) if not isinstance(device, torch.device) else device

        self.verts = self.verts.to(self.device)
        self.tris = self.tris.to(self.device)
        self.vnorms = self.vnorms.to(self.device)
        self.vert_tris = self.vert_tris.to(self.device)
        self.vert_tri_inds = self.vert_tri_inds.to(self.device)
        self.nbrs_2hop = self.nbrs_2hop.to(self.device)
        
    def _get_face_normal_data(self, X):
        """
        Computes the normal (N), corner angles (a), and the corresponding Jacobians (JN and Ja) for
        each face in the mesh with respect to its corner vertices. Each array first has the
        following dimensions:
        -  N: [nt x nd]
        - JN: [nt x nc x nd x nd]
        -  a: [nt x nc]
        - Ja: [nt x nc x nd]

        Before exiting, the arrays are reshaped to correspond to each vertex and its associated
        triangles (represented by nnbrs) as such:
        -  N: [nv x nn x nd x  1]
        - JN: [nv x nn x nd x nd]
        -  a: [nv x nn x  1 x  1]
        - Ja: [nv x nn x  1 x nd]
        """
        nv = self.nverts
        nt = self.ntris
        nd = self.ndims
        nc = 3 # because triangle
        nn = self.nvert_tris.max()

        # Get face edges and Jacobians (I3) (consistent w/ surfa notation)
        X = X[self.tris]
        e1 = X[:, 1, :] - X[:, 0, :]
        e2 = X[:, 2, :] - X[:, 1, :]
        e3 = X[:, 2, :] - X[:, 0, :]

        I3 = _expand(torch.eye(nd, dtype=torch.float64, device=self.device), (0,), (nt, 1, 1))

        # Face normals
        mN = torch.cross(e1, e2)
        LN = _norm(mN,1)
        N = mN/LN
        """
        Je1e2 = torch.cross(I3, _expand(e2, (-1,), (1, 1, nd)))
        Je2e1 = torch.cross(I3, _expand(e1, (-1,), (1, 1, nd)))
        JmN = torch.stack([Je1e2, -(Je1e2 + Je2e1), Je2e1], dim=1)

        mN = _expand(mN, (1, -1))
        LN = _expand(LN, (1, -1))
        JN = (1/torch.pow(LN, 2)) * ((LN * JmN) - (mN @ mN.transpose(-1, -2) @ JmN))
        """
        # Face angles
        e_1 = torch.stack([e3, -e1, e2], dim=1)
        e_2 = torch.stack([e1, e2, e3], dim=1)
        en_1 = _norm(e_1, -1)
        en_2 = _norm(e_2, -1)

        ma = _dot(e_1, e_2, -1)
        La = _norm(e_1, -1) * _norm(e_2,-1)
        a = torch.arccos(ma / La)
        """
        Jma = - (e_1 + e_2)
        Ja = (1/torch.pow(La,2)) * ((La * Jma) - (ma @ ma.transpose(-1, -2) @ Jma))
        """
        # Convert from face encoded to vertex encoded
        N = torch.where(
            _expand(self.vert_tris, (-1,), (1, 1, nd)) > -1,
            N[self.vert_tris, ...], 0
        ).unsqueeze(-1)
        """
        JN = torch.where(
            _expand(self.vert_tris, (-1, -1), (1, 1, nd, nd)) > -1,
            JN[self.vert_tris, self.vert_tri_inds, ...], 0
        )
        """
        a = torch.where(
            _expand(self.vert_tris, (-1,), (1, 1, 1)) > -1,
            a[self.vert_tris, self.vert_tri_inds], 0
        ).unsqueeze(-1)
        """
        Ja = torch.where(
            _expand(self.vert_tris, (-1,), (1, 1, nd)) > -1,
            Ja[self.vert_tris, self.vert_tri_inds, ...], 0
        ).unsqueeze(-2)
        """
        #return N, JN, a, Ja
        return N, a

    def _get_rotation_matrices(self, X):
        """
        Computes the rotation matrix G (and its Jacobian JG) that will transform a neighborhood of
        points into the normal/tangent coordinate system for the center vertex (p0). Note that JG
        is the Jacobian of G wrt to the center vertex. G and JG should have the following
        dimensions:
        -  G = [ e1  e2  N]: [nv x nd x nd]
        - JG = [Je1 Je2 JN]: [nv x nd x nd x nd]

        Steps to calculate G:
        1. Calculate face normals (Nf), face angles (af), and their Jacobians (JNf and Jaf).
        2. Calculate the vertex normal N = sum(Nf_i * w_i) / norm(N), where i is an iterator over
           all  faces that have p0 as a corner vertex. Here, w_i = af_i / sum(af_i), so this is
           effectively just the average of the face normals, weighted by their face angles. Note
           that sum(af_i) is always 2*pi.
        3. Calculate the tangent e1 = [-Ny, Nx, 0] / norm(e1) or [-Nz, 0, Nx] / norm(e1), depending
           whether Ny > Nz for each p0.
        4. Calculate the tangent e2 = (N x e1) / norm(e2).

        Steps to calculate JG:
        1. Calculate JN wrt p0. If we let N = m/L, then JN = (1/L^2) * (L*Jm - m*JL).
        2. Calculate Je1N (Je1 wrt N). If e1 = m1/L1, then Je1N = (1/L1^2) * (L1*Jm1N - m1*JL1N).
           Then Je1 wrt p0 = Je1N * JN (chain rule!)
        3. Calculate  Je2N (Je2 wrt N). If e2 = m21/L2, then Je2N = (1/L2^2) * (L2*Jm2N - m2*JL2N).
           Then Je2 wrt p0 = Je2N * JN.
        """
        nv = self.nverts
        nt = self.ntris
        nn = self.nvert_tris.max().item()
        nd = self.ndims

        #Nf, JNf, af, Jaf = self._get_face_normal_data(X)
        Nf, af = self._get_face_normal_data(X)
        
        # Get vertex normals and their Jacobians wrt the corresponding center vertex
        mN = _dot(Nf, af, 1).squeeze(1) / (2 * torch.pi) #.sum(dim=1) / (2*torch.pi)
        LN = _norm(mN, 1)
        N = (mN / LN).squeeze()
        self.vnorms = N

        """
        JmN = (1 / (2 * torch.pi)) * ((Nf @ Jaf) + (af * JNf)).sum(dim=1)
        JN = (1 / torch.pow(LN, 2)) * ((LN * JmN) - (mN @ mN.transpose(-2, -1) @ JmN))
        """
        
        # Define e1 and e2 (tangent vectors) to build G
        yz_mask = _expand(N[..., 1].abs() > N[...,2].abs(), (1,), (1, nd))
        e_yx = torch.stack([-N[..., 1], N[...,0], torch.zeros((nv), device=self.device)], dim=1)
        e_zx = torch.stack([-N[..., 2], torch.zeros((nv), device=self.device), N[..., 0]], dim=1)

        m1 = torch.where(yz_mask, e_yx, e_zx)
        L1 = _norm(m1, 1)
        e1 = m1 / L1

        m2 = torch.cross(N, m1)
        L2 = _norm(m2, 1)
        e2 = m2 / L2

        """
        # Get Jacobians of tanget vectors wrt the corresponding center vertex
        Je_yx = _expand(
            torch.stack([torch.tensor([0, 1, 0]),
                         torch.tensor([-1, 0, 0]),
                         torch.tensor([0, 0, 0])], dim=-1
            ), (0,), (nv, 1, 1)
        ).to(self.device)

        Je_zx = _expand(
            torch.stack([torch.tensor([0, 0, 1]),
                         torch.tensor([0, 0, 0]),
                         torch.tensor([-1, 0, 0])], dim=-1
            ), (0,), (nv, 1, 1)
        ).to(self.device)

        m1 = m1.unsqueeze(-1)
        L1 = L1.unsqueeze(-1)
        Jm1N = torch.where(_expand(yz_mask, (-1,), (1, 1, nd)), Je_yx, Je_zx).to(torch.float64)
        Je1 = (1 / torch.pow(L1, 2)) * ((L1 * Jm1N) - (m1 @ m1.transpose(-1, -2) @ Jm1N)) * JN

        m2 = m2.unsqueeze(-1)
        L2 = L2.unsqueeze(-1)

        Jm2N = (
            torch.cross(JN, _expand(m1, (), (1, 1, nd)))
            + torch.cross(_expand(N, (-1,), (1, 1, nd)), Je1)
        )
        Je2 = (1 / torch.pow(L2, 2)) * ((L2 * Jm2N) - (m2 @ m2.transpose(-1, -2) @ Jm2N)) * JN
        """
        # Stack to create set of rotation matrices
        G = torch.stack([e1, e2, N], dim=-1)
        #JG = torch.stack([Je1, Je2, JN], dim=-1)

        return G #, JG

    def _mean_curvature(self, X=None, do_grad=False):
        """
        Calculate mean curvature/gradient at each vertex
        """
        # Curvature calculations
        C = torch.tensor([1, 0, 1], device=self.device, dtype=torch.float64).unsqueeze(0)
        X = self.verts if X is None else X.to(self.device)

        nv = self.nverts
        nd = self.ndims
        nn = self.nnbrs_2hop.max().item()

        # Get vertex coords and neighbors
        nbrs = _expand(self.nbrs_2hop, (-1,), (1, 1, nd))
        V = _expand(X, (1,), (1, nn, 1))
        P = torch.where(nbrs > -1, X[self.nbrs_2hop], V)
        M = P - V

        # Transform coords into normal/tangent space
        M = P - V
        #G, dG = self._get_rotation_matrices(X)
        G = self._get_rotation_matrices(X)
        
        T = M @ G
        u = T[..., 0].unsqueeze(-1)
        v = T[..., 1].unsqueeze(-1)
        w = T[..., 2].unsqueeze(-1)

        # GLM
        Q = torch.cat([u * u, 2 * u * v, v * v], dim=-1)
        Qt = Q.transpose(-2, -1)
        F = torch.linalg.inv(Qt @ Q)
        B = F @ Qt @ w
        H = (C @ B).squeeze(-1)

        return H
    """
        # Gradient calculations
        if do
        I3 = torch.eye(nd, device=self.device, dtype=torch.float64).unsqueeze(0)
        iD = torch.arange(1,nn)

        dM = torch.zeros((nv, nn, nd, nn, nd), device=self.device, dtype=torch.float64)
        dM[:, 0, :, 1:, :] = -1 * _expand(I3, (2,), (nv, 1, nn - 1, 1))
        dM[:, iD, :, iD, :] = _expand(I3, (), (nv, 1, 1))

        u = _expand(u, (-1, -1), (1, 1, nd, nn, 1))
        v = _expand(v, (-1, -1), (1, 1, nd, nn, 1))
        w = _expand(w, (-1, -1), (1, 1, nd, nn, 1))
        Q = _expand(Q, (1, 1), (1, nn, nd, 1, 1))
        Qt = _expand(Qt, (1, 1), (1, nn, nd, 1, 1))
        F = _expand(F, (1, 1), (1, nn, nd, 1, 1))

        # Calculate Jacobians
        M_dG_u = torch.einsum('vni,vij->vnj', M, dG[..., 0])
        M_dG_v = torch.einsum('vni,vij->vnj', M, dG[..., 1])
        M_dG_w = torch.einsum('vni,vij->vnj', M, dG[..., 2])
        
        du = (dM @ _expand(G[...,0], (1, 1, -1), (1, nn, nd, 1, 1)))
        dv = (dM @ _expand(G[...,1], (1, 1, -1), (1, nn, nd, 1, 1)))
        dw = (dM @ _expand(G[...,2], (1, 1, -1), (1, nn, nd, 1, 1)))

        du[:, :, :, 0, :] += M_dG_u.unsqueeze(-1)
        dv[:, :, :, 0, :] += M_dG_v.unsqueeze(-1)
        dw[:, :, :, 0, :] += M_dG_w.unsqueeze(-1)

        dQ = torch.cat([2 * du * u, 2 * (dv * u + du * v), 2 * dv * v], dim=-1)
        dQt = dQ.transpose(-2, -1)
        dF = -F @ ((dQt @ Q) + (Qt @ dQ)) @ F
        dB = (dF @ Qt @ w) + (F @ dQt @ w) + (F @ Qt @ dw)
        J_H = (C @ dB).squeeze()
        
        return H, J_H
    """

    def _update_verts(self, X):
        self.verts = X
        

# --------------------------------------------------------------------------------------------------

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
    if rpts is not None:
        assert (
            len(rpts) == len(x.shape) + len(unsq),
            "In _expand(), len(rpts) must equal len(x.shape) + len(unsq)"
        )
    for d in unsq:
        x = x.unsqueeze(d)
    return x if rpts is None else x.repeat(rpts)

def _norm(x, d):
    """
    Custom function to normalize a tensor without reducing its number of dimensions. Also mostly
    just helps to keep the code cleaner.
    """
    return torch.norm(x, dim=d).unsqueeze(d)
