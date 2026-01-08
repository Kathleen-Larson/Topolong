import os, time
import numpy as np
import surfa as sf
import torch

from torch.nn.utils.rnn import pad_sequence as pad_seq
from utils import _norm, _expand, _dot



###############

class MeanCurvature:
    def __init__(self,
                 mesh,
                 compute_times=True,
                 opt_n_iters=10,
                 opt_step_size=0.1,
                 device='cpu',
                 **kwargs
    ):
        self.compute_times = compute_times
        if self.compute_times: self.time__init = time.time()
        self.device = torch.device('cuda' if device=='gpu' and torch.cuda.is_available() else 'cpu')
        self.ftype = torch.float64
        self.itype = torch.int64
        self.ndims = 3

        self.mesh = self._to_device(mesh) if self.device.type == 'cuda' else mesh
        self.curv = torch.empty((self.mesh.nverts, 1), device=self.device, dtype=self.ftype)
        self.grad = torch.empty(self.mesh.verts.shape, device=self.device, dtype=self.ftype)

    def _to_device(self, mesh):
        """
        Converts necessary mesh attributes to correct device
        """
        keys_to_convert = ['verts', 'tris', 'vert_tris', 'vert_tri_inds', 'nbrs_1hop', 'nbrs_2hop']
        for key in mesh.__dict__:
            if key in keys_to_convert:
                mesh.__dict__[key] = mesh.__dict__[key].to(self.device)
        return mesh

    def _get_face_normal_data(self):
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
        nv = self.mesh.nverts
        nt = self.mesh.ntris
        nd = self.ndims
        nc = 3 # because triangle
        nn = self.mesh.nvert_tris.max()

        # Get face edges and Jacobians (I3) (consistent w/ surfa notation)
        X = self.mesh.verts[self.mesh.tris]
        e1 = X[:, 1, :] - X[:, 0, :]
        e2 = X[:, 2, :] - X[:, 1, :]
        e3 = X[:, 2, :] - X[:, 0, :]

        I3 = _expand(torch.eye(nd, dtype=self.ftype, device=self.device), (0,), (nt, 1, 1))
        
        # Face normals
        mN = torch.cross(e1, e2)
        LN = _norm(mN,1)
        N = mN/LN

        Je1e2 = torch.cross(I3, _expand(e2, (-1,), (1, 1, nd)))
        Je2e1 = torch.cross(I3, _expand(e1, (-1,), (1, 1, nd)))
        JmN = torch.stack([Je1e2, -(Je1e2 + Je2e1), Je2e1], dim=1)

        mN = _expand(mN, (1, -1))
        LN = _expand(LN, (1, -1))
        JN = (1/torch.pow(LN, 2)) * ((LN * JmN) - (mN @ mN.transpose(-1, -2) @ JmN))

        # Face angles
        e_1 = torch.stack([e3, -e1, e2], dim=1)
        e_2 = torch.stack([e1, e2, e3], dim=1)
        en_1 = _norm(e_1, -1)
        en_2 = _norm(e_2, -1)

        ma = _dot(e_1, e_2, -1)  #(e_1 * e_2).sum(dim=-1).unsqueeze(-1)
        La = _norm(e_1, -1) * _norm(e_2,-1)
        a = torch.arccos(ma / La)

        Jma = - (e_1 + e_2)
        Ja = (1/torch.pow(La,2)) * ((La * Jma) - (ma @ ma.transpose(-1, -2) @ Jma))

        # Convert from face encoded to vertex encoded
        N = torch.where(
            _expand(self.mesh.vert_tris, (-1,), (1, 1, nd)) > -1,
            N[self.mesh.vert_tris, ...], 0
        ).unsqueeze(-1)
        JN = torch.where(
            _expand(self.mesh.vert_tris, (-1, -1), (1, 1, nd, nd)) > -1,
            JN[self.mesh.vert_tris, self.mesh.vert_tri_inds, ...], 0
        )
        a = torch.where(
            _expand(self.mesh.vert_tris, (-1,), (1, 1, 1)) > -1,
                        a[self.mesh.vert_tris, self.mesh.vert_tri_inds], 0
        ).unsqueeze(-1)
        Ja = torch.where(
            _expand(self.mesh.vert_tris, (-1,), (1, 1, nd)) > -1,
            Ja[self.mesh.vert_tris, self.mesh.vert_tri_inds, ...], 0
        ).unsqueeze(-2)

        return N, JN, a, Ja


    
    def _get_rotation_matrices(self):
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
        nv = self.mesh.nverts
        nt = self.mesh.ntris
        nn = self.mesh.nvert_tris.max().item()
        nd = self.ndims

        Nf, JNf, af, Jaf = self._get_face_normal_data()

        # Get vertex normals and their Jacobians wrt the corresponding center vertex
        mN = _dot(Nf, af, 1).squeeze(1) / (2 * torch.pi) #.sum(dim=1) / (2*torch.pi)
        LN = _norm(mN, 1)
        N = (mN / LN).squeeze()

        JmN = (1 / (2 * torch.pi)) * ((Nf @ Jaf) + (af * JNf)).sum(dim=1)
        JN = (1 / torch.pow(LN, 2)) * ((LN * JmN) - (mN @ mN.transpose(-2, -1) @ JmN))    

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
        Jm1N = torch.where(_expand(yz_mask, (-1,), (1, 1, nd)), Je_yx, Je_zx).to(self.ftype)
        Je1 = (1 / torch.pow(L1, 2)) * ((L1 * Jm1N) - (m1 @ m1.transpose(-1, -2) @ Jm1N)) * JN
        
        m2 = m2.unsqueeze(-1)
        L2 = L2.unsqueeze(-1)

        Jm2N = (
            torch.cross(JN, _expand(m1, (), (1, 1, nd)))
            + torch.cross(_expand(N, (-1,), (1, 1, nd)), Je1)
        )
        Je2 = (1 / torch.pow(L2, 2)) * ((L2 * Jm2N) - (m2 @ m2.transpose(-1, -2) @ Jm2N)) * JN

        # Stack to create set of rotation matrices 
        G = torch.stack([e1, e2, N], dim=-1)
        JG = torch.stack([Je1, Je2, JN], dim=-1)

        return G, JG

    def _mean_curvature(self, X=None, grad_flag=True, output=False, **kwargs):
        """
        to-do: add readme
        """        
        # Curvature calculations
        #self.mesh = self.mesh if self.mesh is None else self.mesh
        X = self.mesh.verts if X is None else X
        C = torch.tensor([1, 0, 1], device=self.device, dtype=self.ftype).unsqueeze(0)

        nv = self.mesh.nverts
        nd = self.ndims
        nn = self.mesh.nnbrs_2hop.max().item()

        # Get vertex coords and neighbors
        nbrs = _expand(self.mesh.nbrs_2hop, (-1,), (1, 1, nd))
        V = _expand(X, (1,), (1, nn, 1))
        P = torch.where(nbrs > -1, self.mesh.verts[self.mesh.nbrs_2hop], V)
        M = P - V

        # Transform coords into normal/tangent space
        M = P - V
        G, dG = self._get_rotation_matrices()

        T = M @ G
        u = T[..., 0].unsqueeze(-1)
        v = T[..., 1].unsqueeze(-1)
        w = T[..., 2].unsqueeze(-1)

        # GLM
        Q = torch.cat([u * u, 2 * u * v, v * v], dim=-1)
        Qt = Q.transpose(-2, -1)
        F = torch.linalg.inv(Qt @ Q)
        B = F @ Qt @ w
        self.curv = (C @ B).squeeze(-1)
        
        # Gradient calculations
        if grad_flag:
            # Initialize/resize matrices
            I3 = torch.eye(nd, device=self.device, dtype=self.ftype).unsqueeze(0)
            iD = torch.arange(1,nn)

            dM = torch.zeros((nv, nn, nd, nn, nd), device=self.device, dtype=self.ftype)
            dM[:, 0, :, 1:, :] = -1 * _expand(I3, (2,), (nv, 1, nn - 1, 1))
            dM[:, iD, :, iD, :] = _expand(I3, (), (nv, 1, 1)) #.repeat(nv, 1, 1)

            u = _expand(u, (-1, -1), (1, 1, nd, nn, 1))
            v = _expand(v, (-1, -1), (1, 1, nd, nn, 1))
            w = _expand(w, (-1, -1), (1, 1, nd, nn, 1))
            Q = _expand(Q, (1, 1), (1, nn, nd, 1, 1))
            Qt = _expand(Qt, (1, 1), (1, nn, nd, 1, 1))
            F = _expand(F, (1, 1), (1, nn, nd, 1, 1))

            # Calculate Jacobians
            du = (dM @ _expand(G[...,0], (1, 1, -1), (1, nn, nd, 1, 1)))
            dv = (dM @ _expand(G[...,1], (1, 1, -1), (1, nn, nd, 1, 1)))
            dw = (dM @ _expand(G[...,2], (1, 1, -1), (1, nn, nd, 1, 1)))

            dQ = torch.cat([2 * du * u, 2 * (dv * u + du * v), 2 * dv * v], dim=-1)
            dQt = dQ.transpose(-2, -1)
            dF = -F @ ((dQt @ Q) + (Qt @ dQ)) @ F
            dB = (dF @ Qt @ w) + (F @ dQt @ w) + (F @ Qt @ dw)
            self.grad = (C @ dB).sum(dim=1).squeeze()

        if output:
            if grad_flag: return self.curv, self.grad
            else: return self.curv

