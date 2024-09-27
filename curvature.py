import os, time
import numpy as np
import surfa as sf
import torch

import torch.linalg as tla
from torch.nn.utils.rnn import pad_sequence as pad_seq

###############

class MeanCurvature:
    def __init__(self, mesh, n_neighbors:int=None, compute_times=True):
        self.compute_times = compute_times
        self.time__init = time.time()
        #self.device = 'cpu'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ftype = torch.float
        self.itype = torch.int
        self.ndims = 3
        
        ## Times (for debugging and optimization)
        self.compute_times = compute_times
        if self.compute_times: self.time__2hop_connectivity_neighbors = 0
        if self.compute_times: self.time__normal_tangent_rotation = 0
        if self.compute_times: self.time__glm_coeffs = 0
        if self.compute_times: self.time__init = time.time()
        
        # Convert mesh data to tensors
        self.verts = torch.tensor(mesh.vertices, device=self.device, dtype=self.ftype)
        self.vert_normals = torch.tensor(mesh.vertex_normals, device=self.device, dtype=self.ftype)
        self.nverts = len(self.verts)
        
        ## Get 2 hop neighbors (make sure 1st element in each list is center vertex)
        tris = torch.tensor(mesh.faces, device=self.device, dtype=self.itype)
        nbrs_1hop = [torch.unique(tris[torch.where(tris==p)[0]].flatten()) for p in range(self.nverts)]
        nbrs_2hop = [torch.unique(torch.cat([nbrs_1hop[p1] for p1 in nbrs_1hop[p]])) for p in range(self.nverts)]
        self.vert_nbrs = pad_seq([torch.cat([torch.tensor([p], device=self.device), nbrs_2hop[p][torch.where(nbrs_2hop[p]!=p)]]) \
                                  for p in range(self.nverts)], batch_first=True, padding_value=-1)
        self.nnbrs = (self.vert_nbrs!=-1).sum(dim=1).max()
        
        
        ## Initialize curvature calculations
        self.curv = torch.zeros((self.nverts, 1), device=self.device, dtype=self.ftype)
        self.grad = torch.zeros((self.nverts, self.ndims), device=self.device, dtype=self.ftype)
        
        if self.compute_times:
            self.time__init = (time.time() - self.time__init)
            print(f'MeanCurvature initialized in {self.time__init:>.2f} s')
            
            

    def _rotation_matrices(self):
        # Calculate rotation matrices to tranform into normal/tangent space
        N = self.vert_normals.unsqueeze(dim=1)
        e1 = torch.cat([N[...,1], -N[...,0], torch.zeros((self.nverts,1), device=self.device)], dim=1).unsqueeze(dim=1)
        e2 = tla.cross(N, e1).to(self.device)
        R = torch.cat([e1/tla.norm(e1, dim=-1).repeat(1,self.ndims).unsqueeze(dim=1),
                       e2/tla.norm(e2, dim=-1).repeat(1,self.ndims).unsqueeze(dim=1),
                       N], dim=1)
        return R.transpose(1,2)
        


    def _mean_curvature(self): ### Fastest (no loops)
        t = time.time()

        # Initialize vectors
        Eu = torch.tensor([1, 0, 0], device=self.device, dtype=self.grad.dtype).unsqueeze(dim=1)
        Ev = torch.tensor([0, 1, 0], device=self.device, dtype=self.grad.dtype).unsqueeze(dim=1)
        Ew = torch.tensor([0, 0, 1], device=self.device, dtype=self.grad.dtype).unsqueeze(dim=1)
        C = torch.tensor([1, 0, 1], device=self.device, dtype=self.grad.dtype).unsqueeze(dim=0)
        G = self._rotation_matrices().unsqueeze(dim=1).repeat(1,self.ndims,1,1)
        
        # Transform coords of 2 hop neighbor vertices into respective normal/tangent spaces
        nbrs = self.vert_nbrs.unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1,self.ndims,1,self.ndims)
        Vnbrs = self.verts[self.vert_nbrs].unsqueeze(dim=1).repeat(1,self.ndims,1,1)
        V = self.verts.unsqueeze(dim=1).unsqueeze(dim=1).repeat(1,self.ndims,self.nnbrs,1)
        P = torch.where(nbrs > -1, Vnbrs, V)
        
        M = P - V
        T = M @ G
        u = T[...,0]
        v = T[...,1]
        w = T[...,2].unsqueeze(dim=-1)

        # Calculate curvature w/ GLM
        Q = torch.stack([u * u, 2 * u * v, v * v], dim=-1)
        Qt = Q.transpose(-2,-1)
        D = Qt @ Q
        F = tla.inv(D)
        B = F @ Qt @ w
        k = C @ B

        # Initialize matrices for gradient calculation
        I3 = torch.eye(self.ndims, device=self.device).unsqueeze(0)
        dM = torch.zeros((self.nverts, self.nnbrs, self.ndims, self.nnbrs, self.ndims), device=M.device, dtype=M.dtype)
        dM[:,0,:,1:,:] = -1 * I3.unsqueeze(dim=2).repeat(self.nverts,1,self.nnbrs-1,1)
        dM[:,torch.arange(1,self.nnbrs),:,torch.arange(1,self.nnbrs),:] = I3.repeat(self.nverts,1,1)
        
        # Calculate curvature gradient
        def _expand(x, n):
            if n==5: return x.unsqueeze(dim=1).repeat(1,self.nnbrs,1,1,1)
            elif n==4: return x.unsqueeze(dim=1).repeat(1,self.nnbrs,1,1)
            
        du = (dM @ _expand(G @ Eu, 5)).squeeze()
        dv = (dM @ _expand(G @ Ev, 5)).squeeze()
        dw = (dM @ _expand(G @ Ew, 5))

        dQ = torch.stack([2*du*_expand(u,4), 2*dv*_expand(u,4) + 2*du*_expand(v,4), 2*dv*_expand(v,4)], dim=-1)
        dQt = dQ.transpose(-2,-1)
        dD = (dQt @ _expand(Q,5)) + (_expand(Qt,5) @ dQ)
        dF = -_expand(F,5) @ dD @ _expand(F,5)
        dB = (dF @ _expand(Qt,5) @ _expand(w,5)) + (_expand(F,5) @ dQt @ _expand(w,5)) + (_expand(F,5) @ _expand(Qt,5) @ dw)
        dk = C @ dB

        # Update
        print(f'{(time.time() - t):>.2f} s')
        self.curv = k.squeeze()
        self.grad = dk.sum(dim=1).squeeze()
        

        
    def _mean_curvature_slow(self):  ## Slower (does all vertices simultaneously, but loops over neighbors/dimensions to calculate gradients)
        t = time.time()
        
        # Transform coords of 2 hop neighbor vertices into respective normal/tangent spaces
        V = self.verts.unsqueeze(dim=1).repeat(1,self.nnbrs,1)
        P = torch.where(self.vert_nbrs.unsqueeze(dim=-1).repeat(1,1,self.ndims) > -1, self.verts[self.vert_nbrs], V)
        M = P - V
        G = self._rotation_matrices()
        T = M @ G
        
        # Calculate curvature w/ GLM
        Q = torch.stack([T[...,0]*T[...,0], 2*T[...,1]*T[...,0], T[...,1]*T[...,1]], dim=-1)
        Qt = Q.transpose(1,2)
        D = Qt @ Q
        F = tla.inv(D)
        betas = F @ Qt @ T[...,2].unsqueeze(dim=-1)

        # Initialize matrices for gradients
        grad = torch.zeros((self.nverts, self.nnbrs, self.ndims), device=self.device, dtype=self.grad.dtype)        
        Eu = torch.tensor([1, 0, 0], device=self.device, dtype=self.grad.dtype).unsqueeze(dim=1)
        Ev = torch.tensor([0, 1, 0], device=self.device, dtype=self.grad.dtype).unsqueeze(dim=1)
        Ew = torch.tensor([0, 0, 1], device=self.device, dtype=self.grad.dtype).unsqueeze(dim=1)
        C = torch.tensor([1, 0, 1], device=self.device, dtype=self.grad.dtype).unsqueeze(dim=0)

        for p in range(self.nnbrs):
            for d in range(self.ndims):
                dM = torch.zeros((self.nverts, self.nnbrs, self.ndims), device=P.device, dtype=P.dtype)
                if p == 0:
                    dM[:,1:,d] = -1
                else:
                    dM[:,p,d] = 1

                # Solve for gradient
                du = (dM @ (G @ Eu)).squeeze()
                dv = (dM @ (G @ Ev)).squeeze()
                dw = (dM @ (G @ Ew))

                dQ = torch.stack([2*T[...,0]*du, 2*(T[...,0]*dv + T[...,1]*du), 2*T[...,1]*dv], dim=-1)
                dQt = dQ.transpose(1,2)
                dD = (dQt @ Q) + (Qt @ dQ)
                dF = -F @ dD @ F
                dB = (dF @ Qt @ T[...,2].unsqueeze(dim=-1)) + (F @ dQt @ T[...,2].unsqueeze(dim=-1)) + (F @ Qt @ dw)
                
                dk = C @ dB
                grad[:,p,d] += dk.squeeze()

        # Update
        print(f'{(time.time() - t):>.2f} s')
        self.curv = (betas[:,0] + betas[:,2]).squeeze()
        self.grad = grad.sum(dim=1)

        
        
    def _mean_curvature_slowest(self):  ### Slowest (loops over all vertices, and also over neighbors/dimensions w/in gradient calculation)
        t = time.time()

        # Initialize vectors for gradient calculation
        Eu = torch.tensor([1, 0, 0], device=self.device, dtype=self.grad.dtype).unsqueeze(dim=1)
        Ev = torch.tensor([0, 1, 0], device=self.device, dtype=self.grad.dtype).unsqueeze(dim=1)
        Ew = torch.tensor([0, 0, 1], device=self.device, dtype=self.grad.dtype).unsqueeze(dim=1)
        C = torch.tensor([1, 0, 1], device=self.device, dtype=self.grad.dtype).unsqueeze(dim=0)
        G_all = self._rotation_matrices()
        
        grad = torch.zeros((self.nverts, self.nnbrs, self.ndims), device=self.device, dtype=self.grad.dtype)
        for p0 in range(self.nverts):
            # Transform coords of 2 hop neighbor vertices into respective normal/tangent spaces
            V = self.verts[p0].repeat(self.nnbrs,1)
            P = torch.where(self.vert_nbrs[p0].unsqueeze(dim=-1).repeat(1,self.ndims) > -1,
                            self.verts[self.vert_nbrs[p0]], V)
            M = P - V
            G = G_all[p0,...]
            T = M @ G

            # Calculate curvature w/ GLM
            u = T[:,0].unsqueeze(dim=1)
            v = T[:,1].unsqueeze(dim=1)
            w = T[:,2].unsqueeze(dim=1)
            Q = torch.cat([u * u, 2 * u * v, v * v], dim=1)
            Qt = Q.transpose(0,1)
            D = Qt @ Q
            F = tla.inv(D)
            B = F @ Qt @ w
            self.curv[p0] = (C @ B).item()

            for d in range(self.ndims):
                for p1 in range(self.nnbrs):
                    # Initialize gradient matrix
                    dM = torch.zeros(P.shape, device=P.device, dtype=P.dtype)
                    if p1 == 0: dM[1:,d] = -1
                    else: dM[p1,d] = 1

                    # Solve for gradients
                    du = dM @ (G @ Eu)
                    dv = dM @ (G @ Ev)
                    dw = dM @ (G @ Ew)
                    
                    dQ = torch.cat([2 * u * du, 2 * (u * dv + v * du), 2 * v * dv], dim=-1)
                    dD = (dQ.transpose(0,1) @ Q) + (Qt @ dQ)
                    dF = -F @ dD @ F
                    dB = (dF @ Qt @ w) + (F @ dQ.transpose(0,1) @ w) + (F @ Qt @ dw)
                    
                    dk = C @ dB
                    grad[p0,p1,d] += dk.item()
                    
        print(f'{(time.time() - t):>.2f} s')
        self.grad = grad.sum(dim=1)
