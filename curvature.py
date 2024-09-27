import os, time
import numpy as np
import surfa as sf
import torch

import torch.linalg as tla
from torch.nn.utils.rnn import pad_sequence as pad_seq

###############
class MeanCurvature:
    def __init__(self,
                 input_path:str,
                 target_path:str=None,
                 compute_times=True
    ):
        ## Class utilities
        self.compute_times = compute_times
        if self.compute_times: self.time__init = time.time()
        self.device = 'cpu'
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ftype = torch.float64
        self.itype = torch.int64
        self.ndims = 3
        
        ## Times (for debugging and optimization)
        self.compute_times = compute_times
        if self.compute_times: self.time__2hop_connectivity_neighbors = 0
        if self.compute_times: self.time__normal_tangent_rotation = 0
        if self.compute_times: self.time__glm_coeffs = 0
        if self.compute_times: self.time__init = time.time()

        ## Input mesh data
        self.mesh, self.meshdata = self._load_mesh(input_path, get_vert_nbrs=True)
        self.meshdata.curv = torch.zeros((self.meshdata.nverts, 1), device=self.device, dtype=self.ftype)
        self.meshdata.grad = torch.zeros((self.meshdata.nverts, self.ndims), device=self.device, dtype=self.ftype)
        
        ## Target mesh data
        if target_path is not None:
            self.targ, self.targdata = self._load_mesh(target, get_vert_nbrs=False)
            self.targdata.curv = torch.zeros((self.targdata.nverts, 1), device=self.device, dtype=self.ftype)
            self.targdata.grad = torch.zeros((self.targdata.nverts, self.ndims), device=self.device, dtype=self.ftype)
        else:
            self.targ, self.targdata = [None, None]
            
        if self.compute_times:
            self.time__init = (time.time() - self.time__init)
            print(f'MeanCurvature initialized in {self.time__init:>.2f} s')    

            
    class MeshData:
        def __init__(meshdata, verts, tris, normals, device, get_vert_nbrs:bool=True):
            # Convert necessary mesh data to tensor
            meshdata.verts = torch.tensor(verts, device=device)
            meshdata.tris = torch.tensor(tris, device=device)
            meshdata.vert_normals = torch.tensor(normals, device=device)
            meshdata.nverts = verts.shape[0]
            
            # Initialize curvature and gradient tensors
            meshdata.curv = None
            meshdata.grad = None

            ## Get 2 hop neighbors (make sure 1st element in each list is center vertex)
            if get_vert_nbrs:
                nbrs_1hop = [torch.unique(meshdata.tris[torch.where(meshdata.tris==p)[0]].flatten()) \
                             for p in range(meshdata.nverts)]
                nbrs_2hop = [torch.unique(torch.cat([nbrs_1hop[p1] for p1 in nbrs_1hop[p]])) \
                             for p in range(meshdata.nverts)]
                meshdata.vert_nbrs = pad_seq([torch.cat([torch.tensor([p], device=device),
                                                         nbrs_2hop[p][torch.where(nbrs_2hop[p]!=p)]]) \
                                              for p in range(meshdata.nverts)], batch_first=True, padding_value=-1)
                meshdata.nnbrs = (meshdata.vert_nbrs!=-1).sum(dim=1).max()
                                

    def _load_mesh(self, fname, get_vert_nbrs:bool=True):
        ## Load mesh data w/ surfa and initialize meshdata class
        mesh = sf.load_mesh(fname)
        meshdata = self.MeshData(mesh.vertices, mesh.faces, mesh.vertex_normals, self.device, get_vert_nbrs)        
        return mesh, meshdata
        

    def _mean_curvature(self, MD):
        """
        This function takes a meshdata class as input (MD) and 
        calculates the curvature/gradient at all vertices
        """
        t = time.time()

        # Initialize vectors
        Eu = torch.tensor([1, 0, 0], device=self.device, dtype=self.ftype).unsqueeze(1)
        Ev = torch.tensor([0, 1, 0], device=self.device, dtype=self.ftype).unsqueeze(1)
        Ew = torch.tensor([0, 0, 1], device=self.device, dtype=self.ftype).unsqueeze(1)
        C = torch.tensor([1, 0, 1], device=self.device, dtype=self.ftype).unsqueeze(0)

        # Get rotation matrices for normal/tangent rotation of each vertex
        e1 = torch.stack([MD.vert_normals[...,1], -MD.vert_normals[...,0],
                          torch.zeros((MD.nverts), device=self.device)], dim=1)
        e2 = tla.cross(MD.vert_normals, e1)
        G = torch.stack([e1/(tla.norm(e1,dim=-1).unsqueeze(-1)),
                         e2/(tla.norm(e2,dim=-1).unsqueeze(-1)),
                         MD.vert_normals], dim=1).transpose(1,2).unsqueeze(1)
        
        # Transform coords of 2 hop neighbor vertices
        nbrs = MD.vert_nbrs.unsqueeze(1).unsqueeze(-1).repeat(1,self.ndims,1,self.ndims)
        Vnbrs = MD.verts[MD.vert_nbrs].unsqueeze(1).repeat(1,self.ndims,1,1)
        V = MD.verts.unsqueeze(1).unsqueeze(1).repeat(1,self.ndims,MD.nnbrs,1)
        P = torch.where(nbrs > -1, Vnbrs, V)        
        M = P - V
        T = M @ G
        u = T[...,0]
        v = T[...,1]
        w = T[...,2].unsqueeze(-1)

        # Calculate curvature w/ GLM
        Q = torch.stack([u * u, 2 * u * v, v * v], dim=-1)
        Qt = Q.transpose(-2,-1)
        D = Qt @ Q
        F = tla.inv(D)
        B = F @ Qt @ w
        k = C @ B

        # Initialize matrices for gradient calculation
        I3 = torch.eye(self.ndims, device=self.device, dtype=self.ftype).unsqueeze(0)
        iD = torch.arange(1,MD.nnbrs)
        dM = torch.zeros((MD.nverts, MD.nnbrs, self.ndims, MD.nnbrs, self.ndims), device=M.device, dtype=M.dtype)
        dM[:,0,:,1:,:] = -1 * I3.unsqueeze(2).repeat(MD.nverts,1,MD.nnbrs-1,1)
        dM[:,iD,:,iD,:] = I3.repeat(MD.nverts,1,1)
        
        # Calculate curvature gradient
        def _expand(x, n):
            if n==5: return x.unsqueeze(1).repeat(1,MD.nnbrs,1,1,1)
            elif n==4: return x.unsqueeze(1).repeat(1,MD.nnbrs,1,1)
            
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
        MD.curv = k.squeeze()
        MD.grad = dk.sum(dim=1).squeeze()

        
