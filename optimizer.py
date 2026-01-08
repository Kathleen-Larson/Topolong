import os
import numpy as np
import torch

from utils import _dot, _expand, _norm


#---------------------------------------------------------------------------------------------------

class MeshLBFGS:
    def __init__(self,
                 moving,              # Mesh to optimize w.r.t. curvature
                 target,              # Target mesh
                 c1=1e-4,             # Armijo rule constant
                 c2=0.9,              # Curavture condition constant
                 history_size=10,     # max no. iters saved in memory
                 max_n_ls_iters=25,     # max no. iters for each line search
                 max_n_opt_iters=50,    # max no. iters for total optimization
                 tol=1e-10,            # tolerance
                 device=None
    ):
        # Parse args
        self.device = 'cuda' if device == 'gpu' else 'cpu'
        
        self.moving_mesh = moving
        self.target_mesh = target
        self.f_targ, _ = self.target_mesh._mean_curvature(output=True)
        
        self.c1 = c1
        self.c2 = c2
        self.eps = 1e-10
        self.tol = tol
        
        self.history_size = history_size
        self.max_n_ls_iters = max_n_ls_iters
        self.max_n_opt_iters = max_n_opt_iters
                
        # Storage
        self.s = []
        self.y = []
        self.rho = []
        self.eps = 1e-10

    def _cost(self, x0):
        """
        cost = ||curv - curv_targ||^2
        grad(cost) = 2 * grad(curv) * ||curv - curv_targ||
        """
        f, g = self.moving_mesh._mean_curvature(x0, output=True)
        eps = f - self.f_targ
        cost = torch.pow(eps, 2).sum()
        grad = (2 * g * _expand(eps, (), (1, g.shape[-1])))

        return cost, grad

    def _Wolfe_line_search(self, x0, f0, g0, z):
        """
        Strong Wolfe line search to find optimal step size (alpha)
        - x0: current parameter values (e.g., mesh coordinates)
        - f0: current obj_fn values (e.g., mesh curvature)
        - g0: current gradient values (e.g., grad(curv))
        - z: search direction
        """
        # Initialize
        alpha = 1.0

        if torch.sum(g0 * z) >= 0:
            print(f'Warning: search direction not descent (g * z = {gz:.4f})')
            return 0., x0, f0, g0

        # Iterate
        for _ in range(self.max_n_ls_iters):
            # Get updated values
            x1 = x0 + alpha * z
            f1, g1 = self._cost(x1)

            # Check conditions
            _armijo_ok = (
                f1 <= f0 + self.c1 * alpha * torch.sum(g0 * z)
            )
            _wolfe_ok = (
                abs(torch.sum(g1 * z)) <= self.c2 * abs(torch.sum(g0 * z))
            )

            # Update alpha
            if _armijo_ok and _wolfe_ok:
                return alpha, x1, f1, g1
            elif not _armijo_ok:
                alpha *= 0.5
            else:
                alpha *= 2.1

            if alpha < self.eps:
                print(f'Warning: step size too small (alpha = {alpha:.4e})')

        return alpha, x1, f1, g1

    def _get_search_direction(self, g):
        """
        Calculate search direction based on gradient
        """
        # Initialize
        if len(self.s) == 0:
            return -g

        q = g.clone()
        
        # Compute right product (first loop)
        alpha = []

        for i in reversed(range(len(self.s))):
            s_i = self.s[i]
            y_i = self.y[i]
            rho_i = self.rho[i]

            alpha_i = self.rho[i] * torch.sum(self.s[i] * q)
            alpha.append(alpha_i)

            q -= alpha_i * self.y[i]

        # Scale initial Hessian approximation
        if len(self.s) > 0:
            s_last = self.s[-1]
            y_last = self.y[-1]

            sy = torch.sum(s_last * y_last)
            yy = torch.sum(y_last * y_last)
            H0k = (sy / yy) if yy > self.eps else 1.0
            z = H0k * q
        else:
            z = q

        # Compute left produce
        alpha.reverse()
        
        for i in range(len(self.s)):
            beta = self.rho[i] * torch.sum(self.y[i] * z)
            z += self.s[i] * (alpha[i] - beta)

        return -z
    
    def _update_history(self, s, y):
        """
        Update storage for s = (x1 - x0), y = (g1 - g0)
        """

        ys = torch.sum(s * y)
        
        if ys > self.eps:
            # Compute rho
            rho = 1. / ys
            
            # Maintain history
            if len(self.s) >= self.history_size:
                self.s.pop(0)
                self.y.pop(0)
                self.rho.pop(0)
                
                self.s.append(s.clone())
                self.y.append(y.clone())
                self.rho.append(rho)
        else:
            print(f'Warning: skipping update.. y^T * s = {ys:.3e} < eps')

    def optimize(self, x0, verbose=True):
        """
        Perform optimization of input parameters x0, given an objective function (self._cost) and 
        target value (self.target)
        """
        
        # Initialize
        x0 = x0.clone().to(self.device)
        f0, g0 = self._cost(x0)

        # Track best?
        x_best = x0.clone()
        f_best = f0.clone()
        
        for it in range(self.max_n_opt_iters):
            # Check convergence
            gN = _norm(g0, -1).max()
            if gN < self.tol:
                if verbose:
                    print(f'Converged at iteration {it}: max |g| = {gN:.3e}')
                break

            # Line search
            z = self._get_search_direction(g0)
            alpha, x1, f1, g1 = self._Wolfe_line_search(x0, f0, g0, z)

            # Update
            if alpha > 0:
                s = x1 - x0
                y = g1 - g0
                self._update_history(s, y)

                x0 = x1
                f0 = f1
                g0 = g1

            # Track best
            if f1 < f_best:
                f_best = f1.clone()
                x_best = x1.clone()

            if verbose:
                print(f'Iteration {it:3d}: loss = {f1:.6f}, max |g| = {gN:.3e}, alpha = {alpha:3e}')
                    
        return x_best
