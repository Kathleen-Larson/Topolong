import os, time
import numpy as np
import surfa as sf
import torch
import time

import utils
from tensor_mesh import TensorMesh
from tensor_image import TensorImage
from border_stats import BorderStatsGenerator


# --------------------------------------------------------------------------------------------------

class MRISPlaceSurface:
    def __init__(
            self,
            in_pial=None,       # input pial surface to optimiate (surfa mesh)
            in_white=None,      # input white surface to optimate (surfa mesh)
            image=None,         # surfa volume to use for intensity optimization
            mesh_target=None,   # surfa mesh to use as target for curvature optimization
            seg=None,           # Label map used for calculating boundary intensities (aseg)
            wm=None,            # wm.mgz, probably remove after debugging
            lut=None,           # Lookup table to determine left/right labels in seg
            adaptive=False,     # flag to change loss weightings during optimization
            device=None,        # device to store all tensor-based classes/operations
            l_curvature=None,   # weighting for curvature loss
            l_intensity=None,   # weighting for intensity loss
            l_nspring=None,     # weighting for spring energy loss (normal component)
            l_tspring=None,     # weighting for spring energy loss (tangential component)
            l_repulsion=None,   # weighting for repulsion energy loss
            n_grad_avgs=1,      # no. averages to start for loss smoothing (intensity only)
            proximity_ratio=0.05,  # % of mesh bbox to include in proximity trees (for debugging)
            separate_loss_types=False,
            smoothing_sigma=2.,  #
            targets_fname=None,  # name of file containing target intensities (probs debug only)
            borders_dict=None,  # dict containing border intensities
            surf='white',       # surface (pial or white)
            hemi='lh',          # hemisphere (lh or rh)
            verbose=False,      # flag to print out losses at each iteration
            log=None            # log to output text
    ):
        """
        Wrapper class to optimize vertices of moving mesh w.r.t. target mesh, using the LBFGS
        algorithm
        """
        self.device = torch.device(
            'cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu'
        ) if not isinstance(device, torch.device) else device

        # Optimization parameters
        self.adaptive = adaptive
        self.verbose = verbose

        self.log = log
        if self.log is not None and os.path.isfile(self.log):
            open(self.log, 'w').close()

        # Set up loss dictionaries
        if all(l is None for l in [l_curvature, l_intensity, l_nspring, l_tspring, l_repulsion]):
            utils.fatal('[MRISPlaceSurface] error: must provide at least one non-zero weighting '
                        'for possible losses')

        self.manual_grad_losses_dict = {}
        if l_intensity is not None and l_intensity > 0:
            self.manual_grad_losses_dict['cost_intensity'] = l_intensity

        do_intensity_loss = 'cost_intensity' in self.manual_grad_losses_dict.keys()
            
        self.autograd_losses_dict = {}
        if l_curvature is not None and l_curvature > 0:
            self.autograd_losses_dict['cost_curvature'] = l_curvature
        if l_repulsion is not None and l_repulsion > 0:
            self.autograd_losses_dict['cost_repulsion'] = l_repulsion
        if (l_nspring is not None and l_nspring > 0) or (l_tspring is not None and l_tspring > 0):
            self.autograd_losses_dict['cost_spring'] = [
                0. if l_nspring is None else l_nspring, 0. if l_tspring is None else l_tspring
            ]

        do_curvature_loss = 'cost_curvature' in self.autograd_losses_dict.keys()
        do_repulsion_loss = 'cost_repulsion' in self.autograd_losses_dict.keys()
        do_spring_loss = 'cost_spring' in self.autograd_losses_dict.keys()

        self.separate_loss_types = separate_loss_types
        
        # Initialize mesh to be optimized
        self.surf = surf
        self.hemi = hemi
        self.fix_mtl = (self.surf == 'pial')
        self.use_curvature_residuals = mesh_target is None

        if in_pial is None and in_white is None:
            utils.fatal('[MRISPlaceSurface] error: must provide at least one input surface')
        if in_pial is not None and in_white is not None:
            utils.fatal('[MRISPlaceSurface] error: joint placement of pial and white surfaces not '
                        'implemented :(')

        self.mesh = in_white.copy() if in_pial is None else in_pial.copy()
        self.tmesh = TensorMesh(
            mesh=self.mesh,
            device=device,
            verts_requires_grad=True,
            compute_curvature=do_curvature_loss,
            compute_curvature_residuals=self.use_curvature_residuals,
            compute_repulsion_energy=do_repulsion_loss,
            compute_spring_energy=do_spring_loss,
            proximity_ratio=proximity_ratio
        )

        # Target curvature?
        if mesh_target:
            tmesh_target = TensorMesh(mesh=mesh_target, compute_proximity_tree=False)
            M = tmesh_target.neighbor_displacements(pair_type='2hop', return_mask=False)
            u = (M * tmesh_target.vert_tangents0.unsqueeze(dim=1)).sum(dim=-1, keepdims=True)
            v = (M * tmesh_target.vert_tangents1.unsqueeze(dim=1)).sum(dim=-1, keepdims=True)
            w = (M * tmesh_target.vert_norms.unsqueeze(dim=1)).sum(dim=-1, keepdims=True)
            B = utils.GLM(torch.cat([u * u, 2 * u * v, v * v], dim=-1), w)
            self.target_curvature = B[:, 0] + B[:, 2]
        else:
            self.target_curvature = None

        # Set fixed (aka ripped) vertices
        tseg = TensorImage(seg, dtype='int', device=self.device)
        self._set_fixed_vertices(seg=tseg, lut=lut)
        
        # # Determine target intensities for vertex placement
        if do_intensity_loss:
            self.n_grad_avgs = n_grad_avgs
            self.smoothing_sigma = smoothing_sigma

            # Image for intensity loss
            if image is None:
                utils.fatal('[MRISPlaceSurface] error: must provide image if using intensity loss')
            if seg is None:
                utils.fatal('[MRISPlaceSurface] error: must provide label map if using intensity '
                            'loss')
            self.image = image
            self.timage = TensorImage(image, do_smoothing=True, device=device)
            
            """
            Right now this is hard coded from autodet.gw.stats.lh.dat, but will replace with 
            border_stats.py after debugging
            """

            with torch.no_grad():
                white_dict = {
                    'border_hi': 110.363342,
                    'border_lo': 61.000000,
                    'outside_hi': 110.363342,
                    'outside_lo': 48.894100,
                    'inside_hi': 120.000000
                }
                pial_dict = {
                    'border_hi': 48.894100,
                    'border_lo': 24.682297,
                    'outside_hi': 42.841148,
                    'outside_lo': 10.000000,
                    'inside_hi': 91.636658
                }
                self.borders_dict = (
                    borders_dict if borders_dict is not None
                    else white_dict if surf == 'white'
                    else pial_dict
                )
                """
                self.borders_dict = BorderStatsGenerator(
                    T1=self.timage.data, seg=tseg.data, wm=None, lut=lut
                ).compute_border_stats(gm_lo=30, gm_hi=110, use_modes=True, which='white')
                """
                if not targets_fname:
                    self.target_intensities = torch.tensor(
                        np.loadtxt('target_vals.txt'), device=self.device, dtype=torch.float
                    ).unsqueeze(dim=-1)
                    self.has_valid_intensity = (~self.rip_verts_flag)
                else:
                    self._calculate_target_intensities(n_avg_iters=5)
                
    def _set_fixed_vertices(self, seg, lut):
        """
        Determines vertices that should remain fixed in initial position during surface placement.
        Based off of MRISripMidline in utils/mrisurf.cpp. Essentially, samples aseg along surface 
        normal w/in 2mm of each vertex, sets as ok if ipsilateral GM/WM, and flags if subcortical 
        or CC. Also checks if putamen is close to vertex and will record distance so that the 
        surface does not accidentally go in there.
        """

        nverts = self.tmesh.nverts
        rip_flag = torch.zeros((nverts,), dtype=torch.bool)

        exclude_list = [
            'vessel', 'choroid-plexus', 'Optic-Chiasm', 'Thalamus', 'Pallidum', 'Caudate',
            'CC', 'Brain-Stem', 'VentralDC', '3rd-Ventricle', '4th-Ventricle'
        ] + ['-'.join(['Right' if self.hemi == 'lh' else 'Left', 'Cerebral-White-Matter'])]

        ipsi_wm = '-'.join(['Left' if self.hemi == 'lh' else 'Right', 'Cerebral-White-Matter'])
        ipsi_gm = '-'.join(['Left' if self.hemi == 'lh' else 'Right', 'Cerebral-Cortex'])
        
        # Generate normal profiles and sample intensities
        step = 0.5
        n_pts = 4

        pos = torch.arange(0, (n_pts * step) + step, step).repeat([nverts, 1]).to(self.device)
        pts = self.tmesh.compute_points_along_normals(N=n_pts, stepsize=step)
        pt_labels = seg._interp(pts.reshape(-1, 3), mode='nearest').reshape(nverts, -1)
        
        # Define separate search directions
        pt_labels_out = pt_labels[:, n_pts:]
        M_out = torch.full_like(pos, -1, dtype=torch.int)

        pt_labels_in = pt_labels[:, :(n_pts + 1)].flip(dims=(-1,))
        M_in = torch.full_like(pos, -1, dtype=torch.int)
        
        def check_pt_labels(pt_labels, labels):
            labels = torch.tensor(utils.search_lut(lut, labels)).to(self.device)
            return torch.isin(pt_labels, labels)
        
        # Make label/distance specific checks
        """
        1. If any points contain a label in the exclude_list
        2. If ventricle is anywhere except >= 1mm away on inner profile
        3. Hippocampus/amygdala if doing fix_mtl (outward only)
        4. If accumbens is anywhere (white only)
        5. If hypointensities or lesions (white/outward only)
        6. If the putamen is less than 1.1 on inward profile (white only)
        7. Reinclude if profile contains ispilateral gm/wm (at specific distances)
        """
        M_out[check_pt_labels(pt_labels_out, exclude_list)] = 0
        M_in[check_pt_labels(pt_labels_in, exclude_list)] = 0

        M_out[check_pt_labels(pt_labels_out, 'Lateral-Ventricle')] = 0
        M_in[check_pt_labels(pt_labels_in, 'Lateral-Ventricle') & (pos <= 1)] = 0

        if self.fix_mtl:
            M_out[check_pt_labels(pt_labels_out, ['Hippocampus', 'Amygdala'])] = 0

        if self.surf == 'white':
            M_out[check_pt_labels(pt_labels_out, ['Accumbens-area'])] = 0
            M_in[check_pt_labels(pt_labels_in, ['Accumbens-area'])] = 0

            M_out[check_pt_labels(pt_labels_out, ['hypointensities', 'Lesion'])] = 0
            M_in[check_pt_labels(pt_labels_in, ['lesion', 'hypointensities']) & (pos < 1)] = 0

            M_in[check_pt_labels(pt_labels_in, ['Putamen']) & (pos < 1.1)] = 0

        M_out[check_pt_labels(pt_labels_out, ipsi_gm) & (pos > 0)] = 1
        M_in[check_pt_labels(pt_labels_in, [ipsi_gm, ipsi_wm]) & (pos < 1.1)] = 1

        # Fix verts if a bad label was hit before an ipsilateral wm/gm label
        K = n_pts + 1
        col_pos = torch.arange(n_pts + 1, device=self.device).unsqueeze(0)

        fix_out = torch.where(M_out == 0, col_pos, K).min(dim=1).values
        keep_out = torch.where(M_out == 1, col_pos, K).min(dim=1).values

        fix_in = torch.where(M_in == 0, col_pos, K).min(dim=1).values
        keep_in = torch.where(M_in == 1, col_pos, K).min(dim=1).values

        self.rip_verts_flag = (fix_out < keep_out) | (fix_in < keep_in)

        # Check for the putamen in the superior direction up to 10mm away (white only)
        if self.surf == 'white':
            z_pts = self.tmesh.verts.unsqueeze(dim=1) + (
                torch.tensor([0, 0, 1.], device=self.device)
                * torch.arange(0, 10 + step, step, device=self.device).unsqueeze(dim=-1)
            )
            z_pt_labels = seg._interp(z_pts.reshape(-1, 3), mode='nearest').reshape(nverts, -1)
            M = check_pt_labels(z_pt_labels, 'Putamen')

            self.rip_verts_flag |= (M[:, :3].any(dim=-1) & (M.float().mean(dim=-1) > 0.5))

        if self.surf == 'white':
            M = check_pt_labels(
                pt_labels_in, ['Putamen', 'Accumbens-area', 'Claustrum']
            ).any(dim=-1)

            self.rip_verts_flag |= M

        # Perform dilation/erosion to get rid of holes in the rip flag overlay
        n_iters = 3

        for _ in range(n_iters):
            neighbor_flag = torch.where(
                self.tmesh.vert_neighbors_1hop >= 0,
                self.rip_verts_flag[self.tmesh.vert_neighbors_1hop.clamp(min=0)], False
            ).any(dim=-1)
            self.rip_verts_flag = (self.rip_verts_flag | neighbor_flag)

        for _ in range(n_iters):
            neighbor_flag = torch.where(
                self.tmesh.vert_neighbors_1hop >= 0,
                self.rip_verts_flag[self.tmesh.vert_neighbors_1hop.clamp(min=0)], True
            ).all(dim=-1)
            self.rip_verts_flag = (self.rip_verts_flag & neighbor_flag)

        # Store rip_verts_flag as tmesh attribute
        self.tmesh._freeze_verts(torch.where(self.rip_verts_flag)[0])
            
    def _calculate_target_intensities(self, max_dist=10., n_avg_iters=0):
        """
        This is doing what MRISComputeBorderValues_new() does
        sigma=2 for both pial_sigma and white_sigma
        """
        nverts = self.tmesh.nverts
        eps = 1e-7

        # Sampling points
        step_sz = 0.5 * self.timage.voxsize[0]
        upsample = (step_sz / 0.1).int().item()
        step_sz_up = step_sz / upsample
        
        pts = self.tmesh.compute_points_along_normals(
            N=np.ceil(max_dist / step_sz).int(), stepsize=step_sz
        )
        pts_up = self.tmesh.compute_points_along_normals(
            N=np.ceil(max_dist / step_sz_up).int(), stepsize=step_sz_up, offset=True
        )

        n_pts = pts.shape[1]
        center_idx = (pts.shape[1] - 1) // 2
        
        # Intensity profiles (between border_lo and border_hi, constant at all sigmas)
        I = self.timage._interp(pts)
        I_up = self.timage._interp(pts_up)
        I_up_ok = (
            (I_up >= self.borders_dict['border_lo']) & (I_up <= self.borders_dict['border_hi'])
        )
        
        # Distance profiles
        pos = torch.arange(start=(-max_dist), end=(max_dist + step_sz), step=step_sz)
        pos_up = torch.arange(
            start=(-max_dist - step_sz_up / 2),
            end=(max_dist + step_sz_up / 2) + step_sz_up / 2,
            step=step_sz_up
        )

        # Iterate over increasing sigma values to find distance bounds (skip ripped verts)
        pos_up_ok = torch.zeros_like(I_up, dtype=torch.bool)
        found_range = self.rip_verts_flag.clone()

        remaining_idxs = torch.where(torch.logical_not(found_range))[0]
        n_remaining = nverts - found_range.sum()

        sigma = self.smoothing_sigma
        self.grad_sigmas = torch.zeros((nverts,))
        self.grad_sign = torch.zeros((nverts,))

        while n_remaining > 0 and sigma <= (10 * self.smoothing_sigma):
            # Gradient mask (changes w/ sigma)
            g = self.timage._interp_derivative(
                pts[remaining_idxs], self.tmesh.vert_norms[remaining_idxs], sigma=sigma
            )
            g_neg = g < -eps

            # Intensity criteria
            I_band_lo = I[remaining_idxs] >= self.borders_dict['border_lo']
            I_band_hi = I[remaining_idxs] <= self.borders_dict['border_hi']
        
            # Parse criteria in inward direction (towards WM)
            in_ok = (g_neg[:, :center_idx] & I_band_hi[:, :center_idx]).flip(dims=(-1,))
            in_steps = torch.cumprod(in_ok.to(torch.int8), dim=-1).sum(dim=-1)
            in_idxs = center_idx - in_steps
            in_pos_up_ok = pos_up >= (pos[in_idxs] - step_sz / 2).unsqueeze(dim=1)

            # Parse criteria in outward direction (towards CSF)
            out_ok = g_neg[:, (center_idx + 1):] & I_band_lo[:, (center_idx + 1):]
            out_steps = torch.cumprod(out_ok.to(torch.int8), dim=-1).sum(dim=-1)
            out_idxs = center_idx + out_steps
            out_pos_up_ok = pos_up <= (pos[out_idxs] + step_sz / 2).unsqueeze(dim=1)

            # Double check if center is ok
            center_ok = (
                g_neg[:, center_idx] & (I_band_hi[:, center_idx] | I_band_lo[:, center_idx])
            )            

            # Combine and flag verts with no range found at current sigma
            pos_up_ok_it = (in_pos_up_ok & out_pos_up_ok)
            pos_up_ok_it[~center_ok] = False
            found_range_it = pos_up_ok_it.any(dim=1)

            # Update
            pos_up_ok[remaining_idxs[found_range_it]] = pos_up_ok_it[found_range_it]
            found_range[remaining_idxs[found_range_it]] = True
            self.grad_sigmas[remaining_idxs[found_range_it]] = sigma

            remaining_idxs = torch.where(torch.logical_not(found_range))[0]
            n_remaining = nverts - found_range.sum()
            sigma *= 2

        # Create 1mm look ahead w/ outside bounds
        offset = max(1, int(round(1.0 / step_sz_up.item())))
        I_shift = torch.roll(I_up, shifts=-offset, dims=-1)
        valid_shift = torch.zeros_like(I_up, dtype=torch.bool)
        valid_shift[:, :-offset] = True

        lookahead_ok = (
            valid_shift
            & (I_shift >= self.borders_dict['outside_lo'])
            & (I_shift <= torch.as_tensor(
                [self.borders_dict['border_hi'], self.borders_dict['outside_hi']]
            ).min())
        )

        has_candidate = I_up_ok & pos_up_ok & lookahead_ok
        found_idx = has_candidate.any(dim=1)
        
        # Find local minima of upsampled gradient criteria (expensive)
        g_up = torch.zeros_like(I_up)
        g_up[found_idx] = self.timage._interp_derivative(
            pts_up[found_idx], self.tmesh.vert_norms[found_idx], sigma=2
        ).to(I_up.dtype)
        g_is_minima = torch.nn.functional.pad(
            (g_up[:, 1:-1] < g_up[:, :-2]) & (g_up[:, 1:-1] < g_up[:, 2:]),
            (1, 1), value=0
        ) & (g_up.abs() > eps)

        # Combine everything to find idx of target intensity
        has_local_grad_min = pos_up_ok & I_up_ok & lookahead_ok & g_is_minima
        found_local_grad_min = has_local_grad_min.any(dim=1)

        target_idxs = torch.where(
            found_local_grad_min,
            torch.where(has_local_grad_min, g_up, torch.inf).argmin(dim=1),
            torch.where(has_candidate, g_up, torch.inf).argmin(dim=1)
        )

        self.target_intensities = torch.zeros((nverts,), dtype=I.dtype, device=self.device)
        self.target_intensities[found_idx] = I_up[found_idx, target_idxs[found_idx]]

        self.target_pts = self.tmesh.verts.clone()
        self.target_pts[found_idx] = pts_up[found_idx, target_idxs[found_idx]]

        self.target_pos = torch.zeros_like(self.target_intensities)
        self.target_pos[found_idx] = pos_up[target_idxs[found_idx]]

        # Average values across neighbors
        self.has_valid_intensity = (found_idx & ~self.rip_verts_flag)
        self.target_intensities = self.tmesh.smooth_over_neighbors(
            self.target_intensities, mask=self.has_valid_intensity, N_avgs=5, n_hops=1, signed=False
        )

    # COST FUNCTIONS -------------------------------------------------------------------------------

    def cost_curvature(self, weight=1.):
        """
        Curvature constraint
        """
        # Transform sets of 2 hop neighbors into normal/tangent space w.r.t. center vertex
        M, valid = self.tmesh.neighbor_displacements(pair_type='2hop', return_mask=True)
        valid[:, 0] = 0
        
        u = (M * self.tmesh.vert_tangents0.unsqueeze(dim=1)).sum(dim=-1, keepdims=True)
        v = (M * self.tmesh.vert_tangents1.unsqueeze(dim=1)).sum(dim=-1, keepdims=True)
        w = (M * self.tmesh.vert_norms.unsqueeze(dim=1)).sum(dim=-1, keepdims=True)

        # GLM
        if self.target_curvature is None:
            ones = valid.unsqueeze(dim=-1).to(u.dtype)
            B = utils.GLM(torch.cat([u * u, v * v, u, v, ones], dim=-1), w)
            eps = B[:, -1]
        else:
            B = utils.GLM(torch.cat([u * u, 2 * u * v, v * v], dim=-1), w)
            curv = B[:, 0] + B[:, 2]
            eps = self.target_curvature - curv

        # Exclude ripped center verts
        valid_centers = ~self.rip_verts_flag
        n_valid = valid_centers.sum()

        return weight * (0.5 /  n_valid) * (eps[valid_centers] ** 2).sum()

    def cost_intensity(self, weight=1., step_sz=0.1, max_delta=5.):
        """
        Intensity constraint (actually is a constraint to minimize distance to point with the 
        target intensity)
        """
        # Current vs. target intensity difference
        valid = self.has_valid_intensity & (~self.rip_verts_flag)
        n_valid = valid.sum()

        I = self.target_intensities.clone()
        I[valid] = self.timage._interp(self.tmesh.verts[valid]).transpose(0, 1)
        delta_I = (self.target_intensities - I)
        
        if self.separate_loss_types:
            # Project differences onto normal and smooth over neighbors
            N_proj = delta_I.clamp(min=-max_delta, max=max_delta) * self.tmesh.vert_norms
            N_proj_smoothed = self.tmesh.smooth_over_neighbors(
                N_proj, mask=valid, N_avgs=self.n_grad_avgs, n_hops=1, signed=True
            )

            # Compute loss
            cost = (delta_I ** 2).sum()
            return weight * (0.5 / n_valid) * cost, weight * N_proj_smoothed

        else:
            # Compute loss
            cost = torch.where(
                delta_I.abs() <= max_delta,
                0.5 * (delta_I.abs() ** 2),
                max_delta * (delta_I.abs() - 0.5 * max_delta)
            ).sum()

            return weight * (1. / n_valid) * cost
        
    def cost_repulsion(self, weight=1.):
        """
        Repulsion energy cost
        """
        # Q: how to calculate bin ???
        min_dist = 1.0 #* 10 (uncomment for phantoms)
        repulse_k = 1.
        repulse_e = 0.25

        # Distance between vertices in each other's proximity trees
        disps, valid = self.tmesh.neighbor_displacements(
            pair_type='prox', exclude_ripped=True, return_mask=True
        )
        dists = disps.norm(dim=-1) + repulse_e
        n_valid = valid.sum()
        
        # Filter ones outside min_dist
        mask = (dists <= (min_dist - repulse_e)) & valid
        num = mask.sum(dim=1).clamp(min=1)

        # Repulsion energy
        repulsion = ((2. / 3.) * repulse_k / num) * (mask / (dists ** 6)).sum(dim=1)

        return weight * (1. / n_valid) * repulsion.sum() # / n_repulse

    def cost_spring(self, weight=[1., 1.]):
        """
        Spring force constraint
        """
        # Distance between 1hop neighbors
        disps, valid = self.tmesh.neighbor_displacements(
            pair_type='1hop', exclude_ripped=True, return_mask=True
        )

        # Normal component
        N = self.tmesh.vert_norms.unsqueeze(dim=1)
        spring_N = ((N * disps).sum(dim=-1) ** 2).sum(dim=-1)

        # Tangential component
        e0 = self.tmesh.vert_tangents0.unsqueeze(dim=1)
        e1 = self.tmesh.vert_tangents1.unsqueeze(dim=1)
        spring_T = (((e0 * disps).sum(dim=-1) ** 2) + ((e1 * disps).sum(dim=-1) ** 2)).sum(dim=-1)
        
        # Combine
        n_valid = (~self.rip_verts_flag).sum()

        return (0.5 / n_valid) * ((weight[0] * spring_N) + (weight[1] * spring_T)).sum()
    
    # OPTIMIZERS -----------------------------------------------------------------------------------
    def lbfgs_optimize(self, lr=1., history_size=10, max_iters=10, max_steps=50):
        """
        Vertex placement with LBFGS optimizer
        """
        # Closure function that computes at each iteration
        def closure():
            self.losses_dict = self.manual_grad_losses_dict | self.autograd_losses_dict
            n_losses = len(self.losses_dict)
            n_printed = 0
            
            opt.zero_grad()
            self.tmesh._update_vert_properties(X)
            
            loss_str = '' if self.verbose else None
            n_printed = 0

            L_total = X.new_zeros(())
            
            if self.separate_loss_types:
                # Gradient contributions w/ autograd
                for func, weight in self.autograd_losses_dict.items():
                    loss = eval(f'self.{func}(weight)')
                    L_total = L_total + loss

                    if self.verbose:
                        if n_printed > 0:
                            loss_str += ' + '
                        loss_str += f'{loss.item():>.2e} ({"_".join(func.split("_")[1:])})'
                        n_printed += 1
                
                # Manually do the rest (probably just intensity)
                for func, weight in self.manual_grad_losses_dict.items():
                    loss, grad = eval(f'self.{func}(weight)')
                    L_total = L_total + _ManualGradTerm.apply(X, loss.detach(), grad.detach())

                    if self.verbose:
                        if n_printed > 0:
                            loss_str += ' + '
                        loss_str += f'{loss.item():>.2e} ({"_".join(func.split("_")[1:])})'
                        n_printed += 1

                # Back-propagate
                L_total.backward()

            else:
                # Just compute everything with autograd
                for func, weight in self.losses_dict.items():
                    loss = eval(f'self.{func}(weight)')
                    L_total = L_total + loss

                    if self.verbose:
                        if n_printed > 0:
                            loss_str += ' + '
                        loss_str += f'{loss.item():>.2e} ({"_".join(func.split("_")[1:])})'
                        n_printed += 1

                L_total.backward()

            with torch.no_grad():
                X.grad[self.rip_verts_flag] = 0

            if self.verbose:
                if n_losses > 1:
                    loss_str = f'{L_total.item():>.2e} = ' + loss_str
                if self.log is not None:
                    with open(self.log, 'a') as f:
                        f.write(f'{loss_str}\n')
                else:
                    print('Loss =', loss_str)
            
            return L_total

        # Initialize
        X = self.tmesh.verts
        X_init = X.clone()

        opt = torch.optim.LBFGS(
            [X], lr=lr,
            history_size=history_size,
            max_iter=max_iters,
            line_search_fn="strong_wolfe"
        )

        # Run
        for i in range(max_steps):
            if self.verbose:
                if self.log is not None:
                    with open(self.log, 'a') as f:
                        f.write(f'Iteration {i + 1}:')
                else:
                    print(f'Iteration {i + 1}:')

            # Update parameters if adaptive = True
            if i == max_steps // 2 and self.adaptive:
                """
                if self.autograd_losses_dict.get('cost_intensity') is not None:
                    if self.autograd_losses_dict.get('cost_intensity').get('step_size') is not None:
                        self.autograd_losses_dict.get('cost_intensity')['step_size'] /= 4
                    else:
                        self.autograd_losses_dict.get('cost_intensity')['step_size'] = 0.5 / 4
                """
            # Closure function
            opt.step(closure)

            # Refresh cached properties
            if 'cost_repulsion' in self.losses_dict:
                with torch.no_grad():
                    self.tmesh._compute_proximity_trees()
            """
            if 'cost_intensity' in self.losses_dict:
                with torch.no_grad():
                    self._calculate_target_intensities()
            """
        # Update input surfa mesh
        self.mesh.vertices = self.tmesh.verts.detach().cpu()

    def sgd_optimize(self, max_steps=100, lr=0.1, momentum=0.9, nesterov=True):
        """
        Vertex placement with SGD optimizer
        """
        X = self.tmesh.verts
        opt = torch.optim.SGD([X], lr=lr, momentum=momentum, nesterov=nesterov)

        self.losses_dict = self.manual_grad_losses_dict | self.autograd_losses_dict
        n_losses = len(self.losses_dict)

        for i in range(max_steps):
            if self.verbose:
                print(f'Iteration {i + 1}:')
            
            opt.zero_grad()
            self.tmesh._update_vert_properties(X)

            loss_str = '' if self.verbose else None
            n_printed = 0

            L_total = 0.
            #for func, weight in self.losses_dict.items():
            for func, weight in self.autograd_losses_dict.items():
                loss = eval(f'self.{func}(weight)')
                L_total = L_total + loss

                if self.verbose:
                    if n_printed > 0:
                        loss_str += ' + '
                    loss_str += f'{loss.item():>.2e} ({"_".join(func.split("_")[1:])})'
                    n_printed += 1

            L_total.backward()

            for func, weight in self.autograd_losses_dict.items():
                with torch.no_grad():
                    grad = eval(f'self.{func}(weight)')
                    X.grad = X.grad + grad
            
            with torch.no_grad():
                X.grad[self.rip_verts_flag] = 0

            opt.step()

            if self.verbose:
                if n_losses > 1:
                    loss_str = f'{L_total.item():>.2e} = ' + loss_str
                print('Loss =', loss_str)
            
        # Update input surfa mesh
        self.mesh.vertices = self.tmesh.verts.detach().cpu()


# --------------------------------------------------------------------------------------------------
        
class _ManualGradTerm(torch.autograd.Function):
    """
    Forward returns a precomputed scalar cost; backward injects a precomputed gradient w.r.t. X and 
    bypasses autograd
    """
    @staticmethod
    def forward(ctx, X, cost, grad):
        ctx.save_for_backward(grad)
        return cost
    
    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad_output * grad, None, None
