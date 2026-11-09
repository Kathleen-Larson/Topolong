import os, time
import numpy as np
import surfa as sf
import torch

import utils
from tensor_image import TensorImage

VALID_HEMIS = ['lh', 'rh']
VALID_SURFS = ['white', 'pial']

# --------------------------------------------------------------------------------------------------

class BorderStatsGenerator:
    def __init__(
            self,
            T1,                 # T1 image
            seg,                # Label map image to define borders
            wm=None,            # WM image (probably will remove later)
            lut=None,           # Lookup table for values in seg
            hemi='lh',          # hemisphere (lh or rh)
            csf_min=10.,        # Min value for CSF class
            csf_max=40.,        # Max value for CSF class
            gm_min=30.,         # Min GM value
            gm_max=95.,         # Max GM value
            gm_pial_min=40.,    # Min GM value at pial surface
            gm_pial_max=75.,    # Max GM value at pial surface
            gm_white_min=70.,   # Min GM value at white surface
            wm_max=120.,        # Max WM value
            wm_white_min=85.,   # Min WM value at white surface
            wm_white_max=105.,  # Max WM value at white surface
            variable_sigma=3.,  # Scaling parameter from AutoDetGWStats
            scale_down=0.2,     # Scaling parameter from AutoDetGWStats
            device=None,
    ):
        # Parse inputs
        self.lut = lut
        self.hemi = hemi if hemi in VALID_HEMIS else utils.fatal(
            f'[BorderStatsGenerator] error: must provide valid hemi ({hemi} was input)'
        )

        self.device = torch.device(
            'cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu'
        ) if not isinstance(device, torch.device) else device

        self.T1 = T1.to(self.device)
        self.wm = wm.to(self.device) if wm is not None else None
        self.seg = seg.to(self.device)
        
        # Template intensity values
        self.csf_min = csf_min
        self.csf_max = csf_max
        self.gm_min = gm_min
        self.gm_max = gm_max
        self.gm_white_min = gm_white_min
        self.gm_pial_min = gm_pial_min
        self.gm_pial_max = gm_pial_max
        self.wm_max = wm_max
        self.wm_white_min = wm_white_min
        self.wm_white_max = wm_white_max

        self.var_sigma = variable_sigma
        self.scale = scale_down
        
        # Get label values
        self.wm_labels = torch.tensor(
            utils.search_lut(lut, 'Cerebral-White-Matter')
        ).to(self.device)
        self.gm_labels = torch.tensor(
            utils.search_lut(lut, 'Cerebral-Cortex')
        ).to(self.device)
        
    def create_border_masks(self):
        """
        Create masks of tissue borders
        """
        # Set up dilation filter
        structuring_element = torch.zeros((3,) * 3, dtype=torch.int)
        structuring_element[
            torch.tensor([1, 1, 2, 0, 1, 1, 1]),
            torch.tensor([1, 1, 1, 1, 1, 0, 2]),
            torch.tensor([1, 2, 1, 1, 0, 1, 1])
        ] = 1.
        dilate_filter = utils.init_convolution(
            in_shape=self.seg.unsqueeze(dim=0).unsqueeze(dim=0).shape,
            out_shape=self.seg.unsqueeze(dim=0).unsqueeze(dim=0).shape,
            conv_weight_data=structuring_element,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            device=self.device,
        )

        # Tissue masks
        wm_mask = torch.isin(self.seg, self.wm_labels).to(self.seg.dtype)
        gm_mask = torch.isin(self.seg, self.gm_labels).to(self.seg.dtype)
        csf_mask = torch.where(self.seg == 0, 1, 0).to(self.seg.dtype)
        
        # Get adjacent boundary voxels
        white_border = torch.logical_and(
            utils.apply_convolution(wm_mask, dilate_filter) > 0,
            utils.apply_convolution(gm_mask, dilate_filter) > 0
        ).to(torch.int)

        wm_white = wm_mask * white_border
        gm_white = gm_mask * white_border
        
        pial_border = torch.logical_and(
            utils.apply_convolution(gm_mask, dilate_filter) > 0,
            utils.apply_convolution(csf_mask, dilate_filter) > 0
        ).to(torch.int)

        gm_pial = gm_mask * pial_border
        csf_pial = csf_mask * pial_border

        """
        # Now compare to using wm.mgz
        wm_mask_2 = torch.where(self.wm > 5, 1, 0)
        gm_mask_2 = torch.where(self.wm <= 5, 1, 0)

        white_border_2 = torch.logical_and(
            utils.apply_convolution(wm_mask_2, dilate_filter) > 0,
            utils.apply_convolution(gm_mask_2, dilate_filter) > 0
        ).to(torch.int)

        wm_white_2 = wm_mask_2 * white_border_2
        gm_white_2 = gm_mask_2 * white_border_2
        """
        return wm_white, gm_white, gm_pial, csf_pial
        
        
    def compute_border_stats(self, use_modes=True, gm_lo=None, gm_hi=None, which=None):
        """
        A simplified version of the FS AutoDetGMStats::AutoDetStats() function
        """

        if which is not None and which not in ['white', 'pial']:
            utils.fatal(
                '[BorderStatsGenerator.compute_border_stats()] error: must input which="white" or '
                'which="pial" if using (default is None)'
            )
        
        # Create masks of borders
        gm_lo = self.gm_min if gm_lo is None else gm_lo
        gm_hi = self.gm_max if gm_hi is None else gm_hi
        
        wm_white, gm_white, gm_pial, csf_pial = self.create_border_masks()

        wm_white[self.T1 < (gm_lo + gm_hi) / 2] = 0
        gm_white[torch.logical_or(self.T1 < gm_lo, self.T1 > gm_hi)] = 0
        
        # Compute statistics for gm/wm boundary
        wm_white_mean = (
            self.T1[wm_white == 1].mode().values if use_modes
            else self.T1[wm_white == 1].mean()
        ).item()
        wm_white_std = self.T1[wm_white == 1].std().item()

        gm_white_mean = (
            self.T1[gm_white == 1].mode().values if use_modes
            else self.T1[gm_white == 1].mean()
        ).item()
        gm_white_std = self.T1[gm_white == 1].std().item()

        # Thing with the modes and surfaces (not sure this is necessary...)

        wm_white_min = gm_white_mean
        wm_white_max = wm_white_mean + wm_white_std
        gm_white_min = gm_white_mean - gm_white_std
        gm_pial_min = gm_white_mean - self.var_sigma * gm_white_std
        gm_pial_max = gm_white_mean - gm_white_std
        gm_max = wm_white_mean - wm_white_std
        gm_mid = (gm_max + gm_pial_min) / 2
        csf_max = gm_white_mean - max(0.5, self.var_sigma * gm_white_std)
        
        # Do sanity checks
        def sanity_check(val, targ):
            return val if val > self.scale * targ else self.scale * targ

        wm_white_min = sanity_check(wm_white_min, self.wm_white_min)
        wm_white_max = sanity_check(wm_white_max, self.wm_white_max)
        gm_white_min = sanity_check(gm_white_min, self.gm_white_min)
        gm_pial_min = sanity_check(gm_pial_min, self.gm_pial_min)
        gm_pial_max = sanity_check(gm_pial_max, self.gm_pial_max)
        gm_max = sanity_check(gm_max, self.gm_max)
        csf_max = sanity_check(csf_max, self.csf_max)

        # Compile output dictionary
        white_dict = {
            'border_hi': wm_white_max,
            'border_lo': wm_white_min,
            'outside_hi': wm_white_max,
            'outside_lo': gm_white_min,
            'inside_hi': self.wm_max
        }

        pial_dict = {
            'border_hi': gm_pial_max,
            'border_lo': gm_pial_min,
            'outside_hi': (csf_max + gm_pial_max) / 2,
            'outside_lo': self.csf_min,
            'inside_hi': gm_max
        }

        if which == 'white':
            return white_dict
        if which == 'pial':
            return pial_dict
        else:
            return white_dict, pial_dict
        
# with wm.mgz: white_mean = 96.7222 +/- 9.36334, gray_mean = 65.4349 +/- 12.1059
