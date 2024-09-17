import re
import os
import sys
import logging
import numpy as np
from collections import OrderedDict

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',)

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.plane import convert2patch



class Geometry_MLP(nn.Module):
    def __init__(self, args):
        super(Geometry_MLP, self).__init__()
        self.args = args
        self.reg = nn.Sequential(
            nn.Linear(3,3),
            nn.Linear(3,2),
        )
        
        if args.geo_fusion.lower()=="max":
            self.fusion = nn.AdaptiveMaxPool1d(1)
        elif args.geo_fusion.lower()=="mean":
            self.fusion = nn.AdaptiveAvgPool1d(1)
        else:
            raise Exception(f"{args.geo_fusion} is not supported")
        
    def forward(self, img_coord, flow_up):
        # (1,4,factor*factor,H,W)
        factor = 2 ** self.args.n_downsample
        fit_points = torch.cat([img_coord, flow_up], dim=1)
        fit_points = convert2patch(fit_points, 
                                   patch_size=factor, 
                                   div_last=False)                   # (1,3,factor*factor,H,W)

        A = fit_points[:,:3].permute((0,2,3,4,1))                    # (1,factor*factor,H,W,3)
        ab_proposals = self.reg(A)                                   # (1,factor*factor,H,W,2)
        B,L,H,W,C = ab_proposals.shape
        ab = self.fusion(ab_proposals.view(B,L,-1).transpose(-1,-2)) # (1,H*W*2,1)
        ab = ab.view(B,H,W,C).permute((0,3,1,2))                     # (1,2,H,W)
        geo = torch.cat([disparity[:,:1],ab], dim=1)
        return ab


class Geometry_Conv(nn.Module):
    def __init__(self, args):
        super(Geometry_Conv, self).__init__()
        self.args = args
        self.reg = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 5, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(5, 5, kernel_size=1, padding=0, stride=1),
        )
        
    def forward(self, img_coord, disparity_up, disparity):
        # img_coord: (1,2,H*factor,W*factor)
        # disparity_up: (1,1,H*factor,W*factor)
        # disparity: (1,1,H,W)
        # factor = 2 ** self.args.n_downsample

        # points = torch.cat([img_coord, disparity_up], dim=1)          # (1,3,factor*H,factor*W)
        points = torch.cat([img_coord, disparity_up.detach()], dim=1)   # (1,3,factor*H,factor*W)

        rest_params = self.reg(points)                                  # (1,5,H,W)
        params = torch.cat([disparity,rest_params], dim=1)              # (1,6,H,W)
        return params


class Geometry_Conv_Split(nn.Module):
    def __init__(self, args):
        super(Geometry_Conv_Split, self).__init__()
        self.args = args
        self.encode = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
        )
        self.decode_plane = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, 2, kernel_size=1, padding=0, stride=1),
        )
        self.decode_curvature = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, 3, kernel_size=1, padding=0, stride=1),
        )
        
    def forward(self, img_coord, disparity_up, disparity):
        # img_coord: (1,2,H*factor,W*factor)
        # disparity_up: (1,1,H*factor,W*factor)
        # disparity: (1,1,H,W)
        # factor = 2 ** self.args.n_downsample
        points = torch.cat([img_coord, disparity_up], dim=1)          # (1,3,factor*H,factor*W)

        latten = self.encode(points)                                  # (1,8,factor*H/2,factor*W/2)
        plane_ab = self.decode_plane(latten)                          # (1,2,H,W)
        hessian_g = self.decode_curvature(latten)                     # (1,3,H,W)
        params = torch.cat([disparity,plane_ab,hessian_g], dim=1)     # (1,6,H,W)
        return params


class LBPEncoder(nn.Module):
    """
    Computes the modified Local Binary Patterns (LBP) of an image using custom neighbor offsets.
    """
    def __init__(self, args):
        super(LBPEncoder, self).__init__()
        self.args = args
        self.lbp_neighbor_offsets = self._parse_offsets(self.args.lbp_neighbor_offsets)

        self._build_lbp_kernel()
        self.sigmoid = nn.Sigmoid()
    
    def _build_lbp_kernel(self):
        # Determine the kernel size based on the maximum offset
        self.num_neighbors = len(self.lbp_neighbor_offsets)
        self.max_offset = int(np.abs(self.lbp_neighbor_offsets).max())
        self.kernel_size = 2 * self.max_offset + 1
        self.padding = self.max_offset

        # Initialize the convolution layer for depthwise convolution
        self.lbp_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.num_neighbors,
            kernel_size=self.kernel_size,
            padding=self.padding,
            padding_mode="replicate",
            bias=False,
            groups=1  # Since in_channels=1, groups=1 makes it depthwise
        )

        self.lbp_weight = torch.zeros(self.num_neighbors, 1, 
                                    self.kernel_size, self.kernel_size).float()
        center_y, center_x = self.max_offset, self.max_offset
        for idx, (dy, dx) in enumerate(self.lbp_neighbor_offsets):
            # Compute the position in the kernel for the neighbor
            y, x = center_y + dy, center_x + dx
            if 0 <= y < self.kernel_size and 0 <= x < self.kernel_size:
                self.lbp_weight[idx, 0, y, x] = 1.0
                self.lbp_weight[idx, 0, center_y, center_x] = -1.0
            else:
                raise ValueError(f"Offset ({dy}, {dx}) is out of kernel bounds.")
        
        # Assign the weight to the convolution layer
        self.lbp_conv.weight = nn.Parameter(self.lbp_weight)
        self.lbp_conv.weight.requires_grad = False  # Ensure weights are not updated during training
    
    def _parse_offsets(self, offsets_str):
        """
        Parses a string to extract neighbor offsets.

        Parameters:
            offsets_str (str): String defining neighbor offsets, e.g., "(-1,-1), (1,1), (-1,1), (1,-1)"

        Returns:
            list of tuples: List of neighbor offsets.
        """
        # extract coordinate pairs
        pattern = r'\((-?\d+),\s*(-?\d+)\)'
        matches = re.findall(pattern, offsets_str)
        if not matches:
            raise ValueError(offsets_str + ": not suppoted format, please check it!")
        offsets = [(int(y), int(x)) for y, x in matches]
        return np.array(offsets)
        
        
    def forward(self, img):
        """
        Parameters:
            img (torch.Tensor): Grayscale image tensor of shape [N, 1, H, W].
        Returns:
            torch.Tensor: Modified LBP image of shape [N, C, H, W].
        """
        with torch.no_grad():
            # Apply convolution to compute differences directly
            differences = self.lbp_conv(img)  # Shape: [1, N, H, W] due to padding

            # Apply sigmoid to the differences to get encoding values between 0 and 1
            encoding = self.sigmoid(differences)  # Shape: [1, N, H, W]
        return encoding
