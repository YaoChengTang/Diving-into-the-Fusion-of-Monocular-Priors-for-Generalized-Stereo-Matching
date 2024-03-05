import os
import sys
import logging
import numpy as np
from collections import OrderedDict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',)

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.plane import convert2patch



class Geometry_MLP(nn.Module):
    def __init__(self, args):
        super(Geometry, self).__init__()
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
        factor = 2 ** self.args.n_downsample
        points = torch.cat([img_coord, disparity_up], dim=1)          # (1,3,factor*H,factor*W)

        rest_params = self.reg(points)                                   # (1,5,H,W)
        params = torch.cat([disparity,rest_params], dim=1)             # (1,6,H,W)
        return params