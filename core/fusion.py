import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import Beta

from core.extractor import ResidualBlock



class FusionDepth(nn.Module):
    def __init__(self, args, norm_fn='batch', ):
        super(FusionDepth, self).__init__()
        self.args = args
        self.norm_fn = norm_fn

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=True),
        )
        self.down = nn.Sequential(
            ResidualBlock(4, 2*4, self.norm_fn, stride=2),
            ResidualBlock(2*4, 2*4, self.norm_fn, stride=1)
        )
        self.up   = nn.ConvTranspose2d(2*4, 4, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
        )
        
        
    def forward(self, disp, depth, delta_disp):
        x  = disp
        x1 = self.conv1( torch.cat([disp, depth, delta_disp], dim=1) )

        x2 = self.up(self.down(x1))

        x3 = self.conv2( torch.cat([x1,x2], dim=1) )
        
        return x3


class UpdateHistory(nn.Module):
    def __init__(self, args, in_chans1, in_chans2):
        super(UpdateHistory, self).__init__()
        self.conv = nn.Conv2d(in_chans2, in_chans2, kernel_size=1, stride=1, padding=0)
        self.update = nn.Sequential(nn.Conv2d(in_chans1+in_chans2, in_chans1, kernel_size=3, stride=1, padding=1),)
        
    def forward(self, his, disp):
        hist_update = self.update( torch.cat([his,self.conv(disp)], dim=1) )
        return hist_update


class BetaModulator(nn.Module):
    def __init__(self, args, lbp_dim, norm_fn='batch'):
        super(BetaModulator, self).__init__()
        self.norm_fn = norm_fn
        self.modulation_ratio = args.modulation_ratio
        # self.conv_depth = nn.Sequential(
        #     nn.Conv2d(8, 16, kernel_size=1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True),
        # )
        # self.conv_disp = nn.Sequential(
        #     nn.Conv2d(8, 16, kernel_size=1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True),
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(lbp_dim*2, lbp_dim*2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(lbp_dim*2, lbp_dim*2, kernel_size=3, padding=1, bias=True),
        )
        self.down = nn.Sequential(
            ResidualBlock(lbp_dim*2, 64, self.norm_fn, stride=2),
            ResidualBlock(64, 128, self.norm_fn, stride=1)
        )
        self.up   = nn.ConvTranspose2d(128, lbp_dim*2, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(lbp_dim*4, lbp_dim, kernel_size=3, padding=1, bias=False),
            nn.Softplus(),
            nn.Conv2d(lbp_dim, 2, kernel_size=1, padding=0, bias=False),
            nn.Softplus(),
        )
    
    def forward(self, lbp_disp, lbp_depth, itr_ratio):
        x1 = self.conv1( torch.cat([lbp_disp, lbp_depth], dim=1) )
        x2 = self.up(self.down(x1))
        beta_paras = self.conv2( torch.cat([x1,x2], dim=1) ) + 1  # enforcing alpha>=1, beta>=1

        # build Beta distribution
        alpha, beta = torch.split(beta_paras, 1, dim=1)
        distribution = Beta(alpha, beta)

        if self.training:
            modulation = distribution.rsample()
        else:
            modulation = distribution.mean
        
        # modulation = modulation*2 - 1
        modulation = 1 + modulation * (self.modulation_ratio * itr_ratio)   # we hope modulation has less effect at the first several iterations as the disp is unreliable and the lcoal LBP disp is unreliable
        return modulation
