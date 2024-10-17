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
    def __init__(self, args, lbp_dim, hidden_dim=None, norm_fn='batch'):
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
        if hidden_dim is None:
            hidden_dim = lbp_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(lbp_dim*2, hidden_dim*2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1, bias=True),
        )
        down_dim = 64 if hidden_dim*2<64 else 128
        self.down = nn.Sequential(
            ResidualBlock(hidden_dim*2, down_dim, self.norm_fn, stride=2),
            ResidualBlock(down_dim, 128, self.norm_fn, stride=1)
        )
        self.up   = nn.ConvTranspose2d(128, hidden_dim*2, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim*4, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.Softplus(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1, padding=0, bias=False),
            nn.Softplus(),
        )
    
    def forward(self, lbp_disp, lbp_depth, out_distribution=False):
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
        
        if not out_distribution:
            return modulation
        return modulation, distribution
        
        # # modulation = modulation*2 - 1
        # modulation_rescale = 1 + modulation * (self.modulation_ratio * itr_ratio)   # we hope modulation has less effect at the first several iterations as the disp is unreliable and the lcoal LBP disp is unreliable
        # return modulation_rescale



class RefinementMonStereo(nn.Module):
    def __init__(self, args, norm_fn='batch', hidden_dim=32):
        super(RefinementMonStereo, self).__init__()
        self.args = args

        corr_channel = self.args.corr_levels * (self.args.corr_radius*2 + 1)
        self.conf_estimate = nn.Sequential(
            nn.Conv2d(corr_channel, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1, padding=0),
            nn.)
        self.norm_conf = nn.Sigmoid()
        
        self.mono_params_estimate = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1, padding=0))

        factor = 2**self.args.n_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim+1, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))
        
    def forward(self, disp, depth, hidden, cost_volume, Beta_distribution):
        conf = self.conf_estimate(cost_volume)
        conf_normed = self.norm_conf(conf)

        mono_params = self.mono_params_estimate( torch.cat([disp, depth], dim=1) )
        a, b = torch.split(mono_params, 1, dim=1)
        depth_registered = depth * a + b
        
        disp = disp * conf_normed + (1-conf_normed) * depth_registered

        up_mask= self.mask( torch.cat([hidden, disp], dim=1) )
        
        return disp, up_mask, depth_registered, conf