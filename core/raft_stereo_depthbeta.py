import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update_disp import DispBasicMultiUpdateBlock
from core.extractor import BasicEncoder, ResidualBlock
from core.extractor_depthany import DepthAnyExtractor
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import hor_coords_grid
from core.geometry import LBPEncoder
from core.fusion import BetaModulator


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class RAFTStereoDepthBeta(nn.Module):
    def __init__(self, args):
        super(RAFTStereoDepthBeta, self).__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.cnet = DepthAnyExtractor(model_dir=args.depthany_model_dir,
                                      output_dim=[args.hidden_dims, context_dims], 
                                      norm_fn=args.context_norm, 
                                      downsample=args.n_downsample)
        self.update_block = DispBasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.lbp_encoder = LBPEncoder(args=args)
        self.modulater = BetaModulator(args, lbp_dim=self.lbp_encoder.num_neighbors)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def initialize_disp(self, img):
        """ Disparity is represented as difference between two horizontal coordinate grids disp = hor_coords1 - hor_coords0"""
        N, _, H, W = img.shape

        hor_coords0 = hor_coords_grid(N, H, W).to(img.device)
        hor_coords1 = hor_coords_grid(N, H, W).to(img.device)

        return hor_coords0, hor_coords1

    def upsample_disp(self, disp, mask):
        """ Upsample disp field [H/8, W/8, 1] -> [H, W, 1] using convex combination """
        N, D, H, W = disp.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(factor * disp, [3,3], padding=1)
        up_disp = up_disp.view(N, D, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, D, factor*H, factor*W)

    def rescale_modulation(self, itr, iters):
        # we hope modulation has less effect at the first several iterations as the disp is unreliable and the lcoal LBP disp is unreliable
        if self.args.modulation_alg == "linear":
            ratio = self.args.modulation_ratio * itr / iters
        elif self.args.modulation_alg == "sigmoid":
            ratio = self.args.modulation_ratio * 1 / (1 + np.exp(-2 * (itr - 5)))
        else:
            raise Exception("Not supported modulation_alg: {}".format(self.args.modulation_alg))
        return ratio

    def forward(self, image1, image2, iters=12, disp_init=None, test_mode=False, vis_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                # cnet_list: [[(128,248,360), (128,248,360)], [(128,124,180),(128,124,180)], [(128,62,90),(128,62,90)]]
                cnet_list, depth = self.cnet(image1, num_layers=self.args.n_gru_layers)
                # fmap1: (128,248,360), fmap2: (128,248,360)
                fmap1, fmap2 = self.fnet([image1, image2])
            
            # from IPython import embed
            # embed()

            depth_lbp = self.lbp_encoder(depth)

            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda": # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda": # Faster version of alt
            corr_block = AlternateCorrBlock
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        hor_coords0, hor_coords1 = self.initialize_disp(net_list[0])

        if disp_init is not None:
            hor_coords1 = hor_coords1 + disp_init

        disp_predictions = []
        modulation_predictions = []
        for itr in range(iters):
            hor_coords1 = hor_coords1.detach()
            corr = corr_fn(hor_coords1) # index correlation volume
            disp = hor_coords1 - hor_coords0

            with autocast(enabled=self.args.mixed_precision):
                disp_lbp = self.lbp_encoder(disp)
                modulation = self.modulater(disp_lbp, depth_lbp)

                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_disp = self.update_block(net_list, inp_list, corr, disp, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

                delta_disp = delta_disp * (1 + modulation * self.rescale_modulation(itr, iters)) 

            # F(t+1) = F(t) + \Delta(t)
            hor_coords1 = hor_coords1 + delta_disp

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(hor_coords1 - hor_coords0, up_mask)
            
            disp_predictions.append(disp_up)

            if vis_mode:
                modulation_predictions.append(modulation)

        if test_mode:
            return hor_coords1 - hor_coords0, disp_up

        if vis_mode:
            return {"disp_predictions": disp_predictions, 
                    "depth": depth, 
                    "modulation_predictions": modulation_predictions}

        return disp_predictions