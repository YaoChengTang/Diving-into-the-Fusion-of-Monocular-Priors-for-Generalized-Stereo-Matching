import os
import sys
import logging
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8, LoggerCommon
from core.confidence import OffsetConfidence
from core.refinement import Refinement, UpdateHistory
from core import geometry as GEO
from core.utils.plane import get_pos, convert2patch, predict_disp

logger = LoggerCommon("ARCHI")

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

class RAFTStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
        
        if args.confidence:
            self.confidence_computer = OffsetConfidence(args)

        if args.geo_estimator=="geometry_mlp":
            self.geometry_builder = GEO.Geometry_MLP(args)
        elif args.geo_estimator=="geometry_conv":
            self.geometry_builder = GEO.Geometry_Conv(args)
        elif args.geo_estimator=="geometry_conv_split":
            self.geometry_builder = GEO.Geometry_Conv_Split(args)
        
        if args.refinement is not None and len(args.refinement)>0:
            if self.args.slant is None or len(self.args.slant)==0 :
                dim_disp = 1
            elif self.args.slant in ["slant", "slant_local"] :
                dim_disp = 6

            if args.refinement.lower()=="refinement":
                self.refine = Refinement(args, in_chans=256, dim_fea=96, dim_disp=dim_disp)
            else:
                raise Exception("No such refinement: {}".format(args.refinement))
        
        if self.args.update_his:
            self.update_hist = UpdateHistory(args, 128, dim_disp)

        logger.info(f"RAFTStereo ~ " +\
                    f"Confidence: {args.confidence}, offset_memory_size: {args.offset_memory_size}, " +\
                    f"offset_memory_last_iter: {args.offset_memory_last_iter}, " +\
                    f"slant: {args.slant}, slant_norm: {args.slant_norm}, " +\
                    f"geo estimator: {args.geo_estimator}, geo_fusion: {args.geo_fusion}, " +\
                    f"refine: {args.refinement}, refine_win_size: {args.refine_win_size}, num_heads:{args.num_heads}, " +\
                    f"split_win: {args.split_win}, refine_start_itr: {args.refine_start_itr}, " +\
                    f"update_his: {args.update_his}, U_thold: {args.U_thold}, " +\
                    f"stop_freeze_bn: {args.stop_freeze_bn}" )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)

        img_coord = None
        if self.args.geo_estimator is not None and len(self.args.geo_estimator)>0:
            img_coord = get_pos(H*factor, W*factor, disp=None,
                                slant=self.args.slant,
                                slant_norm=self.args.slant_norm,
                                patch_size=factor,
                                device=flow.device)                                                # (1,2,H*factor,W*factor)
            img_coord = img_coord.repeat(N,1,1,1)
        
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W), img_coord
    
    def upsample_geo(self, mask, params):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = params.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)                                                        # (B,1,9,factor,factor,H,W)

        # d_p = a_q\cdot\Delta u_{q\to p} + b_q\cdot\Delta v_{q\to p} + d_q
        delta_pq = get_pos(H*factor, W*factor, disp=None,
                            slant=self.args.slant,
                            slant_norm=self.args.slant_norm,
                            patch_size=factor,
                            device=params.device)                                                # (1,2,H*factor,W*factor)
        patch_delta_pq = convert2patch(delta_pq, patch_size=factor, div_last=False).detach()     # (1,2,factor*factor,H,W)

        disp = predict_disp(params, patch_delta_pq, patch_size=factor, mul_last=True)            # (B,factor*factor,H,W)
        
        disp = F.unfold(disp, [3,3], padding=1)                                                  # (B,factor*factor*9,H,W)
        disp = disp.view(N, 1, factor, factor, 9, H, W)                                          # (B,1,factor,factor,9,H,W)
        disp = disp.permute((0,1,4,2,3,5,6))                                                     # (B,1,9,factor,factor,H,W)

        disp = torch.sum(mask * disp, dim=2)                                                     # (B,1,factor,factor,H,W)
        disp = disp.permute(0, 1, 4, 2, 5, 3)                                                    # (B,1,H,factor,W,factor)
        return disp.reshape(N, 1, factor*H, factor*W)


    def forward(self, image1, image2, iters=12, flow_init=None, 
                test_mode=False, vis_mode=False, enable_refinement=True):
        """ Estimate optical flow between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])
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

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        disp_predictions = []
        disp_predictions_refine = []
        params_list = []
        params_list_refine = []
        confidence_list = []
        offset_memory = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                ## first-stage in geometry estimation
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

                ## region detection: acquire confidence
                if self.args.confidence:
                    offset_memory.append(delta_flow[:,0:2])
                    if itr<self.args.offset_memory_size:
                        confidence = None
                    else:
                        if self.args.offset_memory_last_iter<0 or itr<=self.args.offset_memory_last_iter:
                            input_offset_mem = offset_memory[-self.args.offset_memory_size:]
                        else:
                            start_itr = self.args.offset_memory_last_iter - self.args.offset_memory_size
                            end_itr   = self.args.offset_memory_last_iter
                            input_offset_mem = offset_memory[start_itr:end_itr]
                        confidence = self.confidence_computer(inp_list[0], input_offset_mem)
                else:
                    confidence = None
                confidence_list.append(confidence)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            flow = coords1 - coords0

            # We do not need to upsample or output intermediate results in test_mode for raftStereo
            if test_mode and itr < iters-1 and \
               (self.args.refinement is None or len(self.args.refinement)==0):
                continue
            
            # upsample disparity map
            if up_mask is None:
                flow_up = upflow8(flow)
            else:
                flow_up, img_coord = self.upsample_flow(flow, up_mask)
            flow_up = flow_up[:,:1]
            flow_predictions.append(flow_up)

            # second-stage in geometry estimation
            geo_params = None
            disparity = -flow[:,:1]
            if self.args.geo_estimator is not None and len(self.args.geo_estimator)>0:
                geo_params = self.geometry_builder(img_coord, -flow_up, disparity)
                
                disp_up = self.upsample_geo(up_mask, params=geo_params)
            params_list.append(geo_params)
            disp_predictions.append(disp_up)

            ## curvature-aware propagation
            disparity_refine = None
            geo_params_refine = None
            if self.args.refinement is not None and len(self.args.refinement)>0 and enable_refinement:
                if itr>=self.args.refine_start_itr:
                    geo_params_refine = self.refine(geo_params, inp_list[0], confidence, 
                                            if_shift=(itr-self.args.refine_start_itr)%2>0)
                    coords1 = coords0 - geo_params_refine[:,:1]
                    disparity_refine = geo_params_refine[:,:1]
                    ### update hidden state
                    if self.args.update_his:
                        net_list[0] = self.update_hist(net_list[0], -disparity_refine)
            params_list_refine.append(geo_params_refine)
            
            # upsample refinement
            disp_up_refine = None
            if geo_params_refine is not None:
                disp_up_refine = self.upsample_geo(up_mask, params=geo_params_refine)
                # disp_up_refine = disp_up_refine[:,:1]
            disp_predictions_refine.append(disp_up_refine)

        if test_mode:
            if self.args.refinement is not None and len(self.args.refinement)>0 and enable_refinement:
                return coords1 - coords0, flow_up_refine
            return coords1 - coords0, flow_up

        if vis_mode:
            return flow_predictions, disp_predictions, disp_predictions_refine, confidence_list

        return flow_predictions, disp_predictions, disp_predictions_refine, confidence_list, params_list, params_list_refine
