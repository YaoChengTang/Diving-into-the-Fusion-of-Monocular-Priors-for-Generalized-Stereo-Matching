import os
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',)

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8
from core.confidence import OffsetConfidence
from core.refinement import Geometry, Refinement, UpdateHistory
from core.utils.plane import get_pos, convert2patch, predict_disp


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

        if args.slant_builder=="geometry":
            self.geometry_builder = Geometry(args)
        
        # # build offset for interpolation in slant plane
        # # d_p = d_q + a_q\cdot\Delta u_{q\to p} + b_q\cdot\Delta v_{q\to p}
        # if self.args.slant in ["slant_local"]:
        #     factor = 2 ** self.args.n_downsample
        #     delta_center = []
        #     for sub_row_idx in range(0, factor):
        #         for sub_col_idx in range(0, factor):
        #             delta = [-factor/2 + 0.5 + sub_row_idx, -factor/2 + 0.5 + sub_col_idx]
        #             delta_center.append(delta)
        #     delta_center = np.array(delta_center)
        #     delta_pq = []
        #     for row_idx in [-1,0,1]:
        #         for col_idx in [-1,0,1]:
        #             delta = delta_center - np.array([row_idx,col_idx])*factor
        #             delta_pq.append(delta)
        #     delta_pq = np.array(delta_pq).reshape(9,factor,factor,2)
        #     delta_pq = torch.Tensor(delta_pq)
        #     self.delta_pq = nn.Parameter(delta_pq, requires_grad=False)   # (9,factor,factor,2)
        
        if args.refinement is not None and len(args.refinement)>0:
            if self.args.slant is None or len(self.args.slant)==0 :
                dim_disp = 2
            elif self.args.slant in ["slant", "slant_local"] :
                dim_disp = 2*3

            if args.refinement.lower()=="refinement":
                self.refine = Refinement(args, in_chans=256, dim_fea=96, dim_disp=dim_disp)
            else:
                raise Exception("No such refinement: {}".format(args.refinement))
        
        if self.args.update_his:
            self.update_hist = UpdateHistory(args, 128, dim_disp)

        if "local_rank" not in args or args.local_rank==0 :
            logging.info(f"RAFTStereo ~ " +\
                         f"Confidence: {args.confidence}, offset_memory_size: {args.offset_memory_size}, " +\
                         f"offset_memory_last_iter: {args.offset_memory_last_iter}, " +\
                         f"slant: {args.slant}, slant_norm: {args.slant_norm}, " +\
                         f"slant builder: {args.slant_builder}, geo_fusion: {args.geo_fusion}, " +\
                         f"refine: {args.refinement}, refine_win_size: {args.refine_win_size}, num_heads:{args.num_heads}, " +\
                         f"split_win: {args.split_win}, refine_start_itr: {args.refine_start_itr}, " +\
                         f"update_his: {args.update_his} U_thold: {args.U_thold}" )

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

    def upsample_flow(self, flow, mask, params=None):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        if (self.args.slant_builder is None or len(self.args.slant_builder)==0) and\
           self.args.slant in ["slant_local"] and params is not None:
            # d_p = a_q\cdot\Delta u_{q\to p} + b_q\cdot\Delta v_{q\to p} + d_q
            delta_pq = get_pos(H*factor, W*factor, disp=None,
                              slant=self.args.slant,
                              slant_norm=self.args.slant_norm,
                              patch_size=factor,
                              device=flow.device)                                                # (1,2,H*factor,W*factor)
            patch_delta_pq = convert2patch(delta_pq, patch_size=factor, div_last=False).detach() # (1,2,factor*factor,H,W)
            a_p, a_p_flow, \
            b_p, b_p_flow, \
            d_p, d_p_flow  = torch.split(params, 1, dim=1)              # (B,1,H,W)
            abd      = torch.cat([a_p, b_p, d_p], dim=1)                # (B,3,H,W)
            abd_flow = torch.cat([a_p_flow, b_p_flow, d_p_flow], dim=1) # (B,3,H,W)
            d_coord      = predict_disp(abd, patch_delta_pq, patch_size=factor, mul_last=True)      # (B,factor*factor,H,W)
            d_coord_flow = predict_disp(abd_flow, patch_delta_pq, patch_size=factor, mul_last=True) # (B,factor*factor,H,W)
            up_flow      = torch.stack([d_coord, d_coord_flow], dim=1)                              # (B,2,factor*factor,H,W)
            up_flow      = up_flow.view(N, 2*factor*factor, H, W)                                   # (B,2*factor*factor,H,W)
            up_flow      = F.unfold(up_flow, [3,3], padding=1)                                      # (B,2*factor*factor*9,H,W)
            up_flow      = up_flow.view(N, 2, factor, factor, 9, H, W)                              # (B,2,factor,factor,9,H,W)
            up_flow      = up_flow.permute((0,1,4,2,3,5,6))                                         # (B,2,9,factor,factor,H,W)

            # up_flow = F.unfold(factor * flow, [3,3], padding=1)
            # up_flow = up_flow.view(N, D, 9, 1, 1, H, W)
            # a = F.unfold(params[:,:2], [3,3], padding=1)
            # a = a.view(N, D, 9, 1, 1, H, W)
            # b = F.unfold(params[:,2:4], [3,3], padding=1)
            # b = b.view(N, D, 9, 1, 1, H, W)
            # delta_pq = self.delta_pq.view(1,1,9,factor,factor,1,1,2)
            # up_flow = up_flow + a*delta_pq[...,1] + b*delta_pq[...,0]
        else:
            up_flow = F.unfold(factor * flow, [3,3], padding=1)
            up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        if self.args.slant_builder is not None and len(self.args.slant_builder)>0:
            delta_pq = get_pos(H*factor, W*factor, disp=None,
                              slant=self.args.slant,
                              slant_norm=self.args.slant_norm,
                              patch_size=factor,
                              device=flow.device)                                                # (1,2,H*factor,W*factor)
            patch_delta_pq = convert2patch(delta_pq, patch_size=factor, div_last=False).detach() # (1,2,factor*factor,H,W)
            patch_delta_pq = patch_delta_pq.repeat(N,1,1,1,1)
            fit_points = up_flow.reshape(N,D,factor*factor,H,W)                                  # (1,2,factor*factor,H,W)
            fit_points = torch.cat([patch_delta_pq,fit_points], dim=1)                           # (1,4,factor*factor,H,W)
        else:
            fit_points = None
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W), fit_points


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
        flow_predictions_refine = []
        raw_params_list = []
        params_list = []
        confidence_list = []
        offset_memory = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                ## GRU-like exploration
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

                ## acquire confidence
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
                        confidence = self.confidence_computer(fmap1, input_offset_mem)
                else:
                    confidence = None
                confidence_list.append(confidence)

            # in stereo mode, project flow onto epipolar
            if self.args.slant_builder is None and \
               self.args.slant in ["slant", "slant_local"]:
                delta_flow[:,1] = 0.0
                delta_flow[:,3] = 0.0
                delta_flow[:,5] = 0.0
            else:
                delta_flow[:,1] = 0.0

            ## compute current position for following exploration
            if self.args.slant_builder is not None or \
               self.args.slant is None or len(self.args.slant)==0 :
                offset = delta_flow
            elif self.args.slant in ["slant", "slant_local"] :
                # d = a*u + b*v + c
                B,_,H,W = coords0.shape
                if self.args.slant=="slant" :
                    if self.args.slant_norm:
                        norm_range = torch.Tensor([W,H])[None,:,None,None].float().to(coords0.device)
                        offset = delta_flow[:,0:1] * coords0 / norm_range + \
                                 delta_flow[:,2:3] * coords0[:,[1,0]] / norm_range[:,[1,0]] + \
                                 delta_flow[:,4:5]
                    else:
                        offset = delta_flow[:,0:1] * coords0 + \
                                 delta_flow[:,2:3] * coords0[:,[1,0]] + \
                                 delta_flow[:,4:5]
                elif self.args.slant=="slant_local" :
                    offset = delta_flow[:,4:6]
                
                if len(raw_params_list)==0:
                    raw_params_list.append(delta_flow)
                else:
                    raw_params_list.append(raw_params_list[-1].detach() + delta_flow)
            else:
                raise Exception(f"No such slant type {self.args.slant}")
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + offset
            disparity = coords1 - coords0

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1 and \
               self.args.slant_builder is None and \
               (self.args.slant is None or len(self.args.slant)==0) and \
               (self.args.refinement is None or len(self.args.refinement)==0 or not enable_refinement):
                continue
            
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(disparity)
            else:
                params = raw_params_list[-1] if self.args.slant_builder is None and \
                                                self.args.slant in ["slant", "slant_local"] else None
                flow_up, fit_points = self.upsample_flow(disparity, up_mask, params=params)
            flow_up = flow_up[:,:1]
            flow_predictions.append(flow_up)

            # compute geometry
            if self.args.slant_builder is not None and len(self.args.slant_builder)>0:
                ab = self.geometry_builder(fit_points)
                geo = torch.cat([disparity[:,:1],ab], dim=1)
                raw_params_list.append(geo)

            ## manifold geometry refinement
            disparity_refine = None
            if self.args.refinement is not None and len(self.args.refinement)>0 and enable_refinement:
                if itr>=self.args.refine_start_itr:
                    disparity_refine = self.refine(disparity, fmap1, confidence, 
                                            if_shift=(itr-self.args.refine_start_itr)%2>0)
                    coords1 = coords0 + disparity_refine

                    if self.args.update_his:
                        net_list[0] = self.update_hist(net_list[0], disparity_refine)
            
            # upsample refinement
            if disparity_refine is not None:
                if up_mask is None:
                    flow_up_refine = upflow8(disparity_refine)
                else:
                    params = raw_params_list[-1] if self.args.slant in ["slant", "slant_local"] else None
                    flow_up_refine, fit_points = self.upsample_flow(disparity_refine, up_mask, params=params)
                flow_up_refine = flow_up_refine[:,:1]
            else:
                flow_up_refine = None
            flow_predictions_refine.append(flow_up_refine)

            # upsample paramaters
            if self.args.slant_builder is not None:
                params_list = raw_params_list
            elif self.args.slant is not None and len(self.args.slant)>0 and not test_mode:
                # if up_mask is None:
                #     params = upflow8(raw_params_list[-1])
                # else:
                #     params = self.upsample_flow(raw_params_list[-1], up_mask)
                # params = torch.concat([params[:,0:1], params[:,2:3], params[:,4:5]], dim=1)
                # params_list.append(params)
                params_list = [torch.concat([params[:,0:1], params[:,2:3], params[:,4:5]], dim=1) for params in raw_params_list]

        if test_mode:
            if disparity_refine is not None:
                return coords1 - coords0, flow_up_refine
            return coords1 - coords0, flow_up

        if vis_mode:
            return flow_predictions, flow_predictions_refine, confidence_list

        return flow_predictions, flow_predictions_refine, confidence_list, params_list
