import os
import sys
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.utils import coords_grid, disparity_computation
from core.utils.utils import LoggerCommon

logger = LoggerCommon("LOSS")

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

def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        if not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any():
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
            flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def my_loss(res, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    pass

class Loss(nn.Module):
    def __init__(self, loss_gamma=0.9, max_flow=700, loss_zeta=0.3,
                 smoothness=None, slant=None, slant_norm=False, 
                 ner_kernel_size=3, ner_weight_reduce=False,
                 local_rank=None, mixed_precision=True,
                 args=None):
        super(Loss, self).__init__()
        self.loss_gamma = loss_gamma
        self.loss_zeta = loss_zeta 
        self.max_flow = max_flow
        self.smoothness = smoothness
        self.mixed_precision = mixed_precision
        self.conf_disp = args.conf_disp
        self.args = args

        if self.smoothness is not None and len(self.smoothness)>0:
            self.smooth_loss_computer = SmoothLoss(self.smoothness, 
                                                   slant=slant, 
                                                   slant_norm=slant_norm, 
                                                   kernel_size=ner_kernel_size,
                                                   ner_weight_reduce=ner_weight_reduce)
        
        logger.info(f"smoothness: {smoothness}, " +\
                    f"slant: {slant}, slant_norm: {slant_norm}, " +\
                    f"ner_kernel_size: {ner_kernel_size}, " +\
                    f"ner_weight_reduce: {ner_weight_reduce}, " +\
                    f"conf_disp: {self.conf_disp}. " )
    
    def forward(self, flow_preds, flow_gt, valid, 
                disp_preds=None, disp_preds_refine=None, 
                confidence_list=None, 
                params_list=None, params_list_refine=None,  
                plane_abc=None, 
                imgL=None, imgR=None, 
                global_batch_num=None,):
        """ Loss function defined over sequence of flow predictions """
        n_predictions = len(flow_preds)
        assert n_predictions >= 1
        flow_loss = 0.0
        disp_loss = 0.0
        disp_refine_loss = 0.0
        smooth_loss = 0.0
        confidence_loss = 0.0
        params_loss = 0.0
        params_refine_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()

        # exclude extremly large displacements
        valid = ((valid >= 0.5) & (mag < self.max_flow)).unsqueeze(1)
        assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
        assert not torch.isinf(flow_gt[valid.bool()]).any()

        for i in range(n_predictions):
            assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = self.loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)

            # confidence loss
            if confidence_list[i] is not None and \
               (self.args.offset_memory_last_iter<0 or \
               (self.args.offset_memory_last_iter>0 and i<=self.args.offset_memory_last_iter)):
                with autocast(enabled=self.mixed_precision):
                    gt_error = (flow_preds[i].detach() - flow_gt).abs().detach()
                    gt_error = F.interpolate(gt_error,scale_factor=1/4,mode='bilinear')
                    # confidence_loss += i_weight * F.smooth_l1_loss(confidence_list[i], gt_error)
                    # confidence_loss += i_weight * F.binary_cross_entropy_with_logits(confidence_list[i], 
                    #                                                      torch.sigmoid(gt_error-4))
                    gt_conf = (gt_error>4).float()
                    weight = torch.pow(F.sigmoid(confidence_list[i])-gt_conf, 2)
                    tmp_confidence_loss = (1+gt_conf*0.5) * weight *\
                                          F.binary_cross_entropy_with_logits(confidence_list[i], 
                                                                            gt_conf, reduction='none')
                    confidence_loss += i_weight * tmp_confidence_loss.mean()

            # flow loss
            i_loss = (flow_preds[i] - flow_gt).abs()
            if self.conf_disp and global_batch_num>3 and confidence_list[i] is not None:
                weight = F.interpolate(confidence_list[i],scale_factor=4,mode='bilinear')
                i_loss = i_loss * (F.sigmoid(weight.detach()/3)*1.5 + 1)
            assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
            flow_loss += i_weight * i_loss[valid.bool()].mean()

            # disparity loss
            if disp_preds is not None and len(disp_preds)>0 and disp_preds[i] is not None:
                i_loss = (-disp_preds[i] - flow_gt).abs()
                disp_loss += i_weight * i_loss[valid.bool()].mean()
            
            # plane loss
            if params_list is not None and len(params_list)>0 and plane_abc is not None and plane_abc.shape[1]==3:
                # print("~"*30, params_list[-1].shape, plane_abc.shape)
                i_loss = (params_list[i] - plane_abc).abs()
                params_loss += i_weight * 0.5 * i_loss.mean()

            # refinement loss
            if disp_preds_refine is not None and len(disp_preds_refine)>0 and disp_preds_refine[i] is not None:
                i_loss = (-disp_preds_refine[i] - flow_gt).abs()
                disp_refine_loss += i_weight * i_loss[valid.bool()].mean()
            
            # plane loss
            if params_list_refine is not None and len(params_list_refine)>0 and plane_abc is not None and plane_abc.shape[1]==3:
                # print("~"*30, params_list_refine[-1].shape, plane_abc.shape)
                i_loss = (params_list_refine[i] - plane_abc).abs()
                params_refine_loss += i_weight * 0.5 * i_loss.mean()

            if i>n_predictions//2:
                with autocast(enabled=self.mixed_precision):
                    if self.smoothness=="gradient":
                        smooth_loss += i_weight * self.smooth_loss_computer(flow_preds[i], imgL).mean()
                    elif self.smoothness=="curvature":
                        smooth_loss += i_weight * self.smooth_loss_computer(params_list[i], imgL).mean()

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        if disp_preds is not None and len(disp_preds)>0 and disp_preds[-1] is not None:
            epe = torch.sum((-disp_preds[-1] - flow_gt)**2, dim=1).sqrt()
            epe = epe.view(-1)[valid.view(-1)]
            metrics.update({'epe_disp': epe.mean().item(),
                            '3px_disp': (epe < 3).float().mean().item(),})
        
        if disp_preds_refine is not None and len(disp_preds_refine)>0 and disp_preds_refine[-1] is not None:
            epe = torch.sum((-disp_preds_refine[-1] - flow_gt)**2, dim=1).sqrt()
            epe = epe.view(-1)[valid.view(-1)]
            metrics.update({'epe_disp_refine': epe.mean().item(),
                            '3px_disp_refine': (epe < 3).float().mean().item(),})

        if self.smoothness is not None and len(self.smoothness)>0:
            loss = flow_loss + disp_loss + params_loss + disp_refine_loss + params_refine_loss + confidence_loss + self.loss_zeta * smooth_loss
        else:
            loss = flow_loss + disp_loss + params_loss + disp_refine_loss + params_refine_loss + confidence_loss
            smooth_loss = torch.Tensor([0.0]).to(flow_loss.device)
        
        return loss, metrics, flow_loss, disp_loss, disp_refine_loss, confidence_loss, smooth_loss, params_loss, params_refine_loss


class SmoothLoss(nn.Module):
    """Smooth constaint for prediction.
    - gradient-based smooth regularization:
        \psi_{pq}  = max(w_{pq},\epsilon) min(\hat{\psi}_{pq}(f_p,f_q), \tau_{dis}) \\
        w_{pq}     = e^{-||I_L(p)-I_L(q)||_1 / \eta} \\
        \hat{\psi}_{pq} = |d_p(f_p) - d_q(f_q)| \\
        d_p(f_p)   = a_p p_u + b_p p_v + c_p \\
        d_q(f_q)   = a_q q_u + b_q q_v + c_q
    - curvature-based smooth regularization:
        \psi_{pq}  = max(w_{pq},\epsilon) min(\hat{\psi}_{pq}(f_p,f_q), \tau_{dis}) \\
        w_{pq}     = e^{-||I_L(p)-I_L(q)||_1 / \eta} \\
        \hat{\psi}_{pq} = |d_p(f_p) - d_p(f_q)| + |d_q(f_q) - d_q(f_p)| \\
        d_p(f_p)   = a_p p_u + b_p p_v + c_p \\
        d_p(f_q)   = a_p q_u + b_p q_v + c_p
    """
    def __init__(self, smoothness, slant=None, slant_norm=False, kernel_size=3, 
                 ner_weight_reduce=False, epsilon=0.01, tau=3, eta=10):
        super(SmoothLoss, self).__init__()
        self.smoothness = smoothness
        self.slant = slant
        self.slant_norm = slant_norm

        self.eta = eta
        self.tau = tau
        self.epsilon = epsilon

        self.reduce = ner_weight_reduce
        self.img_ner_extractor = NerghborExtractor(3, kernel_size, reduce=self.reduce)
        self.coord_ner_extractor = NerghborExtractor(2, kernel_size)
        self.params_ner_extractor = NerghborExtractor(3, kernel_size)
    
    def forward(self, params, imgL):
        """Function: compute smoothe loss 
        args:
            params: (B,3,H,W)
            imgL: (B,3,H,W)
            coordL: (B,2,H,W)
            corrdR: (B,2,H,W)
        """
        img_ner    = self.img_ner_extractor(imgL)         # B,3,N,H,W
        B, _, H, W = imgL.shape
        coord      = coords_grid(B, H, W).to(imgL.device) # B,2,H,W
        coord_ner  = self.coord_ner_extractor(coord)      # B,2,N,H,W
        coord      = coord.unsqueeze(2)                   # B,2,1,H,W
        params_ner = self.params_ner_extractor(params)    # B,3,N,H,W
        params     = params.unsqueeze(2)                  # B,3,1,H,W

        # w_{pq} = e^{-||I_L(p)-I_L(q)||_1 / \eta}
        if not self.reduce:
            weight = torch.exp(-torch.abs(img_ner-imgL.unsqueeze(2)).mean(dim=1) / self.eta)   # B,N,H,W
        else:
            weight = torch.exp(-torch.abs(img_ner).mean(dim=1) / self.eta)   # B,N,H,W

        if self.smoothness=="gradient":
            # \hat{\psi}_{pq} = |d_p(f_p) - d_q(f_q)|
            psi_p = disparity_computation(params, coords0=coord, 
                                        slant=self.slant, slant_norm=self.slant_norm) - \
                    disparity_computation(params_ner, coords0=coord_ner, 
                                        slant=self.slant, slant_norm=self.slant_norm)
            psi   = torch.abs(psi_p)                            # B,N,H,W
        elif self.smoothness=="curvature":
            # |d_p(f_p) - d_p(f_q)|
            psi_p = disparity_computation(params, coords0=coord, 
                                        slant=self.slant, slant_norm=self.slant_norm) - \
                    disparity_computation(params, coords0=coord_ner, 
                                        slant=self.slant, slant_norm=self.slant_norm)
            # d_q(f_q) - d_q(f_p)
            psi_q = disparity_computation(params_ner, coords0=coord_ner, 
                                        slant=self.slant, slant_norm=self.slant_norm) - \
                    disparity_computation(params_ner, coords0=coord, 
                                        slant=self.slant, slant_norm=self.slant_norm)
            # \hat{\psi} = |d_p(f_p) - d_p(f_q)| + |d_q(f_q) - d_q(f_p)|
            psi   = torch.abs(psi_p) + torch.abs(psi_q)         # B,N,H,W
        
        # \psi_{pq}  = max(w_{pq},\epsilon) min(\hat{\psi_{pq}(f_p,f_q)}, \tau_{dis})
        smooth_loss = torch.clip(weight, min=self.epsilon,) * \
                      F.sigmoid(psi/self.tau*8-4) * self.tau
        smooth_loss = smooth_loss.mean()
        return smooth_loss


def diamond(n):
    a = np.arange(n)
    b = np.minimum(a,a[::-1])
    return (b[:,None]+b)>=(n-1)//2
def diamond_edge(n):
    arr = np.diagflat(np.ones(n//2+1), n//2)
    arr = np.maximum(arr,np.flip(arr,1))
    return np.maximum(arr,np.flip(arr,0))
kernel_dict = {}
kernel_dict["diamond"] = diamond
kernel_dict["diamond_edge"] = diamond_edge

class NerghborExtractor(nn.Module):
    """Extarct the neighbors of each pixel using depthwise convolution. 
       Input: (B,C,H,W), Output: (B,C,N,H,W).
    """
    def __init__(self, input_channel, kernel_size=3, reduce=False):
        super(NerghborExtractor, self).__init__()
        self.reduce = reduce
        self.input_channel = input_channel

        # build kernel matrix
        if isinstance(kernel_size, int):
            H, W = kernel_size, kernel_size
            self.neighbors_num = kernel_size*kernel_size
            neighbor_kernel = np.zeros((self.neighbors_num, H, W), dtype=np.float16)
            for idx in range(self.neighbors_num):
                neighbor_kernel[idx, idx//H, idx%W] = 1

        elif isinstance(kernel_size, str):
            ## obatin the compressed kernel
            kernel_type, size = kernel_size.split("-")
            kernel_size = int(size)
            compressed_kernel = kernel_dict[kernel_type](kernel_size)
            ## decode the compressed kernel into a series of kernels
            H, W = compressed_kernel.shape
            self.neighbors_num = np.count_nonzero(compressed_kernel)
            neighbors_pos = np.nonzero(compressed_kernel)
            neighbor_kernel = np.zeros((self.neighbors_num, H, W), dtype=np.float16)
            for idx_k, (idx_h, idx_w) in enumerate(zip(neighbors_pos[0],neighbors_pos[1])):
                neighbor_kernel[idx_k, idx_h, idx_w] = compressed_kernel[idx_h, idx_w]
        else:
            raise Exception("kernel_size currently only supports integer")
        if self.reduce:
            neighbor_kernel[:, H//2, W//2] = -1

        if not self.reduce:
            neighbor_kernel = np.tile(neighbor_kernel, (input_channel,1,1))
            neighbor_kernel = neighbor_kernel[:,np.newaxis]                     # in*neighbors_num, 1, k, k
            output_channel  = input_channel*self.neighbors_num
            groups = input_channel
        else:
            neighbor_kernel = np.tile(neighbor_kernel[:, np.newaxis], 
                                      (1,input_channel,1,1))                    # neighbors_num, in, k, k
            output_channel  = self.neighbors_num
            groups = 1
        
        # extract neighbors through depthwise conv
        self.conv = nn.Conv2d(input_channel, output_channel, 
                              kernel_size=kernel_size, padding=kernel_size//2, bias=False, 
                              groups=groups, padding_mode="reflect")
        neighbor_kernel  = torch.Tensor(neighbor_kernel)
        self.conv.weight = nn.Parameter(neighbor_kernel, requires_grad=False)
        
    def forward(self, x):
        B,C,H,W = x.shape
        neighbors = self.conv(x)
        neighbors = neighbors.reshape((B,-1,self.neighbors_num,H,W))
        if self.reduce:
            neighbors = neighbors / self.input_channel
        return neighbors
