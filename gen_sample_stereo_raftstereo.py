from __future__ import print_function, division
import sys
sys.path.insert(0,'core')
sys.path.append('core/utils')

import os
import argparse
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

from core.raft_stereo import RAFTStereo, autocast
from core.raft_stereo_disp import RAFTStereoDisp
from core.raft_stereo_mast3r import RAFTStereoMast3r
from core.raft_stereo_depthany import RAFTStereoDepthAny
from core.raft_stereo_noctx import RAFTStereoNoCTX
from core.raft_stereo_depthfusion import RAFTStereoDepthFusion
from core.raft_stereo_depthbeta import RAFTStereoDepthBeta
from core.raft_stereo_depthbeta_nolbp import RAFTStereoDepthBetaNoLBP
from core.raft_stereo_depthmatch import RAFTStereoDepthMatch
from core.raft_stereo_depthbeta_refine import RAFTStereoDepthBetaRefine
from core.raft_stereo_depth_postfusion import RAFTStereoDepthPostFusion

import stereo_datasets as datasets
from core.utils.vis import Visualizer
from core.utils.utils import InputPadder, LoggerCommon


NODE_RANK    = os.getenv('NODE_RANK', default=0)
LOCAL_RANK   = os.getenv("LOCAL_RANK", default=0)
LOG_ROOT     = os.getenv('LOG_ROOT', default="logs")

logger = LoggerCommon("GENERATION")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evalute_metric(flow_pr, flow_gt, valid_gt, d1_thold, dataset_name=None):
    assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
    epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

    epe_flattened = epe.flatten()
    out = (epe_flattened > d1_thold)
    if dataset_name.lower()=="middlebury":
        val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)
    elif dataset_name.lower()=="eth3d":
        val = valid_gt.flatten() >= 0.5
    elif dataset_name.lower()=="flyingthings3d":
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
    elif dataset_name.lower()=="kitti":
        val = valid_gt.flatten() >= 0.5
    elif dataset_name.lower()=="booster":
        val = valid_gt.flatten() >= 0.5      # we mainly focus on transparent areas in vis
    else:
        raise Exception(f"{dataset_name} is not supported")
    image_out = out[val].float().mean().item()
    image_epe = epe_flattened[val].mean().item()
    return image_epe, image_out


@torch.no_grad()
def evalute(atom_dict, 
            image1, image2, flow_gt, valid_gt, imageGT_file, 
            padder, viser, dataset_name, d1_thold):
    flow_pr_sequence        = atom_dict.get("disp_predictions", [])
    depth                   = atom_dict.get("depth", None)
    depth_registered        = atom_dict.get("depth_registered", None)
    depth_registered_up     = atom_dict.get("depth_registered_up", None)
    modulation_predictions  = atom_dict.get("modulation_predictions", [])
    flow_pr_refine_sequence = atom_dict.get("disp_refine_predictions", [])
    conf_fusion             = atom_dict.get("conf_fusion", None)

    # fill None in confidence_list with zero-matrix and
    # unpad confidence map
    confidence_list = modulation_predictions
    if confidence_list is not None and len(confidence_list)>0:
        for idx, conf in enumerate(confidence_list):
            if conf is not None:
                # confidence_list[idx] = F.sigmoid(conf)
                pass
            else:
                confidence_list[idx] = torch.zeros_like(flow_pr_sequence[-1])
            confidence_list[idx] = padder.unpad(confidence_list[idx]).cpu().squeeze(0)
    
    # fill None in flow_pr_refine_sequence with zero-matrix
    if flow_pr_refine_sequence is not None and len(flow_pr_refine_sequence)>0:
        for idx, flow_pr_refine in enumerate(flow_pr_refine_sequence):
            if flow_pr_refine is None:
                flow_pr_refine_sequence[idx] = torch.zeros_like(flow_pr_sequence[idx])
    
    # merge flow_pr_sequence and flow_pr_refine_sequence to process them at once
    len_sequence = len(flow_pr_sequence)
    flow_pr_sequence += flow_pr_refine_sequence

    # compute epe and d1 for each iteration
    vis_epe_sequence = []
    vis_xpx_sequence = []
    for idx, flow_pr in enumerate(flow_pr_sequence):
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        flow_pr_sequence[idx] = flow_pr

        image_epe, image_out = evalute_metric(flow_pr, flow_gt, valid_gt,
                                              d1_thold=d1_thold, dataset_name=dataset_name)
        vis_epe_sequence.append(image_epe)
        vis_xpx_sequence.append(image_out)
    
    # split the results from flow_pr_sequence and flow_pr_refine_sequence
    flow_pr_refine_sequence = flow_pr_sequence[len_sequence:]
    flow_pr_sequence = flow_pr_sequence[:len_sequence]
    vis_refine_epe_sequence = vis_epe_sequence[len_sequence:]
    vis_epe_sequence = vis_epe_sequence[:len_sequence]
    vis_refine_xpx_sequence = vis_xpx_sequence[len_sequence:]
    vis_xpx_sequence = vis_xpx_sequence[:len_sequence]

    # save prediction and the corresponding path for visualization
    viser.save_pred_vis(-flow_pr.data.numpy()[0], imageGT_file)
    image1 = padder.unpad(image1).cpu().squeeze(0).permute(1,2,0)
    image2 = padder.unpad(image2).cpu().squeeze(0).permute(1,2,0)

    vis1 = [{"name": "Left Image", 
             "img_list": [image1.data.numpy().astype(np.uint8)], "cmap": None},
            {"name": "Right Image", 
             "img_list": [image2.data.numpy().astype(np.uint8)], "cmap": None},
            {"name": "GT Disp", "img_list": [-flow_gt.data.numpy()[0]], "cmap": "jet"},
            {"name": "Pr Disp -1", 
             "img_list": [-flow_pr_sequence[-1].data.numpy()[0]], 
             "cmap": "jet",
             "GT": [-flow_gt.data.numpy()[0]],
             "error_map": True,},
            {"name": "Pr Disp -3", 
             "img_list": [-flow_pr_sequence[-3].data.numpy()[0]], 
             "cmap": "jet",
             "GT": [-flow_gt.data.numpy()[0]],
             "error_map": True,},]
    if depth is not None:
        vis1.append( {"name": "Mono Depth", "img_list": [depth.cpu().squeeze(0).data.numpy()[0]], "cmap": "jet"} )
    if depth_registered is not None:
        vis1.append( {"name": "Mono depth_registered", "img_list": [depth_registered.cpu().squeeze(0).data.numpy()[0]], "cmap": "jet"} )
    if depth_registered_up is not None:
        depth_registered_up = padder.unpad(depth_registered_up).cpu().squeeze(0)
        vis1.append( {"name": "Mono depth_registered_up", 
                      "img_list": [depth_registered_up.data.numpy()[0]], 
                      "cmap": "jet",
                      "GT": [-flow_gt.data.numpy()[0]],
                      "error_map": True,} )
    if conf_fusion is not None:
        vis1.append( {"name": "Mono conf_fusion", "img_list": [conf_fusion.cpu().squeeze(0).data.numpy()[0]], "cmap": "viridis"} )
    viser.analyze(vis1, imageGT_file, in_one_fig=True)

    # vis2 = [{"name": "Disp", 
    #          "img_list": [-flow_pr.data.numpy()[0] for flow_pr in flow_pr_sequence], 
    #          "cmap": "jet",
    #          "epe_list": vis_epe_sequence,
    #          f"{d1_thold}px_list": vis_xpx_sequence,
    #          "GT": [-flow_gt.data.numpy()[0]],
    #          "stop_idx": 20,
    #          "improvement": viser.args.improvement,
    #          "movement": viser.args.movement,
    #          "error_map": True,
    #          "acceleration": viser.args.acceleration,
    #          "mask": viser.args.mask,
    #          "binary_thold": viser.args.binary_thold},]
    # viser.analyze(vis2, imageGT_file, in_one_fig=False)

    if viser.args.mask and confidence_list is not None and len(confidence_list)>0 :
        vis3 = [{"name": "Encourage", 
                "img_list": [conf.data.numpy()[0] for conf in confidence_list], 
                "cmap": "gray",
                "epe_list": None,
                f"{d1_thold}px_list": None,
                "GT": None,
                "stop_idx": 20,
                "improvement": False,
                "movement": False,
                "error_map": False,
                "acceleration": False,
                "mask": True,
                "binary_thold": viser.args.binary_thold},]
        viser.analyze(vis3, imageGT_file, in_one_fig=False)

    return image_epe, image_out


@torch.no_grad()
def validate_eth3d(model, iters=32, root="", sv_root="", mixed_prec=False, args=None):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params, root=root)
    viser = Visualizer(root, sv_root, "eth3d", scratch=False, args=args, logger=logger)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, imageGT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            atom_dict = model(image1, image2, iters=iters, test_mode=False, vis_mode=True)

        image_epe, image_out = evalute(atom_dict, 
                                       image1, image2, flow_gt, valid_gt, imageGT_file, 
                                       padder, viser, dataset_name="ETH3D", d1_thold=1.0)
        epe_list.append(image_epe)
        out_list.append(image_out)
        logger.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
    
    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    logger.info("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, root="", sv_root="", mixed_prec=False, args=None):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, root=root, image_set='training')
    torch.backends.cudnn.benchmark = True
    viser = Visualizer(root, sv_root, "kitti2015", scratch=False, args=args, logger=logger)

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, imageGT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            atom_dict = model(image1, image2, iters=iters, test_mode=False, vis_mode=True)
            end = time.time()
        
        image_epe, image_out = evalute(atom_dict, 
                                       image1, image2, flow_gt, valid_gt, imageGT_file, 
                                       padder, viser, dataset_name="kitti", d1_thold=3.0)
        epe_list.append(image_epe)
        out_list.append(image_out)
        if val_id < 9 or (val_id+1)%10 == 0:
            logger.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. " +\
                         f"EPE {round(image_epe,4)} D1 {round(image_out,4)}. " +\
                         f"Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
    
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    logger.info(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_things(model, iters=32, root='', sv_root="", mixed_prec=False, args=None):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', root=root, things_test=True, eval=True)
    viser = Visualizer(root, sv_root, "FlyingThings3D", scratch=False, args=args, logger=logger)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        (imageL_file, imageR_file, imageGT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            atom_dict = model(image1, image2, iters=iters, test_mode=True)
        
        image_epe, image_out = evalute(atom_dict, 
                                       image1, image2, flow_gt, valid_gt, imageGT_file, 
                                       padder, viser, dataset_name="FlyingThings3D",
                                       d1_thold=3.0)
        epe_list.append(image_epe)
        out_list.append(image_out)
        if val_id%100==0:
            logger.info(f"FlyingThings3D Iter {val_id+1} out of {len(val_dataset)}. " +\
                         f"EPE {round(image_epe,4)} D1 {round(image_out,4)}")

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    logger.info("Validation FlyingThings: %f, %f" % (epe, d1))
    return {'things-epe': epe, 'things-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', root="", sv_root="", mixed_prec=False, args=None):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, root=root, split=split)
    viser = Visualizer(root, sv_root, "middlebury", scratch=False, args=args, logger=logger)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, imageGT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            atom_dict = model(image1, image2, iters=iters, test_mode=False, vis_mode=True)

        image_epe, image_out = evalute(atom_dict, 
                                       image1, image2, flow_gt, valid_gt, imageGT_file, 
                                       padder, viser, dataset_name="Middlebury", d1_thold=2.0)
        epe_list.append(image_epe)
        out_list.append(image_out)
        logger.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. " +\
                     f"EPE {round(image_epe,4)} D1 {round(image_out,4)}")

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    logger.info(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


@torch.no_grad()
def validate_booster(model, iters=32, root="", mixed_prec=False, sv_root="", image_set="train/balanced", args=None):
    """ Peform validation using the Booster (TRAIN balanced) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Booster(aug_params, root=root, image_set=image_set)
    viser = Visualizer(root, sv_root, "booster", scratch=False, args=args, logger=logger)

    epe_list = []
    bad1_list, bad2_list, bad3_list, bad5_list = [], [], [], []
    epe_trans_list, epe_notrans_list = [],[]
    bad1_trans_list, bad2_trans_list, bad3_trans_list, bad5_trans_list = [], [], [], []
    bad1_notrans_list, bad2_notrans_list, bad3_notrans_list, bad5_notrans_list = [], [], [], []
    # epe_trans_b_list = []

    trans_b_bad1, trans_b_bad2, trans_b_bad3, trans_b_bad5, trans_b_sum = 0,0,0,0,0
    trans_f_bad1, trans_f_bad2, trans_f_bad3, trans_f_bad5, trans_f_sum = 0,0,0,0,0
    notrans_bad1, notrans_bad2, notrans_bad3, notrans_bad5, notrans_sum = 0,0,0,0,0

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, imageGT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        image1 = F.interpolate(image1, scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True)
        image2 = F.interpolate(image2, scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True)
        flow_gt = F.interpolate(flow_gt.unsqueeze(0), scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True).squeeze(0)
        flow_gt /= 4
        trans_mask = (valid_gt == 3).float()   # get transparent surfaces
        trans_mask = F.interpolate(trans_mask.unsqueeze(0).unsqueeze(0), scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
        
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            atom_dict = model(image1, image2, iters=iters, test_mode=False, vis_mode=True)
        
        image_epe, image_out = evalute(atom_dict, 
                                       image1, image2, flow_gt, trans_mask, imageGT_file, 
                                       padder, viser, dataset_name="Booster", d1_thold=2.0)
        epe_list.append(image_epe)
        out_list.append(image_out)
        logger.info(f"Booster-{image_set} Iter {val_id+1} out of {len(val_dataset)}. " +\
                     f"EPE {round(image_epe,4)} D1 {round(image_out,4)}")

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    logger.info(f"Validation Booster-{image_set}: EPE {epe}, D1 {d1}")
    return {f'Booster-{image_set}-epe': epe, f'Booster-{image_set}-d1': d1}
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help="dataset root", default=None)
    parser.add_argument('--sv_root', help="visualization root", default=None)
    parser.add_argument('--test_exp_name', default='', help="name your experiment in testing")
    parser.add_argument('--model_name', default='RaftStereo', help="name your model: raftstereo, raftstereodisp, RAFTStereoMast3r, RAFTStereoDepthAny, raftstereodepthfusion, RAFTStereoDepthBeta, RAFTStereoDepthBetaNoLBP")
    parser.add_argument('--mast3r_model_path', default='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', help="pretrained model path for MaSt3R")
    parser.add_argument('--depthany_model_dir', default='/data5/yao/pretrained', help="directory of pretrained model path for DepthAnything")
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True, choices=["eth3d", "kitti", "things", "booster"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--eval', action='store_true', help='evaluation mode')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    parser.add_argument('--lbp_neighbor_offsets', default='(-1,-1), (1,1), (1,-1), (-1,1)', help="determine the neighbors used in LBP encoder")
    parser.add_argument('--modulation_ratio', type=float, default=1., help="hyperparameters for modulation")
    parser.add_argument('--modulation_alg', choices=["linear", "sigmoid"], default="linear", help="rescale modulation")
    parser.add_argument('--conf_from_fea', action='store_true', help="confidence in refinement not only from cost volume but also from other features")
    parser.add_argument('--refine_pool', action='store_true', help="use pooling in refinement")

    parser.add_argument('--improvement', action='store_true', help="visualize improvement map (error_map[i] - error_map[i-1])")
    parser.add_argument('--movement', action='store_true', help="visualize movement map (flow_pr[i] - flow_pr[i-1])")
    parser.add_argument('--acceleration', action='store_true', help="visualize acceleration map (movement_map[i] - movement_map[i-1])")
    parser.add_argument('--mask', action='store_true', help="visualize mask")
    parser.add_argument('--binary_thold', type=float, default=0.5, help="visualize binary mask")

    args = parser.parse_args()

    args.eval = True
    assert args.sv_root is not None, "Please specify the visualization root"
    if args.sv_root is None:
        raise Exception("Please specify sv_root")
    args.sv_root = os.path.join(args.sv_root, 
                        args.restore_ckpt.split("/")[-2], args.test_exp_name)

    # 重新设定日志文件位置
    logger.set_log_path(args.sv_root, "GENERATION-{}".format(args.test_exp_name))

    logger.print_args(args)

    if args.model_name.lower() == "raftstereo":
        model  = RAFTStereo(args)
    elif args.model_name.lower() == "raftstereodisp":
        model  = RAFTStereoDisp(args)
    elif args.model_name.lower() == "raftstereomast3r":
        model = RAFTStereoMast3r(args)
    elif args.model_name.lower() == "raftstereodepthany":
        model = RAFTStereoDepthAny(args)
    elif args.model_name.lower() == "raftstereonoctx":
        model = RAFTStereoNoCTX(args)
    elif args.model_name.lower() == "raftstereodepthfusion":
        model = RAFTStereoDepthFusion(args)
    elif args.model_name.lower() == "raftstereodepthbeta":
        model = RAFTStereoDepthBeta(args)
    elif args.model_name.lower() == "RAFTStereoDepthBetaNoLBP".lower():
        model = RAFTStereoDepthBetaNoLBP(args)
    elif args.model_name.lower() == "RAFTStereoDepthMatch".lower():
        model = RAFTStereoDepthMatch(args)
    elif args.model_name.lower() == "RAFTStereoDepthBetaRefine".lower():
        model = RAFTStereoDepthBetaRefine(args)
    elif args.model_name.lower() == "RAFTStereoDepthPostFusion".lower():
        model = RAFTStereoDepthPostFusion(args)
    else :
        raise Exception("No such model: {}".format(args.model_name))
    model = torch.nn.DataParallel(model, device_ids=[0])

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(".tar")
        logger.info(f"Loading checkpoint from {args.restore_ckpt}")
        checkpoint = torch.load(args.restore_ckpt)
        # model.load_state_dict(checkpoint, strict=True)
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.find("lbp_encoder.lbp_conv") != -1:
                continue
            new_state_dict[key] = value
        # model.load_state_dict(new_state_dict, strict=True)
        model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Done loading checkpoint from {args.restore_ckpt}")

    model.cuda()
    model.eval()

    logger.info(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':
        if args.root is None:
            args.root = "/data6/ETH3D"
        validate_eth3d(model, iters=args.valid_iters, root=args.root, 
                       sv_root=args.sv_root, mixed_prec=use_mixed_precision,
                       args=args)

    elif args.dataset == 'kitti':
        if args.root is None:
            args.root = "/data6/KITTI2015"
        validate_kitti(model, iters=args.valid_iters, root=args.root, 
                       sv_root=args.sv_root, mixed_prec=use_mixed_precision,
                       args=args)
    
    elif args.dataset == 'kitti2012':
        if args.root is None:
            args.root = "/data6/KITTI2012"
        validate_kitti(model, iters=args.valid_iters, root=args.root, 
                       sv_root=args.sv_root, mixed_prec=use_mixed_precision,
                       args=args)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        if args.root is None:
            args.root = "/data6/Middlebury"
        validate_middlebury(model, iters=args.valid_iters, root=args.root, split=args.dataset[-1], 
                            sv_root=args.sv_root, mixed_prec=use_mixed_precision,
                            args=args)

    elif args.dataset == 'things':
        if args.root is None:
            args.root = "/data6/sceneflow/sceneflow"
        validate_things(model, iters=args.valid_iters, root=args.root, 
                        sv_root=args.sv_root, mixed_prec=use_mixed_precision,
                        args=args)
    
    elif args.dataset == 'booster':
        if args.root is None:
            args.root = "./datasets/Booster"
        res = validate_booster(model, iters=args.valid_iters, root=args.root, 
                               sv_root=args.sv_root, mixed_prec=use_mixed_precision,
                               args=args)
