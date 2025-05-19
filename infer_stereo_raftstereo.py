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
from PIL import Image

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


@torch.no_grad()
def visualize(atom_dict, 
              image1, image2, imageGT_file, 
              padder, viser):
    flow_pr_sequence        = atom_dict.get("disp_predictions", [])
    depth                   = atom_dict.get("depth", None)
    depth_registered        = atom_dict.get("depth_registered", None)
    depth_registered_up     = None # atom_dict.get("depth_registered_up", None)
    modulation_predictions  = [] # atom_dict.get("modulation_predictions", [])
    flow_pr_refine_sequence = [] # atom_dict.get("disp_refine_predictions", [])
    conf_fusion             = None # atom_dict.get("conf_fusion", None)

    for idx, flow_pr in enumerate(flow_pr_sequence):
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        flow_pr_sequence[idx] = flow_pr

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
    
    # save prediction and the corresponding path for visualization
    viser.save_pred_vis(-flow_pr.data.numpy()[0], imageGT_file)
    image1 = padder.unpad(image1).cpu().squeeze(0).permute(1,2,0)
    image2 = padder.unpad(image2).cpu().squeeze(0).permute(1,2,0)

    # print("-"*30, (-flow_pr_sequence[-1].data.numpy()[0]).max(), (-flow_pr_sequence[-1].data.numpy()[0]).min(),
    #       (-flow_gt.data.numpy()[0]).max(), (-flow_gt.data.numpy()[0]).min())

    vis1 = [
            {"name": "Left Image", 
             "img_list": [image1.data.numpy().astype(np.uint8)], "cmap": None},
            {"name": "Right Image", 
             "img_list": [image2.data.numpy().astype(np.uint8)], "cmap": None},

            # {"name": "GT Disp", "img_list": [-flow_gt.data.numpy()[0]], "cmap": "jet", "vmin": vmin, "vmax": vmax},

            {"name": "Final Fused Disp", 
             "img_list": [-flow_pr_sequence[-1].data.numpy()[0]], 
             "cmap": "jet",
             "error_map": False,
             "vmin": None, "vmax": None,},

            {"name": "Disp Before Fusion", 
             "img_list": [-flow_pr_sequence[-3].data.numpy()[0]], 
             "cmap": "jet",
             "error_map": False,
             "vmin": None, "vmax": None,},
            ]

    if depth_registered_up is not None:
        depth_registered_up = padder.unpad(depth_registered_up).cpu().squeeze(0)
        vis1.append( {"name": "Upsampled Registered Mono Depth", 
                      "img_list": [depth_registered_up.data.numpy()[0]], 
                      "cmap": "jet",
                      "GT": [-flow_gt.data.numpy()[0]],
                      "error_map": True,
                      "vmin": vmin, "vmax": vmax,} )
    if conf_fusion is not None:
        vis1.append( {"name": "Mono Confidence for Fusion", "img_list": [conf_fusion.cpu().squeeze(0).data.numpy()[0]], "cmap": "viridis"} )
    if depth is not None:
        vis1.append( {"name": "Mono Depth from DepthAnything", 
                      "img_list": [depth.cpu().squeeze(0).data.numpy()[0]], 
                      "cmap": "jet", "vmin": None, "vmax": None,} )
    if depth_registered is not None:
        vis1.append( {"name": "Registered Mono Depth", 
                      "img_list": [depth_registered.cpu().squeeze(0).data.numpy()[0]], 
                      "cmap": "jet", "vmin": None, "vmax": None,} )

    viser.analyze(vis1, imageGT_file, in_one_fig=True)

    if viser.args.mask and confidence_list is not None and len(confidence_list)>0 :
        vis3 = [{"name": "Encourage", 
                "img_list": [conf.data.numpy()[0] for conf in confidence_list], 
                "cmap": "gray",
                "epe_list": None,
                f"{bad_thold}px_list": None,
                "GT": None,
                "stop_idx": 20,
                "improvement": False,
                "movement": False,
                "error_map": False,
                "acceleration": False,
                "mask": True,
                "binary_thold": viser.args.binary_thold},]
        viser.analyze(vis3, imageGT_file, in_one_fig=False)

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path_txt', help="txt saving image paths", default=None)
    parser.add_argument('--root', help="dataset root", default=None)
    parser.add_argument('--sv_root', help="visualization root", default=None)
    parser.add_argument('--test_exp_name', default='', help="name your experiment in testing")
    parser.add_argument('--model_name', default='RaftStereo', help="name your model: raftstereo, raftstereodisp, RAFTStereoMast3r, RAFTStereoDepthAny, raftstereodepthfusion, RAFTStereoDepthBeta, RAFTStereoDepthBetaNoLBP")
    parser.add_argument('--mast3r_model_path', default='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', help="pretrained model path for MaSt3R")
    parser.add_argument('--depthany_model_dir', default='/data5/yao/pretrained', help="directory of pretrained model path for DepthAnything")
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True)
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
    parser.add_argument('--refine_unet', action='store_true', help="use EfficientUnet in refinement")

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


    viser = Visualizer(args.root, args.sv_root, "common", scratch=False, args=args, logger=logger)

    # Loading paths
    if args.img_path_txt is not None and args.root is not None:
        if os.path.exists(args.img_path_txt):
            with open(args.img_path_txt, 'r') as f:
                lines = f.readlines()

            left_img_path_list, right_img_path_list = [], []
            gt_path_list = []
            for line in lines:
                left_img_path, right_img_path, gt_path = line.strip().split()
                left_img_path_list.append( os.path.join(args.root, left_img_path) )
                right_img_path_list.append( os.path.join(args.root, right_img_path) )
                gt_path_list.append( os.path.join(args.root, gt_path) )
            print(f"There are {len(left_img_path_list)} pairs of images in {args.img_path_txt}")
        else:
            raise Exception("Please provide valid path for img_path_txt")
            
    for left_img_path, right_img_path in tqdm(zip(left_img_path_list,right_img_path_list), total=len(left_img_path_list), desc="Processing Data"):
        image1 = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        image2 = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).cuda()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).cuda()

        # image1 = F.interpolate(image1, scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True)
        # image2 = F.interpolate(image2, scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=use_mixed_precision):
            atom_dict = model(image1, image2, iters=args.valid_iters, test_mode=False, vis_mode=True)
        
        visualize(atom_dict, 
                 image1, image2, left_img_path, 
                 padder, viser)


        
