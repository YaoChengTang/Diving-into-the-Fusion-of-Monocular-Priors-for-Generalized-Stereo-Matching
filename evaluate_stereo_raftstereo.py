from __future__ import print_function, division
import sys
sys.path.insert(0,'core')
sys.path.append('core/utils')

import os
import argparse
import time
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime

from core.raft_stereo import RAFTStereo, autocast
from core.raft_stereo_disp import RAFTStereoDisp
from core.raft_stereo_mast3r import RAFTStereoMast3r
from core.raft_stereo_depthany import RAFTStereoDepthAny
from core.raft_stereo_noctx import RAFTStereoNoCTX
from core.raft_stereo_depthfusion import RAFTStereoDepthFusion

import stereo_datasets as datasets
from core.utils.utils import InputPadder, LoggerCommon


NODE_RANK    = os.getenv('NODE_RANK', default=0)
LOCAL_RANK   = os.getenv("LOCAL_RANK", default=0)
LOG_ROOT     = os.getenv('LOG_ROOT', default="logs")

logger = LoggerCommon("EVAL")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, root="", mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params, root=root)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logger.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}" +\
                     f"\r\n{imageL_file}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    logger.info("Validation ETH3D: EPE %f, D1 %f" % (round(epe,4), round(d1,4)))
    logger.info("\r\n"*3)
    return {'eth3d-epe': round(epe,4), 'eth3d-d1': round(d1,4)}


@torch.no_grad()
def validate_kitti(model, iters=32, root="", mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, root=root, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logger.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    logger.info(f"Validation KITTI: EPE {round(epe,4)}, D1 {round(d1,4)}, FPS {format(1/avg_runtime, '.2f')}, Time ({format(avg_runtime, '.3f')}s)")
    logger.info("\r\n"*3)
    return {'kitti-epe': round(epe,4), 'kitti-d1': round(d1,4)}


@torch.no_grad()
def validate_kitti2012(model, iters=32, root="", mixed_prec=False):
    """ Peform validation using the KITTI-2012 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI2012(aug_params, root=root, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logger.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    logger.info(f"Validation KITTI: EPE {round(epe,4)}, D1 {round(d1,4)}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    logger.info("\r\n"*3)
    return {'kitti-epe': round(epe,4), 'kitti-d1': round(d1,4)}

@torch.no_grad()
def validate_things(model, iters=32, root='', mixed_prec=False, args=None, eval=False, info=""):
    """ Peform validation using the FlyingThings3D (TEST) split """
    eval = args.eval if args is not None else eval
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', root=root, things_test=True, eval=True)

    out_list_1, epe_list = [], []
    out_list_2, out_list_3 = [], []
    tqdm_disable = args is not None and (args.silence or args.local_rank>0 or int(NODE_RANK)>0)
    for val_id in tqdm(range(len(val_dataset)), disable=tqdm_disable):
        paths, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe = epe.flatten()
        val = (valid_gt.flatten() >= -0.5) & (flow_gt.abs().flatten() < 192)
        # val_nocc = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out_1 = (epe > 1.0)
        out_2 = (epe > 2.0)
        out_3 = (epe > 3.0)
        image_out_1 = out_1[val].float().mean().item()
        image_out_2 = out_2[val].float().mean().item()
        image_out_3 = out_3[val].float().mean().item()
        image_epe   = epe[val].mean().item()

        # avoid corrupted data
        if val.sum()<10 or image_epe>20 or image_out_1>0.95:
            logger.info(f"Corrupted data: {paths}")
            continue

        epe_list.append(image_epe)
        out_list_1.append(image_out_1)
        out_list_2.append(image_out_2)
        out_list_3.append(image_out_3)
        
        # if not eval and val_id>10:
        #     break
        
        logger.info(f"FlyingThings Iter {val_id+1} out of {len(val_dataset)}. " + \
                     f"EPE {round(image_epe,4)}, BAD1 {round(image_out_1,4)}, " +\
                     f"BAD2 {round(image_out_2,4)}, BAD3 {round(image_out_3,4)} ")
        

    epe_list = np.array(epe_list)
    out_list_1 = np.array(out_list_1)
    out_list_2 = np.array(out_list_2)
    out_list_3 = np.array(out_list_3)

    epe = np.mean(epe_list)
    bad1 = 100 * np.mean(out_list_1)
    bad2 = 100 * np.mean(out_list_2)
    bad3 = 100 * np.mean(out_list_3)

    logger.info("Validation FlyingThings: %f, %f, %f, %f" % (round(epe,4), round(bad1), round(bad2), round(bad3)))
    logger.info("\r\n"*3)
    
    return {info+'things-epe': round(epe), 
            info+'things-bad1': round(bad1), 
            info+'things-bad2': round(bad2), 
            info+'things-bad3': round(bad3)}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', root="", mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, root=root, split=split)

    out_list, epe_list = [], []
    out_nocc_list, epe_nocc_list = [], []
    out_mask_list, epe_mask_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)
        val_nocc = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) > -1000)

        mask_analysis = ( (valid_gt.reshape(-1) >= 0.5) | (epe_flattened <= 2.0) ) & \
                        (flow_gt[0].reshape(-1) > -1000)


        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        image_out_nocc = out[val_nocc].float().mean().item()
        image_epe_nocc = epe_flattened[val_nocc].mean().item()
        image_out_mask = (out[mask_analysis].float().sum() / val.sum()).item()
        image_epe_mask = (epe_flattened[mask_analysis].sum() / val.sum()).item()
        logger.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. " + \
                     f"EPE {round(image_epe,4)} D1 {round(image_out,4)} " + \
                     f"EPE_nocc {round(image_epe_nocc,4)} D1_nocc {round(image_out_nocc,4)} " + \
                     f"EPE_mask {round(image_epe_mask,4)} D1_mask {round(image_out_mask,4)} " + \
                     f"\r\n{imageL_file}")
        epe_list.append(image_epe)
        out_list.append(image_out)
        epe_nocc_list.append(image_epe_nocc)
        out_nocc_list.append(image_out_nocc)
        epe_mask_list.append(image_epe_mask)
        out_mask_list.append(image_out_mask)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)
    epe_nocc_list = np.array(epe_nocc_list)
    out_nocc_list = np.array(out_nocc_list)
    epe_mask_list = np.array(epe_mask_list)
    out_mask_list = np.array(out_mask_list)

    epe = np.mean(epe_list)
    d1  = 100 * np.mean(out_list)
    epe_nocc = np.mean(epe_nocc_list)
    d1_nocc  = 100 * np.mean(out_nocc_list)
    epe_mask = np.mean(epe_mask_list)
    d1_mask  = 100 * np.mean(out_mask_list)

    logger.info(f"Validation Middlebury{split}: EPE {round(epe,4)}, D1 {round(d1,4)}, " + \
                 f"EPE_nocc {round(epe_nocc,4)}, D1_nocc {round(d1_nocc,4)}, " + \
                 f"EPE_mask {round(epe_mask,4)}, D1_mask {round(d1_mask,4)}")
    return {f'middlebury{split}-epe': round(epe,4), f'middlebury{split}-d1': round(d1,4)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help="dataset root", default=None)
    parser.add_argument('--test_exp_name', default='', help="name your experiment in testing")
    parser.add_argument('--model_name', default='RaftStereo', help="name your model: raftstereo, raftstereodisp, RAFTStereoMast3r, RAFTStereoDepthAny, raftstereodepthfusion")
    parser.add_argument('--mast3r_model_path', default='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', help="pretrained model path for MaSt3R")
    parser.add_argument('--depthany_model_dir', default='/data5/yao/pretrained', help="directory of pretrained model path for DepthAnything")
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True, choices=["eth3d", "kitti", 'kitti2012', "things"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    parser.add_argument('--silence', action='store_true', help='no output of training/eval process')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "abs_reg", "alt", "abs_alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    args = parser.parse_args()

    # 重新设定日志文件位置
    log_path = os.path.join(LOG_ROOT, args.restore_ckpt.split("/")[-2])
    logger.set_log_path(log_path, "TEST-{}".format(args.test_exp_name))

    logger.print_args(args)

    args.eval = True
    
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
    else :
        raise Exception("No such model: {}".format(args.model_name))
    model = torch.nn.DataParallel(model, device_ids=[0])

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(".tar")
        logger.info(f"Loading checkpoint from {args.restore_ckpt}")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logger.info(f"Done loading checkpoint")

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
        res = validate_eth3d(model, iters=args.valid_iters, root=args.root, 
                             mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti':
        if args.root is None:
            args.root = "/data6/KITTI2015"
        res = validate_kitti(model, iters=args.valid_iters, root=args.root, 
                             mixed_prec=use_mixed_precision)
    
    elif args.dataset == 'kitti2012':
        if args.root is None:
            args.root = "/data6/KITTI2012"
        res = validate_kitti2012(model, iters=args.valid_iters, root=args.root, 
                                 mixed_prec=use_mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        if args.root is None:
            args.root = "/data6/Middlebury"
        res = validate_middlebury(model, iters=args.valid_iters, root=args.root, split=args.dataset[-1], 
                                  mixed_prec=use_mixed_precision)

    elif args.dataset == 'things':
        if args.root is None:
            args.root = "/data6/sceneflow/sceneflow"
        res = validate_things(model, iters=args.valid_iters, root=args.root, 
                              mixed_prec=use_mixed_precision)
    
    
    # write results into excel
    res["Model"] = args.test_exp_name + " - " + os.path.basename(args.restore_ckpt)
    row = pd.DataFrame([res])
    df = None
    file_path = os.path.join(LOG_ROOT,"eval.xlsx")
    if os.path.exists(file_path):
        sheet_to_df_map = pd.read_excel(file_path, sheet_name=None)
        if args.dataset in sheet_to_df_map:
            df = sheet_to_df_map[args.dataset]
    df_update = pd.concat([df,row])
    
    if not os.path.exists(file_path):
        writer = pd.ExcelWriter(file_path, mode='w', engine="openpyxl")
    else:
        writer = pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace")
    df_update.to_excel(writer, sheet_name=args.dataset, index=False)
    writer.close()


