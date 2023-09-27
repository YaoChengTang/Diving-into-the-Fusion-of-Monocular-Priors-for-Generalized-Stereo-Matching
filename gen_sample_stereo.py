from __future__ import print_function, division
import sys
sys.path.append('core')

import os
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

from raft_stereo import RAFTStereo, autocast
import stereo_datasets as datasets
from utils.utils import InputPadder
from utils.vis import Visualizer


LOG_ROOT = os.getenv('LOG_ROOT', default="")
LOG_PATH = os.path.join("logs" if LOG_ROOT is None or len(LOG_ROOT)==0 else LOG_ROOT, 
                        '{}-{}.log'.format(os.path.basename(__file__), datetime.now().strftime("%y%m%d_%H%M%S")))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    handlers = [logging.FileHandler(LOG_PATH), 
                                logging.StreamHandler()]
                   )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, root="", sv_root="", mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params, root=root)
    viser = Visualizer(root, sv_root, "eth3d", scratch=False)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, imageGT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr_sequence = model(image1, image2, iters=iters, test_mode=False, vis_mode=True)
        
        vis_epe_sequence = []
        vis_xpx_sequence = []
        for idx, flow_pr in enumerate(flow_pr_sequence):
            flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
            flow_pr_sequence[idx] = flow_pr

            assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
            epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

            epe_flattened = epe.flatten()
            val = valid_gt.flatten() >= 0.5
            out = (epe_flattened > 1.0)
            image_out = out[val].float().mean().item()
            image_epe = epe_flattened[val].mean().item()

            vis_epe_sequence.append(image_epe)
            vis_xpx_sequence.append(image_out)

        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

        # save prediction and the corresponding path for visualization
        viser.save_pred_vis(-flow_pr.data.numpy()[0], imageGT_file)
        image1 = padder.unpad(image1).cpu().squeeze(0)
        image2 = padder.unpad(image2).cpu().squeeze(0)
        viser.analyze([-flow_pr.data.numpy()[0] for flow_pr in flow_pr_sequence], 
                      image1.data.numpy(), 
                      image2.data.numpy(), 
                      -flow_gt.data.numpy()[0], 
                      valid_gt.data.numpy(),
                      imageGT_file,
                      vis_epe_sequence,
                      vis_xpx_sequence)
    
    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    logging.info("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, root="", sv_root="", mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, root=root, image_set='training')
    torch.backends.cudnn.benchmark = True
    viser = Visualizer(root, sv_root, "kitti2015", scratch=False)

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, imageGT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr_sequence = model(image1, image2, iters=iters, test_mode=False, vis_mode=True)
            end = time.time()
        
        vis_epe_sequence = []
        vis_xpx_sequence = []
        for idx, flow_pr in enumerate(flow_pr_sequence):

            if val_id > 50:
                elapsed_list.append(end-start)
            flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
            flow_pr_sequence[idx] = flow_pr

            assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
            epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

            epe_flattened = epe.flatten()
            val = valid_gt.flatten() >= 0.5

            out = (epe_flattened > 3.0)
            image_out = out[val].float().mean().item()
            image_epe = epe_flattened[val].mean().item()

            vis_epe_sequence.append(image_epe)
            vis_xpx_sequence.append(image_out)
            
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        # save prediction and the corresponding path for visualization
        viser.save_pred_vis(-flow_pr.data.numpy()[0], imageGT_file)
        image1 = padder.unpad(image1).cpu().squeeze(0)
        image2 = padder.unpad(image2).cpu().squeeze(0)
        viser.analyze([-flow_pr.data.numpy()[0] for flow_pr in flow_pr_sequence], 
                      image1.data.numpy(), 
                      image2.data.numpy(), 
                      -flow_gt.data.numpy()[0], 
                      valid_gt.data.numpy(),
                      imageGT_file,
                      vis_epe_sequence,
                      vis_xpx_sequence)
    
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    logging.info(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_things(model, iters=32, root='', sv_root="", mixed_prec=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', root=root, things_test=True)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
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
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe > 1.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    logging.info("Validation FlyingThings: %f, %f" % (epe, d1))
    return {'things-epe': epe, 'things-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', root="", sv_root="", mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, root=root, split=split)
    viser = Visualizer(root, sv_root, "middlebury", scratch=False)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, imageGT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr_sequence = model(image1, image2, iters=iters, test_mode=False, vis_mode=True)

        vis_epe_sequence = []
        vis_xpx_sequence = []
        for idx, flow_pr in enumerate(flow_pr_sequence):
            flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
            flow_pr_sequence[idx] = flow_pr

            assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
            epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

            epe_flattened = epe.flatten()
            val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)

            out = (epe_flattened > 2.0)
            image_out = out[val].float().mean().item()
            image_epe = epe_flattened[val].mean().item()

            vis_epe_sequence.append(image_epe)
            vis_xpx_sequence.append(image_out)
            
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

        # save prediction and the corresponding path for visualization
        viser.save_pred_vis(-flow_pr.data.numpy()[0], imageGT_file)
        image1 = padder.unpad(image1).cpu().squeeze(0)
        image2 = padder.unpad(image2).cpu().squeeze(0)
        viser.analyze([-flow_pr.data.numpy()[0] for flow_pr in flow_pr_sequence], 
                      image1.data.numpy(), 
                      image2.data.numpy(), 
                      -flow_gt.data.numpy()[0], 
                      valid_gt.data.numpy(),
                      imageGT_file,
                      vis_epe_sequence,
                      vis_xpx_sequence)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    logging.info(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help="dataset root", default=None)
    parser.add_argument('--sv_root', help="visualization root", default=None)
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True, choices=["eth3d", "kitti", "things"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

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
    parser.add_argument('--slant', type=str, default=None, help="use slanted stereo matching")
    parser.add_argument('--slant_norm', action='store_true', help="use normalization in slanted stereo matching")
    parser.add_argument('--confidence', action='store_true', help="use confidence learning")
    parser.add_argument('--offset_memory_size', type=int, default=2, help="size of offset memory in confidence learning")
    parser.add_argument('--detach_in_confidence', action='store_true', help="detach for feature and offset in confidence learning")
    args = parser.parse_args()

    assert args.sv_root is not None, "Please specify the visualization root"

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(".tar")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint from {args.restore_ckpt}")

    model.cuda()
    model.eval()

    logging.info(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':
        if args.root is None:
            args.root = "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/ETH3D"
        validate_eth3d(model, iters=args.valid_iters, root=args.root, 
                       sv_root=args.sv_root, mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti':
        if args.root is None:
            args.root = "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/KITTI2015"
        validate_kitti(model, iters=args.valid_iters, root=args.root, 
                       sv_root=args.sv_root, mixed_prec=use_mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        if args.root is None:
            args.root = "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Middlebury"
        validate_middlebury(model, iters=args.valid_iters, root=args.root, split=args.dataset[-1], 
                            sv_root=args.sv_root, mixed_prec=use_mixed_precision)

    elif args.dataset == 'things':
        if args.root is None:
            args.root = "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Sceneflow"
        validate_things(model, iters=args.valid_iters, root=args.root, 
                        sv_root=args.sv_root, mixed_prec=use_mixed_precision)
