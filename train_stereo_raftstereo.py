from __future__ import print_function, division

import os
import sys
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0,'core')
sys.path.insert(0,'core/utils')

DATASET_ROOT = os.getenv('DATASET_ROOT', default="")
NODE_RANK    = os.getenv('NODE_RANK', default=0)
LOCAL_RANK   = os.getenv("LOCAL_RANK", default=0)
LOG_ROOT     = os.getenv('LOG_ROOT', default="logs")
TB_ROOT      = os.getenv('TB_ROOT', default="")
CKPOINT_ROOT = os.getenv('CKPOINT_ROOT', default="")

from core.loss import sequence_loss
from core.raft_stereo import RAFTStereo
from core.stereo_datasets import fetch_dataloader
from core.utils.ddp import ddp_init, ddp_close, get_model_ddp
from core.utils.utils import LoggerTraining, init_directories, delete_directories_if_static
from evaluate_stereo_raftstereo import *

logger = LoggerTraining("TRAIN", None, None)

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def train(args):
    model = get_model_ddp(args)

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    
    logger.set_training(model, scheduler)
    logger.info("Parameter Count: %d" % count_parameters(model))

    model.cuda()
    model.train()
    if not args.stop_freeze_bn:
        model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 10000

    scaler = GradScaler(enabled=args.mixed_precision)

    total_steps = 0
    should_keep_training = True
    global_batch_num = 0
    tqdm_disable = args is not None and (args.silence or args.local_rank>0 or int(NODE_RANK)>0)

    while should_keep_training:
        
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader, disable=tqdm_disable)):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            assert model.training
            flow_predictions = model(image1, image2, iters=args.train_iters)
            assert model.training

            corrupted = True
            n_predictions = len(flow_predictions)
            for i in range(n_predictions):
                if not torch.isnan(flow_predictions[i]).any() and not torch.isinf(flow_predictions[i]).any():
                    corrupted = False
            if corrupted:
                continue

            loss, metrics = sequence_loss(flow_predictions, flow, valid)
            if args.local_rank==0 and int(NODE_RANK)==0:
                logger.push(metrics)
                logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
                logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            
            global_batch_num += 1
            scaler.scale(loss).backward()
            # logger.info(model.module.fnet.layer1[0].weight.grad)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()



            if total_steps % validation_frequency == validation_frequency - 1:
                if args.local_rank==0 and int(NODE_RANK)==0:
                    save_path = os.path.join(CKPOINT_ROOT, 
                                    '%d_%s.pth' % (total_steps + 1, args.exp_name))
                    logger.info(f"Saving file {save_path}")
                    torch.save(model.state_dict(), save_path)

                # middlebury_root = "/".join(DATASET_ROOT.split("/")[:-1]) + "/Middlebury"
                # results = validate_middlebury(model.module, iters=args.valid_iters, split="H", root=middlebury_root, )
                results = validate_things(model.module, iters=args.valid_iters)
                if args.local_rank==0 and int(NODE_RANK)==0:
                    logger.write_dict(results)

                model.train()
                model.module.freeze_bn()

            total_steps += 1
            if total_steps > args.num_steps:
                should_keep_training = False
                break

    if args.local_rank==0 and int(NODE_RANK)==0:
        logger.close()
        PATH = os.path.join(CKPOINT_ROOT, '%s.pth' % args.exp_name)
        torch.save(model.state_dict(), PATH)
        print("FINISHED TRAINING")

    return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='raft-stereo', help="name your experiment")
    parser.add_argument('--model_name', default='RaftStereo', help="name your model: raftstereo, raftstereodisp, RAFTStereoMast3r, RAFTStereoDepthAny, RAFTStereoNoCTX, raftstereodepthfusion")
    parser.add_argument('--mast3r_model_path', default='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', help="pretrained model path for MaSt3R")
    parser.add_argument('--depthany_model_dir', default='/data5/yao/pretrained', help="directory of pretrained model path for DepthAnything")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    parser.add_argument('--silence', action='store_true', help='no output of training/eval process')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--num_workers', type=int, default=8, help="number of worker used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
    
    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "abs_reg", "alt", "abs_alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--stop_freeze_bn', action='store_true', help="stop freeze BN")
    parser.add_argument('--lbp_neighbor_offsets', default='(-1,-1), (1,1), (1,-1), (-1,1)', help="determine the neighbors used in LBP encoder")
    parser.add_argument('--modulation_ratio', type=float, default=1., help="hyperparameters for modulation")
    
    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')

    # DDP setting
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--local-rank", type=int, default=os.getenv("LOCAL_RANK"))
    parser.add_argument('--world-size', type=int, default=os.getenv("WORLD_SIZE"))
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK"))
    parser.add_argument('--world_size', type=int, default=os.getenv("WORLD_SIZE"))

    args = parser.parse_args()
    logger.print_args(args)
    
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    ddp_init(args)
    init_directories([LOG_ROOT, TB_ROOT, CKPOINT_ROOT])

    train(args)

    # try:
    #     train(args)
    # except Exception as err:
    #     delete_directories_if_static([LOG_ROOT, TB_ROOT, CKPOINT_ROOT])
    #     raise Exception(err)

    ddp_close()
