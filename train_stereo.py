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
from torch.utils.tensorboard import SummaryWriter

LOG_ROOT     = os.getenv('LOG_ROOT', default="")
TB_ROOT      = os.getenv('TB_ROOT', default="")
CKPOINT_ROOT = os.getenv('CKPOINT_ROOT', default="")
logging.basicConfig(filename=os.path.join("logs" if LOG_ROOT is None or len(LOG_ROOT)==0 else LOG_ROOT, 
                                          'log-{}.log'.format(datetime.now().strftime("%y%m%d_%H%M%S"))), 
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

from evaluate_stereo import *
from core.loss import Loss
from core.raft_stereo import RAFTStereo
from core.stereo_datasets import fetch_dataloader
from core.utils.ddp import ddp_init, ddp_close, get_model_ddp

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


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir='runs' if TB_ROOT is None or len(TB_ROOT)==0 else TB_ROOT)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs' if TB_ROOT is None or len(TB_ROOT)==0 else TB_ROOT)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs' if TB_ROOT is None or len(TB_ROOT)==0 else TB_ROOT)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = get_model_ddp(args)
    if args.local_rank==0 :
        logging.info("Parameter Count: %d" % count_parameters(model))

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    if args.local_rank==0:
        logger = Logger(model, scheduler)

    model.cuda()
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    myLoss = Loss(loss_gamma=0.9, max_flow=700, loss_zeta=0.5,
                    smoothness=args.loss_smooth, 
                    slant=args.slant, slant_norm=args.slant_norm,
                    ner_kernel_size=args.ner_kernel_size,
                    ner_weight_reduce=args.ner_weight_reduce,
                    local_rank=args.local_rank,
                    mixed_precision=args.mixed_precision)
    device  = torch.device("cuda", args.local_rank)
    myLoss  = myLoss.to(device)
                    
    validation_frequency = 10000

    scaler = GradScaler(enabled=args.mixed_precision)

    total_steps = 0
    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:
        
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader, disable=args.local_rank>0)):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            assert model.training
            if args.slant:
                flow_predictions, params_list = model(image1, image2, iters=args.train_iters)
            else:
                flow_predictions = model(image1, image2, iters=args.train_iters)
            assert model.training

            try:
                loss, metrics, \
                flow_loss, smooth_loss = myLoss(flow_predictions, flow, valid, 
                                                params_list=params_list, 
                                                imgL=image1, imgR=None)
            except Exception as err:
                if args.local_rank==0:
                    debug_info = ""
                    n_predictions = len(flow_predictions)
                    for i in range(n_predictions):
                        if torch.isnan(flow_preds[i]).any():
                            debug_info += f" {i}-iter contains NAN."
                        if torch.isinf(flow_preds[i]).any():
                            debug_info += f" {i}-iter contains INF."
                    for name, param in model.named_parameters():
                        if param.requires_grad and torch.isnan(param).any():
                            debug_info += f" NAN found in parameter: {name}"
                        if param.requires_grad and torch.isinf(param).any():
                            debug_info += f" INF found in parameter: {name}"
                    logging.info(debug_info)
                raise Exception(err)

            if args.local_rank==0:
                logger.push(metrics)
                logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
                logger.writer.add_scalar("live_flow_loss", flow_loss.item(), global_batch_num)
                logger.writer.add_scalar("live_smooth_loss", smooth_loss.item(), global_batch_num)
                logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if total_steps % validation_frequency == validation_frequency - 1:
                if args.local_rank==0:
                    save_path = os.path.join(CKPOINT_ROOT, 
                                    'checkpoints/%d_%s.pth' % (total_steps + 1, args.name))
                    logging.info(f"Saving file {save_path}")
                    torch.save(model.state_dict(), save_path)
                results = validate_things(model.module, iters=args.valid_iters, args=args)

                if args.local_rank==0:
                    logger.write_dict(results)

                model.train()
                model.module.freeze_bn()

            total_steps += 1
            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if args.local_rank==0 and len(train_loader) >= 10000:
            save_path = os.path.join(CKPOINT_ROOT, 
                            'checkpoints/%d_epoch_%s.pth.gz' % (total_steps + 1, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    if args.local_rank==0:
        logger.close()
        PATH = os.path.join(CKPOINT_ROOT, 'checkpoints/%s.pth' % args.name)
        torch.save(model.state_dict(), PATH)
        print("FINISHED TRAINING")

    return None


def init_directory(args):
    if args.local_rank==0 :
        if not os.path.exists( os.path.join(CKPOINT_ROOT, 'checkpoints') ):
            os.makedirs( os.path.join(CKPOINT_ROOT, 'checkpoints') )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

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
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--slant', action='store_true', help="use slanted stereo matching")
    parser.add_argument('--slant_norm', action='store_true', help="use normalization in slanted stereo matching")
    
    # Loss parameters
    parser.add_argument('--loss_smooth', type=str, default=None, choices=["", "gradient", "curvature"], help="use smoothness loss")
    parser.add_argument('--ner_kernel_size', default=3, help="nerghborhood size used in smooth loss")
    parser.add_argument('--ner_weight_reduce', action='store_true', help="pre-reduce in the nerghborhood computation of smooth loss")

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

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    ddp_init(args)
    init_directory(args)

    train(args)

    ddp_close()
