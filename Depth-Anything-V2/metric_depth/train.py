import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.stereo_datasets import Fooling3DDataset
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss, AffineInvariantLoss
from util.metric import eval_depth
from util.utils import init_log

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



parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='hypersim', choices=['hypersim', 'vkitti', 'Fooling3D'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')

parser.add_argument('--depth_as_mask', action='store_true', help='using depth>0 as mask')


def main():
    args = parser.parse_args()
    
    warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    rank, world_size = setup_distributed(port=args.port)
    
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('dataset/splits/vkitti2/train.txt', 'train', size=size)
    elif args.dataset.lower() == "fooling3d":
        aug_params = {'crop_size': (args.img_size, args.img_size), 
                      'min_scale': args.spatial_scale[0], 
                      'max_scale': args.spatial_scale[1], 
                      'do_flip': False, 
                      'yjitter': not args.noyjitter,
                      'resize': (args.img_size, args.img_size)}
        if hasattr(args, "saturation_range") and args.saturation_range is not None:
            aug_params["saturation_range"] = args.saturation_range
        if hasattr(args, "img_gamma") and args.img_gamma is not None:
            aug_params["gamma"] = args.img_gamma
        if hasattr(args, "do_flip") and args.do_flip is not None:
            aug_params["do_flip"] = args.do_flip
        trainset = Fooling3DDataset(aug_params, args=args, root='./datasets/Fooling3D')
    else:
        raise NotImplementedError
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)
    
    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    elif args.dataset == 'vkitti':
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    elif args.dataset.lower() == "fooling3d":
        aug_params = {}
        valset = Fooling3DDataset(aug_params, root="./datasets/Fooling3D", image_set="testing")
    else:
        raise NotImplementedError
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    if args.pretrained_from:
        new_state_dict = {k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loading {len(new_state_dict.keys())} layers")
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)
    
    # criterion = SiLogLoss().cuda(local_rank)
    criterion = AffineInvariantLoss().cuda(local_rank)
    
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    scaler = GradScaler(enabled=False)

    total_iters = args.epochs * len(trainloader)
    
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    
    for epoch in range(args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
            logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                            epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'], 
                            previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))
        
        trainloader.sampler.set_epoch(epoch + 1)
        
        model.train()
        total_loss = 0
        
        # for i, sample in enumerate(trainloader):
        #     optimizer.zero_grad()
            
        #     img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()

        mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
        
        for i, (path_info, *data_blob) in enumerate(trainloader):
            optimizer.zero_grad()
            
            with torch.no_grad():
                image1, image2, flow, valid = [x.cuda() for x in data_blob]

                img = (image1/255.0 - mean) / std
                depth = -flow[:,0]
                if args.depth_as_mask:
                    valid_mask = depth>0
                else:
                    valid_mask = valid
                # print(f"img: {img.shape}, depth: {depth.shape}, valid_mask: {valid_mask.shape}")
                
                if random.random() < 0.5:
                    img = img.flip(-1)
                    depth = depth.flip(-1)
                    valid_mask = valid_mask.flip(-1)
            
            try:
                pred = model(img)
                # print(f"pred: {pred.shape}")

                loss = criterion(pred, depth, (valid_mask == 1) & (depth > 0))

                # if torch.isnan(loss):
                #     print(f"{rank} Skipping iteration {i} due to NaN loss: {path_info[0]}")
                # loss_is_nan = torch.tensor(float(not torch.isfinite(loss)), device="cuda")
                # dist.all_reduce(loss_is_nan, op=dist.ReduceOp.SUM)
                # if loss_is_nan.item() > 0:
                #     optimizer.zero_grad()
                #     is_nan = torch.isnan(loss).any().float()
                #     continue

                is_nan = torch.isnan(loss).any().float()
                if is_nan == 1.0:
                    current_memory = torch.cuda.memory_allocated()
                    max_memory = torch.cuda.max_memory_allocated()
                    print(f"NaN loss detected at {i} iteration: {path_info[0]}" + \
                            f", valid+flow: {((valid >= 0.5) & (torch.sum(flow**2, dim=1).sqrt() < 1000)).unsqueeze(1).sum()}" + \
                            f", valid: {(valid >= 0.5).sum()}"  + \
                            f", flow: {(torch.sum(flow**2, dim=1).sqrt() < 1000).sum()}" + \
                            f", {valid.shape}, {flow.shape}" + \
                            f", flow min: {torch.sum(flow**2, dim=1).sqrt().min()}" + \
                            f", flow max: {torch.sum(flow**2, dim=1).sqrt().max()}" + \
                            f", pred min: {pred.min()}" + \
                            f", pred max: {pred.max()}" + \
                            f", cur mem: {current_memory / 1024**2:.2f} MB" + \
                            f", max mem: {max_memory / 1024**2:.2f} MB")
                    if rank == 0:
                        # print("-"*10, (valid >= 0.5).unsqueeze(1).float().dtype)
                        vutils.save_image(image1[0]/255, "image1.png")
                        vutils.save_image((valid >= 0.5).unsqueeze(1).float()[0], "output_valid.png")
                        vutils.save_image((-flow[:,0]).float()[0] / (-flow[:,0]).float()[0].max(), "output_flow.png")
                dist.all_reduce(is_nan, op=dist.ReduceOp.MAX)
                if is_nan.item() == 1.0:
                    optimizer.zero_grad()
                    # del loss, img, pred  # 释放 batch 数据
                    # torch.cuda.empty_cache()
                    continue
                
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # optimizer.step()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                # scheduler.step()
                scaler.update()

            except Exception as err:
                current_memory = torch.cuda.memory_allocated()
                max_memory = torch.cuda.max_memory_allocated()
                raise Exception(err, f"iteration: {i}, path_info[0]: {path_info[0]}, img: {img.shape}, depth: {depth.shape}, valid_mask: {valid_mask.shape}" + \
                                f", {current_memory / 1024**2:.2f} MB" + \
                                f", {max_memory / 1024**2:.2f} MB")
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0
            
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)
            
            if rank == 0 and i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))
                writer.add_images('train/image1', image1[:4]/255, iters)
                writer.add_images('train/depth', depth[:4].unsqueeze(1)/depth[:4].max(), iters)
                writer.add_images('train/pred', pred[:4].unsqueeze(1), iters)
                writer.add_images('train/valid_mask', valid_mask[:4].unsqueeze(1), iters)
            # break


        model.eval()
        
        results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
        nsamples = torch.tensor([0.0]).cuda()
        
        # for i, sample in enumerate(valloader):
            
        #     img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]

        for i, (path_info, *data_blob) in enumerate(valloader):
            image1, image2, flow, valid = [x.cuda() for x in data_blob]
            
            image1 = F.interpolate(image1, size=(args.img_size, args.img_size), mode='bilinear', align_corners=True)
            # flow   = F.interpolate(flow, size=(args.img_size, args.img_size), mode='bilinear', align_corners=True)
            # valid  = F.interpolate(valid.unsqueeze(1), size=(args.img_size, args.img_size), mode='nearest')
            
            img = (image1/255.0 - mean) / std
            depth = -flow[:,0]
            if args.depth_as_mask:
                valid_mask = depth>0
            else:
                valid_mask = valid
            # print(f"img: {img.shape}, depth: {depth.shape}, valid_mask: {valid_mask.shape}")
            
            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True).squeeze(1)
                # print(f"pred: {pred.shape}")
            
            valid_mask = (valid_mask == 1) & (depth > 0)
            
            if valid_mask.sum() < 10:
                continue
            
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1
        
        torch.distributed.barrier()
        
        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)
        
        if rank == 0:
            logger.info('==========================================================================================')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
            logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
            logger.info('==========================================================================================')
            print()
            
            for name, metric in results.items():
                writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
        
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))


if __name__ == '__main__':
    main()