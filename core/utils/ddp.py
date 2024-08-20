# ==============================================================================
# Copyright (c) 2022 The PersFormer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import random
import logging
import subprocess
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',)

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import DataParallel as DP
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS

from core.raft_stereo import RAFTStereo
from core.raft_stereo_disp import RAFTStereoDisp


def setup_distributed(args):
    args.rank = int(os.getenv("RANK"))
    args.local_rank = int(os.getenv("LOCAL_RANK"))
    args.world_size = int(os.getenv("WORLD_SIZE"))
    # print("-"*10, "local_rank: {}, world_size:{}".format(args.local_rank, args.world_size),
    #      " - {}, {}".format(dist.get_rank(), dist.get_world_size()))  # they result in the same value
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    torch.set_printoptions(precision=10)

def ddp_init(args):
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) >= 1

    if args.distributed:
        setup_distributed(args)

    # deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.local_rank)
    np.random.seed(args.local_rank)
    random.seed(args.local_rank)

    print("complete initialization of local_rank:{}".format(args.local_rank))

def ddp_close():
    dist.destroy_process_group()

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def reduce_tensors(*tensors, world_size):
    return [reduce_tensor(tensor, world_size) for tensor in tensors]


def get_loader(dataset, args):
    """
        create dataset from ground-truth
        return a batch sampler based ont the dataset
    """
    if args.distributed:
        if args.local_rank == 0:
            print('use distributed sampler')
        data_sampler = DS(dataset, shuffle=True, drop_last=True)
        data_loader = DataLoader(dataset,
                                batch_size=args.batch_size, 
                                sampler=data_sampler,
                                num_workers=args.num_workers, 
                                pin_memory=True,
                                persistent_workers=True)
    else:
        if args.local_rank == 0:
            print("use default sampler")
        # data_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idx)
        # data_loader = DataLoader(transformed_dataset,
        #                         batch_size=args.batch_size, sampler=data_sampler,
        #                         num_workers=args.nworkers, pin_memory=True,
        #                         persistent_workers=True,
        #                         worker_init_fn=seed_worker,
        #                         generator=g)
        train_loader = DataLoader(train_dataset, 
								  batch_size=args.batch_size, 
								  pin_memory=True, shuffle=True, 
                                  num_workers=args.num_workers, 
                                  drop_last=True)
    return data_loader

NODE_RANK    = os.getenv('NODE_RANK', default=0)
LOCAL_RANK   = os.getenv("LOCAL_RANK", default=0)

def get_model_ddp(args):
    if args.model_name.lower() == "raftstereo":
        model  = nn.SyncBatchNorm.convert_sync_batchnorm(RAFTStereo(args))
    elif args.model_name.lower() == "raftstereodisp":
        model  = nn.SyncBatchNorm.convert_sync_batchnorm(RAFTStereoDisp(args))
    else :
        raise Exception("No such model: {}".format(args.model_name))
    
    device = torch.device("cuda", args.local_rank)
    model  = model.to(device)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(".tar")
        if args.local_rank==0 :
            logging.info("Loading checkpoint from {} ...".format(args.restore_ckpt))
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        if args.local_rank==0 :
            logging.info(f"Done loading checkpoint")
            
    dist.barrier()

    # DDP setting
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, 
                    find_unused_parameters=True)
    else:
        model = DP(RAFTStereo(args))
    return model
