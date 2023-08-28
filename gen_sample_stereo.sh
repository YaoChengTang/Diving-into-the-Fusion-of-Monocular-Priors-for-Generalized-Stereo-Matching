# /usr/bin/bash

CUDA_VISIBLE_DEVICE=1 python gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset middlebury_H
CUDA_VISIBLE_DEVICE=1 python gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset kitti
CUDA_VISIBLE_DEVICE=1 python gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset eth3d
