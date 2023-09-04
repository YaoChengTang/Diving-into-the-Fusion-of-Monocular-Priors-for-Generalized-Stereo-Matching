# /usr/bin/bash

# CUDA_VISIBLE_DEVICE=1 python3 gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=1 python3 gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset kitti
# CUDA_VISIBLE_DEVICE=1 python3 gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset eth3d

CUDA_VISIBLE_DEVICE=1 python3 gen_sample_stereo.py --restore_ckpt pretrained/raft-stereo-raw.pth --sv_root /home/users/chengtang.yao/Output/raftstereo-raw --dataset middlebury_H
CUDA_VISIBLE_DEVICE=1 python3 gen_sample_stereo.py --restore_ckpt pretrained/raft-stereo-raw.pth --sv_root /home/users/chengtang.yao/Output/raftstereo-raw --dataset kitti
CUDA_VISIBLE_DEVICE=1 python3 gen_sample_stereo.py --restore_ckpt pretrained/raft-stereo-raw.pth --sv_root /home/users/chengtang.yao/Output/raftstereo-raw --dataset eth3d
