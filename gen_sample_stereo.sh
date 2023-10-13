# /usr/bin/bash

# CUDA_VISIBLE_DEVICE=1 python3 gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=1 python3 gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset kitti
# CUDA_VISIBLE_DEVICE=1 python3 gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset eth3d

# CUDA_VISIBLE_DEVICE=1 python3 gen_sample_stereo.py --restore_ckpt pretrained/raft-stereo-raw.pth --sv_root /home/users/chengtang.yao/Output/raftstereo-raw --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=1 python3 gen_sample_stereo.py --restore_ckpt pretrained/raft-stereo-raw.pth --sv_root /home/users/chengtang.yao/Output/raftstereo-raw --dataset kitti
# CUDA_VISIBLE_DEVICE=1 python3 gen_sample_stereo.py --restore_ckpt pretrained/raft-stereo-raw.pth --sv_root /home/users/chengtang.yao/Output/raftstereo-raw --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-NS.tar --dataset middlebury_H --sv_root /home/users/chengtang.yao/Output/raftstereo-nerf
# CUDA_VISIBLE_DEVICE=2 python3 gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-NS.tar --dataset kitti --sv_root /home/users/chengtang.yao/Output/raftstereo-nerf
# CUDA_VISIBLE_DEVICE=2 python3 gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-NS.tar --dataset eth3d --sv_root /home/users/chengtang.yao/Output/raftstereo-nerf

# CUDA_VISIBLE_DEVICE=2 python3 gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --dataset middlebury_H --sv_root /home/users/chengtang.yao/Output/raftstereo-confidence
# CUDA_VISIBLE_DEVICE=2 python3 gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --dataset kitti --sv_root /home/users/chengtang.yao/Output/raftstereo-confidence
# CUDA_VISIBLE_DEVICE=2 python3 gen_sample_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --dataset eth3d --sv_root /home/users/chengtang.yao/Output/raftstereo-confidence

CUDA_VISIBLE_DEVICE=2 python3 gen_sample_stereo.py --restore_ckpt pretrained/raft-stereo-Refine_Conf_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 4 --refine_start_itr 3 --dataset middlebury_H --sv_root /home/users/chengtang.yao/Output/raftstereo-Refine_Conf_loss