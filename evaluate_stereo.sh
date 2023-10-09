# /usr/bin/bash

# CUDA_VISIBLE_DEVICE=1 python3 evaluate_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=1 python3 evaluate_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset kitti
# CUDA_VISIBLE_DEVICE=1 python3 evaluate_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset eth3d

# CUDA_VISIBLE_DEVICE=1 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-raw.pth --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=1 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-raw.pth --dataset kitti
# CUDA_VISIBLE_DEVICE=1 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-raw.pth --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-largeBatch.pth --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-largeBatch.pth --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-largeBatch.pth --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raftstereo-NS.tar --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raftstereo-NS.tar --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raftstereo-NS.tar --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slant.pth --slant "slant" --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slant.pth --slant "slant" --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slant.pth --slant "slant" --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange.pth --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange.pth --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange.pth --slant "slant" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange2.pth --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange2.pth --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange2.pth --slant "slant" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange-smooth.pth --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange-smooth.pth --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange-smooth.pth --slant "slant" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNormRange3.pth --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNormRange3.pth --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNormRange3.pth --slant "slant" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNormRange-SmoothDiamond_edge-5.pth --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNormRange-SmoothDiamond_edge-5.pth --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNormRange-SmoothDiamond_edge-5.pth --slant "slant" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-LocalSlant.pth --slant "slant_local" --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-LocalSlant.pth --slant "slant_local" --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-LocalSlant.pth --slant "slant_local" --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantLocal-moreiter.pth --slant "slant_local" --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantLocal-moreiter.pth --slant "slant_local" --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantLocal-moreiter.pth --slant "slant_local" --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-OffsetConfidence.pth --confidence --offset_memory_size 3 --detach_in_confidence --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-OffsetConfidence.pth --confidence --offset_memory_size 3 --detach_in_confidence --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-OffsetConfidence.pth --confidence --offset_memory_size 3 --detach_in_confidence --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-OffsetConfidence_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-OffsetConfidence_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-OffsetConfidence_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --dataset eth3d

