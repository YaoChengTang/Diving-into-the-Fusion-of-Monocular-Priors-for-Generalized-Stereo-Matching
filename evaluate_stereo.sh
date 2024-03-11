# /usr/bin/bash

CUDA_VISIBLE_DEVICES=2 python3 evaluate_stereo.py --restore_ckpt pretrained/rebuild-raft-stereo-3090-20240307-165740.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset middlebury_H
CUDA_VISIBLE_DEVICES=2 python3 evaluate_stereo.py --restore_ckpt pretrained/rebuild-raft-stereo-3090-20240307-165740.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset kitti
CUDA_VISIBLE_DEVICES=2 python3 evaluate_stereo.py --restore_ckpt pretrained/rebuild-raft-stereo-3090-20240307-165740.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset eth3d


CUDA_VISIBLE_DEVICES=2 python3 evaluate_stereo.py --restore_ckpt pretrained/rebuild-raft-stereo-a10-4-20240307-170859.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset middlebury_H
CUDA_VISIBLE_DEVICES=2 python3 evaluate_stereo.py --restore_ckpt pretrained/rebuild-raft-stereo-a10-4-20240307-170859.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset kitti
CUDA_VISIBLE_DEVICES=2 python3 evaluate_stereo.py --restore_ckpt pretrained/rebuild-raft-stereo-a10-4-20240307-170859.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset eth3d