# /usr/bin/bash

CUDA_VISIBLE_DEVICES=2 python3 evaluate_stereo.py --restore_ckpt pretrained/rebuild-raft2-noSF-singleMask.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset middlebury_H
CUDA_VISIBLE_DEVICES=2 python3 evaluate_stereo.py --restore_ckpt pretrained/rebuild-raft2-noSF-singleMask.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset kitti
CUDA_VISIBLE_DEVICES=2 python3 evaluate_stereo.py --restore_ckpt pretrained/rebuild-raft2-noSF-singleMask.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset eth3d



python3 merge_sheet.py