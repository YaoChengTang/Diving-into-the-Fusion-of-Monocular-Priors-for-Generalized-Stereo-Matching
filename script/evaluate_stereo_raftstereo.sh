# /usr/bin/bash

export LOG_ROOT="/data5/yao/runs/log"
export TB_ROOT="/data5/yao/runs/tboard"
export CKPOINT_ROOT="/data5/yao/runs/ckpoint"

export CUDA_VISIBLE_DEVICES=2

python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/raftstereo_20240821_142156/raftstereo.pth --dataset middlebury_H --model_name "RaftStereo" --test_exp_name "final"
python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/raftstereo_20240821_142156/90000_raftstereo.pth --dataset middlebury_H --model_name "RaftStereo" --test_exp_name "90000itr"
python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/raftstereo_20240821_142156/80000_raftstereo.pth --dataset middlebury_H --model_name "RaftStereo" --test_exp_name "80000itr"
python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/raftstereo_20240821_142156/70000_raftstereo.pth --dataset middlebury_H --model_name "RaftStereo" --test_exp_name "70000itr"
# python3 evaluate_stereo_raftstereo.py --restore_ckpt pretrained/rebuild-raft2-noSF-singleMask.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset kitti
# python3 evaluate_stereo_raftstereo.py --restore_ckpt pretrained/rebuild-raft2-noSF-singleMask.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset eth3d

python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDisp_20240821_142314/RaftStereoDisp.pth --dataset middlebury_H --model_name "RaftStereoDisp" --test_exp_name "final"
python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDisp_20240821_142314/90000_RaftStereoDisp.pth --dataset middlebury_H --model_name "RaftStereoDisp" --test_exp_name "90000itr"
python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDisp_20240821_142314/80000_RaftStereoDisp.pth --dataset middlebury_H --model_name "RaftStereoDisp" --test_exp_name "80000itr"
python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDisp_20240821_142314/70000_RaftStereoDisp.pth --dataset middlebury_H --model_name "RaftStereoDisp" --test_exp_name "70000itr"




python3 merge_sheet.py