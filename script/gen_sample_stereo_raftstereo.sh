# /usr/bin/bash

export LOG_ROOT="/data5/yao/runs/vis"
export TB_ROOT="/data5/yao/runs/tboard"
export CKPOINT_ROOT="/data5/yao/runs/ckpoint"



# CUDA_VISIBLE_DEVICES=1 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK3_20240917_122628/RaftStereoDepthBetaK3.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_H --sv_root /data5/yao/runs/vis --model_name "RAFTStereoDepthBeta" --lbp_neighbor_offsets "(-3,-3), (3,3), (3,-3), (-3,3)" --modulation_ratio 1.0 --test_exp_name "final-N3"

CUDA_VISIBLE_DEVICES=1 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthAny_20240908_125231/RaftStereoDepthAny.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_H --sv_root /data5/yao/runs/vis --model_name "RAFTStereoDepthAny" --test_exp_name "final"
