# /usr/bin/bash

export LOG_ROOT="/data5/yao/runs/vis"
export TB_ROOT="/data5/yao/runs/tboard"
export CKPOINT_ROOT="/data5/yao/runs/ckpoint"



# CUDA_VISIBLE_DEVICES=0 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK3_20240917_122628/RaftStereoDepthBetaK3.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_H --sv_root /data5/yao/runs/vis --model_name "RAFTStereoDepthBeta" --lbp_neighbor_offsets "(-3,-3), (3,3), (3,-3), (-3,3)" --modulation_ratio 1.0 --test_exp_name "final-N3"

# CUDA_VISIBLE_DEVICES=1 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthAny_20240908_125231/RaftStereoDepthAny.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_H --sv_root /data5/yao/runs/vis --model_name "RAFTStereoDepthAny" --test_exp_name "final"

# CUDA_VISIBLE_DEVICES=2 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53_20240920_165346/80000_RaftStereoDepthBetaK53.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_H --sv_root /data5/yao/runs/vis --model_name "RAFTStereoDepthBeta" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --test_exp_name "8000-N53"

# CUDA_VISIBLE_DEVICES=3 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53_20240920_165346/80000_RaftStereoDepthBetaK53.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_H --sv_root /data5/yao/runs/vis --model_name "RAFTStereoDepthBeta" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --test_exp_name "8000-N53" --mask

# CUDA_VISIBLE_DEVICES=3 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32_20241024_172455/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_F --sv_root /data5/yao/runs/vis --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-10,-10), (10,10), (10,-10), (-10,10), (-6,0), (6,0), (0,-6), (0,6)" --modulation_ratio 1.0 --test_exp_name "MF - final-RefineSigmoidPreMonoBatch32" --improvement --movement


# CUDA_VISIBLE_DEVICES=3 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32_20241024_172455/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32.pth --depthany_model_dir "/data5/yao/pretrained" --dataset booster --sv_root /data5/yao/runs/vis --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --test_exp_name "booster-mono-final-RefineSigmoidPreMonoBatch32"

# CUDA_VISIBLE_DEVICES=3 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32_20241024_172455/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_F --sv_root /data5/yao/runs/vis --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --test_exp_name "middlebury_F-mono-final-RefineSigmoidPreMonoBatch32"

# CUDA_VISIBLE_DEVICES=3 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_F --sv_root /data5/yao/runs/vis --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --test_exp_name "middlebury_F-50000itr-RefineSigmoidPreMonoBatch48ConfDim"

# CUDA_VISIBLE_DEVICES=4 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth --depthany_model_dir "/data5/yao/pretrained" --dataset booster --sv_root /data5/yao/runs/vis --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --test_exp_name "booster-50000itr-RefineSigmoidPreMonoBatch48ConfDim-visRange"


# CUDA_VISIBLE_DEVICES=1 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/raftstereo_20240821_142156/raftstereo.pth --dataset booster --sv_root /data5/yao/runs/vis --model_name "RaftStereo" --test_exp_name "booster - final-raftstereo"

CUDA_VISIBLE_DEVICES=6 python3 gen_sample_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/raftstereo_20240821_142156/raftstereo.pth --dataset booster --sv_root /data5/yao/runs/vis --model_name "RaftStereo" --test_exp_name "booster-final-raftstereo-visRange"