# /usr/bin/bash

export LOG_ROOT="/data5/yao/runs/vis"
export TB_ROOT="/data5/yao/runs/tboard"
export CKPOINT_ROOT="/data5/yao/runs/ckpoint"


CUDA_VISIBLE_DEVICES=6 python3 vis_intermediate_results.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth --depthany_model_dir "/data5/yao/pretrained" --dataset booster --sv_root /data5/yao/runs/vis/inter_results --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --vis_inter --test_exp_name "SUPP"

CUDA_VISIBLE_DEVICES=6 python3 vis_intermediate_results.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_H --sv_root /data5/yao/runs/vis/inter_results --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --vis_inter --test_exp_name "SUPP"

CUDA_VISIBLE_DEVICES=6 python3 vis_intermediate_results.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth --depthany_model_dir "/data5/yao/pretrained" --dataset eth3d --sv_root /data5/yao/runs/vis/inter_results --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --vis_inter --test_exp_name "SUPP"

CUDA_VISIBLE_DEVICES=6 python3 vis_intermediate_results.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth --depthany_model_dir "/data5/yao/pretrained" --dataset kitti --sv_root /data5/yao/runs/vis/inter_results --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --vis_inter --test_exp_name "SUPP"

CUDA_VISIBLE_DEVICES=6 python3 vis_intermediate_results.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth --depthany_model_dir "/data5/yao/pretrained" --dataset kitti2012 --sv_root /data5/yao/runs/vis/inter_results --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --vis_inter --test_exp_name "SUPP"


# CUDA_VISIBLE_DEVICES=4 python3 vis_intermediate_results.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32_20241024_172455/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32.pth --depthany_model_dir "/data5/yao/pretrained" --dataset booster --sv_root /data5/yao/runs/vis/inter_results --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --vis_inter --test_exp_name "50000itr-RefineSigmoidPreMonoBatch32"


# CUDA_VISIBLE_DEVICES=4 python3 vis_intermediate_results.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32_20241024_172455/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32.pth --depthany_model_dir "/data5/yao/pretrained" --dataset booster --sv_root /data5/yao/runs/vis/inter_results --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --vis_inter --test_exp_name "final-RefineSigmoidPreMonoBatch32"