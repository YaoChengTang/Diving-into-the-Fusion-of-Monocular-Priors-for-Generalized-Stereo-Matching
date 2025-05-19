# /usr/bin/bash

export LOG_ROOT="/data5/yao/runs/vis"
export TB_ROOT="/data5/yao/runs/tboard"
export CKPOINT_ROOT="/data5/yao/runs/ckpoint"



# CUDA_VISIBLE_DEVICES=4 python3 infer_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth --depthany_model_dir "/data5/yao/pretrained" \
# --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea \
# --dataset flicker1024_val --root /data6/Flickr1024 --img_path_txt /home/yao/Document/OpenStereo/OpenStereo-1f93c822e9c9d571a5238c813560478bc4c18662/data/Flicker1024/Flicker1024_validation.txt \
# --sv_root /data5/yao/runs/vis --test_exp_name "ICCV-Rebuttal"

# CUDA_VISIBLE_DEVICES=5 python3 infer_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth --depthany_model_dir "/data5/yao/pretrained" \
# --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea \
# --dataset flicker1024_train --root /data6/Flickr1024 --img_path_txt /home/yao/Document/OpenStereo/OpenStereo-1f93c822e9c9d571a5238c813560478bc4c18662/data/Flicker1024/Flicker1024_train.txt \
# --sv_root /data5/yao/runs/vis --test_exp_name "ICCV-Rebuttal"

# CUDA_VISIBLE_DEVICES=6 python3 infer_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth --depthany_model_dir "/data5/yao/pretrained" \
# --model_name "RAFTStereoDepthBetaRefine" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea \
# --dataset flicker1024_test --root /data6/Flickr1024 --img_path_txt /home/yao/Document/OpenStereo/OpenStereo-1f93c822e9c9d571a5238c813560478bc4c18662/data/Flicker1024/Flicker1024_test.txt \
# --sv_root /data5/yao/runs/vis --test_exp_name "ICCV-Rebuttal"