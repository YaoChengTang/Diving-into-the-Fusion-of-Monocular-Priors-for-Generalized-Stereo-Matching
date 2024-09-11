# /usr/bin/bash

export LOG_ROOT="/data5/yao/runs/log"
export TB_ROOT="/data5/yao/runs/tboard"
export CKPOINT_ROOT="/data5/yao/runs/ckpoint"

export CUDA_VISIBLE_DEVICES=7

# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/raftstereo_20240821_142156/raftstereo.pth --dataset middlebury_H --model_name "RaftStereo" --test_exp_name "final"
# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/raftstereo_20240821_142156/90000_raftstereo.pth --dataset middlebury_H --model_name "RaftStereo" --test_exp_name "90000itr"
# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/raftstereo_20240821_142156/80000_raftstereo.pth --dataset middlebury_H --model_name "RaftStereo" --test_exp_name "80000itr"
# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/raftstereo_20240821_142156/70000_raftstereo.pth --dataset middlebury_H --model_name "RaftStereo" --test_exp_name "70000itr"
# python3 evaluate_stereo_raftstereo.py --restore_ckpt pretrained/rebuild-raft2-noSF-singleMask.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset kitti
# python3 evaluate_stereo_raftstereo.py --restore_ckpt pretrained/rebuild-raft2-noSF-singleMask.pth --slant "slant_local" --slant_norm --geo_estimator "geometry_conv" --dataset eth3d

# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDisp_20240821_142314/RaftStereoDisp.pth --dataset middlebury_H --model_name "RaftStereoDisp" --test_exp_name "final"
# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDisp_20240821_142314/90000_RaftStereoDisp.pth --dataset middlebury_H --model_name "RaftStereoDisp" --test_exp_name "90000itr"
# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDisp_20240821_142314/80000_RaftStereoDisp.pth --dataset middlebury_H --model_name "RaftStereoDisp" --test_exp_name "80000itr"
# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDisp_20240821_142314/70000_RaftStereoDisp.pth --dataset middlebury_H --model_name "RaftStereoDisp" --test_exp_name "70000itr"

# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoMast3r_deep3_20240902_201411/RaftStereoMast3r_deep3.pth --mast3r_model_path "/data5/yao/pretrained/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" --dataset middlebury_H --model_name "RAFTStereoMast3r" --test_exp_name "final"
# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoMast3r_deep3_20240902_201411/90000_RaftStereoMast3r_deep3.pth --mast3r_model_path "/data5/yao/pretrained/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" --dataset middlebury_H --model_name "RAFTStereoMast3r" --test_exp_name "90000itr"
# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoMast3r_deep3_20240902_201411/80000_RaftStereoMast3r_deep3.pth --mast3r_model_path "/data5/yao/pretrained/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" --dataset middlebury_H --model_name "RAFTStereoMast3r" --test_exp_name "80000itr"
# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoMast3r_deep3_20240902_201411/70000_RaftStereoMast3r_deep3.pth --mast3r_model_path "/data5/yao/pretrained/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" --dataset middlebury_H --model_name "RAFTStereoMast3r" --test_exp_name "70000itr"
# python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoMast3r_deep3_20240902_201411/60000_RaftStereoMast3r_deep3.pth --mast3r_model_path "/data5/yao/pretrained/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" --dataset middlebury_H --model_name "RAFTStereoMast3r" --test_exp_name "60000itr"


python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthAny_20240908_125231/RaftStereoDepthAny.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_H --model_name "RAFTStereoDepthAny" --test_exp_name "final"
python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthAny_20240908_125231/90000_RaftStereoDepthAny.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_H --model_name "RAFTStereoDepthAny" --test_exp_name "90000itr"
python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthAny_20240908_125231/80000_RaftStereoDepthAny.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_H --model_name "RAFTStereoDepthAny" --test_exp_name "80000itr"
python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthAny_20240908_125231/70000_RaftStereoDepthAny.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_H --model_name "RAFTStereoDepthAny" --test_exp_name "70000itr"
python3 evaluate_stereo_raftstereo.py --restore_ckpt /data5/yao/runs/ckpoint/RaftStereoDepthAny_20240908_125231/60000_RaftStereoDepthAny.pth --depthany_model_dir "/data5/yao/pretrained" --dataset middlebury_H --model_name "RAFTStereoDepthAny" --test_exp_name "60000itr"




python3 merge_sheet.py