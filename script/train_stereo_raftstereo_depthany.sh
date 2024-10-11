# /usr/bin/bash


# 获取当前 shell 文件名（不包含路径和扩展名）
SCRIPT_NAME=$(basename "$0" .sh)

# 获取当前时间
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

# 如果有参数，使用参数作为文件夹名，否则使用脚本名加时间
if [ -n "$1" ]; then
    FOLDER_NAME="${1}_${CURRENT_TIME}"
    EXP_NAME="${1}"
else
    FOLDER_NAME="${SCRIPT_NAME}_${CURRENT_TIME}"
    EXP_NAME="${SCRIPT_NAME}"
fi


# export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0  # 设置正确的网络接口
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=29501
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7

# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/Sceneflow"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/Middlebury"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/KITTI2015"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/ETH3D"
export DATASET_ROOT="/data6/sceneflow/sceneflow"

export LOG_ROOT="/data5/yao/runs/log/${FOLDER_NAME}"
export TB_ROOT="/data5/yao/runs/tboard/${FOLDER_NAME}"
export CKPOINT_ROOT="/data5/yao/runs/ckpoint/${FOLDER_NAME}"

# 输出新的路径，确认设置正确
echo "LOG_ROOT is set to: $LOG_ROOT"
echo "TB_ROOT is set to: $TB_ROOT"
echo "CKPOINT_ROOT is set to: $CKPOINT_ROOT"



# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthAny" --depthany_model_dir "/data5/yao/pretrained" --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29400 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthFusion" --depthany_model_dir "/data5/yao/pretrained" --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29400 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBeta" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-1,-1), (1,1), (1,-1), (-1,1)" --modulation_ratio 1.0 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBeta" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-3,-3), (3,3), (3,-3), (-3,3)" --modulation_ratio 1.0 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBeta" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29400 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBeta" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-3,-3), (3,3), (3,-3), (-3,3)" --modulation_alg "sigmoid" --modulation_ratio 1.0 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBeta" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-9,-9), (9,9), (9,-9), (-9,9), (-7,0), (7,0), (0,-7), (0,7), (-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29400 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBetaNoLBP" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-3,-3), (3,3), (3,-3), (-3,3)" --modulation_ratio 1.0 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBeta" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-13,0), (13,0), (0,-13), (0,13), (-11,-11), (11,11), (11,-11), (-11,11), (-9,0), (9,0), (0,-9), (0,9), (-7,-7), (7,7), (7,-7), (-7,7), (-5,0), (5,0), (0,-5), (0,5), (-3,-3), (3,3), (3,-3), (-3,3), (-1,0), (1,0), (0,-1), (0,1)" --modulation_ratio 1.0 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29400 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBetaNoLBP" --depthany_model_dir "/data5/yao/pretrained" --noLBP_hidden_dim 32 --modulation_ratio 1.0 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBeta" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 2.0 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29400 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBeta" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 3.0 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29400 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthMatch" --depthany_model_dir "/data5/yao/pretrained" --exp_name "$EXP_NAME"

torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBeta" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --exp_name "$EXP_NAME"