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
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,6
# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1

# export DATASET_ROOT="/data6/sceneflow/sceneflow"
# export DATASET_ROOT="./datasets/Trans"
export DATASET_ROOT="./datasets/FSD"

export LOG_ROOT="./runs/log/${FOLDER_NAME}"
export TB_ROOT="./runs/tboard/${FOLDER_NAME}"
export CKPOINT_ROOT="./runs/ckpoint/${FOLDER_NAME}"

export PRETRAINED_ROOT="./pretrained"

# 输出新的路径，确认设置正确
echo "LOG_ROOT is set to: $LOG_ROOT"
echo "TB_ROOT is set to: $TB_ROOT"
echo "CKPOINT_ROOT is set to: $CKPOINT_ROOT"

echo "DATASET_ROOT is set to: $DATASET_ROOT"
echo "PRETRAINED_ROOT is set to: $PRETRAINED_ROOT"


nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)  # Count the number of GPUs


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

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBeta" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --exp_name "$EXP_NAME"\

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthBetaK53_20240920_165346/80000_RaftStereoDepthBetaK53.pth" --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 16 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthBetaK53_20240920_165346/80000_RaftStereoDepthBetaK53.pth" --lr 0.0005 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 32 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispMonoDepthBatch16_20241021_222325/RaftStereoDepthBetaK53DispMonoDepthBatch16.pth" --lr 0.0005 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 32 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthPostFusion" --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthAny_20240908_125231/RaftStereoDepthAny.pth" --lr 0.0005 --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 48 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispMonoDepthBatch16_20241021_222325/RaftStereoDepthBetaK53DispMonoDepthBatch16.pth" --lr 0.0005 --exp_name "$EXP_NAME"



# # mono pooling
# torchrun --nnode 1 --nproc_per_node 4 --master_port 29501 train_stereo_raftstereo.py --batch_size 48 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --refine_pool --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth" --lr 0.0005 --train_refine_mono --exp_name "$EXP_NAME"

# # w/o mono pooling
# torchrun --nnode 1 --nproc_per_node 4 --master_port 29401 train_stereo_raftstereo.py --batch_size 48 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth" --lr 0.0005 --train_refine_mono --exp_name "$EXP_NAME"

# mono EfficientUnet
# torchrun --nnode 1 --nproc_per_node 4 --master_port 29301 train_stereo_raftstereo.py --batch_size 32 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --refine_unet --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthBetaK53_20240920_165346/80000_RaftStereoDepthBetaK53.pth" --lr 0.0005 --train_refine_mono --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29301 train_stereo_raftstereo.py --batch_size 32 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --refine_unet --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32Unet_20241112_132207/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32Unet.pth" --lr 0.0005 --train_refine_mono --exp_name "$EXP_NAME"

# torchrun --nnode 1 --nproc_per_node 4 --master_port 29301 train_stereo_raftstereo.py --batch_size 48 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --refine_unet --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32Unet_20241114_125423/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch32Unet.pth" --lr 0.0003 --train_refine_mono --exp_name "$EXP_NAME"



# torchrun --nnode 1 --nproc_per_node 2 --master_port 29301 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoMetric3D" --lr 0.0002 --exp_name "RAFTStereoMetric3D"

# torchrun --nnode 1 --nproc_per_node 8 --master_port 29301 train_stereo_raftstereo.py --batch_size 6 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoMetric3D" --lr 0.0001 --restore_ckpt "/data5/yao/runs/ckpoint/RAFTStereoMetric3D_20250305_043320/20000_RAFTStereoMetric3D.pth" --exp_name "RAFTStereoMetric3D_ConfLoss_Lr0001"





# On Trans dataset
# torchrun --nnode 1 --nproc_per_node $nproc_per_node --master_port 29501 train_stereo_raftstereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 10000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth" --lr 0.0005 --fintune_info "tune_raft" --train_datasets "Trans" --exp_name "Trans_RAFTStereoDepthBetaRefine"

# torchrun --nnode 1 --nproc_per_node $nproc_per_node --master_port 29501 train_stereo_raftstereo.py --batch_size 32 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 10000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --restore_ckpt "/home/yao/Document/GeneralizedStereoMatching/clouds/ckpoint/Trans_RAFTStereoDepthBetaRefine_20250321_180351/Trans_RAFTStereoDepthBetaRefine.pth" --lr 0.0005 --fintune_info "tune_refine" --train_datasets "Trans" --exp_name "Trans_RAFTStereoDepthBetaRefine_tuneRefine"


export CUDA_LAUNCH_BLOCKING=1

# On FSD dataset
torchrun --nnode 1 --nproc_per_node $nproc_per_node --master_port 29501 \
train_stereo_raftstereo.py \
    --model_name "RAFTStereoDepthBetaRefine" \
    --train_iters 22 --valid_iters 32 --n_downsample 2 \
    --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" \
    --modulation_ratio 1.0 --conf_from_fea \
    --depthany_model_dir "$PRETRAINED_ROOT" \
    --restore_ckpt "$PRETRAINED_ROOT/MGStereo.pth" \
    --spatial_scale -0.2 0.4 --saturation_range 0 1.4 \
    --lr 0.0005 --batch_size 8 --num_steps 100000 --mixed_precision \
    --fintune_info "tune_raft" --train_datasets "FSD" \
    --exp_name "$EXP_NAME"

