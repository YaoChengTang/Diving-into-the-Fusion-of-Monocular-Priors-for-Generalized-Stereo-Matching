#!/bin/bash

export DATASET_ROOT="./datasets/Fooling3D"

export LOG_ROOT="./clouds/runs/log/${FOLDER_NAME}"
export TB_ROOT="./clouds/runs/tboard/${FOLDER_NAME}"
export CKPOINT_ROOT="./clouds/runs/ckpoint/${FOLDER_NAME}"

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


now=$(date +"%Y%m%d_%H%M%S")

epoch=20
bs=4
lr=0.000005
encoder=vitl
dataset=Fooling3D # hypersim # vkitti
img_size=518
min_depth=0.001
max_depth=20 # 80 for virtual kitti

# pretrained_from=./pretrained/depth_anything_v2_${encoder}.pth
# save_path=./clouds/DepthAnything/fooling3d # exp/hypersim # exp/vkitti

pretrained_from=./pretrained/depth_anything_v2_${encoder}.pth
save_path=./clouds/DepthAnything/fooling3d_maskDepth # exp/hypersim # exp/vkitti



mkdir -p $save_path

# python3 -m torch.distributed.launch \
#     --nproc_per_node=$nproc_per_node \
#     --nnodes 1 \
#     --node_rank=0 \
#     --master_addr=localhost \
#     --master_port=20596 \
#     train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
#     --spatial_scale -0.2 0.4 --saturation_range 0 1.4 \
#     --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
#     --port 20596 2>&1 | tee -a $save_path/$now.log

python3 -m torch.distributed.launch \
    --nproc_per_node=$nproc_per_node \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --depth_as_mask \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
    --port 20596 2>&1 | tee -a $save_path/$now.log
