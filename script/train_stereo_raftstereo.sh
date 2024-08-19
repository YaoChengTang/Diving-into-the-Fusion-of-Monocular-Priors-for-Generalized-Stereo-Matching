# /usr/bin/bash

# export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0  # 设置正确的网络接口
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=29500
export CUDA_VISIBLE_DEVICES=4,5  # 使用两张GPU

# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/Sceneflow"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/Middlebury"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/KITTI2015"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/ETH3D"
export DATASET_ROOT="/data6/sceneflow/sceneflow"

export LOG_ROOT="/data5/yao/log"
export TB_ROOT="/data5/yao/tboard"
export CKPOINT_ROOT="/data5/yao/ckpoint"


torchrun --nnode 1 --nproc_per_node 2 train_stereo_raftstereo.py --batch_size 2 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 200000 --mixed_precision