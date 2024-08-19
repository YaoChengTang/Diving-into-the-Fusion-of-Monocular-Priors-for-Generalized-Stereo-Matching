# /usr/bin/bash

# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/Sceneflow"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/Middlebury"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/KITTI2015"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/ETH3D"
export DATASET_ROOT="/data6/sceneflow/sceneflow"

export LOG_ROOT="/data5/yao/log"
export TB_ROOT="/data5/yao/tboard"
export CKPOINT_ROOT="/data5/yao/ckpoint"

CUDA_VISIBLE_DEVICES=5,6 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 200000 --mixed_precision