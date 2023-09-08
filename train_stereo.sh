# /usr/bin/bash

# "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Sceneflow"
# "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Middlebury"
# "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/KITTI2015"
# "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/ETH3D"
export DATASET_ROOT="/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Sceneflow"

# CUDA_VISIBLE_DEVICE=1,2 python3 train_stereo.py --batch_size 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 200000 --mixed_precision

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant --slant_norm

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant --slant_norm --loss_smooth
