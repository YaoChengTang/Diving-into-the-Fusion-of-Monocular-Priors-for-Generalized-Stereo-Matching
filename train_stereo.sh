# /usr/bin/bash

# "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Sceneflow"
# "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Middlebury"
# "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/KITTI2015"
# "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/ETH3D"
export DATASET_ROOT="/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Sceneflow"

# CUDA_VISIBLE_DEVICE=1,2 python3 train_stereo.py --batch_size 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 200000 --mixed_precision

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant"

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant" --slant_norm

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant" --slant_norm --loss_smooth "curvature" --ner_kernel_size "diamond_edge-5"

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant" --slant_norm --loss_smooth "curvature" --ner_kernel_size "diamond_edge-5" --ner_weight_reduce

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant" --slant_norm --loss_smooth "curvature" --ner_kernel_size "diamond_edge-11" --ner_weight_reduce

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant_local"


# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --confidence --offset_memory_size 3 --detach_in_confidence

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --confidence --offset_memory_size 3 --detach_in_confidence --conf_disp

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --confidence --offset_memory_size 3 --detach_in_confidence --conf_disp --slant "slant" --slant_norm

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --confidence --offset_memory_size 3 --detach_in_confidence --conf_disp --refinement "Refinement" --refine_win_size 4 --refine_start_itr 3

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --confidence --offset_memory_size 3 --detach_in_confidence --conf_disp --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 22 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --confidence --offset_memory_size 20 --detach_in_confidence --conf_disp --refinement "Refinement" --refine_win_size 10 --refine_start_itr 20

# CUDA_VISIBLE_DEVICES=1,3 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 22 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --confidence --offset_memory_size 20 --detach_in_confidence --conf_disp --refinement "Refinement" --refine_win_size 10 --refine_start_itr 20 --update_his

# CUDA_VISIBLE_DEVICES=1,3 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 22 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --confidence --offset_memory_size 20 --detach_in_confidence --conf_disp --refinement "Refinement" --refine_win_size 10 --refine_start_itr 20 --update_his --U_thold 0.5

CUDA_VISIBLE_DEVICES=1,3 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 22 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant" --slant_norm
