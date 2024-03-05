# /usr/bin/bash

# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/Sceneflow"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/Middlebury"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/KITTI2015"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/ETH3D"
export DATASET_ROOT="/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/Sceneflow"


CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 22 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant_local" --slant_norm --geo_estimator "geometry_conv"


# CUDA_VISIBLE_DEVICES=1,3 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 1 --train_iters 22 --valid_iters 22 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --confidence --offset_memory_size 3 --offset_memory_last_iter 10 --detach_in_confidence --conf_disp --refinement "Refinement" --refine_win_size 4 4 --refine_start_itr 3 --update_his --U_thold 0.5 --num_heads 6 --enable_refine_step 0


# CUDA_VISIBLE_DEVICES=6,7 python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 2 train_stereo.py --batch_size 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 50000 --mixed_precision