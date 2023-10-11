# /usr/bin/bash

# "/bucket/output/BasicAlgorithm/Users/chengtang.yao/Sceneflow"
# "/bucket/output/BasicAlgorithm/Users/chengtang.yao/Middlebury"
# "/bucket/output/BasicAlgorithm/Users/chengtang.yao/KITTI2015"
# "/bucket/output/BasicAlgorithm/Users/chengtang.yao/ETH3D"
export DATASET_ROOT="/bucket/output/BasicAlgorithm/Users/chengtang.yao/Sceneflow"
export LOG_ROOT="/job_log"
export TB_ROOT="/job_tboard"
export CKPOINT_ROOT="/job_data"
export SLURM_CPUS_PER_TASK=34

cd $WORKING_PATH

# python3 train_stereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 200000 --mixed_precision

# python3 train_stereo.py --batch_size 24 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant"

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant" --slant_norm

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant" --slant_norm --loss_smooth "curvature"

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant" --slant_norm --loss_smooth "curvature" --ner_kernel_size 3

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant" --slant_norm --loss_smooth "curvature" --ner_kernel_size "diamond_edge-5"

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant" --slant_norm --loss_smooth "curvature" --ner_kernel_size "diamond_edge-5" --ner_weight_reduce

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant" --slant_norm --loss_smooth "curvature" --ner_kernel_size "diamond_edge-11" --ner_weight_reduce

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --slant "slant_local"

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --confidence --offset_memory_size 3 --detach_in_confidence

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --confidence --offset_memory_size 3 --detach_in_confidence --conf_disp


# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --confidence --offset_memory_size 3 --detach_in_confidence --conf_disp --slant "slant" --slant_norm

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --confidence --offset_memory_size 3 --detach_in_confidence --conf_disp --slant "slant" --slant_norm

# python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 300000 --mixed_precision --confidence --offset_memory_size 3 --detach_in_confidence --conf_disp --slant "slant" --slant_norm

python3 -m torch.distributed.launch --master_port=20000 --nproc_per_node 8 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --confidence --offset_memory_size 3 --detach_in_confidence --conf_disp --refinement "Refinement" --refine_win_size 4 --refine_start_itr 3