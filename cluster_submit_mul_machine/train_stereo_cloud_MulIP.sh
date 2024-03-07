# /usr/bin/bash

# "/bucket/output/BasicAlgorithm/Users/chengtang.yao/Sceneflow"
# "/bucket/output/BasicAlgorithm/Users/chengtang.yao/Middlebury"
# "/bucket/output/BasicAlgorithm/Users/chengtang.yao/KITTI2015"
# "/bucket/output/BasicAlgorithm/Users/chengtang.yao/ETH3D"
# export DATASET_ROOT="/bucket/output/BasicAlgorithm/Users/chengtang.yao/Sceneflow"
export DATASET_ROOT="/bucket/output/saturn_v_dev/01_users/chengtang.yao/Sceneflow"
export LOG_ROOT="/job_log"
export TB_ROOT="/job_tboard"
export CKPOINT_ROOT="/job_data"
export SLURM_CPUS_PER_TASK=34

cd $WORKING_PATH

python3 -m torch.distributed.launch --nproc_per_node 4 train_stereo.py --batch_size 4 --num_workers 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 70000 --mixed_precision --slant "slant_local" --slant_norm --geo_estimator "geometry_conv"

