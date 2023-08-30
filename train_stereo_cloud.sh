# /usr/bin/bash

# "/bucket/output/BasicAlgorithm/Users/chengtang.yao/Sceneflow"
# "/bucket/output/BasicAlgorithm/Users/chengtang.yao/Middlebury"
# "/bucket/output/BasicAlgorithm/Users/chengtang.yao/KITTI2015"
# "/bucket/output/BasicAlgorithm/Users/chengtang.yao/ETH3D"
export DATASET_ROOT="/bucket/output/BasicAlgorithm/Users/chengtang.yao/Sceneflow"
export LOG_ROOT="/job_log"
export TB_ROOT="/job_tboard"
export CKPOINT_ROOT="/job_data"

cd $WORKING_PATH

python3 train_stereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 200000 --mixed_precision

sleep 12h