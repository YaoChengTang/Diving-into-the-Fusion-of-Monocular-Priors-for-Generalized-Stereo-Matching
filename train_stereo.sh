# /usr/bin/bash

# "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Sceneflow"
# "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Middlebury"
# "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/KITTI2015"
# "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/ETH3D"
export DATASET_ROOT="/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Sceneflow"

CUDA_VISIBLE_DEVICE=1,2 python3 train_stereo.py --batch_size 4 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 200000 --mixed_precision
