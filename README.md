# Generalized Stereo Matching with Fusion of Monocular Priors

> ⚠️ **Warning**: It is highly recommended to view this markdown in a preview format！
> ⚠️ **Warning**: We strongly recommend researchers retrain the model on GPUs other than A40 for better results.


## Requirements
```Shell
conda env create -f envs/environment_GStereo.yaml
conda activate raftstereo
```


## Required Data
```Shell
├── datasets
    ├── sceneflow
        ├── driving                                               
        │   ├── disparity                                         
        │   ├── frames_cleanpass                                  
        │   └── frames_finalpass                                  
        ├── flying3d                                              
        │   ├── disparity                                         
        │   ├── frames_cleanpass                                  
        │   └── frames_finalpass                                  
        └── monkaa                                                
            ├── disparity                                         
            ├── frames_cleanpass                                                                                             
            └── frames_finalpass
    ├── Kitti15
        ├── testing
        │   ├── image_2
        │   └── image_3
        └── training
            ├── disp_noc_0
            ├── disp_noc_1
            ├── disp_occ_0
            ├── disp_occ_1
            ├── flow_noc
            ├── flow_occ
            ├── image_2
            ├── image_3
            └── obj_map
    ├── Kitti12
        ├── testing
        │   ├── calib
        │   ├── colored_0
        │   ├── colored_1
        │   ├── disp_noc
        │   ├── disp_occ
        │   ├── flow_noc
        │   ├── flow_occ
        │   ├── image_0
        │   └── image_1
        └── training
            ├── calib
            ├── colored_0
            └── colored_1
    ├── Middlebury
        └── MiddEval3  
            ├── testF
            ├── testH
            ├── testQ    
            ├── trainingF                               
            ├── trainingH                                         
            └── trainingQ
    ├── ETH3D
        ├── two_view_testing
        └── two_view_training
            ├── delivery_area_1l
            ├── delivery_area_1s
            ├── delivery_area_2l
    ├── Booster
        ├── test
        │   ├── balanced
        │   └── unbalanced
        └── train
            ├── balanced
            └── unbalanced
```



## Code
All codes are provided here, including DepthAnything v2.
Since we modified `dpt.py` to get intermediate features and depth output, please use the modified code.


- ### Training  
    All training script is presented in [script/train_stereo_raftstereo.sh](script/train_stereo_raftstereo.sh) and [script/train_stereo_raftstereo_depthany.sh](script/train_stereo_raftstereo_depthany.sh).
    Please specify the following variable in scripts before training.
    | variable      | meaning                 |
    |---------------|----------------------|
    | `NCCL_P2P_DISABLE`      | We set `NCCL_P2P_DISABLE=1` as the distributed training went wrong at our `A40` GPU.       |
    | `CUDA_VISIBLE_DEVICES`  | avaliable GPU id, e.g., `CUDA_VISIBLE_DEVICES=0,1,2,3`       |
    | `DATASET_ROOT`  | the training dataset path, e.g., `./datasets/sceneflow`        |
    | `LOG_ROOT`      | path to save log file     |
    | `TB_ROOT`       | path to save tensorboard data        |
    | `CKPOINT_ROOT`  | path to save checkpoint       |
    
    
    In order to reproduce our results, please download `depth_anything_v2_vitl.pth` from DepthAnything v2 before training and specify `--depthany_model_dir` in script shell to path of directory where `depth_anything_v2_vitl.pth` is saved. Here, we do not provide the link as it maybe conflicts to the CVPR guideline.
    We also explain the code for ablation study, in which each experiment is mostly controlled by the `--model_name` used in the training shell.
    | `--model_name`      | meaning                 |
    |-----------------|-------------------------|
    | `RaftStereo`          | Original RaftStereo model       |
    | `RaftStereoDisp`      | The output of GRU is a single channel for disparity instead of two channels for optical flow, `Baseline` in Table 3 of the main text.      |
    | `RAFTStereoMast3r`    | The pre-trained MASt3R is used as the backbone and its features are used for cost volume construction, `RaftStereo + backbone Mast3r` in supplemental text.       |
    | `RaftStereoNoCTX`     | RaftStereo model without context network, `Baseline w/o mono feature` in Table 3 of the main text.   |
    | `RAFTStereoDepthAny`  | RaftStereo model with our monocular encoder, `Baseline + ME` in Table 3 of the main text.       |
    | `RAFTStereoDepthFusion`  | RaftStereo model with our monocular encoder, `Baseline + ME + IDF` in Table 3 of the main text.       |
    | `RAFTStereoDepthBeta`  | RaftStereo model with our monocular encoder and iterative local fusion, `Baseline + ME + ILF` in Table 3 of the main text.       |
    | `RAFTStereoDepthBetaNoLBP`  | RaftStereo model with our monocular encoder and iterative local fusion without LBPEncoder, `L(6)` and `L(7)` in Table 4 of the main text.       |
    | `RAFTStereoDepthMatch`  | RaftStereo model with DepthAnything v2 as feature extractor for cost volume construction, `RaftStereo + backbone DepthAnything` in the supplemental text.       |
    | `RAFTStereoDepthPostFusion`  | RaftStereo model with our monocular encoder, iterative local fusion and post fusion, `Baseline + ME + PF` in Table 3 of the main text.       |
    | `RAFTStereoDepthBetaRefine`  | RaftStereo model with our monocular encoder, iterative local fusion, and global fusion, `Baseline + ME + ILF + GF` in Table 3 of the main text.       |


    |         variable         | meaning                 |
    |--------------------------|-------------------------|
    | `--lbp_neighbor_offsets` | control `LBP Kernel` used in Table 4 of the main text.   |
    | `--modulation_ratio`     | control `r` amplitude parameter used in Table 4 of the main text. |
    | `--conf_from_fea`        | `Cost` or `Hybrid` for `COnfidence` used in Table 4 of the main text. |
    | `--refine_pool`          | learning regitration paramters via pooling in the supplemental text. |


    The training is launched by follows
    ```Shell
    bash ./script/train_stereo_raftstereo_depthany.sh EXP_NAME
    ```
    `EXP_NAME` specifiy the experiment name, we use this name to save each log file, tensorboard data, and checkpoint for different experiments. The corresponding file structure is like follows
    ```Shell
    ├── runs
        ├── ckpoint
        │   ├── RaftStereoDepthAny
        │   ├── RaftStereoMast3r
        │   └── RaftStereoNoCTX
        ├── log
        │   ├── RaftStereoDepthAny
        │   ├── RaftStereoMast3r
        │   └── RaftStereoNoCTX
        └── tboard
            ├── RaftStereoDepthAny
            ├── RaftStereoMast3r
            └── RaftStereoNoCTX
    ```
    > ⚠️ **Warning**: **Please follow the training process mentioned in our main text.** We first train the model without the global fusion module. Then, we train the monocular registration of the global fusion module while keeping the other modules frozen with well-trained model from first stage. Finally, we train the entire global fusion module while keeping the other modules frozen with well-trained model from second stage.

- ### Evaluation  
    Evaluation script is presented in [script/evaluate_stereo_raftstereo.sh](script/evaluate_stereo_raftstereo.sh).
    We use `--test_exp_name` to specifiy the evaluation experiment name.
    The results of each experiment are restored in `LOG_ROOT/eval.xlsx`. We also merge all experiments' result in `LOG_ROOT/merged_eval.xlsx` through `python3 merge_sheet.py`.
    The evaluation metrics remain the same for different methods.
    The `mean ± std` is computed via [tools/get_statistics.py](tools/get_statistics.py).

- ### Visualization  
    We visualize the error map via [script/gen_sample_stereo_raftstereo.sh](script/gen_sample_stereo_raftstereo.sh) and intermediate results via [script/vis_inter_stereo_raftstereo.sh](script/vis_inter_stereo_raftstereo.sh).
    We provide an easy-to-use visualization toolbox to fully understand each module.
