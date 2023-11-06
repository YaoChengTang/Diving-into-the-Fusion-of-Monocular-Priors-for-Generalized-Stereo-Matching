# /usr/bin/bash

# CUDA_VISIBLE_DEVICE=1 python3 evaluate_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=1 python3 evaluate_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset kitti
# CUDA_VISIBLE_DEVICE=1 python3 evaluate_stereo.py --restore_ckpt pretrained/raftstereo-sceneflow.pth --sv_root /home/users/chengtang.yao/Output/raftstereo --dataset eth3d

# CUDA_VISIBLE_DEVICE=1 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-raw.pth --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=1 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-raw.pth --dataset kitti
# CUDA_VISIBLE_DEVICE=1 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-raw.pth --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-largeBatch.pth --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-largeBatch.pth --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-largeBatch.pth --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raftstereo-NS.tar --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raftstereo-NS.tar --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raftstereo-NS.tar --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slant.pth --slant "slant" --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slant.pth --slant "slant" --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slant.pth --slant "slant" --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange.pth --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange.pth --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange.pth --slant "slant" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange2.pth --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange2.pth --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange2.pth --slant "slant" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange-smooth.pth --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange-smooth.pth --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-slantNormRange-smooth.pth --slant "slant" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNormRange3.pth --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNormRange3.pth --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNormRange3.pth --slant "slant" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNormRange-SmoothDiamond_edge-5.pth --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNormRange-SmoothDiamond_edge-5.pth --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNormRange-SmoothDiamond_edge-5.pth --slant "slant" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-LocalSlant.pth --slant "slant_local" --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-LocalSlant.pth --slant "slant_local" --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-LocalSlant.pth --slant "slant_local" --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantLocal-moreiter.pth --slant "slant_local" --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantLocal-moreiter.pth --slant "slant_local" --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantLocal-moreiter.pth --slant "slant_local" --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-OffsetConfidence.pth --confidence --offset_memory_size 3 --detach_in_confidence --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-OffsetConfidence.pth --confidence --offset_memory_size 3 --detach_in_confidence --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-OffsetConfidence.pth --confidence --offset_memory_size 3 --detach_in_confidence --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-OffsetConfidence_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-OffsetConfidence_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-OffsetConfidence_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNorm-Conf_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNorm-Conf_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNorm-Conf_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --slant "slant" --slant_norm --dataset eth3d


# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine_Conf_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 4 --refine_start_itr 3 --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine_Conf_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 4 --refine_start_itr 3 --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine_Conf_loss.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 4 --refine_start_itr 3 --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNorm-Conf_loss-30k.pth --confidence --offset_memory_size 3 --detach_in_confidence --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNorm-Conf_loss-30k.pth --confidence --offset_memory_size 3 --detach_in_confidence --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-SlantNorm-Conf_loss-30k.pth --confidence --offset_memory_size 3 --detach_in_confidence --slant "slant" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine_Conf_loss-W20-B2-20231014-171009.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 20 --refine_start_itr 3 --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine_Conf_loss-W20-B2-20231014-171009.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 20 --refine_start_itr 3 --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine_Conf_loss-W20-B2-20231014-171009.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 20 --refine_start_itr 3 --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine_Conf_loss-W10-B3-20231014-171252.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine_Conf_loss-W10-B3-20231014-171252.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine_Conf_loss-W10-B3-20231014-171252.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/70000_raft-stereo-Refine_Conf_loss-W20-B2-20231014-171009.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 20 --refine_start_itr 3 --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/70000_raft-stereo-Refine_Conf_loss-W20-B2-20231014-171009.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 20 --refine_start_itr 3 --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/70000_raft-stereo-Refine_Conf_loss-W20-B2-20231014-171009.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 20 --refine_start_itr 3 --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine2_Conf_loss-W10-B3-20231016-132008.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine2_Conf_loss-W10-B3-20231016-132008.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine2_Conf_loss-W10-B3-20231016-132008.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine2_detach_mask_Conf_loss-W10-B3-20231016-141357.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine2_detach_mask_Conf_loss-W10-B3-20231016-141357.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine2_detach_mask_Conf_loss-W10-B3-20231016-141357.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine2_last_detach_mask_Conf_loss-W10-B3-20231019-090015.pth --confidence --offset_memory_size 20 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 20 --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine2_last_detach_mask_Conf_loss-W10-B3-20231019-090015.pth --confidence --offset_memory_size 20 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 20 --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine2_last_detach_mask_Conf_loss-W10-B3-20231019-090015.pth --confidence --offset_memory_size 20 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 20 --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine2_detach_mask_Conf_loss_updatehis-W10-B3-20231021-235312.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine2_detach_mask_Conf_loss_updatehis-W10-B3-20231021-235312.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine2_detach_mask_Conf_loss_updatehis-W10-B3-20231021-235312.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine3_mask_Conf_loss_repair_updatehis-W10-B3-20231022-001620.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine3_mask_Conf_loss_repair_updatehis-W10-B3-20231022-001620.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine3_mask_Conf_loss_repair_updatehis-W10-B3-20231022-001620.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine4_focal-W10-B3-20231024-170201.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine4_focal-W10-B3-20231024-170201.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine4_focal-W10-B3-20231024-170201.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine4_focal_conf-20231026-154434.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --U_thold 0.5 --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine4_focal_conf-20231026-154434.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --U_thold 0.5 --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-Refine4_focal_conf-20231026-154434.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --U_thold 0.5 --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-paramsLoss_gt-20231026-205112.pth --slant "slant" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-paramsLoss_gt-20231026-205112.pth --slant "slant" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-paramsLoss_gt-20231026-205112.pth --slant "slant" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-paramsLoss_gt_slantLocal-20231026-220752.pth --slant "slant_local" --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-paramsLoss_gt_slantLocal-20231026-220752.pth --slant "slant_local" --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-paramsLoss_gt_slantLocal-20231026-220752.pth --slant "slant_local" --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-paramsLoss_gt_slantLocal_noNorm-20231026-221215.pth --slant "slant_local" --slant_norm --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-paramsLoss_gt_slantLocal_noNorm-20231026-221215.pth --slant "slant_local" --slant_norm --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-paramsLoss_gt_slantLocal_noNorm-20231026-221215.pth --slant "slant_local" --slant_norm --dataset eth3d

CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-repair_slant_norm-20231101-083700.pth --slant "slant" --slant_norm --dataset middlebury_H
CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-repair_slant_norm-20231101-083700.pth --slant "slant" --slant_norm --dataset kitti
CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-repair_slant_norm-20231101-083700.pth --slant "slant" --slant_norm --dataset eth3d

CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-repair_slantLocal-20231101-083140.pth --slant "slant_local" --dataset middlebury_H
CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-repair_slantLocal-20231101-083140.pth --slant "slant_local" --dataset kitti
CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-repair_slantLocal-20231101-083140.pth --slant "slant_local" --dataset eth3d

CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-repair_slantLocal_norm-20231101-083429.pth --slant "slant_local" --slant_norm --dataset middlebury_H
CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-repair_slantLocal_norm-20231101-083429.pth --slant "slant_local" --slant_norm --dataset kitti
CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-repair_slantLocal_norm-20231101-083429.pth --slant "slant_local" --slant_norm --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-repair_conf_10iter-20231101-095614.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --U_thold 0.5 --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-repair_conf_10iter-20231101-095614.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --U_thold 0.5 --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-repair_conf_10iter-20231101-095614.pth --confidence --offset_memory_size 3 --detach_in_confidence --refinement "Refinement" --refine_win_size 10 --refine_start_itr 3 --update_his --U_thold 0.5 --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-PostGeo_slantl_norm_max-20231104-132129.pth --slant "slant" --slant_norm --slant_builder "geometry" --geo_fusion "max" --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-PostGeo_slantl_norm_max-20231104-132129.pth --slant "slant" --slant_norm --slant_builder "geometry" --geo_fusion "max" --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-PostGeo_slantl_norm_max-20231104-132129.pth --slant "slant" --slant_norm --slant_builder "geometry" --geo_fusion "max" --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-PostGeo_slantlocal_max-20231104-131911.pth --slant "slant_local" --slant_builder "geometry" --geo_fusion "max" --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-PostGeo_slantlocal_max-20231104-131911.pth --slant "slant_local" --slant_builder "geometry" --geo_fusion "max" --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-PostGeo_slantlocal_max-20231104-131911.pth --slant "slant_local" --slant_builder "geometry" --geo_fusion "max" --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-PostGeo_slantlocal_norm_max-20231104-131648.pth --slant "slant_local" --slant_norm --slant_builder "geometry" --geo_fusion "max" --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-PostGeo_slantlocal_norm_max-20231104-131648.pth --slant "slant_local" --slant_norm --slant_builder "geometry" --geo_fusion "max" --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-PostGeo_slantlocal_norm_max-20231104-131648.pth --slant "slant_local" --slant_norm --slant_builder "geometry" --geo_fusion "max" --dataset eth3d

# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-PostGeo_slantlocal_norm-20231104-131340.pth --slant "slant_local" --slant_norm --slant_builder "geometry" --geo_fusion "mean" --dataset middlebury_H
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-PostGeo_slantlocal_norm-20231104-131340.pth --slant "slant_local" --slant_norm --slant_builder "geometry" --geo_fusion "mean" --dataset kitti
# CUDA_VISIBLE_DEVICE=2 python3 evaluate_stereo.py --restore_ckpt pretrained/raft-stereo-PostGeo_slantlocal_norm-20231104-131340.pth --slant "slant_local" --slant_norm --slant_builder "geometry" --geo_fusion "mean" --dataset eth3d
