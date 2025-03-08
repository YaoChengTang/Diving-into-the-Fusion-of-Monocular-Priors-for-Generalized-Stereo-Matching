import os
import sys
import numpy as np
sys.path.insert(0,'Metric3D')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from attrdict import AttrDict

from core.extractor import ResidualBlock
from depth_anything_v2.dpt import DepthAnythingV2
from core.utils.utils import sv_intermediate_results



def resize_tensor(tensor, target_size=512, ratio=16):
    # 获取输入 tensor 的尺寸 (B, C, H, W)
    _, _, H, W = tensor.shape
    
    # 计算 H 和 W 中较长的一边
    if H > W:
        new_H = target_size
        new_W = int(W * (target_size / H))
    else:
        new_W = target_size
        new_H = int(H * (target_size / W))
    
    new_W = (np.ceil(new_W / ratio) * ratio).astype(int)
    new_H = (np.ceil(new_H / ratio) * ratio).astype(int)

    # 使用 interpolate 进行缩放
    resized_tensor = F.interpolate(tensor, size=(new_H, new_W), mode='bicubic', align_corners=False)
    
    return resized_tensor


def resize_to_quarter(tensor, original_size, ratio):
    # 将尺寸缩小为原始尺寸的 1/4
    quarter_H = original_size[0] // ratio
    quarter_W = original_size[1] // ratio
    
    # 使用 interpolate 进行缩小
    resized_tensor = F.interpolate(tensor, size=(quarter_H, quarter_W), mode='bilinear', align_corners=False)
    
    return resized_tensor



from mono.utils.comm import get_func

class Metric3DExtractor(nn.Module):
    def __init__(self, args) -> None:
        super(Metric3DExtractor, self).__init__()
        self.args = args

        cfg = dict(
            model = dict(
                type='DensePredModel',
                backbone=dict(
                    type='vit_large_reg',
                    prefix='backbones.',
                    out_channels=[1024, 1024, 1024, 1024],
                    drop_path_rate = 0.0,
                    checkpoint="./pretrained/metric3d/dinov2_vitl14_reg4_pretrain.pth",
                ),
                decode_head=dict(
                    type='RAFTDepthNormalDPT5',
                    # type='RAFTDepthDPT',
                    prefix='decode_heads.',
                    in_channels=[1024, 1024, 1024, 1024],
                    use_cls_token=True,
                    feature_channels = [256, 512, 1024, 1024], # [2/7, 1/7, 1/14, 1/14]
                    decoder_channels = [128, 256, 512, 1024, 1024], # [4/7, 2/7, 1/7, 1/14, 1/14]
                    up_scale = 7,
                    hidden_channels=[128, 128, 128, 128], # [x_4, x_8, x_16, x_32] [192, 384, 768, 1536]
                    n_gru_layers=3,
                    n_downsample=2,
                    iters=8,
                    slow_fast_gru=True,
                    num_register_tokens=4,
                    # detach=False
                ),
            ),

            data_basic = dict(
                canonical_space = dict(
                    # img_size=(540, 960),
                    focal_length=1000.0,
                ),
                depth_range=(0, 1),
                depth_normalize=(0.1, 200),
                crop_size = (616, 1064),  # %28 = 0
                clip_depth_range=(0.1, 200),
                vit_size=(616,1064)
            ),
        )
        self.cfg = AttrDict(cfg)

        self.encoder = get_func('mono.model.' + self.cfg.model.backbone.prefix + self.cfg.model.backbone.type)(**self.cfg.model.backbone) 
        self.decoder = get_func('mono.model.' + self.cfg.model.decode_head.prefix + self.cfg.model.decode_head.type)(self.cfg)
        # print(get_func('mono.model.' + self.cfg.model.backbone.prefix + self.cfg.model.backbone.type))
        # print(self.encoder)

        self.hidden_dims = self.cfg.model.decode_head.hidden_channels
        self.n_gru_layers = self.cfg.model.decode_head.n_gru_layers
        self.inp_convs = nn.ModuleList([ 
                            nn.Sequential(
                                nn.Conv2d(self.hidden_dims[i]*3, self.hidden_dims[i]*3, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.hidden_dims[i]*3, self.hidden_dims[i]*3, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.hidden_dims[i]*3, self.hidden_dims[i]*3, kernel_size=3, stride=1, padding=1),
                            ) for i in range(self.n_gru_layers)
                        ])
        self.net_convs = nn.ModuleList([
                            nn.Sequential(
                                nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, padding=3//2),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, padding=3//2),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, padding=3//2),
                            ) for i in range(self.n_gru_layers)
                        ])

        load_path = "./pretrained/metric3d/metric_depth_vit_large_800k.pth"
        checkpoint = torch.load(load_path, map_location="cpu")
        state_dict = checkpoint['model_state_dict']

        encoder_state_dict = {k.replace("depth_model.encoder.", ""): v for k, v in state_dict.items() if k.startswith("depth_model.encoder")}
        decoder_state_dict = {k.replace("depth_model.decoder.", ""): v for k, v in state_dict.items() if k.startswith("depth_model.decoder")}

        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)

        self.encoder = self.encoder.to('cuda')
        self.decoder = self.decoder.to('cuda')

        # 冻结 depth_anything 模型的所有参数
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

        
        mean = [123.675, 116.28, 103.53]
        std  = [58.395,   57.12, 57.375]
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor(std).view(1, 3, 1, 1).cuda()
        self.pad_val = torch.tensor(mean).view(1, 3, 1, 1).cuda()


    def forward(self, rgb, intrinsic, baseline=1):

        with torch.no_grad():
            focal_length   = (intrinsic[:, 0] + intrinsic[:, 1]) / 2
            rgb_input, cam_model_stacks, pad, label_scale_factor, (ori_h, ori_w) = self.aug_data(rgb, intrinsic)

            # [f_32, f_16, f_8, f_4]
            features = self.encoder(rgb_input)
            output = self.decoder(features, cam_model=cam_model_stacks)

            # outputs=dict(
            #     prediction=flow_predictions[-1],
            #     predictions_list=flow_predictions,
            #     confidence=conf_predictions[-1],
            #     confidence_list=conf_predictions,
            #     pred_logit=None,
            #     # samples_pred_list=samples_pred_list,
            #     # coord_list=coord_list,
            #     prediction_normal=norma`l_outs[-1],
            #     normal_out_list=normal_outs,
            #     low_resolution_init=low_resolution_init,
            #     net_list = net_list,
            #     inp_list = inp_list,
            # )
            pred_depth, confidence = output['prediction'], output['confidence']
            net_list, inp_list = output['net_list'], output['inp_list']

            B, C, H_new, W_new = pred_depth.shape
            normalize_scale = self.cfg.data_basic.depth_range[1]
            pred_depth = pred_depth[:, :, pad[0] : H_new - pad[1], pad[2] : W_new - pad[3]]
            pred_depth = F.interpolate(pred_depth, [ori_h, ori_w], mode='bilinear') # to original size
            # print("-"*10, f"pred_depth: {pred_depth.shape}, confidence: {confidence.shape}", pred_depth.max(), pred_depth.min())
            pred_depth = pred_depth * normalize_scale / label_scale_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            # print("-"*10, pred_depth.max(), pred_depth.min(), normalize_scale, label_scale_factor, baseline, focal_length)

            pred_disp      = (baseline * focal_length).unsqueeze(1).unsqueeze(1).unsqueeze(1) / pred_depth
            pred_disp_down = F.interpolate(pred_disp, scale_factor=1/2**self.cfg.model.decode_head.n_downsample, mode='bilinear') * (1/2**self.cfg.model.decode_head.n_downsample)
            # print("*"*30, rgb.shape, rgb_input.shape, pred_depth.shape, confidence.shape, pred_disp_down.max(), pred_disp_down.min())

        
        # with autocast(enabled=self.args.mixed_precision):
        net_list = [F.interpolate(x, size=(ori_h//(2**(self.cfg.model.decode_head.n_downsample+i)), 
                                            ori_w//(2**(self.cfg.model.decode_head.n_downsample+i))), 
                                    mode='bilinear', align_corners=False) for i, x in enumerate(net_list)]
        inp_list = [F.interpolate(torch.cat(x,dim=1), 
                                    size=(ori_h//(2**(self.cfg.model.decode_head.n_downsample+i)), 
                                            ori_w//(2**(self.cfg.model.decode_head.n_downsample+i))), 
                                    mode='bilinear', align_corners=False) for i, x in enumerate(inp_list)]
        # Update the hidden states and context features
        net_list = [conv(x) for x, conv in zip(net_list, self.net_convs)]
        inp_list = [list( conv(x).chunk(3, dim=1) ) for x, conv in zip(inp_list, self.inp_convs)]

        return net_list, inp_list, pred_disp_down


    def aug_data(self, rgb, intrinsic):
        B, C, ori_h, ori_w = rgb.shape
        ori_focal = (intrinsic[:,0] + intrinsic[:,1]) / 2
        canonical_focal = self.cfg.data_basic['canonical_space']['focal_length']
        cano_label_scale_ratio = canonical_focal / ori_focal   # Shape: (B,)

        canonical_intrinsic = torch.stack([
            intrinsic[:,0] * cano_label_scale_ratio,
            intrinsic[:,1] * cano_label_scale_ratio,
            intrinsic[:,2],
            intrinsic[:,3],
        ], dim=1)

        # resize
        rgb, cam_model, pad, resize_label_scale_ratio = resize_for_input(rgb, self.cfg.data_basic.crop_size, canonical_intrinsic, [ori_h, ori_w], 1.0, self.pad_val)

        # label scale factor
        label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio     # Shape: (B,)

        rgb = torch.div(((rgb+1)/2*255 - self.mean), self.std)
        
        cam_model = cam_model.permute((0, 3, 1, 2)).float()
        cam_model = cam_model.cuda()
        cam_model_stacks = [
            torch.nn.functional.interpolate(cam_model, size=(cam_model.shape[2]//i, cam_model.shape[3]//i), mode='bilinear', align_corners=False)
            for i in [2, 4, 8, 16, 32]
        ]

        return rgb, cam_model_stacks, pad, label_scale_factor, (ori_h, ori_w)


def resize_for_input(image, output_shape, intrinsic, canonical_shape, to_canonical_ratio, pad_values):
    """
    Resize the input using PyTorch tensors.
    """
    h, w = image.shape[-2:]
    
    resize_ratio_h = output_shape[0] / canonical_shape[0]
    resize_ratio_w = output_shape[1] / canonical_shape[1]
    to_scale_ratio = min(resize_ratio_h, resize_ratio_w)
    
    resize_ratio = to_canonical_ratio * to_scale_ratio
    
    reshape_h = int(resize_ratio * h)
    reshape_w = int(resize_ratio * w)
    
    pad_h = max(output_shape[0] - reshape_h, 0)
    pad_w = max(output_shape[1] - reshape_w, 0)
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    
    # Resize image
    image = F.interpolate(image, size=(reshape_h, reshape_w), mode='bilinear', align_corners=False)
    
    # Padding
    # image = F.pad(image, (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half), value=pad_values)
    image = pad_with_channel_values(image, (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half), pad_values)
    
    # Adjust intrinsic parameters
    intrinsic[:, 2] *= to_scale_ratio  # fx
    intrinsic[:, 3] *= to_scale_ratio  # fy
    
    # Build camera model (dummy implementation, replace with actual function)
    cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
    cam_model = F.pad(cam_model, (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half), value=-1)
    
    pad = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    label_scale_factor = 1 / to_scale_ratio
    
    return image, cam_model, pad, label_scale_factor

def pad_with_channel_values(input_tensor, padding, pad_values):
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    else:
        pad_left, pad_right, pad_top, pad_bottom = padding

    B, C, H, W = input_tensor.shape
    new_H = H + pad_top + pad_bottom
    new_W = W + pad_left + pad_right

    pad_values = pad_values.view(1, C, 1, 1)

    padded_tensor = pad_values.expand(B, C, new_H, new_W).clone()

    # 计算中间区域并复制数据
    h_start, h_end = pad_top, new_H - pad_bottom
    w_start, w_end = pad_left, new_W - pad_right
    padded_tensor[:, :, h_start:h_end, w_start:w_end] = input_tensor

    return padded_tensor


def build_camera_model(H: int, W: int, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Encode the camera intrinsic parameters (focal length and principle point) to a 4-channel map.
    Args:
        H (int): Image height
        W (int): Image width
        intrinsics (torch.Tensor): Tensor of shape (B, 4) containing fx, fy, u0, v0
    Returns:
        torch.Tensor: Camera model tensor of shape (B, H, W, 4)
    """
    B = intrinsics.shape[0]
    fx, fy, u0, v0 = intrinsics[:, 0:1], intrinsics[:, 1:2], intrinsics[:, 2:3], intrinsics[:, 3:4]
    f = (fx + fy) / 2.0   # Shape: (B,1)
    
    # Generate normalized coordinate grids
    x_row = torch.arange(W, dtype=torch.float32, device=intrinsics.device).view(1, W)
    y_col = torch.arange(H, dtype=torch.float32, device=intrinsics.device).view(1, H)
    
    # Normalize based on principal point
    x_center = (x_row - u0) / W  # Shape: (B, W)
    y_center = (y_col - v0) / H  # Shape: (B, H)
    
    # Expand dimensions for batch processing
    x_center = x_center.unsqueeze(1).expand(B, H, W)  # Shape: (B, H, W)
    y_center = y_center.unsqueeze(2).expand(B, H, W)  # Shape: (B, H, W)
    
    # Compute FoV angles
    fov_x = torch.atan(x_center / (f.unsqueeze(1) / W))  # Shape: (B, H, W)
    fov_y = torch.atan(y_center / (f.unsqueeze(1) / H))  # Shape: (B, H, W)
    
    # Stack channels
    cam_model = torch.stack([x_center, y_center, fov_x, fov_y], dim=-1)  # Shape: (B, H, W, 4)
    
    return cam_model

