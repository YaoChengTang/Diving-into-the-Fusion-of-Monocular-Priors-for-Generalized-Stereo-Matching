import os
import sys
import numpy as np
sys.path.insert(0,'mast3r')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from core.extractor import ResidualBlock

try:
    from mast3r.model import AsymmetricMASt3R
except :
    print("Please install mast3r: pip install git+https://github.com/naver/mast3r.git")


# def resize_and_pad_tensor(tensor, target_size=512):
#     # 获取输入 tensor 的尺寸 (B, C, H, W)
#     _, _, H, W = tensor.shape
    
#     # 计算 H 和 W 中较长的一边
#     if H > W:
#         new_H = target_size
#         new_W = int(W * (target_size / H))
#     else:
#         new_W = target_size
#         new_H = int(H * (target_size / W))
    
#     # 使用 interpolate 进行缩放
#     resized_tensor = F.interpolate(tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
    
#     # 计算是否需要填充，使得尺寸可以被16整除
#     pad_H = (16 - new_H % 16) if new_H % 16 != 0 else 0
#     pad_W = (16 - new_W % 16) if new_W % 16 != 0 else 0
    
#     # 进行填充，确保两边可以被16整除
#     padding = (0, pad_W, 0, pad_H)  # (left, right, top, bottom)
#     padded_tensor = F.pad(resized_tensor, padding)
    
#     return padded_tensor

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


class Mast3rExtractor(nn.Module):
    def __init__(self, model_name, output_dim=128, norm_fn='batch', downsample=2):
        super(Mast3rExtractor, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
        #     self.norm1,
        #     nn.ReLU(inplace=True),
        # )

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            self.norm1,
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            self.norm1,
            nn.ReLU(inplace=True),
        )

        self.in_planes = 64
        self.layer2 = self._make_layer(128, stride=1)

        # output convolution
        self.conv = nn.Conv2d(128, output_dim, kernel_size=1)

        # self._init_weights()

        self.mast3r = AsymmetricMASt3R.from_pretrained(model_name).to('cuda')

        # 冻结 Mast3r 模型的所有参数
        for param in self.mast3r.parameters():
            param.requires_grad = False
    
    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
    #             if m.weight is not None:
    #                 nn.init.constant_(m.weight, 1)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    # def _make_layer(self, dim, stride=1):
    #     layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
    #     layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
    #     layers = (layer1, layer2)
        
    #     self.in_planes = dim
    #     return nn.Sequential(*layers)
    
    # def _make_layer(self, dim, stride=1):
    #     layer1 = ResidualBlock(self.in_planes, self.in_planes, self.norm_fn, stride=stride)
    #     layer2 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=1)
    #     layer3 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
    #     layers = (layer1, layer2, layer3)
        
    #     self.in_planes = dim
    #     return nn.Sequential(*layers)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, self.in_planes, self.norm_fn, stride=stride)
        layer1 = ResidualBlock(self.in_planes, self.in_planes, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=1)
        layer3 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layer3 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2, layer3)
        
        self.in_planes = dim
        return nn.Sequential(*layers)
    
    def forward(self, image1, image2, dual_inp=False):
        # resize image
        B, _, H, W = image1.shape
        image1 = resize_tensor(image1)
        image2 = resize_tensor(image2)

        # data format for MaSt3R
        _, _, H1, W1 = image1.shape
        view1 = dict(img=image1, 
                     true_shape=torch.tensor([[H1,W1]], dtype=torch.int32).to(image1.device), 
                     idx=B, instance=str(B))
        view2 = dict(img=image2, 
                     true_shape=torch.tensor([[H1,W1]], dtype=torch.int32).to(image1.device), 
                     idx=B, instance=str(B))

        # mast3r
        with torch.no_grad():
            pred1, pred2 = self.mast3r(view1, view2)

        # fetch features
        ## 3, 1, 24, 1
        fea1 = [pred1['pts3d'], pred1['conf'][...,None], pred1['desc'], pred1['desc_conf'][...,None]]
        fea1 = torch.cat(fea1, dim=-1).permute((0,3,1,2))
        fea1 = torch.cat([image1, fea1], dim=1)   # 32
        fea2 = [pred2['pts3d_in_other_view'], pred2['conf'][...,None], pred2['desc'], pred2['desc_conf'][...,None]]
        fea2 = torch.cat(fea2, dim=-1).permute((0,3,1,2))
        fea2 = torch.cat([image2, fea2], dim=1)   # 32
        x    = torch.cat([fea1,fea2], dim=0)      # 32

        # resize image
        x = resize_to_quarter(x, (H,W), 2**self.downsample)

        # conv
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.conv(x)

        x = x.split(split_size=B, dim=0)

        return x