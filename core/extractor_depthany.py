import os
import sys
import numpy as np
sys.path.insert(0,'Depth-Anything-V2')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from core.extractor import ResidualBlock
from depth_anything_v2.dpt import DepthAnythingV2
from core.utils.utils import sv_intermediate_results, resize_tensor, resize_to_quarter



class DepthAnyExtractor(nn.Module):
    def __init__(self, model_dir, output_dim=[128], norm_fn='batch', downsample=2, args=None):
        super(DepthAnyExtractor, self).__init__()
        self.args = args
        self.norm_fn = norm_fn
        self.downsample = downsample

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.in_planes = 128
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # self._init_weights()

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        encoder = "vitl"
        depth_anything = DepthAnythingV2(**model_configs[encoder])
        depth_anything.load_state_dict(torch.load(os.path.join(model_dir, f'depth_anything_v2_{encoder}.pth'), 
                                                  map_location='cpu'))
        self.depth_anything = depth_anything.to('cuda')

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor(std).view(1, 3, 1, 1).cuda()

        # 冻结 depth_anything 模型的所有参数
        for param in self.depth_anything.parameters():
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

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)
    
    def forward(self, image, dual_inp=False, num_layers=3):
        # resize image
        B, _, H, W = image.shape
        img = resize_tensor(image, target_size=518, ratio=14)
        
        # normalization
        img = ((img+1)/2 - self.mean) / self.std

        # DepthAnything
        with torch.no_grad():
            # out_depth: [1, 1, 518, 756]
            # out_fea: [1, 128, 296, 432]
            depth, depth_fea = self.depth_anything(img)

        # resize image
        # [1, 128, H//4, W//4]
        depth = resize_to_quarter(depth, (H,W), 2**self.downsample)
        x = resize_to_quarter(depth_fea, (H,W), 2**self.downsample)

        if self.args is not None and hasattr(self.args, "vis_inter") and self.args.vis_inter:
            sv_intermediate_results(x, "depthAnything_features", self.args.sv_root)

        x = self.layer1(x)
        outputs08 = [f(x) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08, v) if dual_inp else (outputs08,)

        # [1, 128, H//8, W//8]
        y = self.layer2(x)
        outputs16 = [f(y) for f in self.outputs16]
        if num_layers == 2:
            return (outputs08, outputs16, v) if dual_inp else (outputs08, outputs16)

        # [1, 128, H//16, W//16]
        z = self.layer3(y)
        outputs32 = [f(z) for f in self.outputs32]

        return (outputs08, outputs16, outputs32), depth




class DepthMatchExtractor(nn.Module):
    def __init__(self, model_dir, output_dim=256, norm_fn='batch', downsample=2):
        super(DepthMatchExtractor, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.in_planes = 128
        self.layer2 = self._make_layer(128, stride=1)
        self.conv = nn.Conv2d(128, output_dim, kernel_size=1)

        # self._init_weights()

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        encoder = "vitl"
        depth_anything = DepthAnythingV2(**model_configs[encoder])
        depth_anything.load_state_dict(torch.load(os.path.join(model_dir, f'depth_anything_v2_{encoder}.pth'), 
                                                  map_location='cpu'))
        self.depth_anything = depth_anything.to('cuda')

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor(std).view(1, 3, 1, 1).cuda()

        # 冻结 depth_anything 模型的所有参数
        for param in self.depth_anything.parameters():
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

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)
    
    def forward(self, x, dual_inp=False, num_layers=3):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # resize image
        B, _, H, W = x.shape
        x = resize_tensor(x, target_size=518, ratio=14)
        
        # normalization
        x = ((x+1)/2 - self.mean) / self.std

        # DepthAnything
        with torch.no_grad():
            # out_depth: [1, 1, 518, 756]
            # out_fea: [1, 128, 296, 432]
            depth, depth_fea = self.depth_anything(x)

        # resize image
        # [1, 128, H//4, W//4]
        x = resize_to_quarter(depth_fea, (H,W), 2**self.downsample)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.conv(x)

        if is_list:
            x = x.split(split_size=batch_dim, dim=0)

        return x