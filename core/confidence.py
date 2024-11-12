import os
import sys
import logging
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F



class OffsetConfidence(nn.Module):
    def __init__(self, args):
        super(OffsetConfidence, self).__init__()
        self.detach = args.detach_in_confidence
        self.offset_memory_size = args.offset_memory_size
        self.conv_fea = nn.Conv2d(256, 16, 3, padding=1)
        self.conv_offset = nn.Conv2d(2*args.offset_memory_size, 16, 3, padding=1)
        self.fusion = nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(32, 8, 3, padding=1)),
                        ('relu1', nn.LeakyReLU(inplace=True)),
                        ('conv2', nn.Conv2d(8, 2, 3, padding=1)),
                        ('relu2', nn.LeakyReLU(inplace=True)),
                        ('conv3', nn.Conv2d(2, 1, 1, padding=0)),
                        ]))
        
        if "local_rank" not in args or args.local_rank==0 :
            logging.info(f"OffsetConfidence: " + \
                         f"detach: {args.detach_in_confidence}")

    def forward(self, fea, offset_memory):
        if type(fea) is list:
            fea = torch.cat(fea, dim=1)
        context = self.conv_fea(fea.detach() if self.detach else fea)
        offset_memory = torch.cat([offset.detach() if self.detach else offset for offset in offset_memory], dim=1)
        confidence = self.conv_offset( -offset_memory )
        confidence = self.fusion( torch.cat([confidence,context], dim=1) )
        return confidence



class MBConvBlockSimple(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=1, kernel_size=3, stride=1, se_ratio=0.25):
        super(MBConvBlockSimple, self).__init__()
        
        self.has_se = se_ratio is not None and 0 < se_ratio <= 1
        self.expand_ratio = expand_ratio
        mid_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(mid_channels)
        
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=mid_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        if self.has_se:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se_reduce = nn.Conv2d(mid_channels, se_channels, kernel_size=1)
            self.se_expand = nn.Conv2d(se_channels, mid_channels, kernel_size=1)
        
        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.swish = nn.SiLU(inplace=True)
        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x
        if self.expand_ratio != 1:
            x = self.swish(self.bn0(self.expand_conv(x)))
        
        x = self.swish(self.bn1(self.depthwise_conv(x)))

        if self.has_se:
            se = F.adaptive_avg_pool2d(x, 1)
            se = self.swish(self.se_reduce(se))
            se = torch.sigmoid(self.se_expand(se))
            x = x * se

        x = self.bn2(self.project_conv(x))

        if self.use_residual:
            x = x + identity

        return x


class EfficientNetB1SimpleEncoder(nn.Module):
    def __init__(self, in_C=2):
        super(EfficientNetB1SimpleEncoder, self).__init__()
        
        self.pre_pro = nn.Sequential(
            nn.Conv2d(in_C, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
        )

        # Stem, first downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # EfficientNet-B1 Layers Configuration
        layers_config = [
            (32, 16, 1, 3, 1, 1),  # Stage 1 (no downsampling)
            (16, 24, 6, 3, 2, 2),  # Stage 2 (second downsampling)
            (24, 40, 6, 5, 2, 2),  # Stage 3 (third downsampling)
        ]

        # Building EfficientNet-B1 stages
        self.blocks = nn.ModuleList()
        for in_channels, out_channels, expand_ratio, kernel_size, stride, repeats in layers_config:
            block_layers = []
            block_layers.append(MBConvBlockSimple(in_channels, out_channels, expand_ratio, kernel_size, stride))
            for _ in range(repeats - 1):
                block_layers.append(MBConvBlockSimple(out_channels, out_channels, expand_ratio, kernel_size, stride=1))
            self.blocks.append(nn.Sequential(*block_layers))

    def forward(self, x):
        features = []
        x = self.pre_pro(x)
        features.append(x)  # Store features for skip connections
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
            features.append(x)  # Store features for skip connections
        return features


class EfficientUNetSimple(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientUNetSimple, self).__init__()
        
        # Encoder using EfficientNet-B1 with only three stages
        self.encoder = EfficientNetB1SimpleEncoder()
        
        # Decoder layers (Upsampling)
        self.upconv3 = nn.Conv2d(40, 24, kernel_size=1)
        self.up3 = nn.ConvTranspose2d(24, 24, kernel_size=2, stride=2)
        
        self.upconv2 = nn.Conv2d(24, 16, kernel_size=1)
        self.up2 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        
        self.upconv1 = nn.Conv2d(16, 8, kernel_size=1)
        self.up1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
        
        # Final conv layer
        self.final_conv = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # print("-"*30, features[-1].shape, features[-2].shape, features[-3].shape, features[-4].shape)
        
        # Decoder with skip connections
        x = self.up3(self.upconv3(features[-1])) + features[-2]   # 1/8 ~ 1/4
        x = self.up2(self.upconv2(x)) + features[-3]              # 1/4 ~ 1/2
        x = self.up1(self.upconv1(x)) + features[-4]              # 1/2 ~ 1
        
        # Final output layer
        x = self.final_conv(x)
        return x
