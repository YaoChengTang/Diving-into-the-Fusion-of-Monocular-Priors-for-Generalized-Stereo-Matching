import os
import sys
import logging
import numpy as np
from collections import OrderedDict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',)

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
        context = self.conv_fea(fea.detach() if self.detach else fea)
        offset_memory = torch.cat([offset.detach() if self.detach else offset for offset in offset_memory], dim=1)
        confidence = self.conv_offset( -offset_memory )
        confidence = self.fusion( torch.cat([confidence,context], dim=1) )
        return confidence