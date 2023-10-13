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



def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, np.ndarray):
        return tuple(x)
    return (x,x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim_fea, dim_disp, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim_fea = dim_fea
        self.dim_disp = dim_disp
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qk = nn.Linear(dim_fea, dim_fea * 2, bias=False)
        self.v = nn.Linear(dim_disp, dim_disp, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim_fea))
            self.v_bias = nn.Parameter(torch.zeros(dim_disp))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_disp, dim_disp)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, guidance, shift_mask=None, reliability_mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C_x = x.shape
        qk_bias = None
        v_bias  = None
        if self.q_bias is not None:
            qk_bias = torch.cat((self.q_bias, torch.zeros_like(self.q_bias, requires_grad=False)))
            v_bias  = self.v_bias
        qk = F.linear(input=guidance, weight=self.qk.weight, bias=qk_bias)
        v  = F.linear(input=x, weight=self.v.weight, bias=v_bias)
        qk = qk.reshape(B_, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # (2, B_, nH, N, C_fea/nH)
        v  = v.reshape(B_, N, 1, 1, -1).permute(2, 0, 3, 1, 4)   # (1, B_, 1, N, C_x)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        v    = v.squeeze(0)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # (B_, nH, N, N)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)   # (1, 2*Wh-1, 2*Ww-1, nH)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)   # (B_, nH, N, N)

        if shift_mask is not None:
            nW = shift_mask.shape[0]
            # (B=B_/nW, nW, nH, N, N) + (nW, N, N) + (B=B_/nW, nW, N)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + \
                  shift_mask.unsqueeze(1).unsqueeze(0) +\
                  reliability_mask.view(B_ // nW, nW, N).unsqueeze(2).unsqueeze(-2)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).mean(dim=1)   # (B_, N, C_x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, args, dim_fea, dim_disp, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim_fea = dim_fea
        self.dim_disp = dim_disp
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim_disp)
        self.attn = WindowAttention(
            dim_fea, dim_disp, 
            window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        assert drop_path<=0, "no support for DropPath"
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim_disp)
        mlp_hidden_dim = int(dim_disp * mlp_ratio)
        self.mlp = Mlp(in_features=dim_disp, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def get_shift_mask(self, H, W, device):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            shift_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            shift_mask = shift_mask.masked_fill(shift_mask != 0, float(-100.0)).masked_fill(shift_mask == 0, float(0.0))
        else:
            shift_mask = None
        return shift_mask

    def forward(self, x, guidance, reliability):
        x = x.permute((0,2,3,1))
        guidance = guidance.permute((0,2,3,1))
        reliability = reliability.permute((0,2,3,1))
        
        B, H, W, C_fea = guidance.shape
        _, _, _, C_x = x.shape
        # guidance = guidance.flatten(2).transpose(1, 2)  # (B,H*W,C_fea)
        # x = x.flatten(2).transpose(1, 2)  # (B,H,W,C_x)

        shift_mask = self.get_shift_mask(H,W, x.device)
        shortcut = x
        # x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))   # (80, 180)
            shifted_guidance = torch.roll(guidance, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_reliability = torch.roll(reliability, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_guidance = guidance
            shifted_reliability = reliability

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C_x
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C_x)  # nW*B, window_size*window_size, C_x
        guidance_windows = window_partition(shifted_guidance, self.window_size)  # nW*B, window_size, window_size, C_fea
        guidance_windows = guidance_windows.view(-1, self.window_size * self.window_size, C_fea)  # nW*B, window_size*window_size, C_fea
        reliability_windows = window_partition(shifted_reliability, self.window_size)  # nW*B, window_size, window_size, 1
        reliability_windows = reliability_windows.view(-1, self.window_size * self.window_size)  # nW*B, window_size*window_size

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, guidance_windows, 
                            shift_mask=shift_mask,
                            reliability_mask=reliability_windows)  # nW*B, window_size*window_size, C_x

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C_x)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H, W, C_x)
        # x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        # x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = shortcut + self.drop_path(self.norm2(self.mlp(x)))
        x = x.view(B,H,W,C_x).permute((0,3,1,2))

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Refinement(nn.Module):
    def __init__(self, args, in_chans, dim_fea, dim_disp, num_heads):
        super(Refinement, self).__init__()
        self.window_size = args.refine_win_size
        self.patch_embed = nn.Conv2d(in_chans, dim_fea, kernel_size=3, stride=1, padding=1)
        self.propagation_1 = SwinTransformerBlock(args, dim_fea, dim_disp, num_heads, 
                                window_size=self.window_size, shift_size=0,)
        self.propagation_2 = SwinTransformerBlock(args, dim_fea, dim_disp, num_heads, 
                                window_size=self.window_size, shift_size=self.window_size//2,)
        
        # if "local_rank" not in args or args.local_rank==0 :
        #     logging.info(f"OffsetConfidence: " + \
        #                  f"detach: {args.detach_in_confidence}")

    def forward(self, disparity, fea, confidence=None, if_shift=False):
        guidance = self.patch_embed(fea)
        if confidence is not None :
            uncertainty = F.sigmoid(confidence.detach())
            uncertainty = uncertainty.masked_fill(uncertainty>0.98, float(-100.0)).masked_fill(uncertainty<=0.98, float(1.0))
            reliability = uncertainty.detach()
        else:
            reliability = None
        if not if_shift:
            disparity_refine = self.propagation_1(disparity, guidance, reliability)
        else:
            disparity_refine = self.propagation_2(disparity, guidance, reliability)
        return disparity_refine
    
    