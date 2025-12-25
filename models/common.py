import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse
#from timm.layers._efficientnet_blocks import SqueezeExcite

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from IPython.display import display
from PIL import Image
from torch.cuda import amp

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version, colorstr,
                           increment_path, is_notebook, make_divisible, non_max_suppression, scale_boxes,
                           xywh2xyxy, xyxy2xywh, yaml_load)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, smart_inference_mode

from timm.layers import DropPath, to_2tuple, trunc_normal_

import einops
import numpy as np
from torch import nn, Tensor
from typing import Optional, Dict, Tuple, Union, Sequence
from . import InvertedResidual
from .base_module import BaseModule
from affnet.misc.profiler import module_profile
from affnet.layers import ConvLayer, get_normalization_layer, get_activation_fn
import math
import torch
import torch.fft

import numpy as np
from torch import nn, Tensor
from utils import logger

import torch
from torch.nn import functional as F
from typing import Optional, Dict, Tuple, Union, Sequence

from affnet.modules.base_module import BaseModule
from models.profiler import module_profile
from affnet.layers.layers import ConvLayer, InvertedResidual
from models.sync_batch_norm import SyncBatchNorm
import math
import torch
import torch.fft
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class AConv(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class MP(nn.Module):
    # Max pooling
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class ConvTranspose(nn.Module):
    # Convolution transpose 2d layer
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv_transpose(x)))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class DWConvSE(DWConv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, d, act)
        # 由于使用了timm库的SqueezeExcite，所以不需要再次定义SE
        self.se = SqueezeExcite(c2, 0.5)

    def forward(self, x):
        x = super().forward(x)  # 使用DWConv的forward方法
        x = self.se(x)  # 应用SqueezeExcite模块
        return x

class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))

class DwConv(nn.Module):
    def __init__(self, dim=768):
        super(DwConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DwConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W
class LSK(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv2d(dim // 2, dim, 1)

        # # 上下文感知模块 (Global Context Block)
        # self.context_block = nn.Sequential(
        #     nn.Conv2d(dim, dim, 1),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(dim, dim, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.convl(attn1)
        attn1 = self.conv0_s(attn1)
        attn2 = self.conv1_s(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv_m(attn)

        return x * attn

class TLSK(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv2d(dim // 2, dim, 1)

        # 上下文感知模块 (Global Context Block)
        self.context_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.convl(attn1)
        attn1 = self.conv0_s(attn1)
        attn2 = self.conv1_s(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv_m(attn)
        context = self.context_block(x)
        attn = attn * context
        return x * attn




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP layers with shared weights
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction=reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class CTLSK(nn.Module):  # Changed to CTLSK for consistency with your code
    def __init__(self, dim):
        super(CTLSK, self).__init__()

        # Convolutional layers
        self.conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv2d(dim // 2, dim, 1)

        # Define the missing BatchNorm2d layer (added this to fix the issue)
        self.bn = nn.BatchNorm2d(dim)  # Ensure that batch norm is defined for the residual connection

        # CBAM module
        self.cbam = CBAM(dim)

        # Global Context Block
        self.context_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn1 = self.conv(x)
        attn2 = self.conv1(attn1)
        attn1 = self.conv0_s(attn1)
        attn2 = self.conv1_s(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()

        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv_m(attn)

        # Pass through CBAM
        attn = self.cbam(attn)

        # Apply BatchNorm (fixing the error where bn was missing)
        attn = self.bn(attn)

        # Global Context Block
        context = self.context_block(x)
        attn = attn * context

        return x * attn

    # class Conv2d_BN(nn.Module):
    #     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
    #         super(Conv2d_BN, self).__init__()
    #         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
    #         self.bn = nn.BatchNorm2d(out_channels)
    #
    #     def forward(self, x):
    #         return self.bn(self.conv(x))

    class Conv2d_BN(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bn_weight_init=1):
            super(Conv2d_BN, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
            self.bn = nn.BatchNorm2d(out_channels)
            if bn_weight_init is not None:
                nn.init.constant_(self.bn.weight, bn_weight_init)  # 初始化 BatchNorm 权重

        def forward(self, x):
            return self.bn(self.conv(x))

    class Conv2d_BN(torch.nn.Sequential):
        def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                     groups=1, bn_weight_init=1, resolution=-10000):
            super().__init__()
            self.add_module('c', torch.nn.Conv2d(
                a, b, ks, stride, pad, dilation, groups, bias=False))
            self.add_module('bn', torch.nn.BatchNorm2d(b))
            torch.nn.init.constant_(self.bn.weight, bn_weight_init)
            torch.nn.init.constant_(self.bn.bias, 0)

    # # 定义轻量化的基础卷积层
    # def Conv2d_BN(inp, oup, kernel_size, stride, padding, groups=1):
    #     return nn.Sequential(
    #         nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
    #         nn.BatchNorm2d(oup),
    #         nn.ReLU(inplace=True)
    #     )





    class BAM(nn.Module):
        def __init__(self, in_channels, reduction_ratio=16, dilation_val=4):
            super(BAM, self).__init__()

            # Channel attention
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
            )

            # Spatial attention
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size=3,
                          padding=dilation_val, dilation=dilation_val),
                nn.ReLU(),
                nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1)
            )

            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # Channel attention
            ca = self.channel_attention(x)

            # Spatial attention
            sa = self.spatial_attention(x)

            # Combined attention
            out = ca + sa
            return x * self.sigmoid(out)

    class CBAM_BAM(nn.Module):
        def __init__(self, in_channels, reduction_ratio=16, dilation_val=4):
            super(CBAM_BAM, self).__init__()

            # CBAM Channel Attention
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
                nn.Sigmoid()
            )

            # CBAM Spatial Attention
            self.cbam_spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            )

            # BAM Module (used for spatial attention enhancement)
            self.bam = BAM(in_channels, reduction_ratio=reduction_ratio, dilation_val=dilation_val)

            # Learnable weight alpha for combining CBAM and BAM spatial attentions
            self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize alpha to 0.5

        def forward(self, x):
            # Channel Attention (CBAM)
            ca = self.channel_attention(x)
            x_ca = x * ca

            # CBAM Spatial Attention
            max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]
            avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
            combined = torch.cat([max_pool, avg_pool], dim=1)
            cbam_sa = self.cbam_spatial_attention(combined)

            # BAM Spatial Attention
            bam_sa = self.bam(x_ca)

            # Weighted combination of CBAM and BAM spatial attention
            combined_sa = self.alpha * cbam_sa + (1 - self.alpha) * bam_sa

            # Final output
            out = x_ca * combined_sa
            return out

    class RepVGGDW(nn.Module):
        def __init__(self, ed):
            super(RepVGGDW, self).__init__()
            self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
            self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
            self.bn = nn.BatchNorm2d(ed)
            self.bam_cbam = CBAM_BAM(ed)  # Add the CBAM_BAM module

        def forward(self, x):
            x = self.bn((self.conv(x) + self.conv1(x)) + x)
            x = self.bam_cbam(x)  # Apply the CBAM_BAM module
            return x

    # class RepVGGDW(torch.nn.Module):
    #     def __init__(self, ed) -> None:
    #         super().__init__()
    #         self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
    #         self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
    #         #    self.conv1 = RepConvN(ed, ed, 3, 1, )
    #         self.dim = ed
    #         self.bn = torch.nn.BatchNorm2d(ed)
    #
    # def forward(self, x):
    #     return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


# class RepKKBlock(nn.Module):
#     def __init__(self, inp,  oup, hidden_dim, kernel_size, stride, use_se, use_hs):
#         super(RepKKBlock, self).__init__()
#         assert stride in [1, 2]
#
#         self.identity = stride == 1 and inp == oup
#         assert (hidden_dim == 2 * inp)
#
#         if stride == 2:
#             self.token_mixer = nn.Sequential(
#                 Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
#                 SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
#                 Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
#             )
#             self.channel_mixer = Residual(nn.Sequential(
#                 # pw
#                 Conv2d_BN(oup, 2 * oup, 1, 1, 0),
#                 nn.GELU() if use_hs else nn.GELU(),
#                 # pw-linear
#                 Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
#             ))
#         else:
#             assert (self.identity)
#             self.token_mixer = nn.Sequential(
#                 RepVGGDW(inp),
#                 SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
#             )
#             self.channel_mixer = Residual(nn.Sequential(
#                 # pw
#                 Conv2d_BN(inp, hidden_dim, 1, 1, 0),
#                 nn.GELU() if use_hs else nn.GELU(),
#                 # pw-linear
#                 Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
#             ))
#
#     def forward(self, x):
#         return self.channel_mixer(self.token_mixer(x))

# class BAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, dilation_val=4):
#         super(BAM, self).__init__()
#
#         # Channel attention
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
#         )
#
#         # Spatial attention
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size=3,
#                       padding=dilation_val, dilation=dilation_val),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1)
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # Channel attention
#         ca = self.channel_attention(x)
#
#         # Spatial attention
#         sa = self.spatial_attention(x)
#
#         # Combined attention
#         out = ca + sa
#         return x * self.sigmoid(out)
#
#
# class CBAM_BAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, dilation_val=4):
#         super(CBAM_BAM, self).__init__()
#
#         # CBAM Channel Attention
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#         # CBAM Spatial Attention
#         self.cbam_spatial_attention = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=7, padding=3),
#             nn.Sigmoid()
#         )
#
#         # BAM Module (used for spatial attention enhancement)
#         self.bam = BAM(in_channels, reduction_ratio=reduction_ratio, dilation_val=dilation_val)
#
#         # Learnable weight alpha for combining CBAM and BAM spatial attentions
#         self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize alpha to 0.5
#
#     def forward(self, x):
#         # Channel Attention (CBAM)
#         ca = self.channel_attention(x)
#         x_ca = x * ca
#
#         # CBAM Spatial Attention
#         max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
#         combined = torch.cat([max_pool, avg_pool], dim=1)
#         cbam_sa = self.cbam_spatial_attention(combined)
#
#         # BAM Spatial Attention
#         bam_sa = self.bam(x_ca)
#
#         # Weighted combination of CBAM and BAM spatial attention
#         combined_sa = self.alpha * cbam_sa + (1 - self.alpha) * bam_sa
#
#         # Final output
#         out = x_ca * combined_sa
#         return out


# class RepKKBlock(nn.Module):
#     def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
#         super(RepKKBlock, self).__init__()
#         assert stride in [1, 2]
#
#         self.identity = stride == 1 and inp == oup
#         assert (hidden_dim == 2 * inp)
#
#         # Define BAM (if needed, you can tune the reduction_ratio and dilation_val)
#         self.cbam_bam = CBAM_BAM(oup, reduction_ratio=16, dilation_val=4)
#
#         if stride == 2:
#             self.token_mixer = nn.Sequential(
#                 Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
#                 SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
#                 Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
#             )
#             self.channel_mixer = Residual(nn.Sequential(
#                 # pw
#                 Conv2d_BN(oup, 2 * oup, 1, 1, 0),
#                 nn.GELU() if use_hs else nn.GELU(),
#                 # pw-linear
#                 Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
#             ))
#         else:
#             assert (self.identity)
#             self.token_mixer = nn.Sequential(
#                 RepVGGDW(inp),
#                 SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
#             )
#
#             self.channel_mixer = Residual(nn.Sequential(
#                 # pw
#                 Conv2d_BN(inp, hidden_dim, 1, 1, 0),
#                 nn.GELU() if use_hs else nn.GELU(),
#                 # pw-linear
#                 Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
#             ))
#
#     def forward(self, x):
#         x = self.token_mixer(x)  # First process through token mixer
#         x = self.cbam_bam(x)          # Apply BAM after token mixer
#         return self.channel_mixer(x)  # Finally, process through channel mixer


class GALSK(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # LSK 的卷积层
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv2d(dim // 2, dim, 1)

        # 上下文感知模块 (Global Context Block)
        self.context_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # 注意力模块的投影层
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # LSK 的空间注意力机制
        attn1 = self.conv0(x)
        attn2 = self.convl(attn1)
        attn1 = self.conv0_s(attn1)
        attn2 = self.conv1_s(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv_m(attn)

        # 上下文感知
        context = self.context_block(x)
        attn = attn * context

        # 注意力模块的前馈网络
        x = self.proj_1(x)
        x = self.activation(x)
        x = attn * x  # 注意力加权
        x = self.proj_2(x)

        return x

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSK(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LSKK(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.5, drop_path=0.5, act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x



class LSKModel(nn.Module):
    def __init__(self, dim, depth=12, img_size=224, patch_size=7, stride=4, in_chans=3, num_classes=1000):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(img_size=img_size,
                                             patch_size=patch_size,
                                             stride=stride,
                                             in_chans=in_chans,
                                             embed_dim=dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size // (patch_size ** 2), dim))

        self.d_model = dim
        self.depth = depth
        self.mlp_ratio = 4
        self.norm_cfg = None  # 如果需要，可以设置norm_cfg

        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                mlp_ratio=self.mlp_ratio,
                drop=0.1,
                drop_path=0.1,
                norm_cfg=self.norm_cfg
            ) for _ in range(self.depth)
        ])

        self.norm = nn.BatchNorm2d(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x, H, W = self.patch_embed(x)

        # 将嵌入后的图像转换为序列
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # 添加分类token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # 添加位置编码
        pos_embed = self.pos_embed[:, :, :x.size(2)]
        x += pos_embed

        # 通过所有块
        for blk in self.blocks:
            x = blk(x)

        # 池化和分类头
        x = self.norm(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (B, C, 1, 1)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
class DFL(nn.Module):
    # DFL module
    def __init__(self, c1=17):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))  # / 120.0
        self.c1 = c1
        # self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class BottleneckBase(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(1, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RBottleneckBase(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 1), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNRBottleneckBase(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 1), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Res(nn.Module):
    # ResNet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Res, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, 1, g=g)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class RepNRes(nn.Module):
    # ResNet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(RepNRes, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = RepConvN(c_, c_, 3, 1, g=g)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# =============================
# 基础组件
# =============================
class Conv_BN_Act(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# =============================
# 轻量注意力模块 (SimAM)
# =============================
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n
        e_inv = x_minus_mu_square / (4 * (v + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e_inv)

# =============================
# 改进的 RepNBottleneckDW
# =============================
class RepNBottleneckDW(nn.Module):
    def __init__(self, c1, c2, shortcut=True, dilation=1, e=1.0):
        super().__init__()
        hidden = int(c2 * e)
        self.dwconv = Conv_BN_Act(c1, hidden, 3, 1, p=dilation, g=hidden)  # 深度卷积
        self.pwconv = Conv_BN_Act(hidden, c2, 1, 1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        y = self.pwconv(self.dwconv(x))
        return x + y if self.shortcut else y

# =============================
# 主体模块 RepNCSP
# =============================
class RepSNCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, use_attention=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv_BN_Act(c1, c_, 1, 1)
        self.cv2 = Conv_BN_Act(c1, c_, 1, 1)
        self.m = nn.Sequential(*(RepNBottleneckDW(c_, c_, shortcut, e=1.0) for _ in range(n)))
        self.cv3 = Conv_BN_Act(2 * c_, c2, 1, 1)
        self.att = SimAM() if use_attention else nn.Identity()

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        y = torch.cat((y1, y2), dim=1)
        y = self.att(y)
        return self.cv3(y)

class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class CSPBase(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(BottleneckBase(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class ASPP(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        dilations = [1, 3, 6, 1]
        paddings = [0, 3, 6, 0]
        self.aspp = torch.nn.ModuleList()
        for aspp_idx in range(len(kernel_sizes)):
            conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_sizes[aspp_idx],
                stride=1,
                dilation=dilations[aspp_idx],
                padding=paddings[aspp_idx],
                bias=True)
            self.aspp.append(conv)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.aspp_num = len(kernel_sizes)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(self.aspp_num):
            inp = avg_x if (aspp_idx == self.aspp_num - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


class SPPCSPC(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # self.m = SoftPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class ReOrg(nn.Module):
    # yolo
    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Shortcut(nn.Module):
    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        return x[0] + x[1]


class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()

    def forward(self, x):
        return x


##### GELAN #####

class SPPELAN(nn.Module):
    # spp-elan
    def __init__(self, c1, c2, c3):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class ELAN1(nn.Module):

    def __init__(self, c1, c2, c3, c4):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepSNCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepSNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


#################




class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        #    self.conv1 = RepConvN(ed, ed, 3, 1, )
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepKKBlock(nn.Module):
    def __init__(self, inp,  oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(RepKKBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))




##### YOLOR #####

class ImplicitA(nn.Module):
    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1., std=.02)

    def forward(self, x):
        return self.implicit * x


#################


##### CBNet #####

class CBLinear(nn.Module):
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    def __init__(self, idx):
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out


#################


class DetectMultiBackend(nn.Module):
    # YOLO MultiBackend class for python inference on various backends
    def __init__(self, weights='yolo.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, onnx_end2end, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(
            w)
        fp16 &= pt or jit or onnx or engine  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:  # load metadata dict
                d = json.loads(extra_files['config.txt'],
                               object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                      for k, v in d.items()})
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements('opencv-python>=4.5.4')
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements('openvino')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            executable_network = ie.compile_model(network, device_name="CPU")  # device_name="MYRIAD" for Intel NCS2
            stride, names = self._load_metadata(Path(w).with_suffix('.yaml'))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
            import tensorflow as tf
            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, 'rb') as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                delegate = {
                    'Linux': 'libedgetpu.so.1',
                    'Darwin': 'libedgetpu.1.dylib',
                    'Windows': 'edgetpu.dll'}[platform.system()]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta['stride']), meta['names']
        elif tfjs:  # TF.js
            raise NotImplementedError('ERROR: YOLO TF.js inference is not supported')
        elif paddle:  # PaddlePaddle
            LOGGER.info(f'Loading {w} for PaddlePaddle inference...')
            check_requirements('paddlepaddle-gpu' if cuda else 'paddlepaddle')
            import paddle.inference as pdi
            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).rglob('*.pdmodel'))  # get *.pdmodel file from *_paddle_model dir
            weights = Path(w).with_suffix('.pdiparams')
            config = pdi.Config(str(w), str(weights))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f'Using {w} as Triton Inference Server...')
            check_requirements('tritonclient[all]')
            from utils.triton import TritonRemoteModel
            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            raise NotImplementedError(f'ERROR: {w} is not a supported format')

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLO MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.executable_network([im]).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output['index'])
                    if int8:
                        scale, zero_point = output['quantization']
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None


class AutoShape(nn.Module):
    # YOLO input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        from models.yolo import Detect, Segment
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            if isinstance(m, (Detect, Segment)):
                for k in 'stride', 'anchor_grid', 'stride_grid', 'grid':
                    x = getattr(m, k)
                    setattr(m, k, list(map(fn, x))) if isinstance(x, (list, tuple)) else setattr(m, k, fn(x))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        # Inference from various sources. For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f'image{i}'  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
                files.append(Path(f).with_suffix('.jpg').name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast(autocast):
            # Inference
            with dt[1]:
                y = self.model(x, augment=augment)  # forward

            # Post-process
            with dt[2]:
                y = non_max_suppression(y if self.dmb else y[0],
                                        self.conf,
                                        self.iou,
                                        self.classes,
                                        self.agnostic,
                                        self.multi_label,
                                        max_det=self.max_det)  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


class Detections:
    # YOLO detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1E3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
        s, crops = '', []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(', ')
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                display(im) if is_notebook() else im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip('\n')
            return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    @TryExcept('Showing images is not supported in this environment')
    def show(self, labels=True):
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['ims', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def print(self):
        LOGGER.info(self.__str__())

    def __len__(self):  # override len(results)
        return self.n

    def __str__(self):  # override print(results)
        return self._run(pprint=True)  # print results

    def __repr__(self):
        return f'YOLO {self.__class__} instance\n' + self.__str__()


class Proto(nn.Module):
    # YOLO mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class UConv(nn.Module):
    def __init__(self, c1, c_=256, c2=256):  # ch_in, number of protos, number of masks
        super().__init__()

        self.cv1 = Conv(c1, c_, k=3)
        self.cv2 = nn.Conv2d(c_, c2, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.up(self.cv2(self.cv1(x)))


class Classify(nn.Module):
    # YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))






def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class AFNO2D_channelfirst(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    input shape [B N C]
    """

    def __init__(self, opts, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = getattr(opts, "model.activation.sparsity_threshold", 0.01)
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        # self.norm_layer1 = get_normalization_layer(opts=opts, num_features=out_channels)
        self.act = self.build_act_layer(opts=opts)
        self.act2 = self.build_act_layer(opts=opts)

    @staticmethod
    def build_act_layer(opts) -> nn.Module:
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=1,
        )
        return act_layer

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        # x = self.fu(x)

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])


        o1_real = self.act(
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) - \
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[1]) + \
            self.b1[0, :, :, None, None]
        )

        o1_imag = self.act2(
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0]) + \
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[1]) + \
            self.b1[1, :, :, None, None]
        )

        o2_real = (
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[0]) - \
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[1]) + \
                self.b2[0, :, :, None, None]
        )

        o2_imag = (
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[0]) + \
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[1]) + \
                self.b2[1, :, :, None, None]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, x.shape[3], x.shape[4])

        x = x * origin_ffted
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)

        return x + bias

    def profile_module(
            self, input: Tensor, *args, **kwargs
        ) -> Tuple[Tensor, float, float]:
        # TODO: to edit it
        b_sz, c, h, w = input.shape
        seq_len = h * w

        # FFT iFFT
        p_ff, m_ff = 0, 5 * b_sz * seq_len * int(math.log(seq_len)) * c
        # others
        # params = macs = sum([p.numel() for p in self.parameters()])
        params = macs = self.hidden_size * self.hidden_size_factor * self.hidden_size * 2 * 2 // self.num_blocks
        # // 2 min n become half after fft
        macs = macs * b_sz * seq_len

        # return input, params, macs
        return input, params, macs + m_ff


def remove_edge(img: np.ndarray):
    # // remove the edge of a numpy image
    return img[1:-1, 1:-1]

def save_feature(feature):
    import time
    import matplotlib.pyplot as plt
    import os
    now = time.time()
    feature = feature.detach()
    os.makedirs('visual_example', exist_ok=True)
    for i in range(feature.shape[1]):
        feature_channel = feature[0, i]
        fig, ax = plt.subplots()
        img_channel = ax.imshow(remove_edge(feature_channel.cpu().numpy()), cmap='gray')
        plt.savefig('visual_example/{now}_channel_{i}_feature.png'.format(now=str(now), i=i))
    for i in range(8):
        feature_group = torch.mean(feature[0, i * 8:(i + 1) * 8], dim=1)
        fig, ax = plt.subplots()
        img_group = ax.imshow(remove_edge(feature_group.cpu().numpy()), cmap='gray')
        plt.savefig('visual_example/{now}_group_{i}_feature.png'.format(now=str(now), i=i))

def save_kernel(origin_ffted, H, W):
    import time
    import matplotlib.pyplot as plt
    import os
    now = time.time()
    origin_ffted = origin_ffted.detach()
    kernel = torch.fft.irfft2(origin_ffted, s=(H, W), dim=(2, 3), norm="ortho")
    group_channels = kernel.shape[1] // 8
    os.makedirs('visual_example', exist_ok=True)
    for i in range(kernel.shape[1]):
        kernel_channel = kernel[0, i]
        fig, ax = plt.subplots()
        img_channel = ax.imshow(remove_edge(kernel_channel.cpu().numpy()), cmap='gray')
        plt.savefig('visual_example/{now}_channel_{i}_kernel.png'.format(now=str(now), i=i))
    for i in range(8):
        kernel_group = torch.mean(kernel[0, i*group_channels: (i+1)*group_channels], dim=0)
        fig, ax = plt.subplots()
        img_group = ax.imshow(remove_edge(kernel_group.cpu().numpy()), cmap='gray')
        plt.savefig('visual_example/{now}_group_{i}_kernel.png'.format(now=str(now), i=i))
    kernel_mean = torch.mean(kernel[0], dim=0)
    fig, ax = plt.subplots()
    img_mean = ax.imshow(remove_edge(kernel_mean.cpu().numpy()), cmap='gray')
    plt.savefig('visual_example/{now}_all_kernel.png'.format(now=str(now)))

    abs = origin_ffted.abs()
    abs_group_channels = abs.shape[1] // 8
    os.makedirs('visual_mask_example', exist_ok=True)
    for i in range(abs.shape[1]):
        abs_channel = abs[0, i]
        fig, ax = plt.subplots()
        abs_channel = ax.imshow(abs_channel.cpu().numpy(), cmap='gray')
        plt.savefig('visual_mask_example/{now}_channel_{i}_abs.png'.format(now=str(now), i=i))
    for i in range(8):
        abs_group = torch.mean(abs[0, i*abs_group_channels: (i+1)*abs_group_channels], dim=0)
        fig, ax = plt.subplots()
        img_group = ax.imshow(abs_group.cpu().numpy(), cmap='gray')
        plt.savefig('visual_mask_example/{now}_group_{i}_abs.png'.format(now=str(now), i=i))
    abs_mean = torch.mean(abs[0], dim=0)
    fig, ax = plt.subplots()
    img_mean = ax.imshow(abs_mean.cpu().numpy(), cmap='gray')
    plt.savefig('visual_mask_example/{now}_all_abs.png'.format(now=str(now)))

    real = origin_ffted.real
    real_group_channels = real.shape[1] // 8
    os.makedirs('visual_mask_example', exist_ok=True)
    for i in range(real.shape[1]):
        real_channel = real[0, i]
        fig, ax = plt.subplots()
        real_channel = ax.imshow(real_channel.cpu().numpy(), cmap='gray')
        plt.savefig('visual_mask_example/{now}_channel_{i}_real.png'.format(now=str(now), i=i))
    for i in range(8):
        real_group = torch.mean(real[0, i*real_group_channels: (i+1)*real_group_channels], dim=0)
        fig, ax = plt.subplots()
        img_group = ax.imshow(real_group.cpu().numpy(), cmap='gray')
        plt.savefig('visual_mask_example/{now}_group_{i}_mask.png'.format(now=str(now), i=i))
    real_mean = torch.mean(real[0], dim=0)
    fig, ax = plt.subplots()
    img_mean = ax.imshow(real_mean.cpu().numpy(), cmap='gray')
    plt.savefig('visual_mask_example/{now}_all_real.png'.format(now=str(now)))

    imag = origin_ffted.imag
    imag_group_channels = imag.shape[1] // 8
    os.makedirs('visual_mask_example', exist_ok=True)
    for i in range(8):
        imag_group = torch.mean(imag[0, i*imag_group_channels: (i+1)*imag_group_channels], dim=0)
        fig, ax = plt.subplots()
        img_group = ax.imshow(imag_group.cpu().numpy(), cmap='gray')
        plt.savefig('visual_mask_example/{now}_group_{i}_imag.png'.format(now=str(now), i=i))
    imag_mean = torch.mean(imag[0], dim=0)
    fig, ax = plt.subplots()
    img_mean = ax.imshow(imag_mean.cpu().numpy(), cmap='gray')
    plt.savefig('visual_mask_example/{now}_all_imag.png'.format(now=str(now)))



class Block(nn.Module):
    def __init__(self, opts, dim, hidden_size, num_blocks, double_skip, mlp_ratio=4., drop_path=0., attn_norm_layer='sync_batch_norm', enable_coreml_compatible_fn=False):
        # input shape [B C H W]
        super().__init__()
        self.norm1 = get_normalization_layer(
            opts=opts, norm_type=attn_norm_layer, num_features=dim
        )
        self.filter = AFNO2D_channelfirst(opts=opts, hidden_size=hidden_size, num_blocks=num_blocks, sparsity_threshold=0.01,
                                          hard_thresholding_fraction=1, hidden_size_factor=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = get_normalization_layer(
            opts=opts, norm_type=attn_norm_layer, num_features=dim
        )
        self.mlp = InvertedResidual(
            opts=opts,
            in_channels=dim,
            out_channels=dim,
            stride=1,
            expand_ratio=mlp_ratio,
        )
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        # x = self.filter(x)
        x = self.mlp(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        # x = self.mlp(x)
        x = self.filter(x)
        x = self.drop_path(x)
        x = x + residual
        return x
    def profile_module(
            self, input: Tensor, *args, **kwargs
        ) -> Tuple[Tensor, float, float]:
        b_sz, c, h, w = input.shape
        seq_len = h * w

        out, p_ffn, m_ffn = module_profile(module=self.mlp, x=input)
        # m_ffn = m_ffn * b_sz * seq_len

        out, p_mha, m_mha = module_profile(module=self.filter, x=out)


        macs = m_mha + m_ffn
        params = p_mha + p_ffn

        return input, params, macs


class AFFBlock(BaseModule):

    def __init__(self, in_channels: int, transformer_dim: int=32, ffn_dim: int=64,
                 n_transformer_blocks: Optional[int] = 2,
                 head_dim: Optional[int] = 32,
                 attn_dropout: Optional[float] = 0.0,
                 dropout: Optional[int] = 0.0,
                 ffn_dropout: Optional[int] = 0.0,
                 patch_h: Optional[int] = 8,
                 patch_w: Optional[int] = 8,
                 attn_norm_layer: Optional[str] = "layer_norm_2d",
                 conv_ksize: Optional[int] = 3,
                 dilation: Optional[int] = 1,
                 no_fusion: Optional[bool] = False,
                 *args,
                 **kwargs) -> None:

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=False,
        )
        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = ConvLayer(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                kernel_size=1,  # conv_ksize -> 1
                stride=1,
                use_norm=True,
                use_act=True,
            )
        super().__init__()

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        self.global_rep = [
            Block(
                dim=transformer_dim,
                hidden_size=transformer_dim,
                num_blocks=8,
                double_skip=False,
                mlp_ratio=ffn_dim / transformer_dim,
                attn_norm_layer=attn_norm_layer,
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(
            get_normalization_layer(
                norm_type=attn_norm_layer,
                num_features=transformer_dim,
            )
        )
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)

        repr_str += "\n\t Global representations with patch size of {}x{}".format(
            self.patch_h, self.patch_w
        )
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.global_rep)

        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.conv_proj)

        if self.fusion is not None:
            repr_str += "\n\t Feature fusion"
            if isinstance(self.fusion, nn.Sequential):
                for m in self.fusion:
                    repr_str += "\n\t\t {}".format(m)
            else:
                repr_str += "\n\t\t {}".format(self.fusion)

        repr_str += "\n)"
        return repr_str

    def forward_spatial(self, x: Tensor) -> Tensor:
        res = x

        patches = x

        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        fm = self.conv_proj(patches)

        if self.fusion is not None:
            fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm

    def forward(
            self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tuple) and len(x) == 2:
            return self.forward_temporal(x=x[0], x_prev=x[1])
        elif isinstance(x, Tensor):
            return self.forward_spatial(x)
        else:
            raise NotImplementedError

    def profile_module(
            self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        params = macs = 0.0

        res = input

        b, c, h, w = input.size()

        out, p, m = module_profile(module=self.global_rep, x=input)
        params += p
        macs += m

        out, p, m = module_profile(module=self.conv_proj, x=out)
        params += p
        macs += m

        if self.fusion is not None:
            out, p, m = module_profile(
                module=self.fusion, x=torch.cat((out, res), dim=1)
            )
            params += p
            macs += m

        return res, params, macs


class AFF(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 使用分组卷积来模拟Channel Mixer的效果
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        # 模拟Channel Mixer和Token Mixer的效果
        attn1 = self.conv0(x)
        attn2 = self.convl(attn1)
        attn1 = self.conv0_s(attn1)
        attn2 = self.conv1_s(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv_m(attn)

        return x * attn

# class AFFKBlock(BaseModule):
#     def __init__(self, inp, oup, hidden_dim, kernel_size=3, stride=1, n_transformer_blocks=2, head_dim=32, use_se=False,
#                  use_hs=True):
#         super(AFFKBlock, self).__init__()
#
#         assert hidden_dim == 2 * inp
#         self.identity = stride == 1 and inp == oup
#
#         # Token mixer (局部特征提取)
#         self.token_mixer = nn.Sequential(
#             ConvLayer(inp, inp, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, use_norm=True,
#                       use_act=True),
#             SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
#         )
#
#         # Transformer 用于全局特征提取
#         self.global_rep = nn.Sequential(
#             *[Block(opts=None, dim=hidden_dim, hidden_size=hidden_dim, num_blocks=1, mlp_ratio=2) for _ in
#               range(n_transformer_blocks)]
#         )
#
#         # 全局特征线性层
#         self.fc = nn.Sequential(
#             nn.Linear(inp, hidden_dim),
#             nn.GELU() if use_hs else nn.Identity(),
#             nn.Linear(hidden_dim, oup)
#         )
#
#         # Channel mixer (通道特征提取)
#         self.channel_mixer = Residual(nn.Sequential(
#             ConvLayer(oup, 2 * oup, kernel_size=1, stride=1, use_norm=True, use_act=use_hs),
#             ConvLayer(2 * oup, oup, kernel_size=1, stride=1, use_norm=True, use_act=False),
#         ))
#
#     def forward(self, x):
#         local_features = self.token_mixer(x)  # 获取局部特征
#
#         # 获取全局特征
#         batch_size, channels, height, width = x.size()
#         global_features = self.global_avg_pool(x).view(batch_size, channels)  # 全局平均池化
#         global_features = self.fc(global_features).view(batch_size, -1, 1, 1)  # 线性变换并调整形状
#
#         # 将全局特征输入到 Transformer
#         transformer_input = global_features.view(batch_size, -1)  # 转换为适合 Transformer 的形状
#         transformer_output = self.global_rep(transformer_input)  # 通过 Transformer 提取全局特征
#
#         # 合并局部和全局特征
#         combined_features = local_features + transformer_output.view(batch_size, -1, 1, 1)  # 通过加法或其他方法融合
#
#         return self.channel_mixer(combined_features)


class SSMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads=2, dropout=0.1):
        super(SSMLayer, self).__init__()

        # 状态空间模型的参数定义
        self.hidden_dim = hidden_dim

        # 使用 LSTM 作为 SSM 的一个实现方式，LSTM 本身就可以视作一个状态空间模型
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)

        # 可选：额外的线性层，用于将状态映射回输入维度
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        输入的 x 是形状为 (batch_size, seq_len, input_dim) 的张量
        """
        # LSTM 层的输出：output 是所有时间步的隐状态，(batch_size, seq_len, hidden_dim)
        output, (h_n, c_n) = self.lstm(x)

        # 如果需要，可以进一步通过全连接层映射回原始维度
        output = self.fc(output)

        return output


class AFFBC(BaseModule):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs, n_heads=2):
        super(AFFBC, self).__init__()

        if hidden_dim is None:
            hidden_dim = 2 * inp

        self.identity = stride == 1 and inp == oup  # Identity mapping for stride 1

        # Token mixer (局部特征提取)
        self.token_mixer = nn.Sequential(
            ConvLayer(inp, inp, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, use_norm=True,
                      use_act=True),
            SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),  # 可选的 Squeeze-and-Excitation
        )

        # 使用 SSM 进行全局特征提取
        self.ssm_layer = SSMLayer(input_dim=inp, hidden_dim=hidden_dim, n_heads=n_heads)

        # Channel mixer (通道特征提取)
        self.channel_mixer = nn.Sequential(
            ConvLayer(inp + hidden_dim, hidden_dim, kernel_size=1, stride=1, use_norm=True, use_act=use_hs),
            ConvLayer(hidden_dim, oup, kernel_size=1, stride=1, use_norm=True, use_act=False),
        )

        if stride == 2:
            self.downsample = nn.Sequential(
                ConvLayer(inp, inp, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          use_norm=True, use_act=True),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                ConvLayer(inp, oup, kernel_size=1, stride=1, use_norm=True, use_act=False)
            )
        else:
            self.downsample = nn.Identity()  # stride=1 时，无需下采样

    def forward(self, x: Tensor) -> Tensor:
        # 获取局部特征
        local_features = self.token_mixer(x)

        # 准备 SSM 输入（需要将通道维转换为特征维）
        batch_size, channels, height, width = local_features.size()
        reshaped_features = local_features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)

        # 通过 SSM 进行全局特征提取
        ssm_out = self.ssm_layer(reshaped_features)  # (H*W, B, C)

        # 恢复为 (B, C, H, W)
        ssm_out = ssm_out.permute(1, 2, 0)  # (B, C, H*W)

        # 将 H*W 展开为 H 和 W
        ssm_out1 = ssm_out.reshape(batch_size, channels, height, width)  # (B, C, H, W)

        # 将局部特征与全局特征拼接
        combined_features = torch.cat([local_features, ssm_out1], dim=1)  # (B, C+H, H, W)

        # 通道特征融合
        out = self.channel_mixer(combined_features)

        # 如果 stride 为 2，则添加 downsample 操作
        if isinstance(self.downsample, nn.Sequential):
            out = out + self.downsample(x)  # 在 stride=2 时，执行下采样并加到输出

        return out

class StateSpaceModel(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(StateSpaceModel, self).__init__()
        self.hidden_dim = hidden_dim

        # 状态空间模型的状态转移和观测方程
        self.state_transition = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.observation = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)  # 观测输出

        # 动态滤波器权重，模拟Kalman滤波器的状态更新
        self.filter_weight = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        # 假设输入x是一个特征图
        state = self.state_transition(x)  # 状态转移，获得隐含的全局特征表示

        # 使用滤波器更新状态
        state = state * self.filter_weight.view(1, -1, 1, 1)  # 应用滤波器权重

        # 观测方程，输出全局特征
        global_feature = self.observation(state)  # 使用观测层得到全局特征

        return global_feature


class RepKKBlockSSM(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(RepKKBlockSSM, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

        # 新增的状态空间模型部分
        self.ssm = StateSpaceModel(oup, hidden_dim)  # 使用状态空间模型提取全局特征

        # 局部特征提取层
        self.local_conv = nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, groups=oup, bias=False)  # 深度可分离卷积

    def forward(self, x):
        # 通过 token_mixer 和 channel_mixer
        token_features = self.token_mixer(x)
        channel_features = self.channel_mixer(token_features)

        # 局部特征提取（深度可分离卷积）
        local_features = self.local_conv(channel_features)

        # 使用 SSM 提取全局特征
        global_features = self.ssm(channel_features)

        # 融合局部和全局特征
        fused_features = local_features + global_features  # 加权求和（或加和）

        # 返回融合后的特征图
        return fused_features


class FRepKKBlock(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(FRepKKBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pointwise conv with doubled channels for real and imaginary parts
                Conv2d_BN(2 * oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # linear pointwise conv
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pointwise conv with doubled channels for real and imaginary parts
                Conv2d_BN(2 * inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # linear pointwise conv
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        # Token mixing
        x = self.token_mixer(x)

        # Apply 2D Fourier Transform
        x_fft = torch.fft.fft2(x.to(torch.complex64))  # Use complex64 for Fourier transform compatibility

        # Get real and imaginary parts from the inverse Fourier transform
        x_ifft_real = torch.fft.ifft2(x_fft).real  # Real part
        x_ifft_imag = torch.fft.ifft2(x_fft).imag  # Imaginary part

        # Concatenate real and imaginary parts along the channel dimension
        x_ifft = torch.cat((x_ifft_real, x_ifft_imag), dim=1)

        # Channel mixing with concatenated real and imaginary features
        return self.channel_mixer(x_ifft)


class AFBlock(nn.Module):
    def __init__(self, dim, hidden_size, num_blocks, double_skip, mlp_ratio=4., drop_path=0., attn_norm_layer='sync_batch_norm', enable_coreml_compatible_fn=False):
        # input shape [B C H W]
        super().__init__()
        # self.norm1 = nn.BatchNorm2d(num_features=dim)
        # self.norm1 = nn.SyncBatchNorm.convert_sync_batchnorm()

        if torch.cuda.device_count() < 1 + 1e-10:    # '1 + 1e-10; is in order to make sure that when more than 1 GPU can use Sync-batch-norm.
            # for a CPU-device, Sync-batch norm does not work. So, change to batch norm
            self.norm1 = nn.BatchNorm2d(num_features=dim)
            self.norm2 = nn.BatchNorm2d(num_features=dim)
            logger.info("Using BatchNorm2d")
        else:
            self.norm1 = SyncBatchNorm(normalized_shape=dim, num_features=dim)
            self.norm2 = SyncBatchNorm(normalized_shape=dim, num_features=dim)
            logger.info("Using SyncBatchNorm")
        self.filter = AFNO2D_channelfirst(hidden_size=hidden_size, num_blocks=num_blocks, sparsity_threshold=0.01,
                                          hard_thresholding_fraction=1, hidden_size_factor=1) # if not enable_coreml_compatible_fn else \
            # AFNO2D_channelfirst_coreml(hidden_size=hidden_size, num_blocks=num_blocks, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = InvertedResidual(
            inp=dim,
            oup=dim,
            stride=1,
            expand_ratio=mlp_ratio,
        )
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        # print(f"Block中传入的x.shape：{x.shape}")
        x = self.norm1(x)
        # print(x.shape)
        # x = self.filter(x)
        x = self.mlp(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        # x = self.mlp(x)
        x = self.filter(x)
        x = self.drop_path(x)
        x = x + residual
        return x

    def profile_module(
            self, input: Tensor, *args, **kwargs
        ) -> Tuple[Tensor, float, float]:
        b_sz, c, h, w = input.shape
        seq_len = h * w

        out, p_ffn, m_ffn = module_profile(module=self.mlp, x=input)
        # m_ffn = m_ffn * b_sz * seq_len

        out, p_mha, m_mha = module_profile(module=self.filter, x=out)


        macs = m_mha + m_ffn
        params = p_mha + p_ffn

        return input, params, macs

#
# # 定义轻量化的基础卷积层
# def Conv2d_BN(inp, oup, kernel_size, stride, padding, groups=1):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True)
#     )


# 定义 SE 注意力机制
class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=0.25):
        super(SqueezeExcite, self).__init__()
        reduced_channels = max(1, int(channels * reduction))
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = x.mean((2, 3), keepdim=True)  # 全局平均池化
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        return x * self.sigmoid(se)


# 定义 Residual 模块
class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


# 定义 InvertedResidual（倒残差块）
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, kernel_size=1, stride=1, padding=0))
        layers.extend([
            # dw
            Conv2d_BN(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# 定义主 HybridBlock 模块
class HybridBlock(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se=True, use_hs=True):
        super(HybridBlock, self).__init__()
        self.identity = stride == 1 and inp == oup
        self.use_se = use_se

        # Token mixer (局部特征提取部分) - RepKKBlock部分
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity()
            )

        # 全局特征提取（使用 Block 的频域操作）
        self.global_mixer = Residual(nn.Sequential(
            Conv2d_BN(oup, hidden_dim, 1, 1, 0),
            nn.GELU() if use_hs else nn.ReLU(),
            Conv2d_BN(hidden_dim, oup, 1, 1, 0),
        ))

        # 倒残差结构，进一步提升特征提取能力
        self.channel_mixer = InvertedResidual(inp=oup, oup=oup, stride=1, expand_ratio=4)

    def forward(self, x):
        # Token mixer 部分（局部特征）
        x = self.token_mixer(x)

        # Global mixer 部分（全局特征）
        x = self.global_mixer(x)

        # Channel mixer 部分
        x = self.channel_mixer(x)

        return x