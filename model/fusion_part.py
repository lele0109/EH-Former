import torch
import torch.nn as nn
import numpy as np
import torchsnooper
from timm.models.layers import DropPath
import torch
import torch.nn.functional as F


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)


class Gate_Gap(nn.Module):
    def __init__(self, input_dim, input_dim_fuse, out_dim, fuse_channel, number):
        super(Gate_Gap, self).__init__()
        self.gate_network = nn.Sequential(
            nn.Conv2d(input_dim + input_dim_fuse + fuse_channel, out_dim, kernel_size=1, padding=0),  # xuyaoxiugai
            nn.Sigmoid()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.number = number

    def forward(self, encoder_output_high, encoder_output_low, ex_fuse):
        if self.number > 1:
            combined = torch.cat((encoder_output_high, encoder_output_low, ex_fuse), dim=1)
        else:
            combined = torch.cat((encoder_output_high, encoder_output_low), dim=1)

        combined = self.gap(combined)
        gate_weights = self.gate_network(combined)

        return gate_weights


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class SACA_Gate_V3(nn.Module):
    def __init__(self, uc_channel, out_channel, fuse_channel, number, mode):
        super().__init__()
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.ca = ChannelAttention(uc_channel)
        self.out_conv = conv(uc_channel, out_channel, 3, relu=True)
        self.number = number
        self.mode = mode
        if self.number > 1:
            self.sample = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fuse_conv = conv(fuse_channel, out_channel, 3, relu=True)
            self.gate = Gate_Gap(uc_channel, uc_channel, uc_channel, out_channel, number)
        else:
            self.gate = Gate_Gap(uc_channel, uc_channel, uc_channel, 0, number)

        self.alpha = None

    def forward(self, x_uc, x_opposite, x_ex_fuse=None):
        if self.number > 1:
            if self.mode == 'encoder':
                x_ex_fuse = self.fuse_conv(x_ex_fuse)
                x_ex_fuse = self.sample(x_ex_fuse)
            elif self.mode == 'decoder':
                b, c, h, w = x_uc.shape
                x_ex_fuse = self.fuse_conv(x_ex_fuse)
                x_ex_fuse = F.interpolate(x_ex_fuse, size=(h, w), mode='bilinear', align_corners=True)
        x_uc_weight = self.sa1(x_uc)
        x_opposite_weight = self.sa2(x_opposite)
        gate_weights = self.gate(x_opposite, x_uc, x_ex_fuse)
        x_out = self.alpha * (1 - gate_weights) * x_uc_weight * x_uc + \
                (1 - self.alpha) * gate_weights * x_opposite_weight * x_opposite

        x_out = self.ca(x_out) * x_out
        x_out = self.out_conv(x_out)
        if self.number > 1:
            x_out = torch.add(x_out, x_ex_fuse)
        return {
            "out": x_out,
            "x_weight": F.interpolate(x_uc_weight, size=(224, 224), mode='bilinear', align_corners=True),
            "x_opposite_weight": F.interpolate(x_opposite_weight, size=(224, 224), mode='bilinear', align_corners=True)
        }
