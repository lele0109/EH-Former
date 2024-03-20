import baal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper


class RU_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RU_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x


class RU_double_conv_MCD(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.5):
        super(RU_double_conv_MCD, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            baal.bayesian.dropout.Dropout(p=dropout_rate),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x


class RU_up(nn.Module):
    def __init__(self, out_ch, in_ch, in_ch_skip=0, bilinear=False, with_skip=True):
        super(RU_up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        #  nn.Upsample hasn't weights to learn, but nn.ConvTransposed2d has weights to learn.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            if in_ch_skip == 0 and with_skip:
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            else:
                self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = RU_double_conv(in_ch + in_ch_skip, out_ch)
        self.relu = nn.ReLU(inplace=True)
        group_num = 32
        if out_ch % 32 == 0 and out_ch >= 32:
            if out_ch % 24 == 0:
                group_num = 24
        elif out_ch % 16 == 0 and out_ch >= 16:
            if out_ch % 16 == 0:
                group_num = 16
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch + in_ch_skip, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, out_ch))
        self.with_skip = with_skip

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.with_skip:
            diff_x = x2.size()[-2] - x1.size()[-2]
            diff_y = x2.size()[-1] - x1.size()[-1]

            x1 = F.pad(x1, (diff_y, 0, diff_x, 0))
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)

        return r1


class RU_up_MCD(nn.Module):
    def __init__(self, out_ch, in_ch, in_ch_skip=0, bilinear=False, with_skip=True, dropout_rate=0.5):
        super(RU_up_MCD, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        #  nn.Upsample hasn't weights to learn, but nn.ConvTransposed2d has weights to learn.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            if in_ch_skip == 0 and with_skip:
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            else:
                self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = RU_double_conv_MCD(in_ch + in_ch_skip, out_ch, dropout_rate=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        group_num = 32
        if out_ch % 32 == 0 and out_ch >= 32:
            if out_ch % 24 == 0:
                group_num = 24
        elif out_ch % 16 == 0 and out_ch >= 16:
            if out_ch % 16 == 0:
                group_num = 16
        # print(out_ch, group_num)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch + in_ch_skip, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, out_ch))
        self.with_skip = with_skip

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.with_skip:
            diff_x = x2.size()[-2] - x1.size()[-2]
            diff_y = x2.size()[-1] - x1.size()[-1]

            x1 = F.pad(x1, (diff_y, 0, diff_x, 0))
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)

        return r1


# !!!!!!!!!!!! Universal functions !!!!!!!!!!!!

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

