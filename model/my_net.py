import math
import torch
import torch.nn.functional as F
import os
import torch.nn as nn
import torchsnooper

from model.ru_part import RU_up, outconv, RU_up_MCD
from model.segformer_backbone import mit_b5, mit_b5_MCD
from model.fusion_part import SACA_Gate_V3


class Coarse_SegRes(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, dropout_rate=0.5, is_cam=False):
        super(Coarse_SegRes, self).__init__()
        self.is_cam = is_cam
        self.res_net = mit_b5_MCD(pretrained=False, dropout_rate=dropout_rate)

        self.up4 = RU_up_MCD(in_ch=512, in_ch_skip=320, out_ch=256)
        self.up3 = RU_up_MCD(in_ch=256, in_ch_skip=128, out_ch=128)
        self.up2 = RU_up_MCD(in_ch=128, in_ch_skip=64, out_ch=64)
        self.up1 = RU_up_MCD(in_ch=64, in_ch_skip=0, out_ch=16, with_skip=False)

        self.out_conv1 = outconv(16, n_classes)

    # @torchsnooper.snoop()
    def forward(self, x):
        H, W = x.size(2), x.size(3)
        res_net_outs = self.res_net(x)
        x1 = res_net_outs[0]
        x2 = res_net_outs[1]
        x3 = res_net_outs[2]
        x4 = res_net_outs[3]

        # 上采样支路
        out4 = self.up4(x4, x3)
        out3 = self.up3(out4, x2)
        out2 = self.up2(out3, x1)
        out1 = self.up1(out2, None)
        out_x_1 = self.out_conv1(out1)
        out_x_1 = F.interpolate(out_x_1, size=(H, W), mode='bilinear', align_corners=True)

        if self.is_cam:
            return torch.sigmoid(out_x_1).float()
        else:
            return {
                "region_out": out_x_1
            }


class Segformer_Drop_Branch_Without_Cnet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, feature=False, dk_rate=[], mode='dim_key'):
        super(Segformer_Drop_Branch_Without_Cnet, self).__init__()
        self.feature = feature
        self.res_net = mit_b5(pretrained=False, drop_rate=dk_rate, mode=mode)

        self.up4 = RU_up(in_ch=512, in_ch_skip=320, out_ch=256)
        self.up3 = RU_up(in_ch=256, in_ch_skip=128, out_ch=128)
        self.up2 = RU_up(in_ch=128, in_ch_skip=64, out_ch=64)
        self.up1 = RU_up(in_ch=64, in_ch_skip=0, out_ch=16, with_skip=False)

        self.out_conv1 = outconv(16, n_classes)

    # @torchsnooper.snoop()
    def forward(self, x, uncertain_map):
        H, W = x.size(2), x.size(3)
        res_net_outs = self.res_net(x, uncertain_map)
        x1 = res_net_outs[0]
        x2 = res_net_outs[1]
        x3 = res_net_outs[2]
        x4 = res_net_outs[3]

        # upsampling
        out4 = self.up4(x4, x3)
        out3 = self.up3(out4, x2)
        out2 = self.up2(out3, x1)
        out1 = self.up1(out2, None)
        out_x_1 = self.out_conv1(out1)
        out_x_1 = F.interpolate(out_x_1, size=(H, W), mode='bilinear', align_corners=True)
        if self.feature:
            return {
                "feature": [out4, out3, out2, out1],
                "region_out": out_x_1,
                "en_feature": [x1, x2, x3, x4]
            }
        else:
            return {
                "region_out": out_x_1,
            }


# =========EH-Former=========== #
class Fine_SegRes_EnFuse_SACAV3_Independent(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, dropout_rate=0.5, n_sample=6, model_stage1='SegRes', dk_rate=[], mode='dim_key'):
        super(Fine_SegRes_EnFuse_SACAV3_Independent, self).__init__()
        # 粗网络
        self.c_net = Coarse_SegRes(n_channels=3, n_classes=1, dropout_rate=dropout_rate)
        self.c_net.cuda()
        # 冻结粗网络的权重
        for param in self.c_net.parameters():
            param.requires_grad = False
        self.n_sample = n_sample

        self.fuse1 = SACA_Gate_V3(64, 64, 0, 1, 'encoder')
        self.fuse2 = SACA_Gate_V3(128, 128, 64, 2, 'encoder')
        self.fuse3 = SACA_Gate_V3(320, 320, 128, 3, 'encoder')
        self.fuse4 = SACA_Gate_V3(512, 512, 320, 4, 'encoder')

        self.segres_net = Segformer_Drop_Branch_Without_Cnet(dk_rate=dk_rate, feature=True, mode=mode)

        self.segres_net_opposite = Segformer_Drop_Branch_Without_Cnet(dk_rate=dk_rate, feature=True, mode=mode)

        self.up4 = RU_up(in_ch=512, in_ch_skip=320, out_ch=256)
        self.up3 = RU_up(in_ch=256, in_ch_skip=128, out_ch=128)
        self.up2 = RU_up(in_ch=128, in_ch_skip=64, out_ch=64)
        self.up1 = RU_up(in_ch=64, in_ch_skip=0, out_ch=16, with_skip=False)

        self.out_conv1 = outconv(16, n_classes)

    # @torchsnooper.snoop()
    def forward(self, x):
        H, W = x.size(2), x.size(3)
        if self.training:
            with torch.no_grad():
                # calculate uncertainty map
                sample_list = [torch.sigmoid(self.c_net(x)['region_out']) for i in range(self.n_sample)]
                entropy_list = [-torch.sum(p * torch.log(p + 1e-9), dim=1, keepdim=True) for p in sample_list]
                var_sample = torch.var(torch.stack(sample_list, dim=0), keepdim=True, axis=0)  # 通过方差来刻画不确定性图
                var = var_sample.squeeze(0).cuda()
                mean_sample = torch.mean(torch.stack(entropy_list, dim=0), keepdim=True, axis=0)  # 通过熵均值来刻画不确定性图
                mean = mean_sample.squeeze(0).cuda()
                var = var / torch.max(var)
                mean = mean / torch.max(mean)
                uncertain_map = ((var + mean) / 2).to(x.device)
                uncertain_map_opposite = torch.max(uncertain_map) - uncertain_map
        else:
            uncertain_map = None
            uncertain_map_opposite = None

        # easy stream
        segres_net_outs = self.segres_net(x, uncertain_map)
        x1 = segres_net_outs['en_feature'][0]
        x2 = segres_net_outs['en_feature'][1]
        x3 = segres_net_outs['en_feature'][2]
        x4 = segres_net_outs['en_feature'][3]
        uc_map = uncertain_map
        segres_net_result = segres_net_outs['region_out']

        # hard stream
        segres_net_opposite = self.segres_net_opposite(x, uncertain_map_opposite)
        x1_res = segres_net_opposite['en_feature'][0]
        x2_res = segres_net_opposite['en_feature'][1]
        x3_res = segres_net_opposite['en_feature'][2]
        x4_res = segres_net_opposite['en_feature'][3]
        res_net_result = segres_net_opposite['region_out']

        # fusion
        x1_fuse = self.fuse1(x1, x1_res)
        x1_fuse_out = x1_fuse["out"]
        x2_fuse = self.fuse2(x2, x2_res, x1_fuse_out)
        x2_fuse_out = x2_fuse["out"]
        x3_fuse = self.fuse3(x3, x3_res, x2_fuse_out)
        x3_fuse_out = x3_fuse["out"]
        x4_fuse = self.fuse4(x4, x4_res, x3_fuse_out)
        x4_fuse_out = x4_fuse["out"]

        out4 = self.up4(x4_fuse_out, x3_fuse_out)
        out3 = self.up3(out4, x2_fuse_out)
        out2 = self.up2(out3, x1_fuse_out)
        out1 = self.up1(out2, None)
        out_x_1 = self.out_conv1(out1)
        out_x_1 = F.interpolate(out_x_1, size=(H, W), mode='bilinear', align_corners=True)

        return {
            "region_out": out_x_1,
            "uc_map": uc_map,
            "segres_net_result": segres_net_result,
            "res_net_result": res_net_result,
            "segres_spatial_weight": [x1_fuse["x_weight"], x2_fuse["x_weight"],
                                      x3_fuse["x_weight"], x4_fuse["x_weight"]],
            "res_spatial_weight": [x1_fuse["x_opposite_weight"], x2_fuse["x_opposite_weight"],
                                   x3_fuse["x_opposite_weight"], x4_fuse["x_opposite_weight"]]
        }

    def load_weights(self, path_to_weights):
        pretrained_dict = torch.load(path_to_weights, map_location=torch.device('cpu'))
        self.c_net.load_state_dict(pretrained_dict)


if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    X = torch.randn(1, 3, 224, 224).to(device)
    net = Fine_SegRes_EnFuse_SACAV3_Independent().to(device)
    checkpoint = torch.load(load_path + '/seg_weights.pth', map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    set_params_recursive(net, checkpoint['alpha'])

    out = net(X)
