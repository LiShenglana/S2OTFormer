# -----------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng, for USOT
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from lib.models.modules import ResNet_plus2, Bottleneck

class ResNet50(nn.Module):
    def __init__(self, used_layers=None):
        super(ResNet50, self).__init__()
        if used_layers is None:
            used_layers = [2, 3, 4]
        self.features = ResNet_plus2(Bottleneck, [3, 4, 6, 3],
                                     used_layers=used_layers)

    def forward(self, x):
        x_stages, x = self.features(x)
        return x_stages, x

class backbone_net_stage(nn.Module):
    def __init__(self, used_layers=None):
        super(backbone_net_stage, self).__init__()
        self.features = ResNet_plus2(Bottleneck, [3, 4, 6, 3],
                                     used_layers=used_layers)


    def forward(self, x, stage = None):
        if stage is 'stage1':
            x = self.features.conv1(x)
            x = self.features.bn1(x)
            x = self.features.relu(x)
            x = self.features.maxpool(x)
            x = self.features.layer1(x)
        elif stage is 'stage2':
            x = self.features.layer2(x)
        elif stage is 'stage3':
            x = self.features.layer3(x)
        return x

# class backbone_net_stage2(nn.Module):
#     def __init__(self, used_layers=None):
#         super(backbone_net_stage2, self).__init__()
#         model_t = ResNet_plus2(Bottleneck, [3, 4, 6, 3],
#                                      used_layers=used_layers)
#         self.features = model_t
#
#     def forward(self, x):
#         x = self.features.layer2(x)
#         return x
#
# class backbone_net_stage3(nn.Module):
#     def __init__(self, used_layers=None):
#         super(backbone_net_stage3, self).__init__()
#         if used_layers is None:
#             used_layers = [2, 3, 4]
#         model_base = ResNet_plus2(Bottleneck, [3, 4, 6, 3],
#                                      used_layers=used_layers)
#         model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.features = model_base
#
#     def forward(self, x):
#         x = self.features.layer3(x)
#         return x


if __name__ == '__main__':
    import torch
    net = ResNet50().cuda()
    # print(net)

    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total params: " + str(k/1e6) + "M")

    search = torch.rand(1, 3, 255, 255)
    search = torch.Tensor(search).cuda()
    out = net(search)
    print("output shape: {}".format(out[1].size()))
