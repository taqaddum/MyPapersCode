from collections import OrderedDict

import torch.nn as nn


class CBLResidual(nn.Module):
    def __init__(self, inchannel, outchannel) -> None:
        super().__init__()
        # 1x1卷积核降维
        self.cbl_1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(0.1),
        )
        # 3x3卷积核升维
        self.cbl_2 = nn.Sequential(
            nn.Conv2d(
                outchannel,
                inchannel,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="replicate",
            ),
            nn.BatchNorm2d(inchannel),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        # 残差运算
        y = self.cbl_2(self.cbl_1(x))
        return x + y


class DarkNet53(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        channels = [32, 64, 128, 256, 512, 1024]
        layers = [1, 2, 8, 8, 4]
        self.cbl = nn.Sequential(
            nn.Conv2d(
                channel,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="replicate",
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )
        for i in range(len(channels) - 1):
            layer = self._mklayers(channels[i], channels[i + 1], layers[i])
            setattr(self, f"resblock{i}", layer)

    def _mklayers(self, inchannel, outchannel, num):
        cbl = nn.Sequential(
            nn.Conv2d(
                inchannel,
                outchannel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                padding_mode="replicate",
            ),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(0.1),
        )

        resblock = nn.Sequential(OrderedDict([("cbl", cbl)]))
        for i in range(num):
            resblock.add_module(f"resunit{i}", CBLResidual(outchannel, inchannel))
        return resblock

    def forward(self, x):
        prior = self.resblock1(self.resblock0(self.cbl(x)))
        scal_small = self.resblock2(prior)  # 下采样8倍的小尺度feature map
        scal_middle = self.resblock3(scal_small)  # 下采样16倍的中尺度feature map
        scal_large = self.resblock4(scal_middle)  # 下采样32倍的大尺度feature map
        return scal_small, scal_middle, scal_large
