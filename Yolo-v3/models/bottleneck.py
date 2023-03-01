import torch
import torch.nn as nn


class ConvBL(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, padding=0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            inchannel,
            outchannel,
            kernel_size,
            bias=False,
            padding=padding,
            padding_mode="replicate",
        )
        self.bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        y = self.bn(self.conv(x))
        out = self.relu(y)
        return out


class SppNet(nn.Module):
    def __init__(self):
        # 融合大尺度
        super().__init__()
        self.backbone = nn.Identity(1024, "backbone_large")
        self.cblset_pre = self._mkcbl(1024, [512, 1024], 3)
        self.spp_1 = nn.MaxPool2d(5, stride=1, padding=2)
        self.spp_2 = nn.MaxPool2d(9, stride=1, padding=4)
        self.spp_3 = nn.MaxPool2d(13, stride=1, padding=6)
        self.concate_spp = nn.Identity(2048, "cat[cblset_pre, spp1, spp2, spp3]")
        self.cblset_large = self._mkcbl(2048, [512, 1024], 3)
        # 融合中尺度
        self.cbl_256 = ConvBL(512, 256, 1)
        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.concate_middle = nn.Identity(768, "cat[upsample1, backbone_middle]")
        self.cblset_middle = self._mkcbl(768, [256, 512], 5)
        # 融合小尺度
        self.cbl_128 = ConvBL(256, 128, 1)
        self.upsample_2 = nn.Upsample(scale_factor=2)
        self.concate_small = nn.Identity(384, "cat[upsample2, backbone_small]")
        self.cblset_small = self._mkcbl(384, [128, 256], 5)

    def _mkcbl(self, inchannel, channels, num):
        cblset = nn.Sequential()
        for i in range(num):
            outchannel = channels[i % 2]
            kp = (3, 1) if i % 2 else (1, 0)
            cblset.add_module(
                f"cbl_num{i}", ConvBL(inchannel, outchannel, kp[0], kp[1])
            )
            inchannel = outchannel
        return cblset

    def forward(self, backbone_small, backbone_middle, backbone_large):
        predata = self.cblset_pre(self.backbone(backbone_large))

        sppset = [
            predata,
            self.spp_1(predata),
            self.spp_2(predata),
            self.spp_3(predata),
        ]
        spp_out = self.concate_spp(torch.concat(sppset, dim=1))
        scal_large = self.cblset_large(spp_out)

        middleset = [self.upsample_1(self.cbl_256(scal_large)), backbone_middle]
        scal_middle = self.cblset_middle(
            self.concate_middle(torch.concat(middleset, dim=1))
        )

        smallset = [self.upsample_2(self.cbl_128(scal_middle)), backbone_small]
        scal_small = self.cblset_small(
            self.concate_small(torch.concat(smallset, dim=1))
        )
        return scal_small, scal_middle, scal_large
