from torch import nn
from .bottleneck import ConvBL


class Detector(nn.Module):
    def __init__(self, num_classes, num_anchor):
        super().__init__()
        channel = int((num_classes + 5) * num_anchor)

        conv256 = nn.Conv2d(256, channel, 1, bias=False)
        conv512 = nn.Conv2d(512, channel, 1, bias=False)
        conv1024 = nn.Conv2d(1024, channel, 1, bias=False)

        self.small = nn.Sequential(ConvBL(128, 256, 3, 1), conv256)
        self.middle = nn.Sequential(ConvBL(256, 512, 3, 1), conv512)
        self.large = nn.Sequential(ConvBL(512, 1024, 3, 1), conv1024)

    def forward(self, neck_small, neck_middle, neck_large):
        scal_small = self.small(neck_small)
        scal_middle = self.middle(neck_middle)
        scal_large = self.large(neck_large)
        return scal_small, scal_middle, scal_large
