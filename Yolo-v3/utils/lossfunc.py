from typing import List, Tuple

import torch
import numpy as np
from numpy import ndarray
from torch import Tensor, nn, optim


class YoloLoss(nn.Module):
    def __init__(self, anchors, anchor_num, inshape, num_class, iou_thres=0.5) -> None:
        self.anchors = sorted(anchors, key=lambda x: x[0] * x[1])
        self.width, self.height = inshape
        self.num_class = num_class
        self.anchor_num = anchor_num

    def forward(self, feature: Tensor, targets: List[ndarray]):
        featuresize = feature.size()  # (batch, channel, height, width)
        recepsize = (self.height // featuresize[-2], self.width // featuresize[-1])
        nwanchors = [(w / recepsize[1], h / recepsize[0]) for w, h in self.anchors]

        predict = torch.permute(
            feature.reshape(
                -1,
                len(self.anchor_num),
                5 + self.num_class,
                featuresize[-2],
                featuresize[-1],
            ),
            dims=(0, 1, 3, 4, 2),
        )

        x, y = predict[..., 0], predict[..., 1]
        w, h = predict[..., 2], predict[..., 2]
        conf = predict[..., 4]
        cls = predict[..., 5:]

    def objMask(self, target):
        ...

    def fetchIou(self, anchors, bbox):
        anchors_wh = anchors[:, [2, 3]] - anchors[:, [0, 1]]
        bbox_wh = bbox[:, [2, 3]] - bbox[:, [0, 1]]
        area_anchors = anchors_wh[:, 0] * anchors_wh[:, 1]
        area_bbox = bbox_wh[:, 0] * bbox_wh[:, 1]
        union_area = area_bbox[:, None] + area_anchors

        left_upper = np.maximum(anchors[:, [0, 1]], bbox[:, None, [0, 1]])
        right_bottom = np.minimum(anchors[:, [2, 3]], bbox[:, None, [2, 3]])
        inter_wh = np.clip(right_bottom-left_upper, 0, None)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        return inter_area / (union_area - inter_area)