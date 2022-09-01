from typing import List, Tuple

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor, nn


class YoloLoss:
    def __init__(
        self,
        imgsize: Tuple[int, int],
        anchors: List[Tuple[int, int]],
        batch_gtbox: List[ndarray],
        labels: List[List[int]],
        num_class: int,
        iou_thres: float = 0.5,
    ) -> None:
        self.anchors = sorted(anchors, key=lambda x: x[0] * x[1])
        self.width, self.height = imgsize
        self.num_class = num_class
        self.batch_gtbox = batch_gtbox
        self.labels = labels
        self.iou_thres = iou_thres

    def start(self, feature: Tensor, anchor_no: List[int]):
        """计算总损失

        Parameters
        ----------
        feature : Tensor
            某一尺度下的特征映射
        anchor_no : List[int]
            分配给当前尺度的锚框编号

        Returns
        -------
        Tensor
            返回总损失值
        """
        fsize = feature.size()  # (batch, channel, rows, columns)
        stride = (self.height // fsize[-2], self.width // fsize[-1])
        anchors = np.asarray([(w / stride[1], h / stride[0]) for w, h in self.anchors])

        predict = torch.permute(
            feature.reshape(
                -1,
                len(anchor_no),
                5 + self.num_class,
                fsize[-2],
                fsize[-1],
            ),
            dims=(0, 1, 3, 4, 2),
        )

        x, y = torch.sigmoid(predict[..., 0]), torch.sigmoid(predict[..., 1])
        w, h = predict[..., 2], predict[..., 3]
        conf = torch.sigmoid(predict[..., 4])
        cls = torch.sigmoid(predict[..., 5:])

        target = self.convertBox(anchors, anchor_no, rows=fsize[-2], columns=fsize[-1])
        obj = target[..., 4]
        noobj = self.getNoobj(predict[..., :5], obj, anchors, anchor_no)
        wmulth = target[..., 2] * target[..., 3]

        bceloss = nn.BCELoss(reduction="none")
        mseloss = nn.MSELoss(reduction="none")

        xyloss = (
            obj
            * (2 - wmulth)
            * (bceloss(x, target[..., 0]) + bceloss(y, target[..., 1]))
        )
        whloss = (
            obj
            * (2 - wmulth)
            * (mseloss(w, target[..., 2]) + mseloss(h, target[..., 3]))
        )
        confloss = obj * bceloss(conf, obj) + noobj * bceloss(conf, obj)
        clsloss = bceloss(cls[obj == 1], target[..., 5:][obj == 1])
        loss = (
            0.5 * torch.sum(xyloss)
            + 0.5 * torch.sum(whloss)
            + torch.sum(confloss)
            + torch.sum(clsloss)
        )
        return loss

    def convertBox(
        self,
        anchors: ndarray,
        anchor_no: List[int],
        rows: int,
        columns: int,
    ):
        """转换真实框使其与输出张量相匹配

        Parameters
        ----------
        anchors : ndarray
            缩放后的锚框
        anchor_no : List[int]
            分配给当前尺度的锚框编号
        rows : int
            当前尺度下的特征图行数
        columns : int
            当前尺度下的特征图列数

        Returns
        -------
        Tensor
            返回转换后的真实框
        """
        bs = len(self.batch_gtbox)
        anchor_num = len(anchor_no)
        target = np.zeros((bs, anchor_num, rows, columns, 5 + self.num_class))
        anchorbox = np.concatenate([np.zeros((anchor_num, 2)), anchors], axis=1)

        for b, gtbox in enumerate(self.batch_gtbox):
            if len(gtbox):
                gtbox[:, [0, 2]] *= columns
                gtbox[:, [1, 3]] *= rows
                tranfed_gtbox = np.c_[np.zeros(len(gtbox), 2), gtbox[:, 2], gtbox[:, 3]]
                gtbox_anchor = np.argmax(
                    self.calculateIou(anchorbox, tranfed_gtbox), axis=1
                )

                for k, anchorid in enumerate(gtbox_anchor):
                    if anchorid in anchor_no:
                        n = anchor_no.index(anchorid)
                        tx, i = np.modf(gtbox[:, 0])
                        ty, j = np.modf(gtbox[:, 1])
                        i, j = i.astype(int), j.astype(int)
                        tw = np.log(gtbox[:, 2] / anchors[anchorid, 0])
                        th = np.log(gtbox[:, 3] / anchors[anchorid, 1])
                        target[b, n, j, i, 0] = tx
                        target[b, n, j, i, 1] = ty
                        target[b, n, j, i, 2] = tw
                        target[b, n, j, i, 3] = th
                        target[b, n, j, i, 4] = 1
                        target[b, n, j, i, 5 + self.labels[b][k]] = 1
        return torch.from_numpy(target).cuda()

    @staticmethod
    def calculateIou(anchorbox: ndarray, gtbox: ndarray):
        """计算交并比

        Parameters
        ----------
        anchorbox : ndarray
            锚框，格式为xyxy
        gtbox : ndarray
            真实框，格式为xyxy

        Returns
        -------
        ndarray
            返回二维数组，行是真实框编号，列是锚框编号
        """
        anchorbox_wh = anchorbox[:, [2, 3]] - anchorbox[:, [0, 1]]
        gtbox_wh = gtbox[:, [2, 3]] - gtbox[:, [0, 1]]
        area_anchorbox = anchorbox_wh[:, 0] * anchorbox_wh[:, 1]
        area_gtbox = gtbox_wh[:, 0] * gtbox_wh[:, 1]
        union_area = area_gtbox[:, None] + area_anchorbox

        left_upper = np.maximum(anchorbox[:, [0, 1]], gtbox[:, None, [0, 1]])
        right_bottom = np.minimum(anchorbox[:, [2, 3]], gtbox[:, None, [2, 3]])
        inter_wh = np.clip(right_bottom - left_upper, 0, None)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]
        return inter_area / (union_area - inter_area)

    def getNoobj(
        self, data: Tensor, obj: Tensor, anchors: ndarray, anchor_no: List[int]
    ):
        """过滤IOU大于阈值，但不是最大值的预测框

        Parameters
        ----------
        data : Tensor
            网络输出张量，(bs, anchor_num, rows, columns, 5)
        obj : Tensor
            有目标网格系数
        anchors : ndarray
            放大后的锚框
        anchor_no : List[int]
            尺度对应锚框编号
        Returns
        -------
        Tensor
            返回无目标网格的对应系数
        """
        predict: ndarray = data.clone().detach().numpy()
        noobj: ndarray = ~obj.clone().detach().numpy()
        bx, by = np.meshgrid(range(noobj.shape[-1]), range(noobj.shape[-2]))
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        predict[..., 0] = sigmoid(predict[..., 0]) + bx[None, None, ...]
        predict[..., 1] = sigmoid(predict[..., 1]) + by[None, None, ...]
        predict[..., 2] = (
            np.exp(predict[..., 2]) * anchors[None, anchor_no, 0, None, None]
        )
        predict[..., 3] = (
            np.exp(predict[..., 3]) * anchors[None, anchor_no, 1, None, None]
        )
        predict[..., [0, 1]] = predict[..., [0, 1]] - predict[..., [2, 3]] / 2
        predict[..., [2, 3]] = predict[..., [2, 3]] + predict[..., [0, 1]]

        for i, gtbox in self.batch_gtbox:
            gtbox[:, [0, 2]] *= noobj.shape[-1]
            gtbox[:, [1, 3]] *= noobj.shape[-2]
            gtbox[:, [0, 1]] = gtbox[:, [0, 1]] - gtbox[:, [2, 3]] / 2
            gtbox[:, [2, 3]] = gtbox[:, [2, 3]] + gtbox[:, [0, 1]]

            iouarr: ndarray = np.max(
                self.calculateIou(predict[i].reshape(-1, 4), gtbox), axis=1
            )
            ioumat = iouarr.reshape(noobj.shape[1:])
            noobj[i][ioumat > self.iou_thres] = 0
        return torch.from_numpy(noobj)
