from typing import List, Tuple

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torchvision import ops


class DecodeBox:
    def __init__(
        self,
        anchors: List[Tuple[int, int]],
        anchor_num: List[int],
        class_num: int,
        imgsize: Tuple[int, int],
        inputshape: Tuple[int, int],
    ) -> None:
        self.anchors = sorted(anchors, key=lambda x: x[0] * x[1])
        self.anchor_num = anchor_num
        self.class_num = class_num
        self.imwidth, self.imheight = imgsize
        self.shapew, self.shapeh = inputshape

    def boxformat(self, features):
        assert len(features) != len(self.anchor_num), "锚框数不匹配！！！"

        fore = 0
        outcome = list()
        for i, feature in enumerate(self.features):
            rows, columns = feature.shape[-2], feature.shape[-1]

            rear = fore + self.anchor_num[i]
            step = self.inputshape[0] // rows
            anchorwh = np.array(
                [(aw / step, ah / step) for aw, ah in self.anchors[fore:rear]]
            )
            fore = rear

            bboxattrs: Tensor = feature.reshape(
                -1, len(anchorwh), 5 + self.class_num, rows, columns
            )
            bboxattrs = bboxattrs.permute(0, 1, 3, 4, 2)
            dims = bboxattrs.shape

            tx, ty = bboxattrs[..., 0], bboxattrs[..., 1]
            gridxy = np.meshgrid(range(rows), range(columns))
            bx = torch.from_numpy(np.tile(gridxy[0].astype(float), (*dims[:-1], 1)))
            by = torch.from_numpy(np.tile(gridxy[1].astype(float), (*dims[:-1], 1)))

            tw, th = bboxattrs[..., 2], bboxattrs[..., 3]
            pw = torch.from_numpy(anchorwh[:, 0, None, None])
            ph = torch.from_numpy(anchorwh[:, 1, None, None])

            conf = torch.sigmoid(bboxattrs[..., 4])
            cls = torch.sigmoid(bboxattrs[..., 5:])
            x = torch.sigmoid(tx.data) + bx
            y = torch.sigmoid(ty.data) + by
            w = torch.exp(tw.data) * pw
            h = torch.exp(th.data) * ph

            xywhc = torch.stack(
                [x / columns, y / rows, w / columns, h / rows, conf], dim=-1
            )
            bbox = torch.cat(
                [xywhc, cls], dim=-1
            )  # (batchsize, anchor_num, h, w, 5+class_num)
            outdata = bbox.reshape(
                -1, self.anchor_num[i] * rows * columns, 5 + self.class_num
            )  # (batchsize, anchor_num * h * w, 5 + class_num)
            outcome.append(outdata.data)
            # done:输出s,m,l三个尺度上的boundbox
        return outcome

    def customnms(self, batchtrait: Tensor, confthres=0.5, iouthres=0.4):
        nmsbox = [None for _ in range(batchtrait.size(0))]
        # 从批量中取出样本
        for i, predict in enumerate(batchtrait):
            # predict.shape=(3*13*13+3*26*26+3*52*52, 5 + class_num)
            clsvalue, clslabel = torch.max(predict[:, 5:], dim=1)
            filter = torch.cat(
                (predict[:, :4], clsvalue * predict[:, 4], clslabel), dim=1
            )  # (x, y, w, h, cls_conf, label)，非极大值抑制在该二维张量上进行

            confbox = filter[filter[:, 4] >= confthres]
            for label in torch.unique(confbox[:, 6]):
                singular = confbox[confbox[:, 6] == label]
                indicies = ops.nms(
                    singular[:, :4], singular[:, 5], iou_threshold=iouthres
                )
                nmsbox[i] = (
                    singular[indicies]
                    if nmsbox[i] is None
                    else torch.cat((nmsbox[i], singular[indicies]))
                )
        return nmsbox

    def yolo2voc(self, bbxy: ndarray, bbwh: ndarray, letterbox: bool = True) -> ndarray:
        if letterbox:
            scal = min(self.shapew / self.imwidth, self.shapeh / self.imheight)
            imwh = np.array((self.imwidth, self.imheight))
            rewh = np.round(imwh * scal)
            shapewh = np.array((self.shapew, self.shapeh))
            factor = shapewh / rewh
            offset = (shapewh - rewh) / 2.0 / shapewh
            bbxy = (bbxy - offset) * factor
            bbwh *= factor

        mincorner = (bbxy - bbwh / 2) * imwh
        maxcorner = (bbxy + bbwh / 2) * imwh
        bbox = np.concatenate((mincorner, maxcorner), axis=-1)
        return bbox

    def getbox(self, features):
        with torch.no_grad():
            outcome = self.boxformat(features)
            nmsbox = self.customnms(torch.cat(outcome, dim=1))
            for i, box in enumerate(nmsbox):
                box = box.numpy()
                nmsbox[i] = self.yolo2voc(box[:2], box[2:4])
        return nmsbox

