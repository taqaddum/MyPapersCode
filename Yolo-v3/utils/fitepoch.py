import torch
from torch.optim import Optimizer
from torch import Tensor
from tqdm import tqdm
from utils.lossfunc import YoloLoss
from utils.datasets import YoloDataset
from models import YOLOSpp
from torch.utils import data
from typing import List, Tuple


def collate(items: Tuple):
    images = list()
    boxes = list()
    labels = list()

    for img, box, label in items:
        images.append(img)
        boxes.append(box)
        labels.append(label)
    return images, boxes, labels


def move2gpu(batch: data.DataLoader):
    with torch.no_grad():
        imgs = torch.from_numpy(batch[0]).cuda()
        boxes = torch.from_numpy(batch[1]).cuda()
        labels = batch[2]
    return imgs, boxes, labels


def oneEpoch(
    imgs: Tensor,
    boxes: Tensor,
    labels: List[List[int]],
    lr: float,
    optimizer: Optimizer,
    epoch: int,
    lossfunc: YoloLoss,
    model: YOLOSpp,
):
    model.train()
    x = model(imgs)
