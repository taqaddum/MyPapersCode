import torch
from torch import nn
from torch.utils import data
from torchvision import transforms
from pathlib import Path

# todo:构造yolo格式数据集，包含(图像, 真实框)
class YoloDataset(data.Dataset):
    def __init__(self) -> None:
        super().__init__()

