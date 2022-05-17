from typing import List, Tuple

import albumentations as A
import numpy as np
from numpy import ndarray
from PIL import Image
from torch.utils import data


# done:构造yolo格式数据集，包含(图像, 真实框)
class YoloDataset(data.Dataset):
    def __init__(
        self,
        path: str,
        shape: Tuple[int, int],
        train: bool = True,
        transform: A.Compose = None,
    ) -> None:
        """构造yolo格式数据集

        Parameters
        ----------
        path : str
            train_val文件路径
        shape : Tuple[int, int]
            模型规定宽高
        train : bool, optional
            数据增强开关, by default True
        transform : A.Compose, optional
            几何变换方法, by default None
        """
        super().__init__()
        self.dataset = np.load(path, allow_pickle=True)
        self.shape = shape
        self.train = train
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[ndarray, ndarray, List[str]]:
        """构造数据集

        Parameters
        ----------
        index : int
            数据索引

        Returns
        -------
        Tuple[Image.Image, ndarray, List[str]]
            图片，目标边界框，目标类别标签
        """
        # 读取数据并打乱边界框顺序
        index = index % len(self.dataset)
        orimage = Image.open(self.dataset[index][0])
        img = Image.new("RGB", self.shape, (128, 128, 128))
        boxlabel = np.array(
            self.dataset[index][1], dtype=np.dtype("S40, f4, f4, f4, f4")
        )
        np.random.shuffle(boxlabel)
        # 调整样本输入格式
        label = boxlabel["f0"].tolist()
        box = np.c_[boxlabel["f1"], boxlabel["f2"], boxlabel["f3"], boxlabel["f4"]]
        self.letterbox(orimage, img, box, self.shape)
        # 是否执行数据增强
        if self.train:
            image, box, label = self.augmention(img, box, label, self.transform)
            box = np.asarray(box)
        # 像素归一化
        else:
            image = np.asarray(img)
        # yolo归一化
        if len(box):
            box[:, [2, 3]] -= box[:, [0, 1]]
            box[:, [0, 1]] += box[:, [2, 3]] / 2
            box[:, [0, 2]] /= self.shape[1]
            box[:, [1, 3]] /= self.shape[0]
            image = image / 255
        else:
            print('这张图片没有目标！！！')
        # 关闭文件流
        orimage.close()
        img.close()
        return image, box, label

    def letterbox(
        self,
        origin: Image.Image,
        result: Image.Image,
        box: ndarray,
        shape: Tuple[int, int],
    ) -> None:
        """调整图像大小

        Parameters
        ----------
        origin : Image.Image
            输入图像
        result : Image.Image
            输出图像
        box : ndarray
            目标边界框
        shape : Tuple[int, int]
            输出高宽
        """
        w, h = origin.size  # resize前的原始尺寸
        nh, nw = shape  # resize并填充后的目标尺寸，既网络固定的输入大小
        scale = min(nh / h, nw / w)  # 缩放因子以最长边为基准且宽高比固定
        rw, rh = int(w * scale), int(h * scale)  # resize后的尺寸
        dw = (nw - rw) // 2
        dh = (nh - rh) // 2

        tempimg = origin.resize((rw, rh), resample=Image.CUBIC)
        result.paste(tempimg, box=(dw, dh))

        box[:, [0, 2]] *= rw / w + dw
        box[:, [1, 3]] *= rh / h + dh
        box[:, [0, 1]][box[:, [0, 1]] < 0] = 0
        box[:, 2][box[:, 2] > nw] = nw
        box[:, 3][box[:, 3] > nh] = nh
        filter = np.logical_and(
            (box[:, 2] - box[:, 0]) > 1, (box[:, 3] - box[:, 1]) > 1
        )
        box = box[filter]

    def augmention(
        self,
        image: Image.Image,
        bboxes: ndarray,
        label: List[str],
        transform: A.Compose,
    ) -> Tuple[ndarray, List[List[int]], List[str]]:
        """数据增强

        Parameters
        ----------
        image : Image.Image
            输入图像
        bboxes : ndarray
            目标边界框
        label : List[str]
            目标类别
        transform : A.Compose
            几何变换方法

        Returns
        -------
        Tuple[ndarray, List[List[int]], List[str]]
            变换后的输出图像，变换后的边界框，变换后的类别标签
        """
        image = np.array(image)
        if transform is None:
            bbox_params = A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            )
            flip = A.Sequential([A.HorizontalFlip(), A.VerticalFlip(), A.Rotate()])
            disturb = A.OneOf([A.ISONoise(), A.RandomFog()])
            transform = A.Compose(
                [flip, A.ColorJitter(), A.MotionBlur(), disturb],
                bbox_params=bbox_params,
            )
        transformed = transform(image=image, bboxes=bboxes, class_labels=label)
        return transformed["image"], transformed["bboxes"], transformed["class_labels"]
