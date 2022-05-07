import numpy as np
import albumentations as A
from PIL import Image
from torch.utils import data


# todo:构造yolo格式数据集，包含(图像, 真实框)
class YoloDataset(data.Dataset):
    def __init__(self, path, shape, train=True, transform=None) -> None:
        super().__init__()
        self.dataset = np.load(path, allow_pickle=True)
        self.shape = shape
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
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
            box = np.array(box)
        else:
            image = np.array(img)
        # 关闭文件流
        orimage.close()
        img.close()
        # yolo格式化
        if len(box):
            box[:, [2, 3]] -= box[:, [0, 1]]
            box[:, [0, 1]] += box[:, [2, 3]] / 2
            box[:, [0, 2]] /= self.shape[1]
            box[:, [1, 3]] /= self.shape[0]
            return image, box, label

    def letterbox(self, origin: Image, result: Image, box: np.array, shape: tuple):
        w, h = origin.size  # resize前的原始尺寸
        nh, nw = shape  # resize并填充后的目标尺寸，既网络固定的输入大小
        scale = min(nh / h, nw / w)  # 缩放因子以最长边为基准且宽高比固定
        rw, rh = int(w * scale), int(h * scale)  # resize后的尺寸
        dw = (nw - rw) // 2
        dh = (nh - rh) // 2

        tempimg = origin.resize((rw, rh), resample=Image.CUBIC)
        result.paste(tempimg, box=(dw, dh))

        box[:, [0, 2]] *= nw / w + dw
        box[:, [1, 3]] *= nh / h + dh
        box[:, [0, 1]][box[:, [0, 1]] < 0] = 0
        box[:, 2][box[:, 2] > nw] = nw
        box[:, 3][box[:, 3] > nh] = nh
        filter = np.logical_and((box[:, 2] - box[:, 0]) > 0, (box[:, 3] - box[:, 1]) > 0)
        box = box[filter]

    def augmention(self, image: Image, bboxes, label, transform):
        image = np.array(image)
        if transform is None:
            bbox_params = A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            )
            flip = A.OneOf([A.HorizontalFlip(), A.VerticalFlip(), A.Rotate(),])
            blur = A.OneOf([A.MotionBlur(), A.MedianBlur(), A.RandomFog()])
            transform = A.Compose(
                [flip, A.ColorJitter(), A.RandomGamma(), A.ISONoise(), blur],
                bbox_params=bbox_params,
            )
        transformed = transform(image=image, bboxes=bboxes, class_labels=label)
        return transformed["image"], transformed["bboxes"], transformed["class_labels"]
