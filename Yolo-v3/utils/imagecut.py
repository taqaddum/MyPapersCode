import pandas as pd
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


class OverlapCrop:
    def __init__(self, img, wind_size, overlap_ratio):
        """图像裁剪，有重叠区域

        Parameters
        ----------
        image : PIL
            PIL图片
        wind_size : tuple
            窗口宽高
        overlap_ratio : tuple
            重叠宽/高占滑窗宽/高比例
        """
        assert 0 < overlap_ratio[0] < 1.0, "tuple[0]必须为(0,1)之间的浮点数"
        assert 0 < overlap_ratio[1] < 1.0, "tuple[1]必须为(0,1)之间的浮点数"
        assert isinstance(img, Image.Image), "不是PIL格式"

        self.img = img
        self.wind = wind_size
        self.overlap = (
            int(wind_size[0] * overlap_ratio[0]),
            int(wind_size[1] * overlap_ratio[1]),
        )
        self.boxes = self.get_box()

    def get_box(self):
        """生成滑窗

        Parameters
        ----------
        Returns
        -------
        list
            返回滑窗左上，右下坐标
        """
        im_w = max(self.img.size[0], self.wind[0])
        im_h = max(self.img.size[1], self.wind[1])
        stride_w = self.wind[0] - self.overlap[0]
        stride_h = self.wind[1] - self.overlap[1]
        boxes = list()

        for i in range(self.wind[1], im_h + stride_h, stride_h):
            for j in range(self.wind[0], im_w + stride_w, stride_w):
                right = min(j, im_w)
                lower = min(i, im_h)
                left = right - self.wind[0]
                upper = lower - self.wind[1]
                boxes.append((left, upper, right, lower))
        return boxes

    def crop_save(self, imgpath, targetdir):
        """存储裁剪图片

        Parameters
        ----------
        imgpath : Path
            PIL文件路径
        targetdir : Path
            存储目录
        """
        assert imgpath.is_file(), "文件名错误或文件不存在"
        assert targetdir.is_dir(), "路径不存在"

        df = pd.DataFrame(columns=["image", "left", "upper", "right", "lower"])
        boxes = self.boxes
        iters = tqdm(boxes)
        for n, box in enumerate(iters):
            iters.set_description(f"{imgpath.name}，第{n + 1}次裁剪")
            filename = imgpath.stem + f"_{n + 1:03d}" + imgpath.suffix
            filedir = targetdir.joinpath(filename)
            df.loc[n] = [filename, *box]
            temp = self.img.crop(box)
            temp.save(filedir)
            del temp
        df.to_csv(targetdir.joinpath(imgpath.stem + "_location.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="有重叠的图片分块", allow_abbrev=True)
    parser.add_argument("source", help="图片源路径", type=str)
    parser.add_argument("--targetdir", help="裁剪后存放路径", default="./data/Image", type=str)
    args = parser.parse_args()
    dirs = Path(args.source)
    target = Path(args.targetdir)
    assert dirs.exists(), "路径不存在"
    paths = list(dirs.glob("*.JPG".lower()))

    wind_size = (544, 544)
    overlap_ratio = (0.2, 0.2)

    for path in paths:
        targetdir = target.joinpath(path.stem)
        targetdir.mkdir(parents=True, exist_ok=True)
        image = Image.open(path)
        crop = OverlapCrop(image, wind_size, overlap_ratio)
        crop.crop_save(path, targetdir)
        image.close()
