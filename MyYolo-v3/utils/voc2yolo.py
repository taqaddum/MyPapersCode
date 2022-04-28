from xml.etree import ElementTree as ET
from pathlib import Path

import numpy as np
import argparse


def argument():
    parser = argparse.ArgumentParser(description="pascalvoc convesion to yolo")
    parser.add_argument("source", help="pascalvoc source savedir")
    parser.add_argument("--target", "-o", default="../data", help="yolo target savedir")
    args = parser.parse_args()
    return args


def convert(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    box = list()

    for obj in root.findall("object"):  # 树的层次遍历，只在一级层次中查找标签为object的结点
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xyxy = [float(bndbox.find(s).text) for s in ["xmin", "ymin", "xmax", "ymax"]]
        box.append((name, xyxy))

    if box:
        filename = root.find("filename").text
        dtype = np.dtype(
            [("name", str, 40), ("xyxy", float, 4)]
        )  # note:结构化数组，元素为序列时要指定长度，或是单独指定序列中的每个元素字段
        annotation = np.array(box, dtype=dtype)
        annotation["xyxy"][:, [0, 1]] /= width
        annotation["xyxy"][:, [2, 3]] /= height
        print(filename, "successful!")
        return filename, annotation["name"], annotation["xyxy"]
    else:
        print("不存在真实框")

if __name__ == "__main__":
    args = argument()
    src = Path(args.source)
    tar = Path(args.target)
    files = src.glob("*.xml")
    outcome = list()
    for file in files:
        outcome.append(convert(file))
    outcome = np.asarray(outcome, dtype=object)
    try:
        np.save(tar.joinpath("train_val"), outcome)
    except DeprecationWarning as e:
        print(e)
    else:
        print("over")
