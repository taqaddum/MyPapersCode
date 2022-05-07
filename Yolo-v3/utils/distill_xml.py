import argparse
import os
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


def argument():
    parser = argparse.ArgumentParser(description="pascalvoc convesion to yolo")
    parser.add_argument("--source", help="pascalvoc source savedir", default="../data/Annotation")
    parser.add_argument("--target", "-o", default="../data", help="yolo target savedir")
    args = parser.parse_args(['--source', '../data/Annotation', '--target', '../data'])
    return args


def convert(xmlpath):
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    ground = list()

    for obj in root.findall("object"):  # 树的层次遍历，只在一级层次中查找标签为object的结点
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xyxy = [float(bndbox.find(s).text) for s in ("xmin", "ymin", "xmax", "ymax")]
        xyxy.insert(0, name)
        ground.append(tuple(xyxy))

    if ground:
        filename = root.find("filename").text
        filepath = root.find("path").text

        if not os.path.exists(filepath):
            filepath = filepath.replace('MyYOLO', 'MyPapersCode/Yolo-v3')
        print(filename, "successful!")
        return filepath, ground
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

    try:
        outcome = np.array(outcome, dtype=np.object_)
        np.save(tar.joinpath("train_val"), outcome)
    except DeprecationWarning as e:
        print(e)
    else:
        print("over")
