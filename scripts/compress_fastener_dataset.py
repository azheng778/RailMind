# -*- coding: utf-8 -*-
"""RFDD 扣件数据集全量压缩搬移：1350 张 PNG（约3.1GB）→ 800px JPEG（目标约100MB）。

结构（datasets/fastener/，保持 YOLO 格式可直接训练，本地不入 Git）：
  data.yaml / train_val/{images,labels} / test/{images,labels}
标签为归一化坐标，与分辨率无关，直接复制。运行：python -X utf8 scripts/compress_fastener_dataset.py [max_side] [quality]
"""
import glob
import os
import sys

import cv2
import numpy as np

ROOT = r"E:\BaiduNetdiskDownload\铁轨紧固件损坏检测数据集 YOLO格式\铁轨紧固件损坏检测数据集 YOLO格式"
SRC_YOLO = os.path.join(ROOT, "train&val", "YOLO")
SRC_TEST = os.path.join(ROOT, "test")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "datasets", "fastener")

NAMES = {0: "弹条变形", 1: "弹条断裂", 2: "弹条缺失", 3: "弹条翻转", 4: "正常扣件", 5: "弹条移位"}


def convert(png_path, out_jpg, max_side, quality):
    img = cv2.imdecode(np.fromfile(png_path, dtype="uint8"), cv2.IMREAD_COLOR)  # 中文路径安全
    if img is None:
        return False
    scale = min(1.0, float(max_side) / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    buf.tofile(out_jpg)
    return True


def convert_split(src_img_dir, src_lbl_dir, dst_name, max_side, quality):
    img_out = os.path.join(OUT, dst_name, "images")
    lbl_out = os.path.join(OUT, dst_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    n, fail = 0, 0
    for png in sorted(glob.glob(os.path.join(src_img_dir, "*.png"))):
        stem = os.path.splitext(os.path.basename(png))[0]
        if convert(png, os.path.join(img_out, stem + ".jpg"), max_side, quality):
            n += 1
        else:
            fail += 1
        lbl = os.path.join(src_lbl_dir, stem + ".txt")
        if os.path.exists(lbl):
            open(os.path.join(lbl_out, stem + ".txt"), "w", encoding="utf-8").write(open(lbl, encoding="utf-8").read())
    print(f"{dst_name}: {n} 张（失败 {fail}）")
    return n


def main():
    max_side = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    quality = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    n1 = convert_split(os.path.join(SRC_YOLO, "images"), os.path.join(SRC_YOLO, "labels"), "train_val", max_side, quality)
    n2 = convert_split(os.path.join(SRC_TEST, "images"), os.path.join(SRC_TEST, "labels"), "test", max_side, quality)

    yaml_path = os.path.join(OUT, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(f"path: {OUT}\ntrain: {os.path.join(OUT, 'train_val', 'images')}\n")
        fh.write(f"val: {os.path.join(OUT, 'test', 'images')}\ntest: {os.path.join(OUT, 'test', 'images')}\n")
        fh.write(f"nc: {len(NAMES)}\nnames:\n")
        for i, name in NAMES.items():
            fh.write(f"  {i}: {name}\n")

    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
    print(f"合计 {n1 + n2} 张，总大小 {total / 1048576:.1f} MB → {OUT}")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
