# -*- coding: utf-8 -*-
"""扣件数据集演示子集：从 train&val 抽 150 张缩到 ≤1024px jpg（含标签）→ datasets/fastener_sample。
原始 13.4GB 不动；子集仅用于演示/参赛材料（本地，不入 Git）。"""
import glob
import os
import random

import cv2
import numpy as np

ROOT = r"E:\BaiduNetdiskDownload\铁轨紧固件损坏检测数据集 YOLO格式\铁轨紧固件损坏检测数据集 YOLO格式"
YOLO = os.path.join(ROOT, "train&val", "YOLO")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "fastener_sample")

random.seed(7)
labels = sorted(glob.glob(os.path.join(YOLO, "labels", "*.txt")))
by_cls = {0: [], 1: [], 2: [], 3: [], 5: []}
empty = []
for f in labels:
    content = open(f).read().split()
    if not content:
        empty.append(f)
        continue
    cls = int(content[0])
    if cls in by_cls:
        by_cls[cls].append(f)

# 每类缺陷 24 张 + 正常 30 张 ≈ 150 张，保证缺陷类型覆盖
picked = []
for cls, files in by_cls.items():
    picked += random.sample(files, min(24, len(files)))
picked += random.sample(empty, min(30, len(empty)))
random.shuffle(picked)

n = 0
total = 0
for f in picked:
    stem = os.path.splitext(os.path.basename(f))[0]
    img_path = os.path.join(YOLO, "images", stem + ".png")
    img = cv2.imdecode(np.fromfile(img_path, dtype="uint8"), cv2.IMREAD_COLOR)
    if img is None:
        continue
    scale = min(1.0, 1024.0 / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    out_img = os.path.join(OUT, "images", stem + ".jpg")
    out_lbl = os.path.join(OUT, "labels", stem + ".txt")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
    cv2.imwrite(out_img, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    shutil_size = 1
    if os.path.getsize(f) > 0:
        open(out_lbl, "w", encoding="utf-8").write(open(f, encoding="utf-8").read())
    n += 1
    total += os.path.getsize(out_img)

print(f"subset: {n} images, {total / 1048576:.1f} MB -> {OUT}")
