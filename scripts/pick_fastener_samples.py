# -*- coding: utf-8 -*-
"""挑选扣件演示样本图：断裂/缺失/正常各一张，缩放到 ≤1280px 存入能力包 samples/。"""
import glob
import os
import shutil
import sys

import cv2

ROOT = r"E:\BaiduNetdiskDownload\铁轨紧固件损坏检测数据集 YOLO格式\铁轨紧固件损坏检测数据集 YOLO格式"
TEST_IMG = os.path.join(ROOT, "test", "images")
TEST_LBL = os.path.join(ROOT, "test", "labels")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "railmind", "capabilities", "fastener", "samples")

os.makedirs(OUT, exist_ok=True)

pick = {}
for f in sorted(glob.glob(os.path.join(TEST_LBL, "*.txt"))):
    classes = set(open(f).read().split()[0::5])
    if "1" in classes and "broken" not in pick:
        pick["broken"] = f  # 弹条断裂
    if "2" in classes and "missing" not in pick:
        pick["missing"] = f  # 弹条缺失
    if os.path.getsize(f) == 0 and "normal" not in pick:
        pick["normal"] = f  # 正常

for name, lbl in pick.items():
    img_path = os.path.join(TEST_IMG, os.path.basename(lbl).replace(".txt", ".png"))
    img = cv2.imdecode(__import__("numpy").fromfile(img_path, dtype="uint8"), cv2.IMREAD_COLOR)  # 中文路径安全
    if img is None:
        print("SKIP", img_path)
        continue
    scale = min(1.0, 1280.0 / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    out_path = os.path.join(OUT, f"sample_{name}.jpg")
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{name}: {os.path.basename(img_path)} -> {out_path} ({os.path.getsize(out_path)//1024}KB)")
