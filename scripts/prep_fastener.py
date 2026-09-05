# -*- coding: utf-8 -*-
"""RFDD 扣件缺陷数据集准备：类名映射 + train/val 划分 + data yaml。

数据源（E盘，不入 Git）：E:/BaiduNetdiskDownload/铁轨紧固件损坏检测数据集 YOLO格式/...
train&val 1250 张按 90/10 划分；引用：RFDD, Science Data Bank 2025,
CSTR:149.11.sciencedb.msdc.00071（CC BY 4.0）。
"""
import collections
import glob
import os
import random

ROOT = r"E:\BaiduNetdiskDownload\铁轨紧固件损坏检测数据集 YOLO格式\铁轨紧固件损坏检测数据集 YOLO格式"
YOLO = os.path.join(ROOT, "train&val", "YOLO")
IMG_DIR = os.path.join(YOLO, "images")
LBL_DIR = os.path.join(YOLO, "labels")

# 文件名缺陷码 → 展示名（按 RFDD 数据集约定）
CODE_CN = {"DS": "弹条缺失", "DL": "弹条断裂", "YW": "弹条移位", "BX": "弹条变形", "CS": "弹条损伤", "FZ": "弹条翻转", "N": "正常"}


def code_of(name):
    # 形如 1-L1-WZ-GX3-L-DS-L1-1 → 第 6 段为缺陷码；N 为正常样本
    stem = os.path.splitext(name)[0]
    parts = stem.split("-")
    return parts[5] if len(parts) > 5 else ""


def main():
    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    print(f"images: {len(images)}")
    idx2code = collections.defaultdict(collections.Counter)
    for f in glob.glob(os.path.join(LBL_DIR, "*.txt")):
        cls = open(f).read().split()[0::5]
        code = code_of(os.path.basename(f))
        for c in cls:
            idx2code[int(c)][code] += 1
    names = {}
    for idx in sorted(idx2code):
        top = idx2code[idx].most_common(1)[0][0]
        names[idx] = top
    print("class mapping:", dict(names))
    nc = max(names) + 1
    name_list = [CODE_CN.get(names.get(i, "?"), f"扣件缺陷{i}") for i in range(nc)]
    # 类 4 实例数 6204（每张图含多个正常扣件）→ 正常扣件；其余为文件名缺陷码
    if max(idx2code.get(4, {None: 0}).values()) > 1000:
        name_list[4] = "正常扣件"
    print("names:", name_list)

    random.seed(42)
    with_defect = [f for f in images if os.path.getsize(os.path.join(LBL_DIR, os.path.basename(f).replace(".png", ".txt"))) > 0]
    normal = [f for f in images if f not in with_defect]
    random.shuffle(with_defect)
    val_n = max(50, int(len(images) * 0.1))
    val = with_defect[: val_n // 2] + normal[: val_n - val_n // 2]
    train = [f for f in images if f not in val]
    print(f"train {len(train)} / val {len(val)}")

    split_dir = os.path.join(YOLO, "splits")
    os.makedirs(split_dir, exist_ok=True)
    for name, files in (("train.txt", train), ("val.txt", val)):
        with open(os.path.join(split_dir, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(files))
    yaml_path = os.path.join(split_dir, "fastener.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(f"path: {YOLO}\ntrain: {os.path.join(split_dir, 'train.txt')}\nval: {os.path.join(split_dir, 'val.txt')}\nnc: {nc}\nnames:\n")
        for i, n in enumerate(name_list):
            fh.write(f"  {i}: {n}\n")
    print("yaml:", yaml_path)


if __name__ == "__main__":
    main()
