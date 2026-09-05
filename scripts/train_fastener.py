# -*- coding: utf-8 -*-
"""YOLO26n 扣件缺陷训练（RFDD 6 类，RTX 3080）。运行：python -X utf8 scripts/train_fastener.py [epochs]"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ultralytics import YOLO  # noqa: E402

DATA = r"E:\BaiduNetdiskDownload\铁轨紧固件损坏检测数据集 YOLO格式\铁轨紧固件损坏检测数据集 YOLO格式\train&val\YOLO\splits\fastener.yaml"


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    try:
        model = YOLO("yolo26n.pt")
    except Exception as exc:
        print("yolo26n.pt 下载失败，改用从零训练:", exc, flush=True)
        model = YOLO("yolo26n.yaml")
    model.train(
        data=DATA,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project=os.path.join(REPO, "runs"),
        name="fastener_y26n",
        patience=50,
        verbose=True,
    )
    print("==== fastener_y26n 完成", flush=True)


if __name__ == "__main__":
    main()
