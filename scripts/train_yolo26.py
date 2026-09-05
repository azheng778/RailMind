# -*- coding: utf-8 -*-
"""YOLO26n 训练驱动：受电弓 3 类 → 轨道异物 4 类（串行，RTX 3080）。"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ultralytics import YOLO  # noqa: E402

JOBS = [
    {
        "name": "panto_y26n",
        "src_yaml": os.path.join(REPO, "高体受电弓设备目标检测数据集 YOLO格式", "data.yaml"),
        "root": os.path.join(REPO, "高体受电弓设备目标检测数据集 YOLO格式"),
        "epochs": 30,
    },
    {
        "name": "fod_y26n",
        "src_yaml": os.path.join(REPO, "RailFOD23.v1i.yolov8", "data.yaml"),
        "root": os.path.join(REPO, "RailFOD23.v1i.yolov8"),
        "epochs": 30,
    },
]


def fix_yaml(job):
    """Roboflow 相对路径 ../train/images 改为绝对路径。"""
    fixed = os.path.join(job["root"], "data_fixed.yaml")
    with open(job["src_yaml"], "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    out = []
    for line in lines:
        if line.startswith("train:"):
            out.append(f"train: {os.path.join(job['root'], 'train', 'images')}\n")
        elif line.startswith("val:"):
            out.append(f"val: {os.path.join(job['root'], 'valid', 'images')}\n")
        elif line.startswith("test:"):
            out.append(f"test: {os.path.join(job['root'], 'test', 'images')}\n")
        else:
            out.append(line)
    with open(fixed, "w", encoding="utf-8") as fh:
        fh.writelines(out)
    return fixed


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for job in JOBS:
        if only and only not in job["name"]:
            continue
        data = fix_yaml(job)
        print(f"==== 训练 {job['name']} epochs={job['epochs']} data={data}", flush=True)
        try:
            model = YOLO("yolo26n.pt")  # 预训练权重自动下载；失败则退回 yaml 从零训练
        except Exception as exc:
            print("yolo26n.pt 下载失败，改用从零训练:", exc, flush=True)
            model = YOLO("yolo26n.yaml")
        model.train(
            data=data,
            epochs=job["epochs"],
            imgsz=640,
            batch=16,
            device=0,
            workers=4,
            project=os.path.join(REPO, "runs"),
            name=job["name"],
            patience=50,
            verbose=True,
        )
        print(f"==== {job['name']} 完成", flush=True)


if __name__ == "__main__":
    main()
