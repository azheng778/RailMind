# -*- coding: utf-8 -*-
"""数据集特征文件分析：输入(图像) / 输出(YOLO标签) / 训练脚本特征统计。"""
import os
import sys
from collections import Counter

REPO = r"D:\workspace\RailMind"

DATASETS = [
    ("受电弓目标检测", os.path.join(REPO, "高体受电弓设备目标检测数据集 YOLO格式"),
     {0: "contact_point", 1: "mast", 2: "strip"}),
    ("轨道异物检测", os.path.join(REPO, "RailFOD23.v1i.yolov8"),
     {0: "niaocao 鸟巢", 1: "piaofuwu 漂浮物", 2: "qiqiu 气球", 3: "suliaodai 塑料袋"}),
]


def read_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


IMG_EXT = (".jpg", ".jpeg", ".png")


def analyze(root, names):
    from PIL import Image
    stats = {}
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(root, split, "images")
        lbl_dir = os.path.join(root, split, "labels")
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT)] if os.path.isdir(img_dir) else []
        cls_counter = Counter()
        wh = Counter()
        boxes_wh = []      # 归一化 (w,h)
        objs_per_img = []
        empty = 0
        for f in imgs:
            lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
            if os.path.exists(lbl):
                rows = [l.split() for l in open(lbl, encoding="utf-8").read().splitlines() if l.strip()]
                objs_per_img.append(len(rows))
                for r in rows:
                    cls_counter[int(r[0])] += 1
                    w, h = float(r[3]), float(r[4])
                    boxes_wh.append((w, h))
                    wh["small" if w * h < 0.01 else "medium" if w * h < 0.2 else "large"] += 1
            else:
                empty += 1
                objs_per_img.append(0)
        # 抽样图像尺寸
        sizes = Counter()
        for f in imgs[:: max(1, len(imgs) // 12)]:
            try:
                with Image.open(os.path.join(img_dir, f)) as im:
                    sizes[f"{im.width}x{im.height}"] += 1
            except Exception:
                pass
        big = sorted(boxes_wh, key=lambda x: -x[0] * x[1])
        tiny = sorted(boxes_wh, key=lambda x: x[0] * x[1])
        stats[split] = {
            "images": len(imgs),
            "labels_missing": empty,
            "instances": sum(cls_counter.values()),
            "cls": {names.get(k, k): v for k, v in sorted(cls_counter.items())},
            "objs_per_img": (round(min(objs_per_img)), round(sum(objs_per_img) / max(1, len(objs_per_img)), 2),
                             round(max(objs_per_img))) if objs_per_img else (0, 0, 0),
            "size_buckets": dict(wh),
            "img_sizes": dict(sizes),
            "largest_bbox": (round(big[0][0], 3), round(big[0][1], 3)) if big else None,
            "smallest_bbox": (round(tiny[0][0], 4), round(tiny[0][1], 4)) if tiny else None,
        }
    return stats


for title, root, names in DATASETS:
    print("=" * 70)
    print(f"◆ {title}")
    yaml_cfg = read_yaml(os.path.join(root, "data.yaml"))
    print("  classes:", yaml_cfg["names"], "| nc:", yaml_cfg["nc"])
    s = analyze(root, yaml_cfg["names"] and {i: n for i, n in enumerate(yaml_cfg["names"])})
    for split, d in s.items():
        print(f"  [{split}] images={d['images']} instances={d['instances']} 无标注图={d['labels_missing']}")
        print(f"      类别分布: {d['cls']}")
        print(f"      每图目标数 min/avg/max: {d['objs_per_img']}")
        print(f"      面积分档 small(<1%)/medium/large: {d['size_buckets']}")
        print(f"      图像尺寸抽样: {d['img_sizes']}")
        print(f"      bbox 归一化宽高 最大={d['largest_bbox']} 最小={d['smallest_bbox']}")
