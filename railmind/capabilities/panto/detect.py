"""受电弓部件检测层：YOLO26n（contact_point / mast / strip 三类）。

用于磨耗评估前定位滑板（strip）区域，裁剪后送 VLM 细看；也可独立输出部件检测结果。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import cv2

from railmind.capabilities.yolo_common import YoloDetector

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.environ.get(
    "RAILMIND_PANTO_WEIGHTS",
    os.path.abspath(os.path.join(PKG_DIR, "..", "..", "..", "runs", "panto_y26n", "weights", "best.pt")),
)

NAMES = {0: "接触点", 1: "桅杆", 2: "导电条带(strip)"}

_model: Optional[YoloDetector] = None


def get_model() -> YoloDetector:
    global _model
    if _model is None:
        _model = YoloDetector(DEFAULT_WEIGHTS, NAMES)
    return _model


def detect_strip(image, conf: float = 0.4) -> List[Dict[str, Any]]:
    """返回 strip 类检测结果（置信度降序）。"""
    model = get_model()
    if model.mode != "yolo":
        return []
    dets = [d for d in model.predict(image, conf=conf) if d["class_id"] == 2]
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets


def crop_strip(image, pad_ratio: float = 0.15):
    """裁剪最大 strip 区域（带边距）；未检出返回 None。"""
    dets = detect_strip(image, conf=0.35)
    if not dets:
        return None
    x1, y1, x2, y2 = [int(v) for v in dets[0]["bbox_xyxy"]]
    h, w = image.shape[:2]
    px, py = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return image[y1:y2, x1:x2]
