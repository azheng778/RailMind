"""受电弓部件检测层：YOLO26n（contact_point / mast / strip 三类）。

用于磨耗评估前定位滑板（strip）区域，裁剪后送 VLM 细看；也可独立输出部件检测结果。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from railmind.capabilities.yolo_common import YoloDetector

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(PKG_DIR, "..", "..", ".."))
DEPLOYED_WEIGHTS = os.path.join(PKG_DIR, "weights", "best.pt")
TRAINING_WEIGHTS = os.path.join(REPO_DIR, "runs", "panto_y26n", "weights", "best.pt")

NAMES = {0: "接触点", 1: "桅杆", 2: "导电条带(strip)"}

_model: Optional[YoloDetector] = None
_model_path: Optional[str] = None


def resolve_weights_path() -> str:
    """Resolve an explicit override first, then packaged and training weights."""
    override = os.environ.get("RAILMIND_PANTO_WEIGHTS", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    for path in (DEPLOYED_WEIGHTS, TRAINING_WEIGHTS):
        if os.path.isfile(path):
            return path
    return DEPLOYED_WEIGHTS


def get_model() -> YoloDetector:
    global _model, _model_path
    weights_path = resolve_weights_path()
    if _model is None or _model_path != weights_path:
        _model = YoloDetector(weights_path, NAMES)
        _model_path = weights_path
    return _model


def detector_status() -> Dict[str, Any]:
    """Return model readiness information for health checks and evidence."""
    model = get_model()
    return {
        "mode": model.mode,
        "weights_path": model.weights_path,
        "weights_exists": os.path.isfile(model.weights_path),
        "error": model.error,
    }


def detect_parts(image: np.ndarray, conf: float = 0.4) -> List[Dict[str, Any]]:
    """Detect contact point, mast and strip components, ordered by confidence."""
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("E_INVALID_IMAGE:需要非空图像数组")
    if not 0.0 < conf <= 1.0:
        raise ValueError("E_INVALID_CONF:置信度阈值应在 (0, 1] 范围内")
    model = get_model()
    if model.mode != "yolo":
        return []
    dets = [d for d in model.predict(image, conf=conf) if d.get("class_id") in NAMES]
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets


def detect_strip(image: np.ndarray, conf: float = 0.4) -> List[Dict[str, Any]]:
    """Return strip detections, ordered by confidence."""
    return [d for d in detect_parts(image, conf=conf) if d["class_id"] == 2]


def crop_strip(image: np.ndarray, pad_ratio: float = 0.15):
    """Crop the strongest usable strip region with padding; return None if absent."""
    if not 0.0 <= pad_ratio <= 1.0:
        raise ValueError("E_INVALID_PADDING:裁剪边距比例应在 [0, 1] 范围内")
    dets = detect_strip(image, conf=0.35)
    if not dets:
        return None
    h, w = image.shape[:2]
    usable = []
    for det in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
        y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
        if x2 > x1 and y2 > y1:
            usable.append(((x2 - x1) * (y2 - y1) * det["confidence"], x1, y1, x2, y2))
    if not usable:
        return None
    _, x1, y1, x2, y2 = max(usable, key=lambda item: item[0])
    px, py = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return image[y1:y2, x1:x2]
