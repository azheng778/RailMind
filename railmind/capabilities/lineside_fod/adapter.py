"""internal.lineside.fod —— 轨道异物检测能力（方案 5.6 LineSide-Agent）。

YOLO26n + RailFOD23 数据集（4 类：鸟巢 niaocao / 漂浮物 piaofuwu / 气球 qiqiu / 塑料袋 suliaodai，CC BY 4.0）。
固定影像点/巡检图片输入 → 异物检测 → 风险等级（异物越大越靠近限界风险越高）→ 证据图存档。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import cv2

from railmind.capabilities.yolo_common import YoloDetector

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.environ.get(
    "RAILMIND_FOD_WEIGHTS",
    os.path.abspath(os.path.join(PKG_DIR, "..", "..", "..", "runs", "fod_y26n", "weights", "best.pt")),
)

NAMES = {0: "鸟巢", 1: "漂浮物", 2: "气球", 3: "塑料袋"}

_model: YoloDetector | None = None  # noqa: UP045 —— py3.9 兼容写法


def get_model() -> YoloDetector:
    global _model
    if _model is None:
        _model = YoloDetector(DEFAULT_WEIGHTS, NAMES)
    return _model


def load_image(payload: Dict[str, Any]):
    if payload.get("image_b64"):
        import base64

        buf = payload["image_b64"]
        img = cv2.imdecode(__import__("numpy").frombuffer(base64.b64decode(buf), dtype="uint8"), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("E_INVALID_INPUT:image_b64 解码失败")
        return img
    if payload.get("image_uri"):
        img = cv2.imdecode(__import__("numpy").fromfile(payload["image_uri"], dtype="uint8"), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"E_INVALID_INPUT:图像读取失败 {payload['image_uri']}")
        return img
    raise ValueError("E_INVALID_INPUT:需要 image_uri 或 image_b64")


def infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """payload: {image_uri|image_b64, evidence_dir?} → {diagnosis, evidence}"""
    model = get_model()
    image = load_image(payload)
    h, w = image.shape[:2]
    area = h * w

    if model.mode != "yolo":
        return {
            "diagnosis": {
                "anomaly_type": "foreign_object_unable",
                "severity": "UNKNOWN",
                "confidence": 0.0,
                "degraded": True,
                "degrade_reason": model.error or "weights unavailable",
            },
            "evidence": [],
        }

    detections = model.predict(image, conf=0.4)
    for d in detections:
        bx = d["bbox_xyxy"]
        d["area_ratio"] = round(max(0.0, (bx[2] - bx[0]) * (bx[3] - bx[1]) / area), 4)
        d["level"] = "HIGH" if d["area_ratio"] > 0.05 else "WARNING"

    if detections:
        severity = max((d["level"] for d in detections), key=lambda lv: ["WARNING", "HIGH"].index(lv))
        anomaly = "track_foreign_object"
        note = f"检测到线路侧异物 {len(detections)} 处：{', '.join(sorted({d['class_name'] for d in detections}))}"
    else:
        severity = "NORMAL"
        anomaly = "track_clear"
        note = "未检测到线路侧异物"

    diagnosis = {
        "anomaly_type": anomaly,
        "severity": severity,
        "confidence": round(max((d["confidence"] for d in detections), default=0.0), 3),
        "degraded": False,
        "detections": detections,
        "note": note,
    }

    evidence_uri = None
    if payload.get("evidence_dir"):
        os.makedirs(payload["evidence_dir"], exist_ok=True)
        evidence_uri = os.path.join(payload["evidence_dir"], f"fod_{int(time.time())}.jpg")
        cv2.imencode(".jpg", YoloDetector.draw(image.copy(), detections, severity), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(evidence_uri)

    return {
        "diagnosis": diagnosis,
        "evidence": [{"type": "image", "uri": evidence_uri or payload.get("image_uri", "inline")}],
    }
