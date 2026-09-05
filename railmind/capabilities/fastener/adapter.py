"""internal.lineside.fastener —— 钢轨扣件缺陷检测能力（方案 5.6 LineSide 扩展）。

YOLO26n + RFDD 铁路扣件缺陷数据集（6 类：弹条变形/断裂/缺失/翻转 + 正常扣件 + 弹条移位，
Science Data Bank CSTR:149.11.sciencedb.msdc.00071，CC BY 4.0）。
巡检/固定影像点图片输入 → 扣件状态检测 → 按缺陷类型风险分级 → 证据图存档。

分级规则（试行版）：断裂/缺失直接危及扣压功能 → HIGH；移位/翻转/变形 → WARNING；无缺陷 → NORMAL。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import cv2

from railmind.capabilities.yolo_common import YoloDetector

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGED = os.path.abspath(os.path.join(PKG_DIR, "weights", "best.pt"))
_RUNS = os.path.abspath(os.path.join(PKG_DIR, "..", "..", "..", "runs", "fastener_y26n", "weights", "best.pt"))
DEFAULT_WEIGHTS = os.environ.get(
    "RAILMIND_FASTENER_WEIGHTS",
    _PACKAGED if os.path.exists(_PACKAGED) else _RUNS,
)

NAMES = {0: "弹条变形", 1: "弹条断裂", 2: "弹条缺失", 3: "弹条翻转", 4: "正常扣件", 5: "弹条移位"}
HIGH_CLASSES = {"弹条断裂", "弹条缺失"}  # 扣压功能失效，直接风险
WARNING_CLASSES = {"弹条移位", "弹条翻转", "弹条变形"}


def defect_severity(class_name: str) -> str:
    if class_name in HIGH_CLASSES:
        return "HIGH"
    if class_name in WARNING_CLASSES:
        return "WARNING"
    return "OBSERVE"


_model: YoloDetector | None = None  # noqa: UP045 —— py3.9 兼容写法


def get_model() -> YoloDetector:
    global _model
    if _model is None:
        _model = YoloDetector(DEFAULT_WEIGHTS, NAMES)
    return _model


def load_image(payload: Dict[str, Any]):
    uri = payload.get("fastener_image_uri") or payload.get("image_uri")
    if uri:
        img = cv2.imdecode(__import__("numpy").fromfile(uri, dtype="uint8"), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"E_INVALID_INPUT:图像读取失败 {uri}")
        return img
    if payload.get("image_b64"):
        import base64

        buf = base64.b64decode(payload["image_b64"])
        img = cv2.imdecode(__import__("numpy").frombuffer(buf, dtype="uint8"), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("E_INVALID_INPUT:image_b64 解码失败")
        return img
    raise ValueError("E_INVALID_INPUT:需要 fastener_image_uri 或 image_b64")


def infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """payload: {fastener_image_uri|image_uri|image_b64, evidence_dir?} → {diagnosis, evidence}"""
    model = get_model()
    image = load_image(payload)
    h, w = image.shape[:2]
    area = h * w

    if model.mode != "yolo":
        return {
            "diagnosis": {
                "anomaly_type": "fastener_unable",
                "severity": "UNKNOWN",
                "confidence": 0.0,
                "degraded": True,
                "degrade_reason": model.error or "weights unavailable",
            },
            "evidence": [],
        }

    detections = model.predict(image, conf=0.4)
    defects = []
    for d in detections:
        if d["class_name"] == "正常扣件":
            continue
        bx = d["bbox_xyxy"]
        d["area_ratio"] = round(max(0.0, (bx[2] - bx[0]) * (bx[3] - bx[1]) / area), 4)
        d["level"] = defect_severity(d["class_name"])
        defects.append(d)

    if defects:
        severity = max((d["level"] for d in defects), key=lambda lv: ["OBSERVE", "WARNING", "HIGH"].index(lv))
        anomaly = "fastener_defect"
        kinds = sorted({d["class_name"] for d in defects})
        note = f"检测到扣件缺陷 {len(defects)} 处：{', '.join(kinds)}"
    else:
        severity = "NORMAL"
        anomaly = "fastener_normal"
        note = "扣件状态正常，未检测到缺陷"

    diagnosis = {
        "anomaly_type": anomaly,
        "severity": severity,
        "confidence": round(max((d["confidence"] for d in defects), default=0.0), 3),
        "degraded": False,
        "detections": defects,
        "normal_count": sum(1 for d in detections if d["class_name"] == "正常扣件"),
        "note": note,
    }

    evidence_uri = None
    if payload.get("evidence_dir"):
        os.makedirs(payload["evidence_dir"], exist_ok=True)
        evidence_uri = os.path.join(payload["evidence_dir"], f"fastener_{int(time.time())}.jpg")
        cv2.imencode(".jpg", YoloDetector.draw(image.copy(), defects, severity), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(evidence_uri)

    return {
        "diagnosis": diagnosis,
        "evidence": [{"type": "image", "uri": evidence_uri or payload.get("fastener_image_uri", "inline")}],
    }
