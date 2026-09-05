"""YOLO 能力共享封装：模型加载、推理、结果绘制、降级处理。

权重文件缺失或加载失败时进入 degraded 模式（severity=UNKNOWN），不虚构结论。
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class YoloDetector:
    def __init__(self, weights_path: str, names: Dict[int, str]):
        self.weights_path = weights_path
        self.names = names
        self._lock = threading.Lock()
        self._model = None
        self._mode = "not_loaded"
        self._error: Optional[str] = None
        self.load()

    def load(self) -> bool:
        try:
            from ultralytics import YOLO  # noqa: PLC0415

            with self._lock:
                self._model = YOLO(self.weights_path)
                self._mode = "yolo"
            return True
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._model = None
                self._mode = "degraded"
                self._error = f"{type(exc).__name__}: {exc}"[:200]
            return False

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def error(self) -> Optional[str]:
        return self._error

    def predict(self, image: np.ndarray, conf: float = 0.4) -> List[Dict[str, Any]]:
        if self._mode != "yolo":
            return []
        with self._lock:
            results = self._model.predict(image, conf=conf, verbose=False)
        out: List[Dict[str, Any]] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                out.append({
                    "class_id": cls_id,
                    "class_name": self.names.get(cls_id, str(cls_id)),
                    "confidence": round(float(box.conf.item()), 3),
                    "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                })
        return out

    @staticmethod
    def draw(image: np.ndarray, detections: List[Dict[str, Any]], severity: str = "") -> np.ndarray:
        color_map = {"NORMAL": (80, 220, 80), "OBSERVE": (80, 220, 80), "WARNING": (60, 200, 255),
                     "HIGH": (60, 60, 255), "UNKNOWN": (200, 200, 200)}
        for det in detections:
            color = color_map.get(det.get("level", severity), (255, 160, 60))
            x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{det.get('class_name', '')} {det.get('confidence', 0):.2f}"
            if det.get("level"):
                label += f" {det['level']}"
            cv2.putText(image, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if severity:
            cv2.putText(image, f"severity: {severity}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (60, 60, 255) if severity in ("HIGH", "WARNING") else (80, 220, 80), 2)
        return image
