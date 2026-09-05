"""internal.underbody.door_pose —— 检查门门把手角度检测能力。

复用团队 YOLOv8-pose ONNX 模型（pose_model.py，类别 0=board 门板 4关键点、1=handle 把手 6关键点）：
  门板 4 角点 → 透视几何基准（交点 + 水平斜率）
  把手根部→尖端向量 与 几何垂直方向 的偏差角 = 把手角度
  角度 ≤50° 正常 / 50-75° 警告 / >75° 异常；门板中心区域边缘密度判盖板异常。
输出统一诊断事件（方案 8.2），检测可视化图作为 evidence 存档。
仅适用于 STATION_STOP 场景（方案 5.3）。
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_DIR = os.path.abspath(os.path.join(PKG_DIR, "..", "..", "..", "pose_detect"))

_LEVEL = None  # 延迟解析模型类


def _model_class():
    global _LEVEL
    if _LEVEL is None:
        from railmind.capabilities.door_handle.pose_model import YOLOv8Keypoint

        _LEVEL = YOLOv8Keypoint
    return _LEVEL


class DoorPoseModel:
    """YOLOv8-pose 检查门把手角度模型；线程安全；延迟加载；可降级。"""

    def __init__(self, onnx_path: Optional[str] = None, conf: float = 0.8, nms: float = 0.5):
        self.onnx_path = onnx_path or os.path.join(PKG_DIR, "weights", "best.onnx")
        self.conf = conf
        self.nms = nms
        self._lock = threading.Lock()
        self._model = None
        self._mode = "not_loaded"
        self._error: Optional[str] = None
        self.load()

    def load(self) -> bool:
        try:
            cls = _model_class()
            with self._lock:
                self._model = cls(det_ckpt_path=self.onnx_path, confidence_thres=self.conf, nms_thres=self.nms)
                self._mode = "onnx"
            return True
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._model = None
                self._mode = "degraded"
                self._error = str(exc)
            return False

    @property
    def mode(self) -> str:
        return self._mode

    # ---------- 推理 ----------

    def predict(self, image: np.ndarray, evidence_dir: Optional[str] = None, image_name: str = "input") -> Dict[str, Any]:
        if self._mode != "onnx":
            return {"anomaly_type": "unable_to_judge", "severity": "UNKNOWN", "confidence": 0.0,
                    "degraded": True, "degrade_reason": (self._error or "model unavailable")[:200]}

        with self._lock:
            preds = self._model.process_image(image.copy())
        board = [d for d in preds if int(d[5]) == 0]
        handles = sorted([d for d in preds if int(d[5]) == 1], key=lambda d: d[0])

        result_img = image.copy()
        handles_out: List[Dict[str, Any]] = []
        board_error = False
        anomaly = "door_handle_normal"
        severity = "NORMAL"
        confidence = 0.0

        if not board or len(handles) != 2:
            # 信息不全：按方案 5.3 "结果冲突/信息不足 → 人工复核"，不虚构结论
            return self._draw_and_pack(
                result_img, board, handles, evidence_dir, image_name,
                diagnosis={
                    "anomaly_type": "door_targets_incomplete",
                    "severity": "UNKNOWN",
                    "confidence": float(board[0][4]) if board else 0.0,
                    "degraded": False,
                    "targets": {"board": len(board), "handle": len(handles)},
                    "note": "未完整检测到门板与两个把手，转人工复核",
                },
            )

        from railmind.capabilities.door_handle.pose_model import (
            find_intersection_and_horizontal_slope,
            find_vertical_slope_and_angle,
        )

        arr = np.array([board[0]] + handles)
        geo = find_intersection_and_horizontal_slope(arr, class_id=0)
        if geo is None:
            return self._draw_and_pack(
                result_img, board, handles, evidence_dir, image_name,
                diagnosis={
                    "anomaly_type": "door_angle_uncomputable",
                    "severity": "OBSERVE",
                    "confidence": float(board[0][4]),
                    "degraded": False,
                    "note": "几何交点不可计算（透视退化），建议人工复核该门",
                },
            )

        intersection, h_slope, c23 = geo
        vert = find_vertical_slope_and_angle(handles, intersection, h_slope, c23, class_id=1)
        if not vert:
            return self._draw_and_pack(
                result_img, board, handles, evidence_dir, image_name,
                diagnosis={
                    "anomaly_type": "door_angle_uncomputable",
                    "severity": "OBSERVE",
                    "confidence": float(board[0][4]),
                    "degraded": False,
                    "note": "把手角度不可计算，建议人工复核",
                },
            )

        # 门板盖板异常（边缘密度）
        roi_images = None
        try:
            from railmind.capabilities.door_handle.pose_model import warp_and_crop_roi

            roi_images = warp_and_crop_roi(image, [arr], output_size=500, margin=110)
        except Exception:  # noqa: BLE001
            roi_images = None
        if roi_images:
            gray = cv2.cvtColor(roi_images[0], cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 80, 180)
            edge_count = int(np.sum(edges > 0))
            board_error = edge_count > 3900
        else:
            edge_count = -1

        worst = "NORMAL"
        for i, (_vslope, angle_raw) in enumerate(vert):
            angle = float(abs(angle_raw))
            hconf = float(handles[i][4])
            confidence = max(confidence, hconf)
            level = "NORMAL" if angle <= 50 else ("WARNING" if angle <= 75 else "HIGH")
            worst = max(worst, level, key=lambda lv: ["NORMAL", "OBSERVE", "WARNING", "HIGH"].index(lv))
            k = handles[i][6:].reshape(-1, 3)
            handles_out.append({
                "handle_id": i,
                "angle_deg": round(angle, 1),
                "level": level,
                "confidence": round(hconf, 3),
                "bbox_xyxy": [int(v) for v in handles[i][:4]],
                "root_xy": [float(k[4][0]), float(k[4][1])],
                "tip_xy": [float(k[5][0]), float(k[5][1])],
            })

        if worst == "HIGH" or board_error:
            anomaly, severity = "door_handle_abnormal", "HIGH"
        elif worst == "WARNING":
            anomaly, severity = "door_handle_abnormal", "WARNING"
        else:
            anomaly, severity = "door_handle_normal", "NORMAL"

        diagnosis = {
            "anomaly_type": anomaly,
            "severity": severity,
            "confidence": round(confidence, 3),
            "degraded": False,
            "handles": handles_out,
            "board": {
                "detected": True,
                "cover_abnormal": board_error,
                "edge_density_count": edge_count,
                "bbox_xyxy": [int(v) for v in board[0][:4]],
                "confidence": round(float(board[0][4]), 3),
            },
            "note": "检查门异常：把手角度超限" + ("且盖板边缘异常" if board_error else "") if severity in ("WARNING", "HIGH") else "检查门正常：把手角度与盖板状态均在阈值内",
        }
        return self._draw_and_pack(result_img, board, handles, evidence_dir, image_name, diagnosis,
                                   intersection=intersection, handles_out=handles_out)

    # ---------- 可视化 ----------

    def _draw_and_pack(
        self,
        result_img: np.ndarray,
        board: List[np.ndarray],
        handles: List[np.ndarray],
        evidence_dir: Optional[str],
        image_name: str,
        diagnosis: Dict[str, Any],
        intersection: Optional[np.ndarray] = None,
        handles_out: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        color_map = {"NORMAL": (80, 220, 80), "WARNING": (60, 200, 255), "HIGH": (60, 60, 255)}
        if board:
            b = board[0]
            cv2.rectangle(result_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (70, 70, 255), 2)
            cv2.putText(result_img, f"board {b[4]:.2f}", (int(b[0]), max(14, int(b[1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 255), 2)
        for h_out in handles_out or []:
            color = color_map.get(h_out["level"], (200, 200, 200))
            x1, y1, x2, y2 = h_out["bbox_xyxy"]
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_img, f"{h_out['angle_deg']}deg {h_out['level']}", (x1, max(14, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.line(result_img, (int(h_out["root_xy"][0]), int(h_out["root_xy"][1])),
                     (int(h_out["tip_xy"][0]), int(h_out["tip_xy"][1])), color, 2)
        if intersection is not None:
            cv2.circle(result_img, (int(intersection[0]), int(intersection[1])), 8, (255, 0, 255), -1)
        cv2.putText(result_img, f"severity: {diagnosis.get('severity')}", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 255) if diagnosis.get("severity") in ("HIGH", "WARNING") else (80, 220, 80), 2)

        evidence_uri = None
        if evidence_dir:
            os.makedirs(evidence_dir, exist_ok=True)
            evidence_uri = os.path.join(evidence_dir, f"door_{image_name}_{int(time.time())}.jpg")
            cv2.imwrite(evidence_uri, result_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return {"diagnosis": diagnosis, "evidence": [{"type": "image", "uri": evidence_uri} if evidence_uri else {"type": "image", "note": "no archive dir"}]}


_model: Optional[DoorPoseModel] = None


def get_model() -> DoorPoseModel:
    global _model
    if _model is None:
        _model = DoorPoseModel()
    return _model


def load_image(source: Dict[str, Any]) -> np.ndarray:
    if source.get("image_uri"):
        img = cv2.imdecode(np.fromfile(source["image_uri"], dtype=np.uint8), cv2.IMREAD_COLOR)  # 中文路径兼容
        if img is None:
            raise ValueError(f"E_INVALID_INPUT:图像读取失败 {source['image_uri']}")
        return img
    if source.get("image_b64"):
        import base64

        buf = np.frombuffer(base64.b64decode(source["image_b64"]), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("E_INVALID_INPUT:image_b64 解码失败")
        return img
    raise ValueError("E_INVALID_INPUT:需要 image_uri 或 image_b64")


def infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """SDK 推理入口：{image_uri|image_b64, evidence_dir?} → {diagnosis, evidence}"""
    model = get_model()
    image = load_image(payload)
    image_name = os.path.splitext(os.path.basename(payload.get("image_uri", "inline")))[0][:40]
    return model.predict(image, evidence_dir=payload.get("evidence_dir"), image_name=image_name)
