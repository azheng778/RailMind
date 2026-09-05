"""internal.panto.wear_vlm —— 受电弓滑板磨耗 VLM 评估能力（方案 5.2）。

用远端视觉模型（deepseek-v4-flash-vision-exp，OpenAI 兼容协议）对受电弓图像做
滑板磨耗分级：normal(正常) / light(轻度) / heavy(明显) / unknown(拒判)。
设计要点（方案 5.5 技术要求）：
  * VLM 必须输出固定 JSON；
  * 低置信度（<0.6）时按拒判处理，转人工复核，不虚构结论；
  * 远端不可用时走本地降级路径（degraded=True）。
"""

from __future__ import annotations

import os
from typing import Any, Dict

from railmind.core.vlm import VlmClient, resize_jpeg_bytes

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.abspath(os.path.join(PKG_DIR, "..", "..", "..", "datasets", "pantograph", "samples"))

PROMPT = (
    "你是高速列车受电弓滑板磨耗检测专家。请分析图中受电弓碳滑板（与接触网接触的横条状部件）的磨耗状态。"
    "注意：图中可能带有检测系统的标注文字、线框或测距数字，请忽略这些标注元素，只评估受电弓本体。"
    "只输出一个JSON对象，字段固定为："
    '{"wear_level": "normal|light|heavy|uncertain", '
    '"confidence": 0到1的小数, '
    '"strips_visible": 可见滑板数量, '
    '"description": "一句话中文描述磨耗特征（裂纹/缺块/磨耗面宽度等）"}。'
    "判断标准：滑板表面平整、无贯通裂纹为normal；出现明显磨耗面或浅裂纹为light；"
    "存在缺块、贯通裂纹或滑板剩余厚度明显不足为heavy；图像模糊或滑板不可见为uncertain。"
    "不要输出JSON以外的任何内容。"
)

# 磨耗等级 → 统一严重等级
_SEVERITY = {"normal": "NORMAL", "light": "OBSERVE", "heavy": "WARNING"}


def infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """payload: {image_uri|image_b64} → {diagnosis, evidence}"""
    image_bytes = _load_image_bytes(payload)
    client = VlmClient()

    # 检测引导：受电弓模型定位 strip 后裁剪送 VLM（不可用时退回全图）
    crop_note = "full_image"
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from railmind.capabilities.panto.detect import crop_strip

        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is not None:
            strip = crop_strip(image)
            if strip is not None:
                ok, enc = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    image_bytes = enc.tobytes()
                    crop_note = "strip_crop"
    except Exception:  # noqa: BLE001 —— 检测层缺失不影响 VLM 评估
        pass

    small = resize_jpeg_bytes(image_bytes)
    result = client.analyze_image_with_fallback(small, PROMPT)

    if result.get("degraded"):
        return {
            "diagnosis": {
                "anomaly_type": "pantograph_wear_unable",
                "severity": "UNKNOWN",
                "confidence": 0.0,
                "degraded": True,
                "note": result.get("note", "VLM不可用"),
            },
            "evidence": [{"type": "image", "note": f"input ({crop_note})"}],
        }

    wear = str(result.get("wear_level", "uncertain")).lower()
    confidence = float(result.get("confidence", 0.0) or 0.0)
    if wear == "uncertain" or confidence < 0.6:
        # 低置信度拒判（方案 5.2：低置信度拒判）
        return {
            "diagnosis": {
                "anomaly_type": "pantograph_wear_unable",
                "severity": "UNKNOWN",
                "confidence": round(confidence, 3),
                "degraded": False,
                "wear_level": wear,
                "note": f"磨耗等级低置信度（{wear}, {confidence:.2f}），按拒判处理，转人工复核",
                "description": result.get("description", ""),
            },
            "evidence": [{"type": "image", "note": f"input ({crop_note})"}],
        }

    severity = _SEVERITY.get(wear, "UNKNOWN")
    return {
        "diagnosis": {
            "anomaly_type": "pantograph_wear_" + wear,
            "severity": severity,
            "confidence": round(confidence, 3),
            "degraded": False,
            "wear_level": wear,
            "strips_visible": result.get("strips_visible"),
            "description": result.get("description", ""),
        },
        "evidence": [{"type": "image", "note": f"input ({crop_note})"}],
    }


def _load_image_bytes(payload: Dict[str, Any]) -> bytes:
    if payload.get("image_b64"):
        import base64

        return base64.b64decode(payload["image_b64"])
    if payload.get("image_uri"):
        with open(payload["image_uri"], "rb") as fh:
            return fh.read()
    raise ValueError("E_INVALID_INPUT:需要 image_uri 或 image_b64")
