"""internal.cabin.patrol_vlm —— 车厢巡检 VLM 能力（方案 5.5）。

输入：乘务员巡检视频（第一视角/车厢相机）。
流程：视频 → 4-10s 窗口 × 每窗 6 帧 → VLM 多帧固定 JSON → 相邻窗口一致性校验。
演示模式默认回放预计算缓存（demo_cache.json，真 VLM 产物），可强制实时重分析。

风险等级绝不在本能力生成最终结论（方案 6.3）：这里只给场景判定与证据帧，
升级规则（连续两窗口一致才升级）由 Cabin-Agent 按演示规程确定性执行。
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(PKG_DIR, "demo_cache.json")
FRAME_DIR = os.path.join(PKG_DIR, "frames")

VIDEO_SECONDS = 11.4  # 演示视频有效段（其后为素材片尾，不参与分析）
WINDOWS = [
    {"index": 1, "start": 0.5, "end": 6.0, "n_frames": 6},
    {"index": 2, "start": 6.0, "end": 11.4, "n_frames": 6},
]

PROMPT = (
    "你是高铁车厢智能巡检系统的视觉分析模块。以下 {n} 帧画面来自行驶中高铁车厢的巡检相机，按时间顺序排列。"
    "请只输出一个 JSON 对象（不要多余文字），字段："
    '{"scene": "normal|smoke|person_on_floor|aisle_blocked|door_obstruction|luggage_overhang|abnormal_gathering|violent_motion|unknown", '
    '"confidence": 0到1的小数, "summary": "一句话中文结论", '
    '"events": [{"type": "事件类型", "confidence": 0到1, "frame": 帧序号从1开始}]}. '
    "事件类型仅限：烟雾火光/人员倒地/通道堵塞/车门阻挡/行李架物品伸出/异常聚集/疑似剧烈动作。"
    "注意：不得输出'确认暴力事件'等确定性结论；画面正常则 scene=normal；无法判断则 scene=unknown。"
)

SCENE_CN = {
    "normal": "正常",
    "smoke": "疑似烟雾或火光",
    "person_on_floor": "人员持续倒地",
    "aisle_blocked": "通道持续堵塞",
    "door_obstruction": "车门区域阻挡",
    "luggage_overhang": "行李架物品伸出",
    "abnormal_gathering": "异常聚集",
    "violent_motion": "疑似剧烈动作",
    "unknown": "无法判断",
}

# 方案 5.5 事件类别 → 升级条件 → 处置等级（VLM 不定级，仅给场景；最终等级由 Agent/RiskEngine 复核）
_SCENE_ADVICE = {
    "normal": "车厢秩序正常，保持例行巡检",
    "smoke": "立即通知乘务员现场确认并按火灾预案处置",
    "person_on_floor": "乘务员立即前往现场查看，广播寻医",
    "aisle_blocked": "广播疏导乘客归置行李，恢复通道",
    "door_obstruction": "乘务员清理车门区域，确认不影响开关门",
    "luggage_overhang": "提醒乘客归置行李架物品，防止坠落",
    "abnormal_gathering": "乘务员前往查看聚集原因，维持秩序",
    "violent_motion": "乘务员介入了解情况，必要时联控公安（不下确定性结论）",
    "unknown": "画面置信度不足，重新抽帧或转人工复核",
}

_lock = threading.Lock()


# ---------- VLM 分析（预计算与实时重分析共用） ----------

def _extract_frames(video_path: str, windows: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    import cv2  # noqa: PLC0415

    os.makedirs(FRAME_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"E_OPEN_VIDEO: 无法打开 {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0.0
    out: Dict[int, List[str]] = {}
    for win in windows:
        rel: List[str] = []
        for i in range(win["n_frames"]):
            t = win["start"] + i * (win["end"] - win["start"]) / max(1, win["n_frames"] - 1)
            if duration > 0:
                t = min(t, max(0.0, duration - 0.08))  # 采样点不得落在 EOF（截断版视频末帧恰为窗口端点）
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise ValueError(f"E_READ_FRAME: 窗口{win['index']} 帧{i} 读取失败")
            scale = min(1.0, 720.0 / max(frame.shape[:2]))
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            name = f"w{win['index']}_f{i + 1}.jpg"
            cv2.imwrite(os.path.join(FRAME_DIR, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            rel.append(f"/static/cabin_vlm/frames/{name}")
        out[win["index"]] = rel
    cap.release()
    return out


def analyze_live(video_path: str, write_cache: bool = True) -> Dict[str, Any]:
    """实时重分析：抽帧 → VLM 多帧固定 JSON → 缓存结构。VLM 不可用时逐窗口降级拒判。"""
    from railmind.core.vlm import VlmClient, resize_jpeg_bytes  # noqa: PLC0415

    windows_out: List[Dict[str, Any]] = []
    client = VlmClient(timeout_s=120)
    frames_by_win = _extract_frames(video_path, WINDOWS)
    for win in WINDOWS:
        rel = frames_by_win[win["index"]]
        paths = [os.path.join(FRAME_DIR, os.path.basename(p)) for p in rel]
        result: Dict[str, Any]
        if client.configured:
            images = [resize_jpeg_bytes(open(p, "rb").read(), max_side=512, quality=75) for p in paths]
            result = client.analyze_images_with_fallback(images, PROMPT.replace("{n}", str(win["n_frames"])))
        else:
            result = {"degraded": True, "severity": "UNKNOWN", "confidence": 0.0,
                      "note": "VLM未配置，转人工复核"}
        result.update({"window_index": win["index"], "start_s": win["start"], "end_s": win["end"], "frames": rel})
        windows_out.append(result)
    cache = {"video": "/static/web/assets/cabin_patrol.mp4", "video_seconds": VIDEO_SECONDS,
             "precomputed_at": "live", "windows": windows_out}
    if write_cache:
        with _lock, open(CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=1)
    return cache


def load_cache() -> Optional[Dict[str, Any]]:
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------- 统一诊断输出 ----------

def window_severity(scene: str, degraded: bool) -> str:
    """方案 5.5 事件类别 → 处置等级（单窗口视角；跨窗口升级由 Cabin-Agent 负责）。"""
    if degraded or scene == "unknown":
        return "UNKNOWN"
    if scene == "smoke":
        return "HIGH"
    if scene in ("person_on_floor",):
        return "WARNING"
    if scene in ("aisle_blocked", "door_obstruction", "luggage_overhang"):
        return "WARNING"
    if scene in ("abnormal_gathering", "violent_motion"):
        return "OBSERVE"
    return "NORMAL"


def _normalize_windows(cache: Dict[str, Any]) -> List[Dict[str, Any]]:
    windows = []
    for w in cache.get("windows", []):
        scene = str(w.get("scene") or "unknown")
        degraded = bool(w.get("degraded"))
        if "_raw" in w and "scene" not in w:
            degraded, scene = True, "unknown"
        windows.append({
            "window_index": int(w.get("window_index", 0)),
            "start_s": float(w.get("start_s", 0.0)),
            "end_s": float(w.get("end_s", 0.0)),
            "scene": scene,
            "scene_cn": SCENE_CN.get(scene, scene),
            "confidence": float(w.get("confidence") or 0.0),
            "summary": str(w.get("summary") or w.get("note") or ""),
            "events": w.get("events") or [],
            "frames": w.get("frames") or [],
            "degraded": degraded,
        })
    return windows


def infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """SDK 推理函数：payload {video_uri, source: cache|vlm} → {diagnosis, evidence}。"""
    source = str(payload.get("source") or "cache")
    if source == "vlm" and payload.get("video_uri"):
        # 实时重分析不覆盖预计算缓存：演示回放必须始终稳定（缓存为答辩基线）
        cache = analyze_live(payload["video_uri"], write_cache=False)
    else:
        cache = load_cache()
    if not cache:
        raise ValueError("E_NO_CACHE: demo_cache.json 缺失且未提供 video_uri 实时分析")

    windows = _normalize_windows(cache)
    degraded = any(w["degraded"] for w in windows) or not windows
    scenes = [w["scene"] for w in windows if not w["degraded"]]
    consistent = len(set(scenes)) == 1 and len(scenes) == len(windows) and len(scenes) > 0

    worst = max((window_severity(w["scene"], w["degraded"]) for w in windows),
                key=lambda s: {"HIGH": 4, "WARNING": 3, "OBSERVE": 2, "NORMAL": 1}.get(s, 0))
    if degraded:
        severity = "UNKNOWN"
    elif consistent:
        severity = worst  # 连续两窗口一致 → 按规则升级/维持
    else:
        # 单窗不过半：HIGH/WARNING 降为只观察，不升级（方案 5.5）
        severity = "OBSERVE" if {"HIGH": 4, "WARNING": 3}.get(worst, 0) >= 3 else worst

    confidences = [w["confidence"] for w in windows if not w["degraded"]]
    confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    diagnosis = {
        "anomaly_type": "cabin_patrol",
        "severity": severity,
        "confidence": confidence,
        "windows": windows,
        "consistency": {
            "consistent": consistent,
            "note": "连续两窗口结论一致" if consistent else ("存在降级窗口" if degraded else "相邻窗口结论不一致，需人工复核"),
        },
        "analysis_source": "live_vlm" if source == "vlm" else "cached_vlm",
        "degraded": degraded,
        "model_mode": "vlm",
    }
    evidence = [{"type": "video", "uri": payload.get("video_uri") or cache.get("video", ""),
                 "analysis_source": diagnosis["analysis_source"]}]
    for w in windows:
        for uri in w["frames"]:
            evidence.append({"type": "image", "uri": uri, "window_index": w["window_index"]})
    return {"diagnosis": diagnosis, "evidence": evidence}
