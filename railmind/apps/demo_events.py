"""内置演示事件构造器：前端一键触发三个领域能力的端到端链路。"""

from __future__ import annotations

import glob
import os
import random

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SHM_TRAIN_DIR = os.path.join(_REPO, "datasets", "shm_samples")
DOOR_DIR = os.path.join(_REPO, "pose_detect")
PANTO_SAMPLES = os.path.join(_REPO, "datasets", "pantograph", "samples")
ARC_DATA = os.path.join(_REPO, "datasets", "pantograph", "arcing")


def _pick(patterns, exclude=()):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = [f for f in files if not any(e in os.path.basename(f) for e in exclude)]
    return random.choice(files) if files else None


def build_demo_event(kind: str):
    """kind → 平台统一事件；None 表示未知类型或缺少素材。"""
    if kind == "shm_impact":
        sample = _pick([os.path.join(SHM_TRAIN_DIR, "*_1.00J+*.mat")]) or _pick([os.path.join(SHM_TRAIN_DIR, "*.mat")])
        if not sample:
            return None
        return {
            "domain": "composite_structure",
            "asset": {"train_id": "CRH-07", "carriage_id": "01", "component": "composite_deck_panel", "critical": True},
            "operation_context": {"scene": "TRAIN_RUNNING", "speed_kmh": 301},
            "capability_input": {"signal_uri": sample},
            "note": "车体复合材料区域 Lamb 波传感网触发",
        }

    if kind in ("cabin_patrol", "cabin_live"):
        video = os.path.join(_REPO, "web", "assets", "cabin_patrol.mp4")
        if not os.path.exists(video):
            return None
        live = kind == "cabin_live"
        return {
            "domain": "cabin",
            "asset": {"train_id": "CRH-05", "carriage_id": "04", "component": "cabin_camera_04"},
            "operation_context": {"scene": "TRAIN_RUNNING", "speed_kmh": 301},
            "capability_input": {"video_uri": video, "source": "vlm" if live else "cache"},
            "note": "乘务员巡检 · 车厢内部多帧VLM分析（" + ("实时重分析" if live else "预计算缓存回放") + "）",
        }

    if kind in ("door_normal", "door_open"):
        fname = "frame_00441.jpg" if kind == "door_normal" else "frame_01017.jpg"
        image = os.path.join(DOOR_DIR, fname)
        if not os.path.exists(image):
            return None
        return {
            "domain": "underbody",
            "asset": {"train_id": "CRH-12", "carriage_id": "02", "component": "inspection_door_5"},
            "operation_context": {"scene": "STATION_STOP", "speed_kmh": 0},
            "capability_input": {"image_uri": image, "evidence_dir": os.path.join(_REPO, "data", "door_evidence")},
            "note": "进站停靠轨旁相机快检",
        }

    if kind == "panto_wear":
        image = _pick([os.path.join(PANTO_SAMPLES, "video_frame_0[1-6]*.jpg")], exclude=("video_frame_0000",))
        if not image:
            image = _pick([os.path.join(PANTO_SAMPLES, "video_frame_*.jpg")])
        if not image:
            return None
        return {
            "domain": "pantograph",
            "asset": {"train_id": "CRH-18", "carriage_id": "00", "component": "pantograph_front"},
            "operation_context": {"scene": "STATION_STOP", "speed_kmh": 0},
            "capability_input": {"image_uri": image},
            "note": "受电弓滑板磨耗视觉评估",
        }

    if kind == "panto_arc":
        rec = _pick([
            os.path.join(ARC_DATA, "Trenitalia", "traction", "TI_T_*.mat"),
            os.path.join(ARC_DATA, "MetroMadrid", "traction", "MM_T_*.mat"),
        ])
        rec = rec or _pick([os.path.join(ARC_DATA, "**", "*_T_*.mat")])
        if not rec:
            return None
        name = os.path.splitext(os.path.basename(rec))[0]          # 如 TI_T_3
        prefix, phase_letter = name.split("_")[0], name.split("_")[1]
        folder = {"TI": "Trenitalia", "MM": "MetroMadrid"}[prefix]
        phase = {"T": "traction", "B": "braking"}[phase_letter]
        return {
            "domain": "pantograph",
            "asset": {"train_id": "CRH-03", "carriage_id": "00", "component": "pantograph_rear"},
            "operation_context": {"scene": "TRAIN_RUNNING", "speed_kmh": 285},
            "capability_input": {"mendeley_record": {"folder": folder, "phase": phase, "name": name}},
            "note": "弓网电弧在线监测（真实公开数据集）",
        }

    if kind == "fastener_defect":
        samples = sorted(glob.glob(os.path.join(_REPO, "railmind", "capabilities", "fastener", "samples", "sample_*.jpg")))
        sample = _pick(samples)
        if not sample:
            return None
        return {
            "domain": "lineside",
            "asset": {"train_id": "CRH-07", "carriage_id": "-", "component": "track_k45_plus_fastener"},
            "operation_context": {"scene": "STATION_STOP", "speed_kmh": 0},
            "capability_input": {"fastener_image_uri": sample, "evidence_dir": os.path.join(_REPO, "data", "fastener_evidence")},
            "note": "钢轨扣件缺陷检测（RFDD 数据集 YOLO26n）",
        }

    if kind == "lineside_fod":
        image = _pick([os.path.join(PANTO_SAMPLES, "video_frame_*.jpg")])
        if not image:
            return None
        return {
            "domain": "lineside",
            "asset": {"train_id": "CRH-07", "carriage_id": "-", "component": "track_k45_plus"},
            "operation_context": {"scene": "TRAIN_RUNNING", "speed_kmh": 120},
            "capability_input": {"image_uri": image, "evidence_dir": os.path.join(_REPO, "data", "fod_evidence")},
            "note": "线路侧固定影像点异物筛查",
        }

    return None
