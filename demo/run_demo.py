"""RailMind 端到端演示（三领域 · 真实模型）。

演示链路（对应方案 11.4 + RVID 参赛演示）：
  ① 运行中：SHM 冲击定位（Lamb 波，EnhancedLambWaveNet）→ HIGH → 工单
  ② 进站快检：检查门把手角度（YOLO26? no: YOLOv8-pose ONNX）→ 正常 / 异常 / 拒判三路径
  ③ 运行中：受电弓电弧信号分析（Mendeley 公开数据集，19/19 检出）→ HIGH → 工单
  ④ 线路侧：轨道异物检测（YOLO26n + RailFOD23）
  ⑤ 对话助手：与 LLM 大脑对话查状态

运行：  D:/Anaconda3/envs/railmind/python.exe -X utf8 -m demo.run_demo
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
import time

import numpy as np

from railmind.capabilities.shm_impact.adapter import get_model as get_shm_model
from railmind.platform import RailMindPlatform

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(REPO, "semi-final", "Composite Material_Semi-Final_Dataset", "CompositeMaterial_Semi-Final_TrainingSet")
DOOR_DIR = os.path.join(REPO, "pose_detect")
ARC_DATA = os.path.join(REPO, "datasets", "pantograph", "arcing")


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"■ {title}")
    print("=" * 72)


def main() -> int:
    banner("RailMind 端到端演示 · 三领域能力 + 对话助手")
    shutil.rmtree(os.path.join(REPO, "data", "demo"), ignore_errors=True)
    platform = RailMindPlatform(data_dir=os.path.join(REPO, "data", "demo"), prefer_remote_llm=False, attach_builtin=True)
    for c in platform.registry.all():
        print(f"能力在线: {c.capability_id}  模式={c.mode}")

    # ---------- ① SHM 冲击（真实模型 + 真实样本） ----------
    banner("① 运行中 · 复合材料冲击定位（SHM）")
    files = glob.glob(os.path.join(TRAIN_DIR, "*.mat"))
    sample = next((f for f in files if "1.00J" in f), files[0])
    m = get_shm_model()
    print(f"模型: mode={m.mode} meta={m.meta}")
    summary = platform.handle_event({
        "domain": "composite_structure",
        "asset": {"train_id": "CRH-07", "carriage_id": "01", "component": "composite_deck_panel", "critical": True},
        "operation_context": {"scene": "TRAIN_RUNNING", "speed_kmh": 301},
        "capability_input": {"signal_uri": sample},
    })
    ev = platform.events.query(train_id="CRH-07", anomaly_type="composite_impact", limit=1)[0]
    print(f"风险: {summary['risk']['level']}  分值: {summary['risk']['score']}  工单: {(summary.get('workorder') or {}).get('workorder_id', '-')}")
    print(f"预测: {ev['diagnosis'].get('impacts')}")

    # ---------- ② 检查门（正常 / 异常 / 拒判） ----------
    banner("② 进站停靠 · 检查门把手角度")
    for fname, tag in (("frame_00441.jpg", "关门"), ("frame_01017.jpg", "开门"), ("frame_00993.jpg", "遮挡")):
        summary = platform.handle_event({
            "domain": "underbody",
            "asset": {"train_id": "CRH-12", "carriage_id": "02", "component": "inspection_door_5"},
            "operation_context": {"scene": "STATION_STOP", "speed_kmh": 0},
            "capability_input": {"image_uri": os.path.join(DOOR_DIR, fname), "evidence_dir": os.path.join(REPO, "data", "door_evidence")},
        })
        print(f"{tag}: risk={summary['risk']['level']:8s} 结论={str(summary['specialist_conclusion'].get('conclusion'))[:46]}")

    # ---------- ③ 受电弓电弧（真实公开数据集） ----------
    banner("③ 运行中 · 受电弓电弧信号分析（Mendeley CC BY 4.0）")
    summary = platform.handle_event({
        "domain": "pantograph",
        "asset": {"train_id": "CRH-03", "carriage_id": "00", "component": "pantograph_rear"},
        "operation_context": {"scene": "TRAIN_RUNNING", "speed_kmh": 285},
        "capability_input": {"mendeley_record": {"folder": "Trenitalia", "phase": "traction", "name": "TI_T_1"}},
    })
    print(f"风险: {summary['risk']['level']}  工单: {(summary.get('workorder') or {}).get('workorder_id', '-')}")

    # ---------- ④ 线路侧异物（YOLO26n + RailFOD23） ----------
    banner("④ 线路侧 · 轨道异物检测（YOLO26n）")
    summary = platform.handle_event({
        "domain": "lineside",
        "asset": {"train_id": "CRH-07", "carriage_id": "-", "component": "track_k45_plus"},
        "operation_context": {"scene": "TRAIN_RUNNING", "speed_kmh": 120},
        "capability_input": {"image_uri": os.path.join(REPO, "datasets", "pantograph", "samples", "video_frame_0242.jpg")},
    })
    print(f"风险: {summary['risk']['level']}  结论: {str(summary['specialist_conclusion'].get('conclusion'))[:60]}")

    # ---------- ⑤ 对话助手 ----------
    banner("⑤ 对话助手（LLM 大脑 · 问答）")
    if type(platform.provider).__name__ == "EchoScriptedProvider":
        print("（离线模式：配置 .env 中 RAILMIND_LLM_* 后，对话助手由 LLM 大脑驱动）")
    else:
        try:
            platform.init_chat()
            r = platform.chat_agent.chat("用一句话告诉我当前平台有几个能力在线、几张未关闭工单")
            print("助手:", str(r["reply"])[:160])
            print("工具:", [t["name"] for t in r["tool_trace"]])
        except Exception as exc:  # noqa: BLE001
            print("对话助手异常:", exc)

    banner("工单与能力状态")
    for order in platform.workorders.list():
        print(f"{order['workorder_id']}  {order['status']:8s}  风险={order['risk'].get('level'):8s}  {order.get('component')}")
    for c in platform.registry.all():
        print(f"{c.capability_id:32s} 模式={c.mode:9s} 在线={c.is_online()}  调用={c.call_count}  时延={c.avg_latency_ms:.0f}ms")

    platform.stop()
    print("\n演示结束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
