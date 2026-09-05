# -*- coding: utf-8 -*-
"""演示历史数据种子：生成 7 天连贯运营档案（事件 + 工单全生命周期）。

目的：让平台对话助手/前端查到的历史数据呈现真实运营形态——
  * 事件分布多列车、多部件、多能力，时间连续；
  * 工单具备完整生命周期（DRAFT→OPEN→CLOSED / REJECTED），含处置记录与知识引用；
  * 正常/观察占绝大多数，HIGH/WARNING 少量（真实运营分布）。

运行：python -X utf8 scripts/seed_demo_history.py [--days 7]
注意：会重写 data/events.jsonl 与 data/workorders.json（演示数据，可随时重新生成）。
"""
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")

TRAINS = ["CRH-03", "CRH-05", "CRH-07", "CRH-12", "CRH-18"]
STATIONS = ["北京南", "廊坊", "天津南", "沧州西", "济南西", "徐州东", "南京南", "上海虹桥"]
OPERATORS = ["值班员", "检修员A", "检修员B", "随车机械师"]

# 能力 → (capability_id, 领域, 部件模板, 正常事件生成器, 异常模板)
CAPS = {
    "shm": {
        "capability_id": "internal.shm.impact_locator",
        "domain": "composite_structure",
        "component": "composite_deck_panel",
        "asset_extra": {"critical": True},
        "scene": "TRAIN_RUNNING",
        "speed": (290, 310),
        "input": {"signal_uri": "demo://sample.mat"},
        "anomaly_normal": ("structure_normal", "NORMAL", 0.92, {"note": "复合材料结构状态正常"}),
        "anomaly_types": [
            ("composite_impact", "OBSERVE", 0.82, {"max_energy_j": 0.20}),
            ("composite_impact", "WARNING", 0.86, {"max_energy_j": 0.50}),
            ("composite_impact", "HIGH", 0.90, {"max_energy_j": 0.70}),
        ],
    },
    "door": {
        "capability_id": "internal.underbody.door_pose",
        "domain": "underbody",
        "component": "inspection_door_{}",
        "scene": "STATION_STOP",
        "speed": (0, 0),
        "input": {"image_uri": "/static/door_demo/demo_clean.jpg"},
        "anomaly_normal": ("door_normal_state", "NORMAL", 0.94, {"note": "检查门闭合正常，把手角度 0°"}),
        "anomaly_types": [
            ("door_handle_warning", "WARNING", 0.79, {"handles": [{"angle_deg": 62.0}]}),
            ("door_handle_abnormal", "HIGH", 0.91, {"handles": [{"angle_deg": 85.4}, {"angle_deg": 90.9}]}),
        ],
    },
    "wear": {
        "capability_id": "internal.panto.wear_vlm",
        "domain": "pantograph",
        "component": "pantograph_front",
        "scene": "STATION_STOP",
        "speed": (0, 0),
        "input": {"image_uri": "/static/panto_samples/video_frame_0242.jpg"},
        "anomaly_normal": ("strip_wear_normal", "NORMAL", 0.88, {"note": "滑板磨耗等级 normal"}),
        "anomaly_types": [
            ("strip_wear_light", "OBSERVE", 0.74, {"wear_level": "light"}),
            ("strip_wear_heavy", "WARNING", 0.88, {"wear_level": "heavy"}),
        ],
    },
    "arc": {
        "capability_id": "internal.panto.arc_signal",
        "domain": "pantograph",
        "component": "pantograph_rear",
        "scene": "TRAIN_RUNNING",
        "speed": (270, 305),
        "input": {"mendeley_record": {"folder": "Trenitalia", "phase": "traction", "name": "TI_T_3"}},
        "anomaly_normal": ("arc_normal", "NORMAL", 0.95, {"note": "未检出电弧事件"}),
        "anomaly_types": [
            ("arc_event", "WARNING", 0.81, {"arc_events": 3, "arc_total_ms": 8.2}),
            ("arc_event", "HIGH", 0.93, {"arc_events": 12, "arc_total_ms": 24.6, "arc_score": 78}),
        ],
    },
    "fod": {
        "capability_id": "internal.lineside.fod",
        "domain": "lineside",
        "component": "track_k45_plus",
        "scene": "STATION_STOP",
        "speed": (0, 0),
        "input": {"image_uri": "/static/panto_samples/video_frame_0121.jpg"},
        "anomaly_normal": ("track_clear", "NORMAL", 0.90, {"note": "未检测到线路侧异物"}),
        "anomaly_types": [
            ("track_foreign_object", "WARNING", 0.84, {"detections": [{"class_name": "塑料袋", "area_ratio": 0.012}]}),
            ("track_foreign_object", "HIGH", 0.89, {"detections": [{"class_name": "漂浮物", "area_ratio": 0.062}]}),
        ],
    },
    "fastener": {
        "capability_id": "internal.lineside.fastener",
        "domain": "lineside",
        "component": "track_k45_fastener",
        "scene": "STATION_STOP",
        "speed": (0, 0),
        "input": {"fastener_image_uri": "/static/fastener_demo/samples/sample_broken.jpg"},
        "anomaly_normal": ("fastener_normal", "NORMAL", 0.91, {"note": "扣件状态正常"}),
        "anomaly_types": [
            ("fastener_defect", "WARNING", 0.86, {"detections": [{"class_name": "弹条移位", "level": "WARNING"}]}),
            ("fastener_defect", "HIGH", 0.90, {"detections": [{"class_name": "弹条断裂", "level": "HIGH"}, {"class_name": "弹条缺失", "level": "HIGH"}]}),
        ],
    },
    "cabin": {
        "capability_id": "internal.cabin.patrol_vlm",
        "domain": "cabin",
        "component": "cabin_camera_04",
        "scene": "TRAIN_RUNNING",
        "speed": (295, 305),
        "input": {"video_uri": "/static/web/assets/cabin_patrol.mp4"},
        "anomaly_normal": ("cabin_patrol", "NORMAL", 0.95, {"note": "两窗口场景判定正常，连续一致"}),
        "anomaly_types": [
            ("cabin_patrol", "OBSERVE", 0.71, {"note": "单窗口低置信，转人工复核"}),
        ],
    },
}

KB_BY_DOMAIN = {
    "composite_structure": [{"document_id": "DOC-DEMO-001", "title": "复合材料结构冲击检测作业指引（演示规程）", "chapter": "冲击能量分级与处置", "page": 4}],
    "underbody": [{"document_id": "DOC-DEMO-003", "title": "车底检查门检修说明（演示规程）", "chapter": "检查门未闭合处置", "page": 3}],
    "pantograph": [{"document_id": "DOC-DEMO-002", "title": "受电弓检修规程（演示规程）", "chapter": "电弧异常", "page": 15}],
    "lineside": [{"document_id": "DOC-DEMO-004", "title": "线路侧异物处置办法（演示规程）", "chapter": "异物分级与处置", "page": 7},
                 {"document_id": "DOC-DEMO-005", "title": "钢轨扣件检修规程（演示规程）", "chapter": "扣件缺陷分级处置", "page": 11}],
    "cabin": [{"document_id": "DOC-DEMO-006", "title": "车内异常巡检处置指引（演示规程）", "chapter": "车内事件分级与处置", "page": 5}],
}

WO_NOTES_CLOSE = {
    "composite_impact": "已到现场敲击复核，未见层间损伤，关闭",
    "door_handle_abnormal": "地勤已重新锁闭并更换锁机构，关闭",
    "door_handle_warning": "下一站目视复核正常，关闭",
    "arc_event": "已复核弓网接触状态，滑板更换后燃弧恢复正常，关闭",
    "strip_wear_heavy": "已在入库检修时更换滑板，关闭",
    "track_foreign_object": "工务已现场清理并确认限界安全，关闭",
    "fastener_defect": "天窗点内更换弹条，复检图像正常，关闭",
}
WO_REJECT_NOTE = "人工复核为误报（光影/飞鸟干扰），驳回"


def _rnd_speed(rng, span):
    return rng.randint(*span)


def build(days: int):
    rng = random.Random(42)
    now = time.time()
    events, workorders = [], []
    wo_seq = 0
    ev_seq = 0

    for day in range(days, 0, -1):
        n_events = rng.randint(11, 15)
        # 当天（day==1）留白最近 30 分钟，避免与真实运行事件的时间线冲突
        base = now - day * 86400 + 6 * 3600
        for _ in range(n_events):
            key = rng.choice(list(CAPS))
            cfg = CAPS[key]
            ev_seq += 1
            ts = base + rng.uniform(0, 86400 - 6 * 3600) if day > 1 else now - rng.uniform(1800, 86400)
            anomaly, severity, conf, extra = (
                cfg["anomaly_types"][rng.randrange(len(cfg["anomaly_types"]))] if rng.random() < 0.22 else cfg["anomaly_normal"]
            )
            diagnosis = {
                "anomaly_type": anomaly,
                "severity": severity,
                "confidence": round(conf + rng.uniform(-0.03, 0.02), 3),
                "degraded": False,
            }
            diagnosis.update(extra)
            component = cfg["component"].format(rng.randint(1, 5)) if "{}" in cfg["component"] else cfg["component"]
            event = {
                "schema_version": "1.0",
                "event_id": f"EVT-{time.strftime('%Y%m%d', time.localtime(ts))}-{ev_seq:05d}",
                "ts": ts,
                "source": {"capability_id": cfg["capability_id"], "provider": "RailMind-Team"},
                "asset": {
                    "train_id": rng.choice(TRAINS),
                    "carriage_id": f"{rng.randint(1, 8):02d}",
                    "component": component,
                    **cfg.get("asset_extra", {}),
                },
                "operation_context": {
                    "scene": cfg["scene"],
                    "speed_kmh": _rnd_speed(rng, cfg["speed"]),
                },
                "diagnosis": diagnosis,
                "evidence": [{"type": "image", "uri": next(iter(cfg["input"].values()))}],
                "model_version": "1.0.0",
                "requires_human_review": severity in ("HIGH", "WARNING"),
            }
            events.append(event)

            if severity in ("HIGH", "WARNING") and rng.random() < 0.9:
                wo_seq += 1
                wo_id = f"WO-{time.strftime('%Y%m%d', time.localtime(ts))}-{wo_seq:05d}"
                age_days = (now - ts) / 86400
                if age_days > 1.5:
                    status = "CLOSED" if rng.random() < 0.85 else "REJECTED"
                elif age_days > 0.8:
                    status = "OPEN"
                else:
                    status = "DRAFT"
                created = ts
                history = [{"ts": created, "action": "CREATE_DRAFT"}]
                if status in ("OPEN", "CLOSED", "REJECTED"):
                    history.append({"ts": created + rng.uniform(300, 1800), "action": "OPEN",
                                    "operator": rng.choice(OPERATORS), "note": "确认，安排下一站复核"})
                if status == "CLOSED":
                    history.append({"ts": created + rng.uniform(3600, 20 * 3600), "action": "CLOSED",
                                    "operator": rng.choice(OPERATORS), "note": WO_NOTES_CLOSE.get(anomaly, "已现场复核，关闭")})
                if status == "REJECTED":
                    history.append({"ts": created + rng.uniform(3600, 8 * 3600), "action": "REJECTED",
                                    "operator": rng.choice(OPERATORS), "note": WO_REJECT_NOTE})
                workorders.append({
                    "workorder_id": wo_id,
                    "status": status,
                    "created_at": created,
                    "train_id": event["asset"]["train_id"],
                    "component": component,
                    "anomaly_type": anomaly,
                    "risk": {
                        "level": severity,
                        "score": round(rng.uniform(62, 88) if severity == "HIGH" else rng.uniform(40, 60), 1),
                        "reasons": [f"能力判定 {severity}", "运行中场景加权" if cfg["scene"] == "TRAIN_RUNNING" else "停站场景"],
                        "human_review_required": True,
                    },
                    "agent_conclusion": {"conclusion": f"{component} {anomaly}，等级 {severity}", "severity": severity,
                                         "confidence": diagnosis["confidence"]},
                    "knowledge": KB_BY_DOMAIN.get(cfg["domain"], []),
                    "requires_human_review": True,
                    "history": history,
                })
    return events, workorders


def main():
    # 服务运行时其内存态会把种子文件覆盖回去 —— 必须先停 8900 服务再执行种子
    try:
        import urllib.request

        urllib.request.urlopen("http://127.0.0.1:8900/healthz", timeout=2)
        print("E_SERVER_RUNNING: 8900 端口有服务在运行，会覆盖种子文件。请先停止 railmind.apps.api 再执行本脚本。")
        sys.exit(2)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 连不上即服务未运行，正常继续
        pass
    days = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 7
    events, workorders = build(days)
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "events.jsonl"), "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
    with open(os.path.join(DATA, "workorders.json"), "w", encoding="utf-8") as fh:
        json.dump(workorders, fh, ensure_ascii=False, indent=2, default=str)
    # Chief 摘要与事件同存（EventStore 重启后加载，告警表/对话助手可查完整历史）
    with open(os.path.join(DATA, "events_summaries.jsonl"), "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps({
                "event_id": ev["event_id"],
                "ts": ev["ts"],
                "train_id": ev["asset"]["train_id"],
                "component": ev["asset"]["component"],
                "anomaly_type": ev["diagnosis"]["anomaly_type"],
                "risk_level": ev["diagnosis"]["severity"],
            }, ensure_ascii=False, default=str) + "\n")
    from collections import Counter
    st = Counter(w["status"] for w in workorders)
    sv = Counter(e["diagnosis"]["severity"] for e in events)
    print(f"events: {len(events)} {dict(sv)}")
    print(f"workorders: {len(workorders)} {dict(st)}")


if __name__ == "__main__":
    main()
