"""端到端编排测试（桩能力，不依赖 torch）：完整链路 + 拒判路径 + 工单闭环。"""

import json

import pytest

from railmind.capabilities.sdk import load_descriptor
from railmind.platform import RailMindPlatform

DESCRIPTOR = "railmind/capabilities/shm_impact/capability.yaml"


def stub_warning_result(payload):
    return {
        "diagnosis": {
            "anomaly_type": "composite_impact",
            "severity": "WARNING",
            "confidence": 0.88,
            "impacts": [
                {"x_mm": 150, "y_mm": 250, "energy_j": 0.35, "confidence": 0.9},
                {"x_mm": 300, "y_mm": 100, "energy_j": 0.7, "confidence": 0.86},
            ],
            "max_energy_j": 0.7,
            "degraded": False,
            "model_mode": "stub",
        },
        "evidence": [],
    }


def make_event(signal_uri="demo://sample.mat"):
    return {
        "domain": "composite_structure",
        "asset": {"train_id": "CRH-TEST", "carriage_id": "01", "component": "composite_deck_panel", "critical": True},
        "operation_context": {"scene": "TRAIN_RUNNING", "speed_kmh": 300},
        "capability_input": {"signal_uri": signal_uri},
    }


@pytest.fixture()
def platform(tmp_path):
    pf = RailMindPlatform(data_dir=str(tmp_path), prefer_remote_llm=False)
    pf.attach_local_capability(descriptor_path=DESCRIPTOR, infer_fn=stub_warning_result, heartbeat=False)
    yield pf
    pf.stop()


def test_full_pipeline_creates_workorder(platform):
    summary = platform.handle_event(make_event())
    risk = summary["risk"]
    assert risk["level"] in ("WARNING", "HIGH")
    assert risk["human_review_required"] is True
    assert summary["workorder"] and summary["workorder"]["risk_level"] == risk["level"]
    # 知识引用来自演示规程
    assert summary["knowledge"], "应检索到演示规程引用"
    assert summary["knowledge"][0]["document_id"].startswith("DOC-")
    # 专业Agent结论是结构化JSON
    assert "conclusion" in summary["specialist_conclusion"]
    # trace 记录了完整编排过程
    kinds = [e["kind"] for e in summary["trace"]]
    assert "tool_start" in kinds and "done" in kinds


def test_workorder_confirm_flow(platform):
    summary = platform.handle_event(make_event())
    wo_id = summary["workorder"]["workorder_id"]
    confirmed = platform.workorders.confirm(wo_id, operator="检修员A", note="确认，安排下一站复核")
    assert confirmed["status"] == "OPEN"
    closed = platform.workorders.close(wo_id, operator="检修员A", note="已复核锁闭")
    assert closed["status"] == "CLOSED"
    assert platform.workorders.open_count("CRH-TEST") == 0


def test_offline_capability_refuses(platform):
    platform.registry.mark_offline(platform._capabilities[0].instance_id)
    summary = platform.handle_event(make_event())
    assert summary["capability"] is None
    assert summary["risk"]["level"] == "UNKNOWN"
    assert summary["risk"]["human_review_required"] is True
    assert summary["workorder"] is None
    assert "拒判" in summary["specialist_conclusion"]["conclusion"]


def test_mode_switch_to_observe_disables_dispatch(platform):
    platform.registry.set_mode("internal.shm.impact_locator", "OBSERVE")
    summary = platform.handle_event(make_event())
    assert summary["capability"] is None
    assert summary["risk"]["level"] == "UNKNOWN"


def test_events_recorded_for_history(platform):
    platform.handle_event(make_event())
    rows = platform.events.query(train_id="CRH-TEST", anomaly_type="composite_impact")
    assert len(rows) == 1
    assert rows[0]["diagnosis"]["max_energy_j"] == 0.7


def test_summary_is_valid_json(platform):
    #Chief 的最终输出文本必须是合法 JSON（前端消费）
    summary = platform.handle_event(make_event())
    assert "event_id" in summary
