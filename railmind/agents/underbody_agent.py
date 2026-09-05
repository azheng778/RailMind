"""Underbody-Agent：车底包覆结构检查门领域专业 Agent（方案 5.3 / 6.1）。

输入：门把手角度检测能力的统一诊断事件；
职责：按角度阈值复核把手状态、统计该门历史告警、检索处置知识、输出结构化结论。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from railmind.agents.base import SpecialistAgent
from railmind.core.events import EventStore
from railmind.core.provider import EchoScriptedProvider, LLMProvider
from railmind.core.rag import RagClient
from railmind.core.tools import ToolRegistry

SYSTEM = (
    "你是车底包覆结构检查门领域专业Agent。输入是门把手角度检测能力的诊断事件。"
    "职责：1)核对门把手角度与阈值等级；2)查询该检查门历史告警；3)检索检修处置依据。"
    "最后必须输出JSON：conclusion、severity、confidence、review_points、recommended_action、kb_refs。"
    "信息不全（拒判事件）时如实说明并转人工复核，不得虚构结论。"
)

_ADVICE = {
    "HIGH": "把手角度显著超限或盖板异常：人工现场确认门锁机构状态，未锁闭时重新锁闭并记录",
    "WARNING": "把手角度接近限值：下一停靠站目视复核把手与锁扣",
    "OBSERVE": "记录观察",
    "NORMAL": "检查门状态正常，无后续动作",
}


def _ctx_value(ctx: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for name, payload in ctx.get("tool_results", []):
        if name == tool_name and isinstance(payload, dict) and payload.get("ok"):
            value = payload.get("value") or {}
    return value


class UnderbodyAgent(SpecialistAgent):
    domain = "underbody"

    def __init__(self, event_store: EventStore, rag: RagClient, provider: LLMProvider, on_event=None):
        self.event_store = event_store
        self.rag = rag
        self._session: Dict[str, Any] = {}
        tools = ToolRegistry()
        self._register_tools(tools)
        super().__init__(name="Underbody-Agent", provider=provider, system=SYSTEM, tools=tools, on_event=on_event)

    def _register_tools(self, tools: ToolRegistry) -> None:
        @tools.register(name="query_door_history", description="查询该检查门近1小时告警记录")
        def query_door_history(train_id: str = "", component: str = "") -> Dict[str, Any]:
            train_id = train_id or self._session.get("train_id", "")
            component = component or self._session.get("component", "")
            rows = self.event_store.query(train_id=train_id, component=component, since_ts=time.time() - 3600)
            return {"recent_count": len(rows)}

        @tools.register(name="grade_door", description="按角度阈值与盖板状态做确定性分级")
        def grade_door(max_angle_deg: float = 0.0, cover_abnormal: bool = False, complete: bool = True) -> Dict[str, Any]:
            max_angle_deg = max_angle_deg or float(self._session.get("max_angle", 0.0))
            if not complete:
                return {"severity": "UNKNOWN", "recommended_action": "目标不完整，人工复核该检查门",
                        "review_points": ["确认相机视角与门编号映射"], "max_angle_deg": max_angle_deg}
            severity = "NORMAL"
            if max_angle_deg > 75 or cover_abnormal:
                severity = "HIGH"
            elif max_angle_deg > 50:
                severity = "WARNING"
            return {
                "severity": severity,
                "max_angle_deg": max_angle_deg,
                "recommended_action": _ADVICE.get(severity, _ADVICE["OBSERVE"]),
                "review_points": ["核对把手根部/尖端关键点位置", "比对同车厢其他检查门状态"],
            }

        @tools.register(name="retrieve_kb", description="检索检查门处置知识依据")
        def retrieve_kb() -> Dict[str, Any]:
            result = self.rag.retrieve(
                asset_type="underbody",
                keywords=["检查门", "处置", "锁闭"] if self._session.get("severity") != "NORMAL" else ["检查门", "巡检"],
            )
            if result.refused:
                return {"refused": True, "note": result.refused_text, "citations": []}
            return {"refused": False, "note": "", "citations": result.answer_basis}

    def conclude(self, event: Dict[str, Any], capability_result: Dict[str, Any]) -> Any:
        diag = capability_result.get("diagnosis", {})
        handles = diag.get("handles", [])
        self._session = {
            "train_id": event.get("asset", {}).get("train_id", ""),
            "component": event.get("asset", {}).get("component", ""),
            "max_angle": max([h.get("angle_deg", 0.0) for h in handles], default=float(diag.get("max_angle_deg", 0.0))),
            "severity": diag.get("severity", "UNKNOWN"),
            "complete": diag.get("anomaly_type") not in ("door_targets_incomplete", "door_angle_uncomputable"),
            "confidence": diag.get("confidence", 0.0),
        }
        if isinstance(self.provider, EchoScriptedProvider):
            self.provider.reset(self._script())
        return super().conclude(event, capability_result)

    def _script(self) -> List[Any]:
        def _final(ctx: Dict[str, Any]) -> Any:
            grade = _ctx_value(ctx, "grade_door")
            hist = _ctx_value(ctx, "query_door_history")
            kb = _ctx_value(ctx, "retrieve_kb")
            conclusion = {
                "conclusion": (
                    f"检查门 {_session_component(self._session)} 把手最大角度 "
                    f"{grade.get('max_angle_deg', self._session.get('max_angle', 0)):.1f}°，"
                    f"近1小时第 {hist.get('recent_count', 1)} 次告警，综合等级 {grade.get('severity')}。"
                ),
                "severity": grade.get("severity"),
                "confidence": self._session.get("confidence"),
                "review_points": grade.get("review_points", []),
                "recommended_action": grade.get("recommended_action"),
                "kb_refs": kb.get("citations", []),
                "kb_note": kb.get("note") or None,
            }
            return json.dumps(conclusion, ensure_ascii=False), []

        return [
            ("查询该检查门历史告警并按角度阈值分级", [
                ("query_door_history", {}),
                ("grade_door", {}),
            ]),
            ("检索检查门处置依据", [("retrieve_kb", {})]),
            _final,
        ]


def _session_component(session: Dict[str, Any]) -> str:
    return session.get("component") or "检查门"
