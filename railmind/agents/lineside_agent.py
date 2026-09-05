"""LineSide-Agent：线路侧领域专业 Agent（方案 5.6 / 6.1）。

处理轨道异物检测等线路侧事件：核对历史异物告警、按尺寸/类型分级、检索处置依据。
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
    "你是线路侧监测专业Agent，负责轨道异物等线路侧异常的综合分析。"
    "职责：1)核对该区段历史异物告警频次；2)按异物尺寸与类型分级；3)检索处置依据。"
    "最后输出JSON：conclusion、severity、confidence、review_points、recommended_action、kb_refs。"
)

_ADVICE = {
    "HIGH": "限界内大尺寸异物：立即通知工务与调度，评估是否拦停后续列车",
    "WARNING": "检测到线路侧异物：通知工务现场清理复核",
    "OBSERVE": "记录并持续观察该区段",
}


def _ctx_value(ctx: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for name, payload in ctx.get("tool_results", []):
        if name == tool_name and isinstance(payload, dict) and payload.get("ok"):
            value = payload.get("value") or {}
    return value


class LineSideAgent(SpecialistAgent):
    domain = "lineside"

    def __init__(self, event_store: EventStore, rag: RagClient, provider: LLMProvider, on_event=None):
        self.event_store = event_store
        self.rag = rag
        self._session: Dict[str, Any] = {}
        tools = ToolRegistry()
        self._register_tools(tools)
        super().__init__(name="LineSide-Agent", provider=provider, system=SYSTEM, tools=tools, on_event=on_event)

    def _register_tools(self, tools: ToolRegistry) -> None:
        @tools.register(name="query_fod_history", description="查询该区段近1小时异物告警")
        def query_fod_history(train_id: str = "", component: str = "") -> Dict[str, Any]:
            train_id = train_id or self._session.get("train_id", "")
            component = component or self._session.get("component", "")
            rows = self.event_store.query(train_id=train_id, component=component, anomaly_type="track_foreign_object", since_ts=time.time() - 3600)
            return {"recent_count": len(rows)}

        @tools.register(name="grade_fod", description="按异物数量与最大面积占比分级")
        def grade_fod(count: int = 0, max_area_ratio: float = 0.0) -> Dict[str, Any]:
            count = count or int(self._session.get("count", 0))
            max_area_ratio = max_area_ratio or float(self._session.get("max_area", 0.0))
            if count == 0:
                severity = "NORMAL"
            elif max_area_ratio > 0.05 or count >= 3:
                severity = "HIGH"
            else:
                severity = "WARNING"
            return {
                "severity": severity,
                "count": count,
                "max_area_ratio": max_area_ratio,
                "recommended_action": _ADVICE.get(severity, _ADVICE["OBSERVE"]),
                "review_points": ["核对异物与限界距离", "确认影像点时间戳与位置映射"],
            }

        @tools.register(name="retrieve_kb", description="检索线路侧异物处置依据")
        def retrieve_kb() -> Dict[str, Any]:
            result = self.rag.retrieve(asset_type="lineside", keywords=["异物", "处置", "限界"])
            if result.refused:
                return {"refused": True, "note": result.refused_text, "citations": []}
            return {"refused": False, "note": "", "citations": result.answer_basis}

    def conclude(self, event: Dict[str, Any], capability_result: Dict[str, Any]) -> Any:
        diag = capability_result.get("diagnosis", {})
        dets = diag.get("detections", [])
        self._session = {
            "train_id": event.get("asset", {}).get("train_id", ""),
            "component": event.get("asset", {}).get("component", ""),
            "count": len(dets),
            "max_area": max((d.get("area_ratio", 0.0) for d in dets), default=0.0),
            "note": diag.get("note", ""),
            "confidence": diag.get("confidence", 0.0),
        }
        if isinstance(self.provider, EchoScriptedProvider):
            self.provider.reset(self._script())
        return super().conclude(event, capability_result)

    def _script(self) -> List[Any]:
        def _final(ctx: Dict[str, Any]) -> Any:
            grade = _ctx_value(ctx, "grade_fod")
            hist = _ctx_value(ctx, "query_fod_history")
            kb = _ctx_value(ctx, "retrieve_kb")
            conclusion = {
                "conclusion": (
                    f"线路侧筛查：{self._session.get('note') or '无异物'}；"
                    f"近1小时第 {hist.get('recent_count', 0) + 1} 次告警，综合等级 {grade.get('severity')}。"
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
            ("查询区段异物历史并按尺寸分级", [("query_fod_history", {}), ("grade_fod", {})]),
            ("检索异物处置依据", [("retrieve_kb", {})]),
            _final,
        ]
