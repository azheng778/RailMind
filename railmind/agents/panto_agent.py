"""Panto-Agent：受电弓综合诊断领域专业 Agent（方案 5.2 / 6.1）。

整合两个子能力的诊断事件：滑板磨耗 VLM 评估 + 电弧信号风险分析；
按检修规程综合给出风险等级与处置建议，并检索知识依据。
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
    "你是受电弓综合诊断专业Agent，负责整合滑板磨耗评估与电弧信号分析两类事件。"
    "职责：1)核对磨耗等级与电弧事件指标；2)查询该受电弓历史异常；3)检索检修处置依据；"
    "4)按规程给出综合判断（磨耗等级高且电弧频次升高时提升处置优先级）。"
    "最后必须输出JSON：conclusion、severity、confidence、review_points、recommended_action、kb_refs。"
)

_ADVICE = {
    "HIGH": "显著电弧或明显磨耗：安排人工复核弓网接触状态，评估降弓或限速，尽快安排滑板更换",
    "WARNING": "重点关注：下一停靠站目视复核滑板磨耗与弓网接触痕迹",
    "OBSERVE": "记录并加密观测频次",
}


def _ctx_value(ctx: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for name, payload in ctx.get("tool_results", []):
        if name == tool_name and isinstance(payload, dict) and payload.get("ok"):
            value = payload.get("value") or {}
    return value


class PantoAgent(SpecialistAgent):
    domain = "pantograph"

    def __init__(self, event_store: EventStore, rag: RagClient, provider: LLMProvider, on_event=None):
        self.event_store = event_store
        self.rag = rag
        self._session: Dict[str, Any] = {}
        tools = ToolRegistry()
        self._register_tools(tools)
        super().__init__(name="Panto-Agent", provider=provider, system=SYSTEM, tools=tools, on_event=on_event)

    def _register_tools(self, tools: ToolRegistry) -> None:
        @tools.register(name="query_panto_history", description="查询该受电弓近1小时异常事件")
        def query_panto_history(train_id: str = "", component: str = "") -> Dict[str, Any]:
            train_id = train_id or self._session.get("train_id", "")
            rows = self.event_store.query(train_id=train_id, component=component or None, since_ts=time.time() - 3600)
            arc_rows = [r for r in rows if "arc" in str(r.get("diagnosis", {}).get("anomaly_type", ""))]
            return {"recent_count": len(rows), "recent_arc_count": len(arc_rows)}

        @tools.register(name="grade_event", description="对磨耗/电弧事件做确定性分级并融合")
        def grade_event(wear_level: str = "", arc_score: float = -1.0, arc_events: int = 0) -> Dict[str, Any]:
            wear_level = wear_level or str(self._session.get("wear_level", ""))
            arc_score = arc_score if arc_score >= 0 else float(self._session.get("arc_score", -1))
            arc_events = arc_events or int(self._session.get("arc_events", 0))

            severity = "NORMAL"
            reasons: List[str] = []
            if self._session.get("kind") == "wear":
                mapping = {"normal": "NORMAL", "light": "OBSERVE", "heavy": "WARNING"}
                severity = mapping.get(wear_level, "UNKNOWN")
                reasons.append(f"滑板磨耗等级 {wear_level}")
            else:
                severity = "NORMAL" if arc_score < 20 else ("OBSERVE" if arc_score < 45 else ("WARNING" if arc_score < 70 else "HIGH"))
                reasons.append(f"电弧风险评分 {arc_score}")
            if severity in ("WARNING", "HIGH") and self._session.get("recent_arc_count", 0) > 2:
                severity = "HIGH"
                reasons.append("叠加历史电弧频次升高，按规程提升处置优先级")
            return {
                "severity": severity,
                "reasons": reasons,
                "recommended_action": _ADVICE.get(severity, _ADVICE["OBSERVE"]),
                "review_points": ["核对滑板磨耗面与裂纹", "复核弓网接触状态与燃弧痕迹"],
            }

        @tools.register(name="retrieve_kb", description="检索受电弓检修处置依据")
        def retrieve_kb() -> Dict[str, Any]:
            kw = ["电弧", "滑板磨耗", "处置"] if self._session.get("kind") == "arc" else ["滑板磨耗", "处置"]
            result = self.rag.retrieve(asset_type="pantograph", keywords=kw)
            if result.refused:
                return {"refused": True, "note": result.refused_text, "citations": []}
            return {"refused": False, "note": "", "citations": result.answer_basis}

    def conclude(self, event: Dict[str, Any], capability_result: Dict[str, Any]) -> Any:
        diag = capability_result.get("diagnosis", {})
        anomaly = str(diag.get("anomaly_type", ""))
        self._session = {
            "train_id": event.get("asset", {}).get("train_id", ""),
            "component": event.get("asset", {}).get("component", "pantograph"),
            "kind": "arc" if "arc" in anomaly else "wear",
            "wear_level": diag.get("wear_level", ""),
            "arc_score": diag.get("arc_score", -1),
            "arc_events": diag.get("arc_events", 0),
            "confidence": diag.get("confidence", 0.0),
        }
        if isinstance(self.provider, EchoScriptedProvider):
            self.provider.reset(self._script())
        return super().conclude(event, capability_result)

    def _script(self) -> List[Any]:
        def _final(ctx: Dict[str, Any]) -> Any:
            grade = _ctx_value(ctx, "grade_event")
            hist = _ctx_value(ctx, "query_panto_history")
            kb = _ctx_value(ctx, "retrieve_kb")
            diag_desc = self._session.get("description") or (
                f"电弧事件 {self._session.get('arc_events')} 次 / 评分 {self._session.get('arc_score')}"
                if self._session.get("kind") == "arc" else f"磨耗等级 {self._session.get('wear_level')}"
            )
            conclusion = {
                "conclusion": (
                    f"受电弓综合诊断：{diag_desc}；近1小时相关事件 {hist.get('recent_count', 1)} 次"
                    f"（电弧 {hist.get('recent_arc_count', 0)} 次），综合等级 {grade.get('severity')}。"
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
            ("查询受电弓历史异常并对本次事件分级", [
                ("query_panto_history", {}),
                ("grade_event", {}),
            ]),
            ("检索受电弓检修处置依据", [("retrieve_kb", {})]),
            _final,
        ]
