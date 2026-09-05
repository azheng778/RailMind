"""SHM-Agent：复合材料结构健康监测领域专业 Agent。

输入：第 7 号能力（冲击定位与能量分级）的统一诊断事件；
职责：查询资产历史冲击（时序一致性）、按检修规程给出分级与处置建议、
检索知识引用，输出结构化领域结论。同样走 pi 循环（Echo 脚本驱动）。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from railmind.agents.base import SpecialistAgent
from railmind.core.events import EventStore
from railmind.core.provider import EchoScriptedProvider, LLMProvider
from railmind.core.rag import RagClient, RagResult
from railmind.core.tools import ToolRegistry

SYSTEM = (
    "你是复合材料结构健康监测(SHM)专业Agent。输入是冲击定位能力产生的统一诊断事件。"
    "你的职责：1)核对该资产历史冲击记录，统计1小时内的冲击频次；"
    "2)按冲击能量分级规程给出严重等级与处置建议；3)检索知识依据。"
    "最后必须输出一个JSON对象，字段：conclusion(结论一句话)、severity、confidence、"
    "frequency_1h、review_points(复核要点数组)、recommended_action、kb_refs(引用数组)。"
    "不得虚构规程条款，知识不足时在conclusion中说明。"
)

_ADVICE = {
    "HIGH": "显著冲击：立即安排人工敲击检测或无损检测复核，评估是否限制运行",
    "WARNING": "重点关注冲击：下一停靠站目视复核",
    "OBSERVE": "轻微冲击：记录并持续观察",
}


def _ctx_value(ctx: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    for name, payload in ctx.get("tool_results", []):
        if name == tool_name and isinstance(payload, dict) and payload.get("ok"):
            return payload.get("value") or {}
    return {}


class SHMAgent(SpecialistAgent):
    domain = "composite_structure"

    def __init__(self, event_store: EventStore, rag: RagClient, provider: LLMProvider, on_event=None):
        self.event_store = event_store
        self.rag = rag
        self._session: Dict[str, Any] = {}
        tools = ToolRegistry()
        self._register_tools(tools)
        super().__init__(name="SHM-Agent", provider=provider, system=SYSTEM, tools=tools, on_event=on_event)

    # ---------- 工具（参数可由脚本显式传入；缺省时取会话上下文） ----------

    def _register_tools(self, tools: ToolRegistry) -> None:
        @tools.register(name="query_impact_history", description="查询该列车复合材料结构近1小时冲击记录")
        def query_impact_history(train_id: str = "") -> Dict[str, Any]:
            train_id = train_id or self._session.get("train_id", "")
            since = time.time() - 3600.0
            rows = self.event_store.query(train_id=train_id, anomaly_type="composite_impact", since_ts=since)
            return {"train_id": train_id, "recent_count": len(rows), "window_s": 3600.0}

        @tools.register(name="grade_impact", description="按检修规程对最大冲击能量做确定性分级")
        def grade_impact(max_energy_j: float = 0.0) -> Dict[str, Any]:
            max_energy_j = max_energy_j or float(self._session.get("max_energy_j", 0.0))
            from railmind.capabilities.shm_impact.adapter import energy_to_severity

            severity = energy_to_severity(max_energy_j)
            return {
                "max_energy_j": max_energy_j,
                "severity": severity,
                "recommended_action": _ADVICE[severity],
                "review_points": ["核对定位坐标圈定的复核区域", "对比两侧传感器通道波形一致性"],
            }

        @tools.register(name="retrieve_kb", description="检索复合材料冲击处置知识依据")
        def retrieve_kb() -> Dict[str, Any]:
            energy = float(self._session.get("max_energy_j", 0.0))
            keywords = ["冲击能量分级", "处置"]
            if energy >= 0.50:
                keywords.append("双点冲击")
            result: RagResult = self.rag.retrieve(asset_type="composite_structure", keywords=keywords)
            if result.refused:
                return {"refused": True, "note": result.refused_text, "citations": []}
            return {"refused": False, "note": "", "citations": result.answer_basis}

    # ---------- 编排 ----------

    def conclude(self, event: Dict[str, Any], capability_result: Dict[str, Any]) -> Any:
        diagnosis = capability_result.get("diagnosis", {})
        self._session = {
            "train_id": event.get("asset", {}).get("train_id", ""),
            "max_energy_j": float(diagnosis.get("max_energy_j", 0.0)),
            "degraded": bool(diagnosis.get("degraded", False)),
            "confidence": diagnosis.get("confidence", 0.0),
        }
        if isinstance(self.provider, EchoScriptedProvider):
            self.provider.reset(self._script())
        return super().conclude(event, capability_result)

    def _script(self) -> List[Any]:
        train_id = self._session["train_id"]
        energy = self._session["max_energy_j"]

        def _final(ctx: Dict[str, Any]) -> Any:
            grade = _ctx_value(ctx, "grade_impact")
            hist = _ctx_value(ctx, "query_impact_history")
            kb = _ctx_value(ctx, "retrieve_kb")
            # 当前事件在能力调用时已入事件库，recent_count 已包含本次
            frequency = int(hist.get("recent_count", 1))
            conclusion = {
                "conclusion": (
                    f"复合材料结构监测到冲击，最大冲击能量 {grade.get('max_energy_j', energy):.2f}J，"
                    f"近1小时内第 {frequency} 次，按检修规程等级 {grade.get('severity')}。"
                ),
                "severity": grade.get("severity"),
                "confidence": self._session.get("confidence"),
                "frequency_1h": frequency,
                "review_points": grade.get("review_points", []),
                "recommended_action": grade.get("recommended_action"),
                "kb_refs": kb.get("citations", []),
                "kb_note": kb.get("note") or None,
            }
            return json.dumps(conclusion, ensure_ascii=False), []

        return [
            ("查询历史冲击频次并对最大冲击能量做确定性分级", [("query_impact_history", {"train_id": train_id}), ("grade_impact", {"max_energy_j": energy})]),
            ("按能量等级检索处置知识依据", [("retrieve_kb", {})]),
            _final,
        ]
