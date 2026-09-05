"""Cabin-Agent：车内开放场景巡检领域专业 Agent（方案 5.5）。

输入：车厢巡检 VLM 能力（internal.cabin.patrol_vlm）的统一诊断事件。
职责：1)核对相邻窗口场景判定与一致性；2)按演示规程执行升级规则
（连续两窗口一致才升级，绝不输出"确认暴力事件"等确定性结论）；
3)检索车内处置知识依据。走 pi 循环（Echo 脚本驱动），确定性分级。
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
    "你是高铁车厢巡检专业Agent。输入是车厢巡检VLM能力产生的统一诊断事件。"
    "你的职责：1)核对该车厢近期巡检窗口记录；2)按方案5.5演示规程执行升级规则："
    "连续两个窗口结论一致才升级告警，单窗口异常只观察，低置信度/降级转人工复核，"
    "绝不输出'确认暴力事件'等确定性结论；3)检索车内处置知识依据。"
    "最后必须输出一个JSON对象，字段：conclusion(结论一句话)、severity、confidence、"
    "windows_consistent、review_points(复核要点数组)、recommended_action、kb_refs(引用数组)。"
    "不得虚构规程条款，知识不足时在conclusion中说明。"
)

# 方案 5.5 事件类别表：场景 → (两窗口一致时的等级, 处置建议, 复核要点)
_SCENE_RULES = {
    "smoke": ("HIGH", "立即通知乘务员现场确认并按火灾预案处置",
              ["调取相邻车厢画面交叉确认", "核对烟感/温感系统告警"]),
    "person_on_floor": ("HIGH", "乘务员立即前往现场查看，广播寻医",
                        ["确认旅客意识与伤情", "联控前方车站做好救护准备"]),
    "aisle_blocked": ("WARNING", "广播疏导乘客归置行李，恢复通道",
                      ["下一停靠站复核通道状态"]),
    "door_obstruction": ("WARNING", "乘务员清理车门区域，确认不影响开关门",
                         ["发车前确认车门区域清空"]),
    "luggage_overhang": ("WARNING", "提醒乘客归置行李架物品，防止坠落",
                         ["巡视全列行李架状态"]),
    "abnormal_gathering": ("OBSERVE", "乘务员前往查看聚集原因，维持秩序",
                           ["连续窗口持续跟踪"]),
    "violent_motion": ("OBSERVE", "乘务员介入了解情况（不下确定性结论）",
                       ["必要时联控公安，保留证据帧"]),
    "normal": ("NORMAL", "车厢秩序正常，保持例行巡检", ["按班次例行巡视"]),
    "unknown": ("UNKNOWN", "画面置信度不足，重新抽帧或转人工复核", ["人工调阅原始视频"]),
}


def _ctx_value(ctx: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    for name, payload in ctx.get("tool_results", []):
        if name == tool_name and isinstance(payload, dict) and payload.get("ok"):
            return payload.get("value") or {}
    return {}


class CabinAgent(SpecialistAgent):
    domain = "cabin"

    def __init__(self, event_store: EventStore, rag: RagClient, provider: LLMProvider, on_event=None):
        self.event_store = event_store
        self.rag = rag
        self._session: Dict[str, Any] = {}
        tools = ToolRegistry()
        self._register_tools(tools)
        super().__init__(name="Cabin-Agent", provider=provider, system=SYSTEM, tools=tools, on_event=on_event)

    def _register_tools(self, tools: ToolRegistry) -> None:
        @tools.register(name="query_cabin_history", description="查询该车厢近1小时巡检窗口记录")
        def query_cabin_history(train_id: str = "") -> Dict[str, Any]:
            train_id = train_id or self._session.get("train_id", "")
            since = time.time() - 3600.0
            rows = self.event_store.query(train_id=train_id, anomaly_type="cabin_patrol", since_ts=since)
            return {"train_id": train_id, "recent_windows": len(rows), "window_s": 3600.0}

        @tools.register(name="grade_cabin_scene", description="按方案5.5演示规程执行窗口升级规则")
        def grade_cabin_scene() -> Dict[str, Any]:
            scenes: List[str] = list(self._session.get("scenes", []))
            consistent = bool(self._session.get("consistent"))
            degraded = bool(self._session.get("degraded"))
            if degraded or not scenes or "unknown" in scenes:
                severity, action, points = _SCENE_RULES["unknown"]
            elif consistent:
                severity, action, points = _SCENE_RULES.get(scenes[0], _SCENE_RULES["unknown"])
            else:
                # 相邻窗口结论不一致：不升级，仅观察并人工复核（方案 5.5）
                severity, action, points = "OBSERVE", "相邻窗口结论不一致，暂不升级，转人工复核", ["人工调阅两窗口原始视频"]
            return {"scene": scenes[0] if scenes else "unknown", "windows": len(scenes),
                    "consistent": consistent, "severity": severity,
                    "recommended_action": action, "review_points": points}

        @tools.register(name="retrieve_kb", description="检索车内异常处置知识依据")
        def retrieve_kb() -> Dict[str, Any]:
            scene = (self._session.get("scenes") or ["unknown"])[0]
            keywords = ["车内巡检", "处置"]
            if scene == "smoke":
                keywords.append("火灾")
            result: RagResult = self.rag.retrieve(asset_type="cabin", keywords=keywords)
            if result.refused:
                return {"refused": True, "note": result.refused_text, "citations": []}
            return {"refused": False, "note": "", "citations": result.answer_basis}

    def conclude(self, event: Dict[str, Any], capability_result: Dict[str, Any]) -> Any:
        diagnosis = capability_result.get("diagnosis", {})
        windows = diagnosis.get("windows", [])
        self._session = {
            "train_id": event.get("asset", {}).get("train_id", ""),
            "scenes": [str(w.get("scene", "unknown")) for w in windows],
            "consistent": bool(diagnosis.get("consistency", {}).get("consistent")),
            "degraded": bool(diagnosis.get("degraded", False)),
            "confidence": diagnosis.get("confidence", 0.0),
        }
        if isinstance(self.provider, EchoScriptedProvider):
            self.provider.reset(self._script())
        return super().conclude(event, capability_result)

    def _script(self) -> List[Any]:
        def _final(ctx: Dict[str, Any]) -> Any:
            grade = _ctx_value(ctx, "grade_cabin_scene")
            hist = _ctx_value(ctx, "query_cabin_history")
            kb = _ctx_value(ctx, "retrieve_kb")
            scene_cn = {"normal": "正常", "smoke": "疑似烟雾或火光", "person_on_floor": "人员持续倒地",
                        "aisle_blocked": "通道持续堵塞", "door_obstruction": "车门区域阻挡",
                        "luggage_overhang": "行李架物品伸出", "abnormal_gathering": "异常聚集",
                        "violent_motion": "疑似剧烈动作", "unknown": "无法判断"}.get(grade.get("scene"), grade.get("scene"))
            conclusion = {
                "conclusion": (
                    f"车厢巡检 {grade.get('windows', 0)} 个窗口场景判定「{scene_cn}」，"
                    f"{'连续两窗口结论一致，按演示规程等级 ' + str(grade.get('severity'))}"
                    if grade.get("consistent") else "相邻窗口结论不一致或存在降级，按方案5.5不升级、转人工复核。"
                ),
                "severity": grade.get("severity"),
                "confidence": self._session.get("confidence"),
                "windows_consistent": grade.get("consistent"),
                "review_points": grade.get("review_points", []),
                "recommended_action": grade.get("recommended_action"),
                "kb_refs": kb.get("citations", []),
                "kb_note": kb.get("note") or None,
            }
            return json.dumps(conclusion, ensure_ascii=False), []

        return [
            ("查询近期巡检窗口记录并按演示规程执行升级规则", [("query_cabin_history", {"train_id": ""}), ("grade_cabin_scene", {})]),
            ("检索车内处置知识依据", [("retrieve_kb", {})]),
            _final,
        ]
