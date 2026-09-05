"""RailMind 对话助手：与运维人员对话的 LLM Agent（pi 循环）。

工具 = 平台实时数据（事件/工单/能力/知识检索/触发演示诊断）；
大脑 = 与 Chief 相同的远端 LLM；网络故障时降级为基于平台数据的直接回答。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from railmind.core.events import EventStore
from railmind.core.provider import LLMProvider
from railmind.core.rag import RagClient
from railmind.core.registry import CapabilityRegistry
from railmind.core.tools import ToolRegistry
from railmind.core.workorder import WorkOrderStore
SYSTEM = (
    "你是RailMind列车智能运维平台的对话助手。你可以：查询最近诊断事件与告警、查询工单、"
    "查看能力中心状态、检索运维知识、触发演示诊断（trigger_demo）。"
    "回答要用简洁中文，给出数据依据；涉及风险处置建议时引用知识条目。"
    "用户让你演示/模拟/测试某类诊断时，调用 trigger_demo。"
)


class ChatAgent:
    def __init__(
        self,
        registry: CapabilityRegistry,
        workorders: WorkOrderStore,
        event_store: EventStore,
        rag: RagClient,
        provider: LLMProvider,
        trigger_demo=None,
        on_event=None,
    ):
        self.registry = registry
        self.workorders = workorders
        self.event_store = event_store
        self.rag = rag
        self.trigger_demo_fn = trigger_demo
        self.tools = ToolRegistry()
        self._register_tools(self.tools)
        self.agent = __import__("railmind.core.engine", fromlist=["Agent"]).Agent(
            name="Chat-Agent",
            provider=provider,
            tools=self.tools,
            system=SYSTEM,
            max_iterations=8,
            on_event=on_event,
        )
        self.history: List[Dict[str, Any]] = []

    def _register_tools(self, tools: ToolRegistry) -> None:
        @tools.register(name="query_recent_events", description="查询平台最近的诊断事件与告警（含风险等级与工单号）", parameters={
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "条数，默认8"}},
            "required": [],
        })
        def query_recent_events(limit: int = 8) -> Dict[str, Any]:
            rows = self.event_store.list_summaries(limit=min(limit, 20))
            compact = [
                {
                    "time": r.get("ts"),
                    "train_id": r.get("train_id"),
                    "component": r.get("component"),
                    "risk": r.get("risk_level"),
                    "workorder": r.get("workorder_id"),
                    "conclusion": (r.get("specialist_conclusion") or {}).get("conclusion", ""),
                }
                for r in rows
            ]
            return {"events": compact, "total": len(compact)}

        @tools.register(name="query_workorders", description="查询工单列表与状态", parameters={
            "type": "object", "properties": {}, "required": [],
        })
        def query_workorders() -> Dict[str, Any]:
            rows = self.workorders.list()
            return {
                "count": len(rows),
                "open": self.workorders.open_count(),
                "orders": [
                    {"id": w["workorder_id"], "status": w["status"], "risk": w.get("risk", {}).get("level"),
                     "train": w.get("train_id"), "component": w.get("component")}
                    for w in rows[:12]
                ],
            }

        @tools.register(name="list_capabilities", description="列出能力中心当前注册的能力与运行模式", parameters={
            "type": "object", "properties": {}, "required": [],
        })
        def list_capabilities() -> Dict[str, Any]:
            return {"capabilities": [c.to_public_dict() for c in self.registry.all()]}

        @tools.register(name="kb_search", description="检索运维知识库（规程/处置方法）", parameters={
            "type": "object",
            "properties": {
                "asset_type": {"type": "string", "description": "composite_structure/pantograph/underbody/lineside"},
                "keywords": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["asset_type", "keywords"],
        })
        def kb_search(asset_type: str, keywords: List[str]) -> Dict[str, Any]:
            result = self.rag.retrieve(asset_type=asset_type, keywords=keywords)
            if result.refused:
                return {"refused": True, "text": result.refused_text}
            return {"citations": result.answer_basis}

        @tools.register(name="trigger_demo", description="触发一次演示诊断链路（kind: shm_impact/door_normal/door_open/panto_wear/panto_arc/lineside_fod）", parameters={
            "type": "object",
            "properties": {"kind": {"type": "string", "description": "演示事件类型"}},
            "required": ["kind"],
        })
        def trigger_demo(kind: str) -> Dict[str, Any]:
            if self.trigger_demo_fn is None:
                return {"ok": False, "error": "平台未挂载演示触发器"}
            summary = self.trigger_demo_fn(kind)
            if summary is None:
                return {"ok": False, "error": f"未知演示类型 {kind}"}
            return {
                "ok": True,
                "event_id": summary.get("event_id"),
                "risk": summary.get("risk_level"),
                "workorder": summary.get("workorder_id"),
                "conclusion": (summary.get("specialist_conclusion") or {}).get("conclusion", ""),
            }

    def chat(self, message: str) -> Dict[str, Any]:
        """处理一条用户消息，返回 {reply, tool_trace}。"""
        self.history.append({"role": "user", "content": message})
        result = self.agent.run(message, history=self.history[-8:])
        reply = result.text or "（未能生成回复）"
        self.history.append({"role": "assistant", "content": reply})
        trace = [{"name": t["name"], "ok": t["ok"], "error": t["error"]} for t in result.tool_calls]
        return {"reply": reply, "tool_trace": trace}

    def chat_stream(self, message: str):
        """流式对话：逐段 yield 事件。

        {"type":"tool_start","name"} / {"type":"tool_end","name","ok"}
        {"type":"delta","text"} / {"type":"done","tool_trace"}
        首轮使用远端流式大脑；Provider 故障时降级为一次性本地回答。
        """
        from railmind.core.provider import EchoScriptedProvider  # noqa: PLC0415
        from railmind.core.types import assistant_message, tool_message  # noqa: PLC0415

        self.history.append({"role": "user", "content": message})
        messages = [{"role": "system", "content": SYSTEM}] + self.history[-9:]
        trace: List[Dict[str, Any]] = []

        try:
            for iteration in range(8):
                pending_calls = []
                final = None
                for piece in self.agent.provider.complete_stream(messages, self.tools.specs()):
                    if piece["type"] == "delta":
                        yield {"type": "delta", "text": piece["text"]}
                    else:
                        final = piece["response"]
                resp = final
                messages.append(assistant_message(resp.text, resp.tool_calls or None))
                if not resp.tool_calls:
                    break
                for call in resp.tool_calls:
                    yield {"type": "tool_start", "name": call.name}
                    result = self.tools.dispatch(call.name, call.arguments)
                    yield {"type": "tool_end", "name": call.name, "ok": result.ok}
                    trace.append({"name": call.name, "ok": result.ok, "error": result.error})
                    messages.append(tool_message(call, result))
            else:
                pass
            reply = _last_text(messages) or "（未能生成回复）"
        except Exception as exc:  # noqa: BLE001 —— 网络故障降级
            yield {"type": "delta", "text": "（LLM 大脑暂时不可用，以下为平台数据直查）\n"}
            fallback = self.tools.dispatch("query_recent_events", {})
            fallback2 = self.tools.dispatch("query_workorders", {})
            for r in (fallback, fallback2):
                if r.ok:
                    yield {"type": "delta", "text": json.dumps(r.value, ensure_ascii=False, default=str)[:400] + "\n"}
            reply = f"大脑降级：{exc}"
            trace.append({"name": "fallback", "ok": True, "error": None})

        self.history.append({"role": "assistant", "content": reply})
        yield {"type": "done", "reply": reply, "tool_trace": trace}


def _last_text(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return None
