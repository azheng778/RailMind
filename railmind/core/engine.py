"""Agent 引擎 —— pi 的核心思想：agent 就是一个 while 循环。

    while True:
        resp = provider.complete(messages, tools)
        记录 / 发事件
        if 没有工具调用: break
        执行工具，结果回填 messages

没有图、没有链、没有隐藏状态；编排过程全部经 AgentEvent 暴露，
前端"融合过程"与审计日志直接消费这些事件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from railmind.core.provider import LLMProvider
from railmind.core.tools import ToolRegistry
from railmind.core.types import AgentEvent, EventListener, Message, ToolCall, assistant_message, tool_message


@dataclass
class AgentResult:
    text: Optional[str]
    messages: List[Message] = field(default_factory=list)
    iterations: int = 0
    stop_reason: str = "end_turn"
    events: List[AgentEvent] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # 审计: [{name, arguments, ok, error}]

    @property
    def text_or_json(self) -> Optional[Any]:
        """最终文本若为 JSON 则解析（Agent 间结构化交接用）。"""
        if not self.text:
            return None
        try:
            return json.loads(self.text)
        except (TypeError, json.JSONDecodeError):
            return self.text


class Agent:
    def __init__(
        self,
        name: str,
        provider: LLMProvider,
        tools: Optional[ToolRegistry] = None,
        system: str = "",
        max_iterations: int = 10,
        on_event: Optional[EventListener] = None,
    ):
        self.name = name
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.system = system
        self.max_iterations = max_iterations
        self._listeners: List[EventListener] = []
        if on_event:
            self._listeners.append(on_event)

    def add_listener(self, listener: EventListener) -> None:
        self._listeners.append(listener)

    def _emit(self, kind: str, payload: Dict[str, Any], events: List[AgentEvent]) -> None:
        event = AgentEvent(kind=kind, agent=self.name, payload=payload)
        events.append(event)
        for listener in self._listeners:
            listener(event)

    def run(self, message: Any, history: Optional[List[Message]] = None) -> AgentResult:
        user_content = message if isinstance(message, str) else json.dumps(message, ensure_ascii=False, default=str)
        messages: List[Message] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_content})

        events: List[AgentEvent] = []
        audit: List[Dict[str, Any]] = []
        stop_reason = "max_iterations"
        iterations = 0

        for iterations in range(1, self.max_iterations + 1):
            try:
                resp = self.provider.complete(messages, self.tools.specs())
            except Exception as exc:  # noqa: BLE001 —— Provider 故障不向上抛，转为可审计事件
                stop_reason = "error"
                self._emit("error", {"error": f"E_PROVIDER:{exc}"}, events)
                break

            self._emit(
                "assistant_message",
                {"text": resp.text, "tool_calls": [tc.name for tc in resp.tool_calls], "model": resp.model},
                events,
            )
            messages.append(assistant_message(resp.text, resp.tool_calls or None))

            if not resp.tool_calls:
                stop_reason = resp.stop_reason
                break

            for call in resp.tool_calls:
                self._emit("tool_start", {"name": call.name, "arguments": call.arguments}, events)
                result = self.tools.dispatch(call.name, call.arguments)
                self._emit(
                    "tool_end",
                    {"name": call.name, "ok": result.ok, "error": result.error, "value": _safe_value(result.value)},
                    events,
                )
                audit.append(
                    {"name": call.name, "arguments": call.arguments, "ok": result.ok, "error": result.error}
                )
                messages.append(tool_message(call, result))

        text = _last_text(messages)
        result = AgentResult(
            text=text,
            messages=messages,
            iterations=iterations,
            stop_reason=stop_reason,
            events=events,
            tool_calls=audit,
        )
        self._emit("done", {"stop_reason": stop_reason, "iterations": iterations}, events)
        return result


def _last_text(messages: List[Message]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return None


def _safe_value(value: Any, limit: int = 4000) -> Any:
    """事件里带的工具结果截断，避免大对象刷屏。"""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)[:limit]
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return value
