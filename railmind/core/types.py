"""核心数据类型：消息、工具调用、事件 —— 全部是普通 dataclass，保持内核零魔法。

消息结构沿用 OpenAI 兼容形态（dict），方便 Provider 直接对接任意
OpenAI 兼容推理服务；工具定义用 JSON Schema 描述参数。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


def new_id(prefix: str) -> str:
    """生成带前缀的短 ID，如 EVT-20260904-ab12cd34。"""
    stamp = time.strftime("%Y%m%d")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


@dataclass
class ToolCall:
    """一次工具调用请求（由 Provider 产出）。"""

    id: str
    name: str
    arguments: Dict[str, Any]

    @staticmethod
    def parse_arguments(raw: Any) -> Dict[str, Any]:
        """容忍字符串 / dict 两种入参形态。"""
        if raw is None or raw == "":
            return {}
        if isinstance(raw, dict):
            return raw
        return json.loads(raw)


@dataclass
class ProviderResponse:
    """Provider 单轮补全结果。"""

    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # end_turn | tool_use | max_iterations | error
    model: str = "echo"
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class ToolResult:
    """工具执行结果：ok=False 时 error 为机器可读错误码。"""

    ok: bool
    value: Any = None
    error: Optional[str] = None


# 消息即 dict：{"role", "content", "tool_calls"?, "tool_call_id"?}
Message = Dict[str, Any]


@dataclass
class AgentEvent:
    """Agent 循环过程事件，用于前端展示"融合过程"与审计。"""

    kind: str  # assistant_message | tool_start | tool_end | done | error
    agent: str
    payload: Dict[str, Any]
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "agent": self.agent, "payload": self.payload, "ts": self.ts}


EventListener = Callable[[AgentEvent], None]


def assistant_message(content: Optional[str], tool_calls: Optional[List[ToolCall]] = None) -> Message:
    msg: Message = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
            }
            for tc in tool_calls
        ]
    return msg


def tool_message(call: ToolCall, result: ToolResult) -> Message:
    payload = {"ok": result.ok, "value": result.value} if result.ok else {"ok": False, "error": result.error}
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": json.dumps(payload, ensure_ascii=False, default=str),
    }
