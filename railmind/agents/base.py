"""专业 Agent 基类：pi 内核之上的领域封装。

专业 Agent 与 Chief-Agent 用的是同一个 Agent 循环，差别只在
工具集、系统提示与编排脚本 —— 这是"内核极简、能力分层"的关键。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from typing import Optional

from railmind.core.engine import Agent, AgentResult
from railmind.core.provider import LLMProvider
from railmind.core.tools import ToolRegistry


class SpecialistAgent(Agent):
    domain = "generic"

    def __init__(self, name: str, provider: LLMProvider, system: str, tools: Optional[ToolRegistry] = None, **kwargs):
        super().__init__(name=name, provider=provider, tools=tools or ToolRegistry(), system=system, **kwargs)

    def conclude(self, event: Dict[str, Any], capability_result: Dict[str, Any]) -> AgentResult:
        """对一条统一诊断事件做领域综合分析，产出结构化结论（JSON 文本）。"""
        return self.run({"event": event, "capability_result": capability_result})
