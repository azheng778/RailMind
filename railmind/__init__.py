"""RailMind —— 全空间、动静结合的列车智能运维智能体平台。

极简 Agent 内核（pi 风格）：
    agent = while 循环（LLM/Provider → 工具调用 → 结果回填 → 重复），
    一切领域能力（模型、规则、既有系统）通过能力注册中心以统一协议接入。
"""

__version__ = "0.1.0"

from railmind.core.engine import Agent, AgentResult
from railmind.core.registry import CapabilityRegistry
from railmind.core.risk_engine import RiskEngine

__all__ = ["Agent", "AgentResult", "CapabilityRegistry", "RiskEngine", "__version__"]
