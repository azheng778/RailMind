"""Chief-Agent —— 方案 6.2 工作流（pi 循环驱动）：

接收异常事件 → 判断场景 → 查询能力注册中心 → 调用专业能力/备用能力
→ 专业 Agent 综合分析 → 风险融合（确定性引擎）→ 查询 RAG → 处置建议
→ 高风险生成工单草稿（等人工确认）。

大脑 = 真实 LLM（OpenAICompatibleProvider，自主决定工具调用次序）；
工具参数一律从会话状态读取 —— LLM 负责"何时做什么"，数据不经 LLM 转手，
杜绝转录幻觉；风险等级仍由确定性风险引擎计算（方案 6.3）。
Provider 故障或 LLM 未走完编排时，自动落回 Echo 脚本重跑（演示稳定性）。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from railmind.core.engine import Agent, AgentResult
from railmind.core.events import EventStore
from railmind.core.provider import EchoScriptedProvider, LLMProvider
from railmind.core.rag import RagClient
from railmind.core.registry import CapabilityInstance, CapabilityRegistry
from railmind.core.risk_engine import LEVEL_HIGH, LEVEL_WARNING, RiskEngine
from railmind.core.tools import ToolRegistry
from railmind.core.types import AgentEvent, EventListener
from railmind.core.workorder import WorkOrderStore

_KB_BY_DOMAIN = {
    "composite_structure": {"asset_type": "composite_structure", "keywords": ["处置", "冲击"]},
    "underbody": {"asset_type": "underbody", "keywords": ["检查门", "处置", "锁闭"]},
    "pantograph": {"asset_type": "pantograph", "keywords": ["处置", "电弧", "滑板磨耗"]},
    "lineside": {"asset_type": "lineside", "keywords": ["异物", "处置", "限界"]},
}

SYSTEM = (
    "你是RailMind首席编排Agent（Chief）。对每条诊断事件，严格按以下顺序使用工具完成编排："
    "1) find_capability 定位可用能力；"
    "2) invoke_capability 调用该能力（只调用一次）；"
    "3) consult_specialist 请对应领域专业Agent综合分析（若能力降级/拒判则跳过）；"
    "4) assess_risk 与 retrieve_kb 做风险融合与知识检索；"
    "5) 风险为WARNING/HIGH且需人工复核时调用 draft_workorder 生成工单草稿。"
    "所有工具参数留空即可（平台自动注入上下文）。工具全部完成后，用一句话总结处置结论。"
)


class CapabilityRouter:
    """能力调用器：进程内能力直调，REST 能力走 HTTP（热加载模块）。"""

    def __init__(self) -> None:
        self._local: Dict[str, Any] = {}  # capability_id -> InProcessCapability
        self._rest: Dict[str, Dict[str, Any]] = {}  # capability_id -> {base_url, endpoint, timeout_s}

    def attach_local(self, capability: Any) -> None:
        self._local[capability.descriptor["capability_id"]] = capability

    def attach_rest(self, descriptor: Dict[str, Any], base_url: str) -> None:
        self._rest[descriptor["capability_id"]] = {
            "base_url": base_url.rstrip("/"),
            "endpoint": descriptor.get("invoke", {}).get("endpoint", "/api/v1/inference"),
            "timeout_s": descriptor.get("invoke", {}).get("timeout_ms", 5000) / 1000.0,
        }

    def invoke(self, instance: CapabilityInstance, payload: Dict[str, Any]) -> Dict[str, Any]:
        cap_id = instance.capability_id
        if cap_id in self._local:
            return self._local[cap_id].invoke(payload)
        if cap_id in self._rest:
            import requests  # noqa: PLC0415

            cfg = self._rest[cap_id]
            resp = requests.post(f"{cfg['base_url']}{cfg['endpoint']}", json=payload, timeout=cfg["timeout_s"])
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                raise RuntimeError(f"E_REMOTE_INFERENCE:{data.get('error')}")
            return data["value"]
        raise RuntimeError(f"E_NO_INVOKER:{cap_id} 未注册本地或REST调用通道")


def _ctx_value(ctx: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """取某工具最近一次成功结果（ctx.tool_results 按时间序，取末次匹配）。"""
    value: Dict[str, Any] = {}
    for name, payload in ctx.get("tool_results", []):
        if name == tool_name and isinstance(payload, dict) and payload.get("ok"):
            value = payload.get("value") or {}
    return value


class ChiefAgent:
    def __init__(
        self,
        registry: CapabilityRegistry,
        router: CapabilityRouter,
        risk: RiskEngine,
        rag: RagClient,
        workorders: WorkOrderStore,
        event_store: EventStore,
        specialists: Dict[str, Any],
        provider: LLMProvider,
        on_event: Optional[EventListener] = None,
    ):
        self.registry = registry
        self.router = router
        self.risk = risk
        self.rag = rag
        self.workorders = workorders
        self.event_store = event_store
        self.specialists = specialists  # domain -> SpecialistAgent
        self.trace: List[AgentEvent] = []

        def listener(event: AgentEvent) -> None:
            self.trace.append(event)
            if on_event:
                on_event(event)

        self._session: Dict[str, Any] = {}
        self._tools = ToolRegistry()
        self._impls: Dict[str, Any] = {}
        self._register_tools(self._tools)
        for _n in ("assess_risk", "retrieve_kb", "draft_workorder"):
            self._impls[_n] = self._tools.get(_n).handler
        self._agent = Agent(name="Chief-Agent", provider=provider, tools=self._tools, system=SYSTEM, max_iterations=12, on_event=listener)

    # ---------- 编排工具（参数留空 = 平台注入会话上下文） ----------

    def _register_tools(self, tools: ToolRegistry) -> None:
        @tools.register(name="find_capability", description="按领域/场景定位可分派的诊断能力", parameters={
            "type": "object", "properties": {}, "required": [],
        })
        def find_capability() -> Dict[str, Any]:
            domain = self._session["domain"]
            scene = self._session["scene"]
            self.registry.sweep_offline()
            target = self.registry.dispatch_target(domain=domain, scene=scene)
            if target is None:
                self._session["capability"] = None
                return {"found": False}
            self._session["capability"] = target.to_public_dict()
            return {"found": True, "capability": target.to_public_dict()}

        @tools.register(name="invoke_capability", description="调用已定位的能力执行一次诊断推理", parameters={
            "type": "object", "properties": {}, "required": [],
        })
        def invoke_capability() -> Dict[str, Any]:
            cap = self._session.get("capability")
            if not cap:
                raise RuntimeError("E_NO_CAPABILITY:请先调用 find_capability")
            instance = next(
                (i for i in self.registry.get(cap["capability_id"]) if i.is_online(self.registry.heartbeat_ttl_s)), None
            )
            if instance is None:
                raise RuntimeError(f"E_CAPABILITY_OFFLINE:{cap['capability_id']}")
            result = self.router.invoke(instance, self._session["capability_input"])
            self.event_store.append(result)
            self._session["cap_result"] = result
            return result

        @tools.register(name="consult_specialist", description="请领域专业Agent综合分析本次事件", parameters={
            "type": "object", "properties": {}, "required": [],
        })
        def consult_specialist() -> Dict[str, Any]:
            domain = self._session["domain"]
            agent = self.specialists.get(domain)
            cap_result = self._session.get("cap_result") or {}
            if agent is None:
                diag = cap_result.get("diagnosis", {})
                return {
                    "conclusion": f"{domain} 领域暂无专业Agent，采用能力原始结论：{diag.get('severity', 'UNKNOWN')}",
                    "severity": diag.get("severity", "UNKNOWN"),
                    "confidence": diag.get("confidence", 0.0),
                    "recommended_action": "人工复核",
                }
            agent_result: AgentResult = agent.conclude(self._session["event"], cap_result)
            conclusion = agent_result.text_or_json
            if not isinstance(conclusion, dict):
                conclusion = {"conclusion": str(conclusion)}
            self._session["conclusion"] = conclusion
            return conclusion

        @tools.register(name="assess_risk", description="确定性风险融合（等级由引擎计算，不由大模型生成）", parameters={
            "type": "object", "properties": {}, "required": [],
        })
        def assess_risk() -> Dict[str, Any]:
            event = self._session["event"]
            conclusion = self._session.get("conclusion") or {}
            diag = (self._session.get("cap_result") or {}).get("diagnosis", {})
            cap = self._session.get("capability") or {}
            capability_online = bool(cap) and not diag.get("degraded")
            confidence = conclusion.get("confidence")
            if confidence is None:
                confidence = diag.get("confidence", 0.0)
            assessment = self.risk.assess(
                severity=conclusion.get("severity") or diag.get("severity") or "UNKNOWN",
                confidence=float(confidence),
                frequency_1h=int(conclusion.get("frequency_1h", 1)),
                running=event.get("operation_context", {}).get("scene") == "TRAIN_RUNNING",
                critical_component=bool(event.get("asset", {}).get("critical", False)),
                open_workorders=self.workorders.open_count(event.get("asset", {}).get("train_id")),
                capability_online=capability_online,
                capability_degraded=bool(diag.get("degraded", False)),
                capability_mode=cap.get("mode", "PRIMARY"),
            )
            self._session["risk"] = assessment.to_dict()
            return self._session["risk"]

        @tools.register(name="retrieve_kb", description="检索处置知识依据", parameters={
            "type": "object", "properties": {}, "required": [],
        })
        def retrieve_kb() -> Dict[str, Any]:
            query = _KB_BY_DOMAIN.get(self._session["domain"], {"asset_type": self._session["domain"], "keywords": ["处置"]})
            result = self.rag.retrieve(**query)
            if result.refused:
                self._session["kb"] = []
                return {"refused": True, "text": result.refused_text}
            self._session["kb"] = result.answer_basis
            return {"refused": False, "citations": result.answer_basis}

        @tools.register(name="draft_workorder", description="为高风险事件创建工单草稿（待人工确认）", parameters={
            "type": "object", "properties": {}, "required": [],
        })
        def draft_workorder() -> Dict[str, Any]:
            order = self.workorders.create_draft(
                event=self._session["event"],
                assessment=self._session.get("risk", {}),
                conclusion=self._session.get("conclusion", {}),
                knowledge=self._session.get("kb", []),
            )
            self._session["workorder"] = {"workorder_id": order["workorder_id"], "status": order["status"], "risk_level": order["risk"].get("level")}
            return dict(self._session["workorder"])

    # ---------- 对外入口 ----------

    def handle_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        self.trace = []
        event = dict(event)
        event.setdefault("event_id", f"EVT-{time.strftime('%Y%m%d')}-{int(time.time() * 1000) % 100000:05d}")
        event.setdefault("ts", time.time())
        asset = event.get("asset", {})
        context = event.get("operation_context", {})
        domain = event.get("domain", "composite_structure")
        scene = context.get("scene", "TRAIN_RUNNING")
        capability_input = {**event.get("capability_input", {}), "asset": asset, "operation_context": context}
        self._session = {
            "event": event,
            "domain": domain,
            "scene": scene,
            "capability_input": capability_input,
        }

        self._run()

        # 大脑未完成编排（LLM 提前收尾）→ 从会话状态兜底组装摘要
        if self._session.get("summary") is None:
            self._ensure_summary()

        summary = dict(self._session["summary"])
        summary["trace"] = [e.to_dict() for e in self.trace]

        # 附上告警表需要的扁平字段并存入摘要通道
        summary.update({
            "ts": event["ts"],
            "train_id": asset.get("train_id"),
            "component": asset.get("component"),
            "anomaly_type": (self._session.get("cap_result") or {}).get("diagnosis", {}).get("anomaly_type") or "chief_decision",
            "risk_level": (self._session.get("risk") or {}).get("level", "UNKNOWN"),
            "workorder_id": (self._session.get("workorder") or {}).get("workorder_id"),
        })
        self.event_store.append_summary(summary)
        return summary

    def _run(self) -> None:
        provider = self._agent.provider
        if isinstance(provider, EchoScriptedProvider):
            provider.reset(self._script())
        self._agent.run(self._session["event"])

        # Provider 故障（网络等）→ 本地 Echo 脚本降级重跑（演示稳定性，方案 2.4）
        if self._session.get("summary") is None and not isinstance(provider, EchoScriptedProvider):
            had_provider_error = any(e.kind == "error" for e in self.trace)
            if had_provider_error:
                self.trace = []
                echo = EchoScriptedProvider(self._script())
                old = self._agent.provider
                self._agent.provider = echo
                try:
                    self._agent.run(self._session["event"])
                finally:
                    self._agent.provider = old

    # ---------- 摘要（从会话状态确定性组装，不依赖 LLM 措辞） ----------

    def _ensure_summary(self) -> Dict[str, Any]:
        """闭环兜底：大脑漏掉的编排步骤由平台确定性补齐（方案 6.2 顺序不可缺）。"""
        cap_result = self._session.get("cap_result") or {}
        diag = cap_result.get("diagnosis", {}) if isinstance(cap_result, dict) else {}
        degraded = (not cap_result) or bool(diag.get("degraded")) or diag.get("severity") == "UNKNOWN"

        if not degraded and self._session.get("conclusion") is None:
            agent = self.specialists.get(self._session["domain"])
            if agent is not None:
                agent_result = agent.conclude(self._session["event"], cap_result)
                conclusion = agent_result.text_or_json
                self._session["conclusion"] = conclusion if isinstance(conclusion, dict) else {"conclusion": str(conclusion)}

        if self._session.get("risk") is None:
            self._impls["assess_risk"]()
        if self._session.get("kb") is None:
            self._impls["retrieve_kb"]()

        kb = self._session.get("kb")
        if kb is None:
            kb = self._session["kb"] = []
        wo = self._session.get("workorder")
        risk = self._session.get("risk") or {"level": "UNKNOWN", "score": 0.0, "reasons": ["编排未完成"], "human_review_required": True}
        if risk.get("level") in (LEVEL_WARNING, LEVEL_HIGH) and risk.get("human_review_required") and wo is None:
            self._impls["draft_workorder"]()
            wo = self._session.get("workorder")
        summary = {
            "event_id": self._session["event"].get("event_id"),
            "capability": self._session.get("capability"),
            "specialist_conclusion": self._session.get("conclusion") or {"conclusion": "（大脑未完成专业分析，转人工复核）"},
            "risk": risk,
            "knowledge": kb if isinstance(kb, list) else [],
            "workorder": wo,
            "human_review_required": risk.get("human_review_required", True),
        }
        self._session["summary"] = summary
        return json.dumps(summary, ensure_ascii=False, default=str)

    # ---------- Echo 降级编排脚本（方案 6.2 的线性展开） ----------

    def _script(self) -> List[Any]:
        step_find = ("定位可用诊断能力", [("find_capability", {})])

        def _step_invoke(ctx: Dict[str, Any]) -> Any:
            found = _ctx_value(ctx, "find_capability")
            if not found.get("found"):
                self._session["capability"] = None
                self._session["conclusion"] = {"conclusion": "无在线诊断能力，按拒判处理，转人工复核。"}
                self._session["risk_ready"] = True
                return "无可用能力，风险兜底", [("assess_risk", {}), ("retrieve_kb", {})]
            cap = found["capability"]
            self._session["capability"] = cap
            return f"调用能力 {cap['capability_id']} v{cap['version']}", [("invoke_capability", {})]

        def _step_specialist(ctx: Dict[str, Any]) -> Any:
            if self._session.get("risk_ready"):
                return _decide(ctx)
            cap_result = self._session.get("cap_result") or {}
            diagnosis = cap_result.get("diagnosis", {})
            if not cap_result or diagnosis.get("degraded") or diagnosis.get("severity") == "UNKNOWN":
                self._session["conclusion"] = self._session.get("conclusion") or {
                    "conclusion": "诊断能力处于降级模式，无法给出可靠结论，转人工复核。"
                }
                self._session["risk_ready"] = True
                return "能力降级，风险兜底", [("assess_risk", {}), ("retrieve_kb", {})]
            return "专业Agent综合分析", [("consult_specialist", {})]

        def _step_fuse(ctx: Dict[str, Any]) -> Any:
            conclusion = _ctx_value(ctx, "consult_specialist")
            if conclusion:
                self._session["conclusion"] = conclusion
            if self._session.get("risk_ready"):
                return _decide(ctx)
            return "风险融合与知识检索", [("assess_risk", {}), ("retrieve_kb", {})]

        def _decide(ctx: Dict[str, Any]) -> Any:
            risk = _ctx_value(ctx, "assess_risk")
            if risk:
                self._session["risk"] = risk
            kb = _ctx_value(ctx, "retrieve_kb")
            self._session["kb"] = kb.get("citations", []) if isinstance(kb, dict) else []
            level = self._session["risk"].get("level")
            if level in (LEVEL_WARNING, LEVEL_HIGH) and self._session["risk"].get("human_review_required"):
                return f"风险等级 {level}，生成工单草稿等待人工确认", [("draft_workorder", {})]
            return self._ensure_summary(), []

        def _step_summary(ctx: Dict[str, Any]) -> Any:
            return self._ensure_summary(), []

        return [step_find, _step_invoke, _step_specialist, _step_fuse, _decide, _step_summary]
