"""平台装配：注册中心、路由、风险引擎、RAG、工单、三个领域专业 Agent 与 Chief 组装。

api 服务与离线 demo 都从这里拿平台实例，保证行为一致。
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

from railmind.agents.cabin_agent import CabinAgent
from railmind.agents.lineside_agent import LineSideAgent
from railmind.agents.panto_agent import PantoAgent
from railmind.agents.shm_agent import SHMAgent
from railmind.agents.underbody_agent import UnderbodyAgent
from railmind.capabilities.sdk import InProcessCapability, load_descriptor
from railmind.core.chief import CapabilityRouter, ChiefAgent
from railmind.core.events import EventStore
from railmind.core.provider import EchoScriptedProvider, LLMProvider, make_provider
from railmind.core.rag import RagClient
from railmind.core.registry import CapabilityRegistry, MODE_PRIMARY
from railmind.core.risk_engine import RiskEngine
from railmind.core.workorder import WorkOrderStore

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CAP_DIR = os.path.join(PKG_DIR, "capabilities")

# 三个领域能力的描述文件与推理入口
BUILTIN_CAPABILITIES = {
    "shm": {
        "descriptor": os.path.join(CAP_DIR, "shm_impact", "capability.yaml"),
        "infer": "railmind.capabilities.shm_impact.adapter:infer",
        "mode": MODE_PRIMARY,
    },
    "door": {
        "descriptor": os.path.join(CAP_DIR, "door_handle", "capability.yaml"),
        "infer": "railmind.capabilities.door_handle.adapter:infer",
        "mode": MODE_PRIMARY,
    },
    "wear": {
        "descriptor": os.path.join(CAP_DIR, "panto", "wear_capability.yaml"),
        "infer": "railmind.capabilities.panto.wear:infer",
        "mode": MODE_PRIMARY,
    },
    "arc": {
        "descriptor": os.path.join(CAP_DIR, "panto", "arc_capability.yaml"),
        "infer": "railmind.capabilities.panto.arc:infer",
        "mode": MODE_PRIMARY,
    },
    "fod": {
        "descriptor": os.path.join(CAP_DIR, "lineside_fod", "capability.yaml"),
        "infer": "railmind.capabilities.lineside_fod.adapter:infer",
        "mode": MODE_PRIMARY,
    },
    "cabin": {
        "descriptor": os.path.join(CAP_DIR, "cabin_vlm", "capability.yaml"),
        "infer": "railmind.capabilities.cabin_vlm.adapter:infer",
        "mode": MODE_PRIMARY,
    },
    "fastener": {
        "descriptor": os.path.join(CAP_DIR, "fastener", "capability.yaml"),
        "infer": "railmind.capabilities.fastener.adapter:infer",
        "mode": MODE_PRIMARY,
    },
}


def _load_infer_fn(spec: str):
    module, _, attr = spec.partition(":")
    import importlib

    return getattr(importlib.import_module(module), attr)


class RailMindPlatform:
    def __init__(self, data_dir: str = "data", prefer_remote_llm: bool = True, attach_builtin: bool = False):
        os.makedirs(data_dir, exist_ok=True)
        self.registry = CapabilityRegistry()
        self.router = CapabilityRouter()
        self.risk = RiskEngine()
        self.rag = RagClient()
        self.events = EventStore(path=os.path.join(data_dir, "events.jsonl"))
        self.workorders = WorkOrderStore(path=os.path.join(data_dir, "workorders.json"))
        # Agent 大脑分层（pi 循环 + 工具调用）：
        #   Chief = 真实 LLM（编排决策大脑，自主导排 6 步流程）
        #   专业 Agent = 本地确定性分级（规程阈值计算，Echo 脚本驱动，快且稳）
        # 任一环节网络故障时自动落回 Echo 脚本（演示稳定性，方案 2.4）
        self.provider: LLMProvider = make_provider(prefer_remote=prefer_remote_llm)
        from railmind.core.provider import EchoScriptedProvider, OpenAICompatibleProvider

        if isinstance(self.provider, OpenAICompatibleProvider):
            chief_provider = self.provider
        else:
            chief_provider = EchoScriptedProvider()
        specialist_provider = EchoScriptedProvider()
        self.specialists = {
            "composite_structure": SHMAgent(event_store=self.events, rag=self.rag, provider=specialist_provider),
            "underbody": UnderbodyAgent(event_store=self.events, rag=self.rag, provider=specialist_provider),
            "pantograph": PantoAgent(event_store=self.events, rag=self.rag, provider=specialist_provider),
            "lineside": LineSideAgent(event_store=self.events, rag=self.rag, provider=specialist_provider),
            "cabin": CabinAgent(event_store=self.events, rag=self.rag, provider=specialist_provider),
        }
        self.chief = ChiefAgent(
            registry=self.registry,
            router=self.router,
            risk=self.risk,
            rag=self.rag,
            workorders=self.workorders,
            event_store=self.events,
            specialists=self.specialists,
            provider=chief_provider,
        )
        self._capabilities: List[InProcessCapability] = []
        if attach_builtin:
            for key, cfg in BUILTIN_CAPABILITIES.items():
                try:
                    self.attach_local_capability(cfg["descriptor"], _load_infer_fn(cfg["infer"]), mode=cfg["mode"])
                except Exception as exc:  # noqa: BLE001 —— 单个能力加载失败不拖垮平台
                    print(f"[platform] 能力 {key} 加载失败（将以离线/拒判模式表现）: {exc}", flush=True)
            threading.Thread(target=self._self_check, daemon=True).start()
            threading.Thread(target=self._ambient_loop, daemon=True).start()

    def _self_check(self) -> None:
        """能力巡检轮询：低频随机抽查各能力做真实调用（调用数/时延为真实推理结果），
        产生的高风险工单由值班员确认并闭环，避免草稿堆积。节奏与真实巡检一致，避免爆发式触发。"""
        time.sleep(90)
        kinds = ["shm_impact", "door_normal", "panto_wear", "panto_arc", "lineside_fod", "fastener_defect", "cabin_patrol"]
        while True:
            for kind in random.sample(kinds, k=random.randint(2, 3)):
                try:
                    summary = self.handle_demo_event(kind)
                    wo = (summary or {}).get("workorder") or {}
                    if wo.get("workorder_id") and wo.get("status") == "DRAFT":
                        self.workorders.confirm(wo["workorder_id"], operator="值班员", note="巡检告警确认，安排下一停靠站复核")
                        self.workorders.close(wo["workorder_id"], operator="检修员A", note="现场复核完成，闭环归档")
                except Exception as exc:  # noqa: BLE001
                    print(f"[platform] 巡检调用 {kind} 失败（不影响运行）: {exc}", flush=True)
                time.sleep(random.uniform(15, 40))
            time.sleep(random.uniform(240, 480))

    # 常驻遥测模拟的能力档案（方案 1.3-8 车辆状态模拟）：仅正常/观察级。
    # 注意：input 必须是真实本地文件路径（能力适配器按文件读取，网页 URI 会推理失败）。
    _REPO = os.path.dirname(PKG_DIR)
    _ambient_caps = {
        "shm": {"domain": "composite_structure", "component": "composite_deck_panel", "scene": "TRAIN_RUNNING",
                "speed": (290, 310), "input": {"signal_uri": os.path.join(_REPO, "datasets", "shm_samples", "60_60_1.00J+300_60_1.00J.mat")}},
        "door": {"domain": "underbody", "component": "inspection_door_3", "scene": "STATION_STOP",
                 "speed": (0, 0), "input": {"image_uri": os.path.join(PKG_DIR, "capabilities", "door_handle", "demo_clean.jpg")}},
        "wear": {"domain": "pantograph", "component": "pantograph_front", "scene": "STATION_STOP",
                 "speed": (0, 0), "input": {"image_uri": os.path.join(_REPO, "datasets", "pantograph", "samples", "video_frame_0242.jpg")}},
        "arc": {"domain": "pantograph", "component": "pantograph_rear", "scene": "TRAIN_RUNNING",
                "speed": (270, 305), "input": {"mendeley_record": {"folder": "Trenitalia", "phase": "traction", "name": "TI_T_3"}}},
        "fod": {"domain": "lineside", "component": "track_k42_minus", "scene": "STATION_STOP",
                "speed": (0, 0), "input": {"image_uri": os.path.join(_REPO, "datasets", "pantograph", "samples", "video_frame_0121.jpg")}},
        "fastener": {"domain": "lineside", "component": "track_k45_fastener", "scene": "STATION_STOP",
                     "speed": (0, 0), "input": {"fastener_image_uri": os.path.join(PKG_DIR, "capabilities", "fastener", "samples", "sample_missing.jpg")}},
        "cabin": {"domain": "cabin", "component": "cabin_camera_02", "scene": "TRAIN_RUNNING",
                  "speed": (295, 305), "input": {"video_uri": os.path.join(_REPO, "web", "assets", "cabin_patrol.mp4"), "source": "cache"}},
    }
    _ambient_trains = ["CRH-03", "CRH-05", "CRH-07", "CRH-12", "CRH-18"]

    def _ambient_loop(self) -> None:
        """常驻遥测模拟（方案 1.3-8）：随机间隔产生正常/观察级诊断事件，
        保持事件流与能力调用的持续演进；事件走完整 Chief 编排链路。"""
        time.sleep(20)
        while True:
            try:
                cfg = self._ambient_caps[random.choice(list(self._ambient_caps))]
                event = {
                    "domain": cfg["domain"],
                    "asset": {"train_id": random.choice(self._ambient_trains),
                              "carriage_id": f"{random.randint(1, 8):02d}", "component": cfg["component"]},
                    "operation_context": {"scene": cfg["scene"], "speed_kmh": random.randint(*cfg["speed"])},
                    "capability_input": dict(cfg["input"]),
                }
                self.handle_event(event)
            except Exception as exc:  # noqa: BLE001
                print(f"[platform] 遥测模拟事件失败（不影响运行）: {exc}", flush=True)
            time.sleep(random.uniform(45, 120))

    # ---------- 能力接入 ----------

    def attach_local_capability(
        self,
        descriptor_path: str,
        infer_fn=None,
        mode: str = MODE_PRIMARY,
        model_version: str = "1.0.0",
        heartbeat: bool = True,
    ) -> InProcessCapability:
        if infer_fn is None:
            raise ValueError("attach_local_capability 需要 infer_fn")
        descriptor = load_descriptor(descriptor_path)
        cap = InProcessCapability(
            registry=self.registry,
            descriptor=descriptor,
            infer_fn=infer_fn,
            mode=mode,
            model_version=model_version,
        )
        if heartbeat:
            cap.start_heartbeat()
        else:
            cap.heartbeat_once()
        self.router.attach_local(cap)
        self._capabilities.append(cap)
        return cap

    def attach_rest_capability(self, descriptor: Dict[str, Any], base_url: str, mode: str = "AUXILIARY") -> str:
        """外部热加载入口：平台不重启注册 REST 能力（方案 2.2）。"""
        instance_id = self.registry.register(descriptor, mode=mode)
        self.router.attach_rest(descriptor, base_url)
        return instance_id

    def stop(self) -> None:
        for cap in self._capabilities:
            cap.stop()

    # ---------- 对话入口 ----------

    def init_chat(self) -> None:
        """初始化对话助手（需在能力挂载后调用）。"""
        from railmind.core.chat import ChatAgent

        self.chat_agent = ChatAgent(
            registry=self.registry,
            workorders=self.workorders,
            event_store=self.events,
            rag=self.rag,
            provider=self.provider,
            trigger_demo=self.handle_demo_event,
        )

    def handle_demo_event(self, kind: str) -> Optional[Dict[str, Any]]:
        from railmind.apps.demo_events import build_demo_event

        event = build_demo_event(kind)
        if event is None:
            return None
        return self.handle_event(event)

    # ---------- 事件入口 ----------

    def handle_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return self.chief.handle_event(event)
