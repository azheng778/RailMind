"""能力注册中心 —— 方案第 8 章的落地：动态注册 / 心跳 TTL / 运行模式 / 场景过滤 / 主备。

内存实现（演示与单测用），接口与持久化到 PostgreSQL 的版本保持一致。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 运行模式（方案 8.3）
MODE_OBSERVE = "OBSERVE"        # 观察模式：只记录，不参与分派
MODE_AUXILIARY = "AUXILIARY"    # 辅助模式：参与融合，但结果强制人工确认
MODE_PRIMARY = "PRIMARY"        # 主用模式
MODE_STANDBY = "STANDBY"        # 备用模式：主用离线时接管
MODE_DISABLED = "DISABLED"      # 禁用模式：不接收任务

# 分派排序优先级（越小越优先）
_MODE_RANK = {MODE_PRIMARY: 0, MODE_STANDBY: 1, MODE_AUXILIARY: 2}
_DISPATCHABLE = {MODE_PRIMARY, MODE_STANDBY, MODE_AUXILIARY}

REQUIRED_DESCRIPTOR_FIELDS = ["capability_id", "name", "version", "domain", "supported_scenes", "invoke"]


@dataclass
class CapabilityInstance:
    capability_id: str
    instance_id: str
    version: str
    name: str
    provider: str
    domain: str
    capability_type: str
    supported_scenes: List[str]
    input_types: List[str]
    input_names: List[str]
    descriptor: Dict[str, Any]
    mode: str = MODE_AUXILIARY
    online: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    call_count: int = 0

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "instance_id": self.instance_id,
            "name": self.name,
            "provider": self.provider,
            "version": self.version,
            "type": self.capability_type,
            "domain": self.domain,
            "supported_scenes": self.supported_scenes,
            "input_types": self.input_types,
            "mode": self.mode,
            "online": self.is_online(),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "error_rate": round(self.error_rate, 4),
            "call_count": self.call_count,
        }

    def is_online(self, ttl_s: float = 15.0, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.time()
        return self.online and (now - self.last_heartbeat) <= ttl_s


class RegistrationError(ValueError):
    """注册校验失败（重复 ID / 字段缺失 / 版本不兼容）。"""


class CapabilityRegistry:
    def __init__(self, heartbeat_ttl_s: float = 15.0, clock: Any = time.time):
        self._lock = threading.RLock()
        self._instances: Dict[str, CapabilityInstance] = {}
        self.heartbeat_ttl_s = heartbeat_ttl_s
        self._clock = clock

    # ---------- 注册 / 心跳 ----------

    def register(self, descriptor: Dict[str, Any], mode: str = MODE_AUXILIARY) -> str:
        missing = [f for f in REQUIRED_DESCRIPTOR_FIELDS if f not in descriptor]
        if missing:
            raise RegistrationError(f"E_INVALID_DESCRIPTOR:缺少字段 {missing}")
        cap_id = descriptor["capability_id"]
        with self._lock:
            for inst in self._instances.values():
                if inst.capability_id == cap_id and inst.is_online(self.heartbeat_ttl_s):
                    if inst.version == descriptor["version"]:
                        raise RegistrationError(f"E_DUPLICATE_CAPABILITY:{cap_id} 已有在线同名同版本实例")
                    if _version_lt(descriptor["version"], inst.version):
                        raise RegistrationError(
                            f"E_VERSION_CONFLICT:{cap_id} 在线版本 {inst.version} 高于注册版本 {descriptor['version']}"
                        )
            instance_id = f"{cap_id}#{uuid.uuid4().hex[:6]}"
            self._instances[instance_id] = CapabilityInstance(
                capability_id=cap_id,
                instance_id=instance_id,
                version=str(descriptor["version"]),
                name=descriptor.get("name", cap_id),
                provider=descriptor.get("provider", "unknown"),
                domain=descriptor["domain"],
                capability_type=descriptor.get("capability_type", "inference_tool"),
                supported_scenes=list(descriptor["supported_scenes"]),
                input_types=[i.get("type", "any") for i in descriptor.get("inputs", [])],
                input_names=[i.get("name", "") for i in descriptor.get("inputs", [])],
                descriptor=dict(descriptor),
                mode=mode,
                last_heartbeat=self._clock(),
            )
            return instance_id

    def heartbeat(self, instance_id: str, latency_ms: float = 0.0, error_rate: Optional[float] = None) -> bool:
        with self._lock:
            inst = self._instances.get(instance_id)
            if inst is None:
                return False
            inst.last_heartbeat = self._clock()
            inst.online = True
            if latency_ms > 0:
                inst.avg_latency_ms = inst.avg_latency_ms or latency_ms
            if error_rate is not None:
                inst.error_rate = error_rate
            return True

    def mark_offline(self, instance_id: str) -> bool:
        with self._lock:
            inst = self._instances.get(instance_id)
            if inst is None:
                return False
            inst.online = False
            return True

    def record_invoke(self, instance_id: str, latency_ms: float, ok: bool) -> None:
        with self._lock:
            inst = self._instances.get(instance_id)
            if inst is None:
                return
            inst.call_count += 1
            n = inst.call_count
            inst.avg_latency_ms = inst.avg_latency_ms + (latency_ms - inst.avg_latency_ms) / n
            inst.error_rate = inst.error_rate + ((0.0 if ok else 1.0) - inst.error_rate) / n

    # ---------- 模式管理 ----------

    def set_mode(self, capability_id: str, mode: str) -> int:
        if mode not in {MODE_OBSERVE, MODE_AUXILIARY, MODE_PRIMARY, MODE_STANDBY, MODE_DISABLED}:
            raise ValueError(f"E_INVALID_MODE:{mode}")
        changed = 0
        with self._lock:
            for inst in self._instances.values():
                if inst.capability_id == capability_id:
                    inst.mode = mode
                    changed += 1
        if changed == 0:
            raise KeyError(f"E_CAPABILITY_NOT_FOUND:{capability_id}")
        return changed

    # ---------- 查询与分派（方案 8.5） ----------

    def query(
        self,
        domain: Optional[str] = None,
        scene: Optional[str] = None,
        input_type: Optional[str] = None,
        input_name: Optional[str] = None,
        online_only: bool = True,
        include_observe: bool = False,
    ) -> List[CapabilityInstance]:
        """按领域/场景/输入类型/输入名筛出可分派能力，主用 > 备用 > 辅助，再按时延与错误率。"""
        picked: List[CapabilityInstance] = []
        with self._lock:
            for inst in self._instances.values():
                if online_only and not inst.is_online(self.heartbeat_ttl_s, now=self._clock()):
                    continue
                if inst.mode not in _DISPATCHABLE and not (include_observe and inst.mode == MODE_OBSERVE):
                    continue
                if domain and inst.domain != domain:
                    continue
                if scene and scene not in inst.supported_scenes:
                    continue
                if input_type and input_type not in inst.input_types and "any" not in inst.input_types:
                    continue
                if input_name and inst.input_names and input_name not in inst.input_names:
                    continue
                picked.append(inst)
        picked.sort(key=lambda i: (_MODE_RANK.get(i.mode, 9), i.error_rate, i.avg_latency_ms))
        return picked

    def dispatch_target(
        self,
        domain: str,
        scene: Optional[str] = None,
        input_type: Optional[str] = None,
        input_name: Optional[str] = None,
    ) -> Optional[CapabilityInstance]:
        """选主用；无主用在线时由备用接管（方案 6.4 / 11.2 主备切换）。"""
        candidates = self.query(domain=domain, scene=scene, input_type=input_type, input_name=input_name)
        primaries = [c for c in candidates if c.mode == MODE_PRIMARY]
        if primaries:
            return primaries[0]
        return candidates[0] if candidates else None

    def observe_targets(self, domain: Optional[str] = None, scene: Optional[str] = None) -> List[CapabilityInstance]:
        """观察模式能力：只跑影子评估，不进融合。"""
        picked = [
            inst
            for inst in self._instances.values()
            if inst.mode == MODE_OBSERVE and inst.is_online(self.heartbeat_ttl_s, now=self._clock())
            and (domain is None or inst.domain == domain) and (scene is None or scene in inst.supported_scenes)
        ]
        return picked

    def get(self, capability_id: str) -> List[CapabilityInstance]:
        with self._lock:
            return [i for i in self._instances.values() if i.capability_id == capability_id]

    def all(self) -> List[CapabilityInstance]:
        with self._lock:
            return list(self._instances.values())

    def sweep_offline(self) -> List[str]:
        """把超过 TTL 未心跳的实例标记离线，返回刚离线的 instance_id。"""
        gone: List[str] = []
        with self._lock:
            for inst in self._instances.values():
                if inst.online and not inst.is_online(self.heartbeat_ttl_s, now=self._clock()):
                    inst.online = False
                    gone.append(inst.instance_id)
        return gone


def _version_lt(a: str, b: str) -> bool:
    """粗粒度语义版本比较：a < b。"""

    def parts(v: str) -> List[int]:
        try:
            return [int(x) for x in v.split(".")]
        except ValueError:
            return [0]

    return parts(a) < parts(b)
