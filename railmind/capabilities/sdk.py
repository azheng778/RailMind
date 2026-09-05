"""能力接入 SDK —— 方案 8.4：模块开发者只写核心推理函数，其余交给 SDK。

SDK 负责：能力注册、心跳、健康检查、参数校验、统一诊断事件转换、错误码、
调用指标上报。支持两种运行形态：
  * InProcessCapability：与平台同进程（演示与单测，零网络）；
  * serve_capability()：独立 FastAPI 服务（外部热加载演示，REST 协议）。
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import yaml

from railmind.core.registry import CapabilityRegistry, MODE_PRIMARY
from railmind.core.types import new_id

INFERENCE_TIMEOUT_S = 10.0


def load_descriptor(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _validate_payload(descriptor: Dict[str, Any], payload: Dict[str, Any]) -> Optional[str]:
    """按 descriptor.inputs 做宽松校验：非 optional 的 input 必须出现。"""
    for item in descriptor.get("inputs", []):
        name = item.get("name")
        if name and name not in payload and not item.get("optional", False):
            return f"E_INVALID_INPUT:缺少必填输入 {name}"
    return None


class CapabilityError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}:{detail}")
        self.code = code
        self.detail = detail


class InProcessCapability:
    """进程内能力：注册到 registry + 心跳线程 + 标准化推理包装。"""

    def __init__(
        self,
        registry: CapabilityRegistry,
        descriptor: Dict[str, Any],
        infer_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        mode: str = MODE_PRIMARY,
        heartbeat_interval_s: float = 5.0,
        model_version: str = "0.0.0",
    ):
        self.descriptor = descriptor
        self.infer_fn = infer_fn
        self.heartbeat_interval_s = heartbeat_interval_s
        self.model_version = model_version
        self.instance_id = registry.register(descriptor, mode=mode)
        self.registry = registry
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None

    # ---------- 生命周期 ----------

    def start_heartbeat(self) -> None:
        def _loop() -> None:
            while not self._hb_stop.wait(self.heartbeat_interval_s):
                self.registry.heartbeat(self.instance_id)

        self._hb_thread = threading.Thread(target=_loop, name=f"hb-{self.instance_id}", daemon=True)
        self._hb_thread.start()

    def stop(self) -> None:
        """停心跳并标记离线 —— 用于主备切换 / 降级演练。"""
        self._hb_stop.set()
        self.registry.mark_offline(self.instance_id)

    def heartbeat_once(self) -> None:
        self.registry.heartbeat(self.instance_id)

    # ---------- 推理 ----------

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行一次推理并返回统一诊断事件（方案 8.2）。超时/异常转 CapabilityError。"""
        err = _validate_payload(self.descriptor, payload)
        if err:
            raise CapabilityError(err.split(":")[0], err.split(":", 1)[-1])

        started = time.time()
        result: Dict[str, Any] = {}
        error: Optional[str] = None
        try:
            raw = self.infer_fn(payload)
            result = self.to_diagnosis_event(raw, payload)
        except CapabilityError:
            error = traceback.format_exc(limit=1)
            raise
        except Exception as exc:  # noqa: BLE001
            error = f"E_INFER_ERROR:{exc}"
            raise CapabilityError("E_INFER_ERROR", str(exc)) from exc
        finally:
            latency_ms = (time.time() - started) * 1000.0
            self.registry.record_invoke(self.instance_id, latency_ms, ok=error is None)
            self.registry.heartbeat(self.instance_id, latency_ms=latency_ms)
        return result

    def to_diagnosis_event(self, raw: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """把模块原始输出转换为统一诊断事件；模块也可以直接输出完整事件。"""
        if raw.get("schema_version"):
            return raw
        asset = payload.get("asset", {})
        context = payload.get("operation_context", {"scene": "TRAIN_RUNNING", "speed_kmh": 300})
        return {
            "schema_version": "1.0",
            "event_id": new_id("EVT"),
            "ts": time.time(),
            "source": {
                "capability_id": self.descriptor["capability_id"],
                "instance_id": self.instance_id,
                "provider": self.descriptor.get("provider", "unknown"),
            },
            "asset": asset,
            "operation_context": context,
            "diagnosis": raw.get("diagnosis", {}),
            "evidence": raw.get("evidence", []),
            "model_version": self.model_version,
            "requires_human_review": bool(self.descriptor.get("requires_human_review", True)),
        }


def serve_capability(
    descriptor_path: str,
    infer_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    host: str = "127.0.0.1",
    port: int = 8901,
    model_version: str = "0.0.0",
) -> None:
    """把推理函数包装为独立 REST 服务（外部热加载演示用）。阻塞运行。"""
    from fastapi import FastAPI
    import uvicorn

    descriptor = load_descriptor(descriptor_path)

    app = FastAPI(title=descriptor.get("name", "railmind-capability"))

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"status": "ok", "capability_id": descriptor["capability_id"], "version": descriptor["version"]}

    @app.post(descriptor.get("invoke", {}).get("endpoint", "/api/v1/inference"))
    def inference(payload: Dict[str, Any]) -> Dict[str, Any]:
        err = _validate_payload(descriptor, payload)
        if err:
            return {"ok": False, "error": err}
        try:
            event = {
                "schema_version": "1.0",
                "event_id": new_id("EVT"),
                "ts": time.time(),
                "source": {"capability_id": descriptor["capability_id"], "provider": descriptor.get("provider")},
                "asset": payload.get("asset", {}),
                "operation_context": payload.get("operation_context", {}),
                "diagnosis": infer_fn(payload).get("diagnosis", {}),
                "evidence": [],
                "model_version": model_version,
                "requires_human_review": bool(descriptor.get("requires_human_review", True)),
            }
            return {"ok": True, "value": event}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"E_INFER_ERROR:{exc}"}

    uvicorn.run(app, host=host, port=port, log_level="warning")
