"""Provider 抽象：Agent 内核只依赖 complete(messages, tools) -> ProviderResponse。

- EchoScriptedProvider：本地确定性 Provider（零网络），用脚本步骤驱动循环，
  满足方案 2.4"演示必须有本地降级路径"的要求，也让单测完全可复现。
- OpenAICompatibleProvider：任何 OpenAI 兼容推理服务（GLM/DeepSeek/vLLM…），
  通过环境变量配置；失败时上层可自动落回 Echo。
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from railmind.core.types import ProviderResponse, ToolCall


def _load_dotenv() -> None:
    """极简 .env 加载（不覆盖已有环境变量），仓库根目录。"""
    import os  # noqa: PLC0415

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_dotenv()

# 脚本步骤：既可以是静态 (text, [(tool_name, args), ...])，也可以是
# f(ctx) -> (text, tool_calls) 的条件步骤；ctx 携带此前所有工具结果。
ScriptStep = object

EchoStep = object


class LLMProvider:
    name = "base"

    def complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> ProviderResponse:
        raise NotImplementedError


class EchoScriptedProvider(LLMProvider):
    """按脚本顺序产出"思考文本 + 工具调用"，脚本耗尽即结束回合。

    每步允许二选一：
      * ("文本", [("tool", {...}), ...])
      * callable(ctx) -> 上述元组；ctx = {"messages": ..., "tool_results": [(name, ToolResult)]}
    """

    name = "echo"

    def __init__(self, script: Optional[List[Any]] = None):
        self.script: List[Any] = list(script or [])
        self._cursor = 0

    def reset(self, script: Optional[List[Any]] = None) -> None:
        self.script = list(script or [])
        self._cursor = 0

    def complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> ProviderResponse:
        if self._cursor >= len(self.script):
            return ProviderResponse(text="（脚本已执行完毕，回合结束）", stop_reason="end_turn", model=self.name)

        step = self.script[self._cursor]
        self._cursor += 1
        if callable(step):
            step = step(_build_ctx(messages))
        text, calls = step
        tool_calls: List[ToolCall] = []
        for tool_name, args in calls:
            tool_calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:10]}", name=tool_name, arguments=args))
        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            model=self.name,
        )


def _build_ctx(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_results: List[Tuple[str, Dict[str, Any]]] = []
    for msg in messages:
        if msg.get("role") == "tool":
            try:
                payload = json.loads(msg["content"])
            except (TypeError, json.JSONDecodeError):
                payload = {"raw": msg.get("content")}
            tool_results.append((msg.get("name", ""), payload))
    return {"messages": messages, "tool_results": tool_results}


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI /chat/completions 兼容 Provider。

    环境变量：
      RAILMIND_LLM_BASE_URL  如 https://open.bigmodel.cn/api/paas/v4
      RAILMIND_LLM_API_KEY   密钥
      RAILMIND_LLM_MODEL     如 glm-4-plus
      RAILMIND_LLM_REASONING 思考模式：none 关闭深度思考（低延迟），low 平衡；默认 none
    """

    name = "openai_compatible"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 60,
        temperature: float = 0.2,
        reasoning_effort: Optional[str] = None,
    ):
        self.base_url = (base_url or os.environ.get("RAILMIND_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("RAILMIND_LLM_API_KEY", "")
        self.model = model or os.environ.get("RAILMIND_LLM_MODEL", "glm-4-plus")
        self.reasoning_effort = reasoning_effort or os.environ.get("RAILMIND_LLM_REASONING", "none") or None
        self.timeout_s = timeout_s
        self.temperature = temperature
        if not (self.base_url and self.api_key):
            raise ValueError("OpenAICompatibleProvider 需要 RAILMIND_LLM_BASE_URL / RAILMIND_LLM_API_KEY")

    def complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> ProviderResponse:
        import requests  # noqa: PLC0415

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = tools
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        tool_calls = [
            ToolCall(
                id=tc.get("id", f"call_{uuid.uuid4().hex[:10]}"),
                name=tc["function"]["name"],
                arguments=ToolCall.parse_arguments(tc["function"].get("arguments")),
            )
            for tc in message.get("tool_calls") or []
        ]
        usage = data.get("usage") or {}
        return ProviderResponse(
            text=message.get("content"),
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            model=data.get("model", self.model),
            usage={"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)},
        )


    def complete_stream(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None):
        """流式补全：逐段 yield {"type": "delta", "text"} ；回合结束时最后 yield
        {"type": "final", "response": ProviderResponse}（含拼装好的完整工具调用）。
        与 complete 相同的错误语义（网络/HTTP 异常向上抛，由调用方降级）。"""
        import json as _json  # noqa: PLC0415

        import requests  # noqa: PLC0415

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = tools
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_s,
            stream=True,
        )
        resp.raise_for_status()

        text_parts: List[str] = []
        # 工具调用分片按 index 聚合
        tc_acc: Dict[int, Dict[str, Any]] = {}
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        model_name = self.model
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            payload = raw[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = _json.loads(payload)
            except _json.JSONDecodeError:
                continue
            model_name = chunk.get("model", model_name)
            if chunk.get("usage"):
                usage = chunk["usage"] or usage
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                text_parts.append(delta["content"])
                yield {"type": "delta", "text": delta["content"]}
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                acc = tc_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    acc["name"] = fn["name"]
                if fn.get("arguments"):
                    acc["args"] += fn["arguments"]

        tool_calls = [
            ToolCall(
                id=(acc["id"] or f"call_{uuid.uuid4().hex[:10]}"),
                name=acc["name"],
                arguments=ToolCall.parse_arguments(acc["args"] or "{}"),
            )
            for _, acc in sorted(tc_acc.items())
        ]
        response = ProviderResponse(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            model=model_name,
            usage=usage,
        )
        yield {"type": "final", "response": response}


def make_provider(prefer_remote: bool = True) -> LLMProvider:
    """工厂：配置了远端环境变量则优先远端，否则本地 Echo（演示稳定性兜底）。"""
    if prefer_remote and os.environ.get("RAILMIND_LLM_BASE_URL") and os.environ.get("RAILMIND_LLM_API_KEY"):
        try:
            return OpenAICompatibleProvider()
        except Exception:  # noqa: BLE001
            pass
    return EchoScriptedProvider()
