"""工具注册表：pi 的"少量工具"哲学 —— 注册、声明 JSON Schema、分发，三件事而已。"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional

from railmind.core.types import ToolResult


class Tool:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], handler: Callable[..., Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """线程安全的工具表。handler 异常统一转为 ToolResult(ok=False)，带错误码。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """装饰器注册。parameters 缺省时按函数签名推导一个宽松 schema。"""

        def decorator(fn: Callable[..., Any]) -> Callable:
            tool_name = name or fn.__name__
            if tool_name in self._tools:
                raise ValueError(f"工具名重复: {tool_name}")
            schema = parameters or _infer_parameters(fn)
            self._tools[tool_name] = Tool(tool_name, description or (fn.__doc__ or "").strip(), schema, fn)
            return fn

        return decorator

    def register_function(
        self,
        fn: Callable[..., Any],
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        tool_name = name or fn.__name__
        if tool_name in self._tools:
            raise ValueError(f"工具名重复: {tool_name}")
        schema = parameters or _infer_parameters(fn)
        self._tools[tool_name] = Tool(tool_name, description or (fn.__doc__ or "").strip(), schema, fn)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def specs(self) -> List[Dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"E_TOOL_NOT_FOUND:{name}")
        try:
            value = tool.handler(**(arguments or {}))
            return ToolResult(ok=True, value=value)
        except TypeError as exc:
            return ToolResult(ok=False, error=f"E_INVALID_ARGUMENTS:{name}:{exc}")
        except Exception as exc:  # noqa: BLE001 —— 工具异常不允许打断循环
            return ToolResult(ok=False, error=f"E_TOOL_ERROR:{name}:{exc}")


_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _infer_parameters(fn: Callable[..., Any]) -> Dict[str, Any]:
    """从类型注解推导宽松 JSON Schema；无注解参数按 string 处理。"""
    properties: Dict[str, Any] = {}
    required: List[str] = []
    sig = inspect.signature(fn)
    for param_name, param in sig.parameters.items():
        ann = param.annotation if param.annotation is not inspect.Parameter.empty else str
        properties[param_name] = {"type": _TYPE_MAP.get(ann, "string"), "title": param_name}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    return {"type": "object", "properties": properties, "required": required}


def tool(default_description: str = "", default_parameters: Optional[Dict[str, Any]] = None) -> Callable:
    """便捷组合：functools.partial(ToolRegistry.register)。"""
    return functools.partial(ToolRegistry.register, description=default_description, parameters=default_parameters)
