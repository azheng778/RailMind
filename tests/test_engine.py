"""pi 风格引擎循环测试：脚本驱动、工具分发、错误兜底。"""

from railmind.core.engine import Agent
from railmind.core.provider import EchoScriptedProvider
from railmind.core.tools import ToolRegistry


def make_tools():
    tools = ToolRegistry()

    @tools.register(name="add", description="加法", parameters={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]})
    def add(a: float, b: float) -> dict:
        return {"sum": a + b}

    @tools.register(name="boom", description="必炸")
    def boom() -> dict:
        raise RuntimeError("炸了")

    return tools


def test_loop_executes_tool_then_final():
    provider = EchoScriptedProvider([
        ("先算加法", [("add", {"a": 1, "b": 2})]),
        ("算完了", []),
    ])
    agent = Agent(name="t", provider=provider, tools=make_tools(), system="s")
    result = agent.run("计算1+2")
    assert result.stop_reason == "end_turn"
    assert result.iterations == 2
    assert result.text == "算完了"
    kinds = [e.kind for e in result.events]
    assert "tool_start" in kinds and "tool_end" in kinds and "done" in kinds


def test_tool_error_is_captured_not_raised():
    provider = EchoScriptedProvider([
        ("试试会炸的工具", [("boom", {})]),
        ("工具报错已记录", []),
    ])
    agent = Agent(name="t", provider=provider, tools=make_tools())
    result = agent.run("go")
    failed = [c for c in result.tool_calls if not c["ok"]]
    assert len(failed) == 1
    assert failed[0]["error"].startswith("E_TOOL_ERROR")


def test_unknown_tool_returns_error_result():
    provider = EchoScriptedProvider([
        ("调用不存在工具", [("nope", {})]),
        ("结束", []),
    ])
    agent = Agent(name="t", provider=provider, tools=make_tools())
    result = agent.run("go")
    assert any(c["error"].startswith("E_TOOL_NOT_FOUND") for c in result.tool_calls)


def test_max_iterations_stop():
    # 脚本每步都要求调工具，永不结束 → 触发 max_iterations
    endless = [("继续", [("add", {"a": 1, "b": 1})])] * 10
    agent = Agent(name="t", provider=EchoScriptedProvider(endless), tools=make_tools(), max_iterations=3)
    result = agent.run("go")
    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3


def test_conditional_step_reads_tool_results():
    def conditional(ctx):
        value = None
        for name, payload in ctx["tool_results"]:
            if name == "add" and payload.get("ok"):
                value = payload["value"]["sum"]
        return (f"结果是{value}", [])

    provider = EchoScriptedProvider([
        ("先算", [("add", {"a": 20, "b": 22})]),
        conditional,
    ])
    agent = Agent(name="t", provider=provider, tools=make_tools())
    result = agent.run("go")
    assert result.text == "结果是42"
