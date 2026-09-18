"""平台 HTTP 服务。运行： D:/Anaconda3/envs/railmind/python.exe -m railmind.apps.api """

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from railmind.core.rag import KBChunk, VERSION_DRAFT, VERSION_ACTIVE, VERSION_OBSOLETE
from railmind.core.registry import RegistrationError
from railmind.platform import RailMindPlatform

_PLATFORM: Optional[RailMindPlatform] = None

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_platform() -> RailMindPlatform:
    global _PLATFORM
    if _PLATFORM is None:
        _PLATFORM = RailMindPlatform(data_dir=os.environ.get("RAILMIND_DATA_DIR", "data"), attach_builtin=True)
    return _PLATFORM


def create_app(platform: Optional[RailMindPlatform] = None) -> FastAPI:
    app = FastAPI(title="RailMind 列车智能运维平台", version="0.2.0")
    pf = platform or get_platform()

    # 静态资源：前端页面 + demo 图 + 能力证据图
    web_dir = os.path.join(_REPO, "web")
    if os.path.isdir(web_dir):
        app.mount("/static/web", StaticFiles(directory=web_dir), name="web")
    for mount, src in (
        ("panto_samples", os.path.join(_REPO, "datasets", "pantograph", "samples")),
        ("door_evidence", os.path.join(_REPO, "data", "door_evidence")),
        ("fod_evidence", os.path.join(_REPO, "data", "fod_evidence")),
        ("fastener_evidence", os.path.join(_REPO, "data", "fastener_evidence")),
        ("fastener_demo", os.path.join(_REPO, "railmind", "capabilities", "fastener")),
        ("door_demo", os.path.join(_REPO, "railmind", "capabilities", "door_handle")),
        ("cabin_vlm", os.path.join(_REPO, "railmind", "capabilities", "cabin_vlm")),
    ):
        if os.path.isdir(src):
            app.mount(f"/static/{mount}", StaticFiles(directory=src), name=mount)

    class RegisterBody(BaseModel):
        descriptor: Dict[str, Any] = Field(..., description="能力描述（capability.yaml 内容）")
        base_url: Optional[str] = Field(None, description="REST 能力地址，如 http://127.0.0.1:8901")
        mode: str = Field("AUXILIARY", description="初始运行模式")

    class ModeBody(BaseModel):
        mode: str

    class ReviewBody(BaseModel):
        operator: str = "reviewer"
        note: str = ""

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"status": "ok", "brain": type(pf.provider).__name__, "capabilities_online": sum(1 for c in pf.registry.all() if c.is_online(pf.registry.heartbeat_ttl_s))}

    @app.get("/")
    def index() -> Dict[str, Any]:
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/static/web/index.html")

    @app.get("/api/v1/kb/stats")
    def kb_stats() -> Dict[str, Any]:
        """知识库统计（增强版：文档数/块数/活跃块数/查询数/反馈数/词表大小）。"""
        return pf.rag.stats()

    @app.get("/api/v1/kb/list")
    def kb_list(
        status: Optional[str] = None,
        asset_type: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """知识库条目列表（可过滤状态与资产类型）。"""
        chunks = pf.rag.list_chunks(status=status)
        if asset_type:
            chunks = [c for c in chunks if c.asset_type == asset_type]
        return {
            "total": len(chunks),
            "chunks": [
                {
                    "document_id": c.document_id,
                    "title": c.title,
                    "chapter": c.chapter,
                    "page": c.page,
                    "version": c.version,
                    "status": c.effective_status,
                    "asset_type": c.asset_type,
                    "fault_type": c.fault_type,
                    "tags": c.tags,
                    "text": c.text[:160],
                }
                for c in chunks[:limit]
            ],
        }

    @app.post("/api/v1/kb/search")
    def kb_search(body: Dict[str, Any]) -> Dict[str, Any]:
        """自然语言知识检索。

        Body:
            query: 查询文本（必填）。
            asset_type: 可选的资产类型过滤。
            top_k: 返回条数（默认3）。
        """
        query = str(body.get("query", "")).strip()
        if not query:
            raise HTTPException(status_code=400, detail="query 不能为空")
        result = pf.rag.search(
            query_text=query,
            asset_type=body.get("asset_type"),
            top_k=body.get("top_k"),
        )
        return {
            "refused": result.refused,
            "time_ms": round(result.query_time_ms, 1),
            "citations": result.answer_basis,
            "note": result.refused_text if result.refused else None,
        }

    @app.post("/api/v1/kb/feedback")
    def kb_feedback(body: Dict[str, Any]) -> Dict[str, Any]:
        """记录用户对检索结果的反馈。

        Body:
            query: 用户查询。
            document_id: 被评价的知识条目 ID。
            rating: 评分（1-5）。
            comment: 可选评语。
        """
        query = str(body.get("query", "")).strip()
        doc_id = str(body.get("document_id", "")).strip()
        rating = int(body.get("rating", 3))
        if not query or not doc_id:
            raise HTTPException(status_code=400, detail="query 和 document_id 不能为空")
        pf.rag.record_feedback(query=query, document_id=doc_id, rating=rating, comment=str(body.get("comment", "")))
        return {"ok": True}

    @app.post("/api/v1/kb/rebuild")
    def kb_rebuild() -> Dict[str, Any]:
        """强制重建 TF-IDF 向量索引。"""
        pf.rag.rebuild_index()
        return {"ok": True, "stats": pf.rag.stats()}

    @app.get("/api/v1/kb/queries")
    def kb_queries(limit: int = 20) -> Dict[str, Any]:
        """最近查询日志。"""
        return {"queries": pf.rag.recent_queries(limit=limit)}

    @app.get("/api/v1/kb/feedback")
    def kb_feedback_log(limit: int = 20) -> Dict[str, Any]:
        """最近反馈记录。"""
        return {"feedback": pf.rag.recent_feedback(limit=limit)}

    @app.post("/api/v1/events/demo/{kind}")
    def demo_event(kind: str) -> Dict[str, Any]:
        """一键触发内置演示事件（前端按钮用）。"""
        from railmind.apps.demo_events import build_demo_event

        event = build_demo_event(kind)
        if event is None:
            raise HTTPException(status_code=404, detail=f"未知演示事件类型: {kind}")
        return pf.handle_event(event)

    @app.post("/api/v1/chat")
    def chat(body: Dict[str, Any]) -> Dict[str, Any]:
        """运维对话助手（LLM 大脑 + 平台实时数据工具）。"""
        message = str(body.get("message", "")).strip()
        if not message:
            raise HTTPException(status_code=400, detail="message 不能为空")
        if getattr(pf, "chat_agent", None) is None:
            pf.init_chat()
        return pf.chat_agent.chat(message)

    @app.post("/api/v1/chat/stream")
    def chat_stream(body: Dict[str, Any]):
        """流式对话：SSE 逐段输出（delta/tool/done）。"""
        import json as _json

        from fastapi.responses import StreamingResponse

        message = str(body.get("message", "")).strip()
        if not message:
            raise HTTPException(status_code=400, detail="message 不能为空")
        if getattr(pf, "chat_agent", None) is None:
            pf.init_chat()

        def gen():
            try:
                for piece in pf.chat_agent.chat_stream(message):
                    yield f"data: {_json.dumps(piece, ensure_ascii=False, default=str)}\n\n"
            except Exception as exc:  # noqa: BLE001
                yield f"data: {_json.dumps({'type': 'delta', 'text': f'[错误] {exc}'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/v1/events/raw")
    def list_events_raw(
        limit: int = 20,
        anomaly: Optional[str] = None,
        capability: Optional[str] = None,
        train: Optional[str] = None,
    ) -> Dict[str, Any]:
        """原始能力诊断事件（含 diagnosis 明细，供运行中监测/线路侧页面）。"""
        return {"events": pf.events.list_raw(limit=limit, anomaly_type=anomaly, capability_id=capability, train_id=train)}

    @app.post("/api/v1/workorders/{workorder_id}/close")
    def close_workorder(workorder_id: str, body: ReviewBody) -> Dict[str, Any]:
        try:
            return pf.workorders.close(workorder_id, operator=body.operator, note=body.note)
        except KeyError:
            raise HTTPException(status_code=404, detail="工单不存在")

    @app.get("/api/v1/capabilities")
    def list_capabilities() -> Dict[str, Any]:
        return {"capabilities": [c.to_public_dict() for c in pf.registry.all()]}

    @app.post("/api/v1/capabilities/register")
    def register_capability(body: RegisterBody) -> Dict[str, Any]:
        if body.base_url:
            descriptor = dict(body.descriptor)
            instance_id = pf.attach_rest_capability(descriptor, body.base_url, mode=body.mode)
            return {"ok": True, "instance_id": instance_id, "capability_id": descriptor.get("capability_id")}
        try:
            from railmind.capabilities.sdk import InProcessCapability

            cap = InProcessCapability(pf.registry, body.descriptor, infer_fn=_fallback_infer, mode=body.mode)
            cap.heartbeat_once()
            pf.router.attach_local(cap)
            return {"ok": True, "instance_id": cap.instance_id, "capability_id": body.descriptor.get("capability_id")}
        except RegistrationError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/v1/capabilities/{capability_id}/mode")
    def set_mode(capability_id: str, body: ModeBody) -> Dict[str, Any]:
        try:
            changed = pf.registry.set_mode(capability_id, body.mode)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"能力不存在: {capability_id}")
        return {"ok": True, "capability_id": capability_id, "mode": body.mode, "instances": changed}

    @app.post("/api/v1/events")
    def ingest_event(event: Dict[str, Any]) -> Dict[str, Any]:
        return pf.handle_event(event)

    @app.get("/api/v1/events")
    def list_events(limit: int = 20) -> Dict[str, Any]:
        summaries = pf.events.list_summaries(limit=limit)
        if summaries:
            return {"events": summaries}
        # 原始事件平铺映射（服务重启后摘要清空时的兜底）
        rows = []
        for ev in pf.events.query(limit=limit):
            rows.append({
                "event_id": ev.get("event_id"),
                "ts": ev.get("ts"),
                "train_id": ev.get("asset", {}).get("train_id"),
                "component": ev.get("asset", {}).get("component"),
                "anomaly_type": ev.get("diagnosis", {}).get("anomaly_type"),
                "risk_level": ev.get("diagnosis", {}).get("severity"),
            })
        return {"events": rows}

    @app.get("/api/v1/workorders")
    def list_workorders(status: Optional[str] = None) -> Dict[str, Any]:
        return {"workorders": pf.workorders.list(status=status)}

    @app.post("/api/v1/workorders/{workorder_id}/confirm")
    def confirm_workorder(workorder_id: str, body: ReviewBody) -> Dict[str, Any]:
        try:
            return pf.workorders.confirm(workorder_id, operator=body.operator, note=body.note)
        except KeyError:
            raise HTTPException(status_code=404, detail="工单不存在")

    @app.post("/api/v1/workorders/{workorder_id}/reject")
    def reject_workorder(workorder_id: str, body: ReviewBody) -> Dict[str, Any]:
        try:
            return pf.workorders.reject(workorder_id, operator=body.operator, note=body.note)
        except KeyError:
            raise HTTPException(status_code=404, detail="工单不存在")

    return app


def _fallback_infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """进程内注册（无 infer_fn）时的占位：按拒判输出，避免虚构结论。"""
    return {
        "diagnosis": {"anomaly_type": "unable_to_judge", "severity": "UNKNOWN", "confidence": 0.0, "degraded": True},
        "evidence": [],
    }


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("railmind.apps.api:app", host="127.0.0.1", port=int(os.environ.get("RAILMIND_PORT", "8900")), log_level="info")
