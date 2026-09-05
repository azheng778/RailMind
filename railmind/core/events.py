"""诊断事件存储：内存索引 + JSONL 落盘，供 Agent 查询资产历史（时序一致性/频次统计）。"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional


class EventStore:
    def __init__(self, path: Optional[str] = None):
        self._lock = threading.RLock()
        self.path = path
        # 摘要与原始事件同样落盘（审计/可追溯）：重启后告警表与对话助手仍能看到完整历史
        self._summary_path = os.path.splitext(path)[0] + "_summaries.jsonl" if path else None
        self._events: List[Dict[str, Any]] = []
        self._summaries: List[Dict[str, Any]] = []
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self._events.append(json.loads(line))
        if self._summary_path and os.path.exists(self._summary_path):
            with open(self._summary_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self._summaries.append(json.loads(line))

    def append(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)
            self._flush_line(event)

    def _flush_line(self, event: Dict[str, Any]) -> None:
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def append_summary(self, summary: Dict[str, Any]) -> None:
        """Chief 处理摘要（含风险等级/工单号/知识引用），供告警表与决策面板使用。落盘持久化。"""
        with self._lock:
            self._summaries.append(summary)
            if self._summary_path:
                os.makedirs(os.path.dirname(self._summary_path) or ".", exist_ok=True)
                with open(self._summary_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")

    def list_summaries(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(s) for s in reversed(self._summaries[-limit:])]

    def list_raw(
        self,
        limit: int = 20,
        anomaly_type: Optional[str] = None,
        capability_id: Optional[str] = None,
        train_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """原始能力诊断事件（含完整 diagnosis 明细），供运行中监测/线路侧页面展示。"""
        picked: List[Dict[str, Any]] = []
        with self._lock:
            for ev in reversed(self._events):
                src = ev.get("source", {})
                diag = ev.get("diagnosis", {})
                if anomaly_type and anomaly_type not in str(diag.get("anomaly_type", "")):
                    continue
                if capability_id and src.get("capability_id") != capability_id:
                    continue
                if train_id and ev.get("asset", {}).get("train_id") != train_id:
                    continue
                picked.append(json.loads(json.dumps(ev, default=str)))
                if len(picked) >= limit:
                    break
        return picked

    def query(
        self,
        train_id: Optional[str] = None,
        component: Optional[str] = None,
        anomaly_type: Optional[str] = None,
        since_ts: Optional[float] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        picked: List[Dict[str, Any]] = []
        with self._lock:
            for ev in reversed(self._events):  # 新的在前
                asset = ev.get("asset", {})
                diag = ev.get("diagnosis", {})
                if train_id and asset.get("train_id") != train_id:
                    continue
                if component and asset.get("component") != component:
                    continue
                if anomaly_type and diag.get("anomaly_type") != anomaly_type:
                    continue
                if since_ts and ev.get("ts", 0) < since_ts:
                    continue
                picked.append(ev)
                if len(picked) >= limit:
                    break
        return picked
