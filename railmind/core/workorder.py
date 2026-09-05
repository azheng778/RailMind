"""工单存储：高风险事件经人工确认后生成工单草稿（方案 1.3-6 / 6.2-10）。

状态机：DRAFT → (确认) OPEN → (处置) CLOSED / (驳回) REJECTED
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

from railmind.core.types import new_id


class WorkOrderStore:
    def __init__(self, path: Optional[str] = None):
        self._lock = threading.RLock()
        self.path = path
        self._orders: Dict[str, Dict[str, Any]] = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for row in json.load(fh):
                    self._orders[row["workorder_id"]] = row

    def _flush(self) -> None:
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(list(self._orders.values()), fh, ensure_ascii=False, indent=2, default=str)

    def create_draft(
        self,
        event: Dict[str, Any],
        assessment: Dict[str, Any],
        conclusion: Dict[str, Any],
        knowledge: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        wo_id = new_id("WO")
        order = {
            "workorder_id": wo_id,
            "status": "DRAFT",
            "created_at": event.get("ts"),
            "train_id": event.get("asset", {}).get("train_id"),
            "component": event.get("asset", {}).get("component"),
            "anomaly_type": event.get("diagnosis", {}).get("anomaly_type"),
            "risk": assessment,
            "agent_conclusion": conclusion,
            "knowledge": knowledge,
            "requires_human_review": True,
            "history": [{"ts": event.get("ts"), "action": "CREATE_DRAFT"}],
        }
        with self._lock:
            self._orders[wo_id] = order
            self._flush()
        return dict(order)

    def confirm(self, workorder_id: str, operator: str, note: str = "") -> Dict[str, Any]:
        return self._transition(workorder_id, "OPEN", operator, note)

    def reject(self, workorder_id: str, operator: str, note: str = "") -> Dict[str, Any]:
        return self._transition(workorder_id, "REJECTED", operator, note)

    def close(self, workorder_id: str, operator: str, note: str = "") -> Dict[str, Any]:
        return self._transition(workorder_id, "CLOSED", operator, note)

    def _transition(self, workorder_id: str, status: str, operator: str, note: str) -> Dict[str, Any]:
        with self._lock:
            order = self._orders.get(workorder_id)
            if order is None:
                raise KeyError(f"E_WORKORDER_NOT_FOUND:{workorder_id}")
            if order["status"] == "DRAFT" and status not in ("OPEN", "REJECTED"):
                raise ValueError(f"E_INVALID_TRANSITION:{order['status']}→{status}")
            order["status"] = status
            order["history"].append({"action": status, "operator": operator, "note": note})
            self._flush()
            return dict(order)

    def list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            rows = [dict(o) for o in self._orders.values()]
        if status:
            rows = [o for o in rows if o["status"] == status]
        rows.sort(key=lambda o: str(o.get("created_at")), reverse=True)
        return rows

    def open_count(self, train_id: Optional[str] = None) -> int:
        rows = self.list()
        return len(
            [
                o
                for o in rows
                if o["status"] in ("DRAFT", "OPEN") and (train_id is None or o.get("train_id") == train_id)
            ]
        )
