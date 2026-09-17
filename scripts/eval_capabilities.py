"""能力层批量评测脚本 —— 对应《功能测试与评测方案V1.0》L1 层。

对 7 项已注册能力逐一执行 N 次真实推理，统计：
    成功率 / 平均时延(P50/P95) / severity 分布 / 平均置信度 / 降级率 / 错误

输入构造直接复用 demo_events.build_demo_event，保证与大屏按钮走同一条数据路径。

用法（工作目录：仓库根）::

    python -X utf8 scripts/eval_capabilities.py --repeat 5
    python -X utf8 scripts/eval_capabilities.py --only shm_impact,panto_arc --repeat 3
    python -X utf8 scripts/eval_capabilities.py --out data/eval_report.json

说明：单项能力失败（权重缺失、VLM 未配置、素材缺失）不会中断整体，
会记为 failed 并在报告里给出原因 —— 这正是"降级不虚构"的验证点之一。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# kind -> (演示事件类型, 推理函数导入路径)
CAPABILITIES: Dict[str, str] = {
    "shm_impact": "railmind.capabilities.shm_impact.adapter:infer",
    "door_normal": "railmind.capabilities.door_handle.adapter:infer",
    "panto_wear": "railmind.capabilities.panto.wear:infer",
    "panto_arc": "railmind.capabilities.panto.arc:infer",
    "lineside_fod": "railmind.capabilities.lineside_fod.adapter:infer",
    "fastener_defect": "railmind.capabilities.fastener.adapter:infer",
    "cabin_patrol": "railmind.capabilities.cabin_vlm.adapter:infer",
}


def _load_infer(spec: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    module, _, attr = spec.partition(":")
    import importlib

    return getattr(importlib.import_module(module), attr)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[idx], 1)


def eval_one(kind: str, repeat: int) -> Dict[str, Any]:
    """跑一项能力 repeat 次，返回统计结果。"""
    from railmind.apps.demo_events import build_demo_event

    row: Dict[str, Any] = {
        "kind": kind,
        "repeat": repeat,
        "ok": 0,
        "failed": 0,
        "degraded": 0,
        "latency_ms": [],
        "severities": {},
        "confidences": [],
        "errors": [],
        "last_diagnosis": None,
    }

    event = build_demo_event(kind)
    if event is None:
        row["errors"].append("E_NO_SAMPLE: 演示素材缺失，build_demo_event 返回 None")
        return row

    payload = dict(event.get("capability_input") or {})
    infer = _load_infer(CAPABILITIES[kind])

    for i in range(repeat):
        started = time.time()
        try:
            result = infer(payload)
        except Exception as exc:  # noqa: BLE001 —— 单条失败不中断评测
            row["failed"] += 1
            row["errors"].append(f"第{i + 1}次: {type(exc).__name__}: {exc}"[:300])
            continue
        finally:
            row["latency_ms"].append(round((time.time() - started) * 1000.0, 1))

        diag = (result or {}).get("diagnosis") or {}
        row["ok"] += 1
        if diag.get("degraded"):
            row["degraded"] += 1
        sev = str(diag.get("severity", "UNKNOWN"))
        row["severities"][sev] = row["severities"].get(sev, 0) + 1
        try:
            row["confidences"].append(round(float(diag.get("confidence", 0.0)), 3))
        except (TypeError, ValueError):
            row["confidences"].append(0.0)
        row["last_diagnosis"] = diag

    confs = row.pop("confidences")
    lat = row.pop("latency_ms")
    row.update({
        "success_rate": round(row["ok"] / repeat, 3) if repeat else 0.0,
        "degraded_rate": round(row["degraded"] / row["ok"], 3) if row["ok"] else 0.0,
        "latency_p50_ms": _percentile(lat, 50),
        "latency_p95_ms": _percentile(lat, 95),
        "latency_max_ms": round(max(lat), 1) if lat else 0.0,
        "confidence_avg": round(statistics.fmean(confs), 3) if confs else 0.0,
        "severity_final": max(row["severities"], key=row["severities"].get) if row["severities"] else "UNKNOWN",
        "severity_stable": len(row["severities"]) <= 1,
        "errors": row["errors"][:5],
    })
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="RailMind 能力层批量评测")
    ap.add_argument("--repeat", type=int, default=3, help="每项能力重复次数，默认 3")
    ap.add_argument("--only", type=str, default="", help="只跑指定能力，逗号分隔")
    ap.add_argument("--out", type=str, default=os.path.join("data", "eval_report.json"), help="结果输出路径")
    args = ap.parse_args()

    kinds = [k.strip() for k in args.only.split(",") if k.strip()] or list(CAPABILITIES)
    unknown = [k for k in kinds if k not in CAPABILITIES]
    if unknown:
        print(f"[eval] 未知能力: {unknown}；可选: {list(CAPABILITIES)}")
        return 2

    rows: List[Dict[str, Any]] = []
    print(f"{'能力':<18}{'成功率':>8}{'降级率':>8}{'P50(ms)':>10}{'P95(ms)':>10}{'置信度':>8}  严重等级分布")
    print("-" * 96)
    for kind in kinds:
        row = eval_one(kind, args.repeat)
        rows.append(row)
        print(
            f"{kind:<18}{row['success_rate']:>8.2f}{row['degraded_rate']:>8.2f}"
            f"{row['latency_p50_ms']:>10.1f}{row['latency_p95_ms']:>10.1f}{row['confidence_avg']:>8.3f}  {row['severities']}"
        )
        if row["errors"]:
            print(f"    └─ 错误: {row['errors'][0]}")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repeat": args.repeat,
        "python": sys.version.split()[0],
        "rows": rows,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
        print(f"\n[eval] 报告已写入 {os.path.abspath(args.out)}")

    failed = [r["kind"] for r in rows if r["ok"] == 0]
    if failed:
        print(f"[eval] 完全失败的能力: {failed}（多为素材/权重/VLM 未配置，属可降级项）")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
