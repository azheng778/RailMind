"""Cabin 车厢巡检能力测试（缓存回放 + 方案5.5升级规则，不依赖 VLM 网络）。"""

from railmind.capabilities.cabin_vlm import adapter


def test_cache_infer_normal():
    out = adapter.infer({"source": "cache"})
    d = out["diagnosis"]
    assert d["anomaly_type"] == "cabin_patrol"
    assert d["severity"] == "NORMAL"
    assert d["consistency"]["consistent"] is True
    assert len(d["windows"]) == 2
    assert all(w["scene"] == "normal" for w in d["windows"])
    assert out["evidence"][0]["type"] == "video"
    assert all(e["uri"].startswith("/static/cabin_vlm/frames/") for e in out["evidence"][1:])


def test_window_severity_rules():
    assert adapter.window_severity("smoke", False) == "HIGH"
    assert adapter.window_severity("person_on_floor", False) == "WARNING"
    assert adapter.window_severity("aisle_blocked", False) == "WARNING"
    assert adapter.window_severity("abnormal_gathering", False) == "OBSERVE"
    assert adapter.window_severity("normal", False) == "NORMAL"
    assert adapter.window_severity("unknown", True) == "UNKNOWN"


def _synthetic_cache(scene1, scene2):
    return {"windows": [
        {"window_index": 1, "start_s": 0.5, "end_s": 6.0, "scene": scene1, "confidence": 0.9,
         "summary": "", "events": [], "frames": [], "degraded": False},
        {"window_index": 2, "start_s": 6.0, "end_s": 11.4, "scene": scene2, "confidence": 0.9,
         "summary": "", "events": [], "frames": [], "degraded": False},
    ]}


def test_inconsistent_windows_do_not_escalate(monkeypatch):
    monkeypatch.setattr(adapter, "load_cache", lambda: _synthetic_cache("aisle_blocked", "normal"))
    d = adapter.infer({"source": "cache"})["diagnosis"]
    assert d["consistency"]["consistent"] is False
    assert d["severity"] == "OBSERVE"  # 单窗异常不过半 → 不升级，只观察


def test_consistent_smoke_escalates(monkeypatch):
    monkeypatch.setattr(adapter, "load_cache", lambda: _synthetic_cache("smoke", "smoke"))
    d = adapter.infer({"source": "cache"})["diagnosis"]
    assert d["consistency"]["consistent"] is True
    assert d["severity"] == "HIGH"
