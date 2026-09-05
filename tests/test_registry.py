"""能力注册中心测试：重复ID / TTL / 主备切换 / 场景过滤 / 模式管理。"""

import pytest

from railmind.core.registry import (
    CapabilityRegistry,
    RegistrationError,
    MODE_AUXILIARY,
    MODE_DISABLED,
    MODE_PRIMARY,
    MODE_STANDBY,
)


def make_descriptor(cap_id="cap.test.a", version="1.0.0", scenes=None):
    return {
        "capability_id": cap_id,
        "name": "测试能力A",
        "provider": "test",
        "version": version,
        "domain": "test_domain",
        "supported_scenes": scenes or ["STATION_STOP"],
        "inputs": [{"name": "signal", "type": "matrix"}],
        "invoke": {"protocol": "PYTHON", "endpoint": "x", "timeout_ms": 1000},
    }


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, s):
        self.now += s


def test_register_and_query_by_scene():
    reg = CapabilityRegistry()
    reg.register(make_descriptor(), mode=MODE_PRIMARY)
    assert reg.dispatch_target(domain="test_domain", scene="STATION_STOP") is not None
    assert reg.dispatch_target(domain="test_domain", scene="TRAIN_RUNNING") is None
    assert reg.dispatch_target(domain="other_domain", scene="STATION_STOP") is None


def test_duplicate_same_version_rejected():
    reg = CapabilityRegistry()
    reg.register(make_descriptor())
    with pytest.raises(RegistrationError, match="E_DUPLICATE_CAPABILITY"):
        reg.register(make_descriptor())


def test_heartbeat_ttl_expiry():
    clock = FakeClock()
    reg = CapabilityRegistry(heartbeat_ttl_s=10, clock=clock)
    reg.register(make_descriptor(), mode=MODE_PRIMARY)
    assert reg.dispatch_target(domain="test_domain") is not None
    clock.advance(20)  # 无心跳 → 超时离线
    assert reg.dispatch_target(domain="test_domain") is None
    assert reg.sweep_offline()


def test_standby_takeover_when_primary_offline():
    clock = FakeClock()
    reg = CapabilityRegistry(heartbeat_ttl_s=10, clock=clock)
    inst_a = reg.register(make_descriptor("cap.test.a"), mode=MODE_PRIMARY)
    clock.advance(5)
    inst_b = reg.register(make_descriptor("cap.test.b"), mode=MODE_STANDBY)
    clock.advance(8)  # t=1013：主用 13s 无心跳超时；备用 8s 前注册/心跳仍在线
    reg.sweep_offline()
    primary = reg.dispatch_target(domain="test_domain")
    assert primary.mode == MODE_STANDBY  # 备用接管
    assert primary.capability_id == "cap.test.b"
    assert reg.get("cap.test.a")[0].is_online(reg.heartbeat_ttl_s, now=clock()) is False


def test_disabled_and_observe_excluded():
    reg = CapabilityRegistry()
    reg.register(make_descriptor("cap.test.d"), mode=MODE_DISABLED)
    reg.register(make_descriptor("cap.test.o"), mode="OBSERVE")
    assert reg.dispatch_target(domain="test_domain") is None
    assert len(reg.observe_targets(domain="test_domain")) == 1


def test_set_mode_and_auxiliary_ranking():
    reg = CapabilityRegistry()
    reg.register(make_descriptor("cap.test.a"), mode=MODE_AUXILIARY)
    reg.set_mode("cap.test.a", MODE_PRIMARY)
    assert reg.dispatch_target(domain="test_domain").mode == MODE_PRIMARY
    with pytest.raises(KeyError):
        reg.set_mode("cap.missing", MODE_PRIMARY)


def test_version_conflict_rejected():
    reg = CapabilityRegistry()
    reg.register(make_descriptor(version="2.0.0"))
    with pytest.raises(RegistrationError, match="E_VERSION_CONFLICT"):
        reg.register(make_descriptor(version="1.0.0"))
