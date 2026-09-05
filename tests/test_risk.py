"""确定性风险引擎规则测试。"""

from railmind.core.risk_engine import RiskEngine, LEVEL_HIGH, LEVEL_NORMAL, LEVEL_OBSERVE, LEVEL_UNKNOWN, LEVEL_WARNING


def test_high_severity_strong_confidence_is_high():
    eng = RiskEngine()
    r = eng.assess(severity="HIGH", confidence=0.9, running=True, critical_component=True)
    assert r.level == LEVEL_HIGH
    assert r.human_review_required
    assert r.score >= 65


def test_low_confidence_downgrades_high():
    eng = RiskEngine()
    r = eng.assess(severity="HIGH", confidence=0.55)
    assert r.level == LEVEL_WARNING


def test_minor_impact_is_observe():
    eng = RiskEngine()
    r = eng.assess(severity="OBSERVE", confidence=0.9)
    assert r.level in (LEVEL_NORMAL, LEVEL_OBSERVE)


def test_offline_capability_is_unknown_and_refuses():
    eng = RiskEngine()
    r = eng.assess(severity="HIGH", confidence=0.99, capability_online=False)
    assert r.level == LEVEL_UNKNOWN
    assert r.human_review_required
    assert r.degraded


def test_degraded_caps_high_to_warning():
    eng = RiskEngine()
    r = eng.assess(severity="HIGH", confidence=0.95, capability_degraded=True)
    assert r.level == LEVEL_WARNING
    assert r.degraded
    assert r.human_review_required


def test_frequency_and_multi_source_raise_score():
    eng = RiskEngine()
    base = eng.assess(severity="WARNING", confidence=0.8).score
    more = eng.assess(severity="WARNING", confidence=0.8, frequency_1h=4, multi_source_support=True).score
    assert more > base + 10


def test_auxiliary_mode_forces_review():
    eng = RiskEngine()
    r = eng.assess(severity="NORMAL", confidence=0.99, capability_mode="AUXILIARY")
    assert r.human_review_required


def test_illegal_severity_is_unknown():
    eng = RiskEngine()
    r = eng.assess(severity="极高风险", confidence=0.9)
    assert r.level == LEVEL_UNKNOWN
