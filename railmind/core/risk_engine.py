"""确定性风险引擎 —— 方案 6.3：风险等级绝不由大模型自由生成。

规则可解释，每条评分都留下 reasons 供前端展示。
统一等级：NORMAL / OBSERVE / WARNING / HIGH / UNKNOWN
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LEVEL_NORMAL = "NORMAL"
LEVEL_OBSERVE = "OBSERVE"
LEVEL_WARNING = "WARNING"
LEVEL_HIGH = "HIGH"
LEVEL_UNKNOWN = "UNKNOWN"

_SEVERITY_BASE = {LEVEL_NORMAL: 8.0, LEVEL_OBSERVE: 25.0, LEVEL_WARNING: 55.0, LEVEL_HIGH: 80.0}


@dataclass
class RiskAssessment:
    level: str
    score: float
    reasons: List[str] = field(default_factory=list)
    human_review_required: bool = False
    degraded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "score": round(self.score, 1),
            "reasons": self.reasons,
            "human_review_required": self.human_review_required,
            "degraded": self.degraded,
        }


class RiskEngine:
    """输入模型严重等级 + 置信度 + 上下文因子，输出统一风险等级。"""

    def __init__(
        self,
        unknown_below: float = 20.0,
        observe_below: float = 42.0,
        warning_below: float = 68.0,
        low_confidence: float = 0.5,
    ):
        self.unknown_below = unknown_below
        self.observe_below = observe_below
        self.warning_below = warning_below
        self.low_confidence = low_confidence

    def assess(
        self,
        severity: str,
        confidence: float,
        frequency_1h: int = 1,
        duration_s: float = 0.0,
        multi_source_support: bool = False,
        running: bool = True,
        critical_component: bool = False,
        open_workorders: int = 0,
        capability_online: bool = True,
        capability_degraded: bool = False,
        capability_mode: str = "PRIMARY",
    ) -> RiskAssessment:
        reasons: List[str] = []

        # 拒判路径：能力离线 / 输入非法 → UNKNOWN，强制人工复核（方案 6.4-5/6）
        if not capability_online:
            return RiskAssessment(
                level=LEVEL_UNKNOWN,
                score=0.0,
                reasons=["诊断能力离线，无法给出可靠结论"],
                human_review_required=True,
                degraded=True,
            )
        if severity not in _SEVERITY_BASE or confidence is None:
            return RiskAssessment(
                level=LEVEL_UNKNOWN,
                score=0.0,
                reasons=[f"非法诊断输出 severity={severity!r}，按拒判处理"],
                human_review_required=True,
                degraded=True,
            )

        score = _SEVERITY_BASE[severity]
        reasons.append(f"模型严重等级 {severity} 基础分 {score:.0f}")

        score += float(confidence) * 10.0
        reasons.append(f"置信度 {confidence:.2f} 贡献 {confidence * 10:.1f} 分")

        if frequency_1h > 1:
            bonus = min(3, frequency_1h - 1) * 5.0
            score += bonus
            reasons.append(f"1小时内频次 {frequency_1h} 次 +{bonus:.0f} 分")

        if duration_s >= 60:
            score += 5.0
            reasons.append(f"事件持续 {duration_s:.0f}s ≥60s +5 分")

        if multi_source_support:
            score += 10.0
            reasons.append("多源证据相互支持 +10 分")

        if running and severity in (LEVEL_WARNING, LEVEL_HIGH):
            score += 8.0
            reasons.append("运行中发生且涉及 WARNING/HIGH +8 分")

        if critical_component:
            score += 6.0
            reasons.append("涉及关键部件 +6 分")

        if open_workorders > 0:
            score += 4.0
            reasons.append(f"存在 {open_workorders} 张历史未关闭工单 +4 分")

        score = min(100.0, score)

        level = self._level_from_score(score)
        if level == LEVEL_HIGH and confidence < 0.6:
            level = LEVEL_WARNING
            reasons.append("置信度 <0.6，HIGH 降为 WARNING")

        degraded = capability_degraded
        if degraded:
            if level in (LEVEL_HIGH,):
                level = LEVEL_WARNING
                reasons.append("能力处于降级模式，HIGH 封顶为 WARNING")
            reasons.append("能力降级：结论仅供参考，必须人工复核")

        human_review = level in (LEVEL_WARNING, LEVEL_HIGH) or confidence < self.low_confidence
        if capability_mode == "AUXILIARY":
            human_review = True
            reasons.append("辅助模式结论必须人工确认（方案 8.3）")

        return RiskAssessment(level=level, score=score, reasons=reasons, human_review_required=human_review, degraded=degraded)

    def _level_from_score(self, score: float) -> str:
        if score < self.unknown_below:
            return LEVEL_NORMAL
        if score < self.observe_below:
            return LEVEL_OBSERVE
        if score < self.warning_below:
            return LEVEL_WARNING
        return LEVEL_HIGH
