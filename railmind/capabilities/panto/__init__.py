"""受电弓领域能力：滑板磨耗 VLM 评估 + 电弧风险信号分析（方案 5.2 Panto 子能力）。"""

from railmind.capabilities.panto.arc import analyze_waveform, infer as arc_infer, synthetic_waveform
from railmind.capabilities.panto.wear import infer as wear_infer

__all__ = ["analyze_waveform", "arc_infer", "wear_infer", "synthetic_waveform"]
