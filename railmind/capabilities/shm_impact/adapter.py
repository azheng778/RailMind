"""internal.shm.impact_locator —— 第 7 号能力：复合材料结构冲击定位与能量分级。

复用团队复赛模型（残差 + PAN 融合 + CBAM 注意力，565K 参数），
预处理与复赛提交版 predict_enhanced.py 完全一致：
    .mat['signal'] (5000, 8) → 逐通道 Robust 缩放(中位数/IQR) → 模型 →
    位置回归(X1,Y1,X2,Y2 mm) + 双头能量分类(0.20/0.35/0.50/0.70/1.00 J)

降级路径（方案 2.4）：torch 不可用或权重加载失败时进入 DEGRADED 模式，
输出 severity=UNKNOWN 的拒判结论，交由风险引擎转人工复核，绝不虚构结果。
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

ENERGY_LEVELS = [0.20, 0.35, 0.50, 0.70, 1.00]
# 能力自带权重（复赛提交版 epoch184: RMSE 40.99mm / 能量准确率 90.5%；离线备份在
# 第二届轨道交通比赛/提交材料/semi-final/）
_PACKAGED_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "best_enhanced_model.pth")
# 环境变量 > 能力自带权重
DEFAULT_MODEL_PATH = os.environ.get("RAILMIND_SHM_MODEL_PATH", _PACKAGED_WEIGHTS)

# 能量 → 统一严重等级（与演示规程 DOC-DEMO-001 的分级一致）
def energy_to_severity(max_energy_j: float) -> str:
    if max_energy_j >= 0.70:
        return "HIGH"      # 显著冲击：立即人工敲击/无损检测复核
    if max_energy_j >= 0.35:
        return "WARNING"   # 重点复核：下一停靠站目视复核
    return "OBSERVE"       # 轻微冲击：记录观察


class ShmImpactModel:
    """封装复赛增强模型；线程安全；延迟加载；可降级。"""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._lock = threading.Lock()
        self._model = None
        self._mode = "not_loaded"
        self._meta: Dict[str, Any] = {}
        self.load()

    # ---------- 加载 ----------

    def load(self) -> bool:
        try:
            import torch  # noqa: PLC0415 —— 延迟导入，保证无 torch 时能力仍可注册（降级模式）

            from railmind.capabilities.shm_impact.enhanced_model import create_enhanced_model

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            model = create_enhanced_model(dropout=0.0)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            with self._lock:
                self._model = model
                self._mode = "torch"
                self._meta = {
                    "val_loss": checkpoint.get("val_loss"),
                    "pos_rmse_mm": checkpoint.get("pos_rmse"),
                    "energy_accuracy": checkpoint.get("energy_accuracy"),
                }
            return True
        except Exception as exc:  # noqa: BLE001 —— 降级为拒判模式
            with self._lock:
                self._model = None
                self._mode = "degraded"
                self._meta = {"degrade_reason": str(exc)}
            return False

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def meta(self) -> Dict[str, Any]:
        return dict(self._meta)

    # ---------- 推理 ----------

    def predict_matrix(self, signal: np.ndarray) -> Dict[str, Any]:
        """signal: (5000, 8) float。返回结构化诊断（不含事件包装）。"""
        signal = self._validate(signal)
        if self._mode != "torch":
            return self._degraded_result(reason=self._meta.get("degrade_reason", "model unavailable"))

        import torch  # noqa: PLC0415

        x = self._robust_scale(signal)
        tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # (1, 5000, 8)
        with torch.no_grad():
            positions, energy1_logits, energy2_logits = self._model(tensor)

        pos = positions.squeeze(0).cpu().numpy()
        e1_logits = energy1_logits.squeeze(0).cpu().numpy()
        e2_logits = energy2_logits.squeeze(0).cpu().numpy()
        impacts = self._to_impacts(pos, e1_logits, e2_logits)
        max_energy = max(i["energy_j"] for i in impacts)
        confidence = round(float(np.mean([i["confidence"] for i in impacts])), 3)
        return {
            "anomaly_type": "composite_impact",
            "severity": energy_to_severity(max_energy),
            "confidence": confidence,
            "impacts": impacts,
            "max_energy_j": max_energy,
            "degraded": False,
            "model_mode": "torch",
        }

    # ---------- 输入/预处理 ----------

    @staticmethod
    def _validate(signal: np.ndarray) -> np.ndarray:
        arr = np.asarray(signal, dtype=np.float32)
        if arr.shape != (5000, 8):
            raise ValueError(f"E_INVALID_SHAPE:期望 (5000, 8)，实际 {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("E_INVALID_VALUES:信号含 NaN/Inf")
        return arr

    @staticmethod
    def _robust_scale(signal: np.ndarray) -> np.ndarray:
        """逐通道 (x - median) / IQR，IQR=0 时该通道除以 1。与 RobustScaler 一致。"""
        out = np.empty_like(signal)
        for ch in range(signal.shape[1]):
            col = signal[:, ch]
            med = np.median(col)
            iqr = float(np.percentile(col, 75) - np.percentile(col, 25))
            out[:, ch] = (col - med) / (iqr if iqr > 0 else 1.0)
        return out

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _to_impacts(self, pos: np.ndarray, e1_logits: np.ndarray, e2_logits: np.ndarray) -> List[Dict[str, Any]]:
        """转成 [ {x,y,energy_j,confidence} × 2 ]，能量小的为点1（比赛排序规则）。"""
        p1 = {"x_mm": int(round(float(pos[0]))), "y_mm": int(round(float(pos[1]))),
              "energy_j": round(ENERGY_LEVELS[int(np.argmax(e1_logits))], 2),
              "confidence": round(float(np.max(self._softmax(e1_logits))), 3)}
        p2 = {"x_mm": int(round(float(pos[2]))), "y_mm": int(round(float(pos[3]))),
              "energy_j": round(ENERGY_LEVELS[int(np.argmax(e2_logits))], 2),
              "confidence": round(float(np.max(self._softmax(e2_logits))), 3)}
        if (p1["energy_j"], p1["x_mm"], p1["y_mm"]) > (p2["energy_j"], p2["x_mm"], p2["y_mm"]):
            p1, p2 = p2, p1
        return [p1, p2]

    def _degraded_result(self, reason: str) -> Dict[str, Any]:
        return {
            "anomaly_type": "unable_to_judge",
            "severity": "UNKNOWN",
            "confidence": 0.0,
            "impacts": [],
            "max_energy_j": 0.0,
            "degraded": True,
            "model_mode": "degraded",
            "degrade_reason": reason[:200],
        }


# ---------- 能力入口（SDK infer_fn 签名） ----------

_model: Optional[ShmImpactModel] = None


def get_model() -> ShmImpactModel:
    global _model
    if _model is None:
        _model = ShmImpactModel()
    return _model


def load_signal(source: Dict[str, Any]) -> np.ndarray:
    """payload 支持 signal_uri（.mat 路径）或 signal（嵌套数组）。"""
    if source.get("signal_uri"):
        import scipy.io as sio

        data = sio.loadmat(source["signal_uri"])
        return np.asarray(data["signal"], dtype=np.float32)
    if source.get("signal") is not None:
        return np.asarray(source["signal"], dtype=np.float32)
    raise ValueError("E_INVALID_INPUT:需要 signal_uri 或 signal")


def infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """SDK 推理函数：payload → {diagnosis, evidence}（不含事件包装）。"""
    model = get_model()
    signal = load_signal(payload)
    diagnosis = model.predict_matrix(signal)
    evidence = [
        {
            "type": "signal",
            "uri": payload.get("signal_uri", "inline"),
            "shape": list(signal.shape),
            "model_mode": diagnosis["model_mode"],
        }
    ]
    return {"diagnosis": diagnosis, "evidence": evidence}
