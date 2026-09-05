"""internal.panto.arc_signal —— 受电弓电弧风险信号分析能力（方案 5.2 可解释信号链）。

处理链（确定性，无大模型参与）：
    电压/电流波形 → 滑动窗口 → 突变检测(|dV/dt|分位阈值) → 高频能量分析(FFT 高频带占比)
    → 电弧事件判定(突变+高频能量联合) → 事件持续时长与次数统计 → 电弧风险评分

数据来源：
  * 真实数据：Mendeley "Pantograph Arcing in DC Railway Systems" (DOI 10.17632/74nz86wcgy, CC BY 4.0)
  * 演示数据：DEMO_ONLY 合成波形（正弦工频 + 随机电弧事件注入），用于无数据时的降级演示
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FS_DEFAULT = 100000  # 默认采样率 100 kHz（真实数据加载后按实际覆盖）
WINDOW_MS = 2        # 滑动窗口 2ms
HF_BAND = (5000, 40000)  # 高频带 Hz
DVQuantile = 99.5        # 突变检测分位


def analyze_waveform(voltage: np.ndarray, current: Optional[np.ndarray] = None, fs: int = FS_DEFAULT) -> Dict[str, Any]:
    """对一段连续波形做电弧风险分析，输出可解释指标与评分。

    预处理先去直流/低频漂移（滑动均值），电弧判别用「绝对高频能量相对中位数基线的倍数」——
    电弧噪声是宽带的，能量法比占比法对强直流/工频信号更稳健。
    """
    voltage = np.asarray(voltage, dtype=np.float64)
    win = max(8, int(fs * WINDOW_MS / 1000))
    hop = win // 2

    # 0) 去直流/低频：减去滑动均值（5ms）
    ma_win = max(win, int(fs * 0.005))
    kernel = np.ones(ma_win) / ma_win
    residual = voltage - np.convolve(voltage, kernel, mode="same")

    # 1) 滑动窗口突变检测
    dv = np.abs(np.diff(residual))
    dv_thr = np.percentile(dv, 99.5)
    spike = dv > max(dv_thr, 1e-9)

    # 2) 每窗口高频带绝对能量
    nwin = max(1, (len(voltage) - win) // hop)
    hf_energy = np.zeros(nwin)
    spike_ratio = np.zeros(nwin)
    freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    hf_band = (freqs >= HF_BAND[0]) & (freqs <= HF_BAND[1])
    for i in range(nwin):
        seg = residual[i * hop: i * hop + win]
        if len(seg) < win:
            break
        spec = np.abs(np.fft.rfft(seg * np.hanning(win))) ** 2
        hf_energy[i] = spec[hf_band].sum()
        s = slice(i * hop, i * hop + win - 1)
        spike_ratio[i] = spike[s].mean() if spike[s].size else 0.0

    # 3) 事件判定：高频能量相对稳健基线（中位数*MAD）显著抬升，且窗口内突变密度不低
    med = float(np.median(hf_energy)) + 1e-12
    mad = float(np.median(np.abs(hf_energy - med))) + 1e-12
    hf_thr = med + 12.0 * mad
    sp_thr = 0.02
    events = (hf_energy > hf_thr) & (spike_ratio > sp_thr)

    # 4) 事件时长与次数统计
    runs: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(events):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(events)))
    durations_ms = [(b - a) * hop / fs * 1000.0 for a, b in runs]
    total_arc_ms = float(sum(durations_ms))

    # 5) 风险评分（可解释加权）
    max_hf_ratio = float((hf_energy / med).max()) if len(hf_energy) else 0.0
    score = 0.0
    reasons: List[str] = []
    score += min(len(runs), 10) * 6.0
    if runs:
        reasons.append(f"检出疑似电弧事件 {len(runs)} 次")
    else:
        reasons.append("未检出电弧事件")
    score += min(total_arc_ms / 2.0, 25.0)
    if total_arc_ms > 0:
        reasons.append(f"电弧累计时长 {total_arc_ms:.1f} ms")
    score += min(max_hf_ratio / 50.0, 25.0)
    reasons.append(f"事件窗口高频能量相对基线最高 {max_hf_ratio:.0f} 倍")
    if current is not None and len(current) == len(voltage):
        zero_cross_flat = float(np.mean(np.abs(np.diff(np.asarray(current, dtype=np.float64))) < 1e-6))
        score += zero_cross_flat * 10.0
        reasons.append(f"电流平顶占比 {zero_cross_flat:.2f}（电弧限流特征）")

    score = min(100.0, score)
    severity = "NORMAL" if score < 20 else ("OBSERVE" if score < 45 else ("WARNING" if score < 70 else "HIGH"))
    return {
        "anomaly_type": "pantograph_arc" if runs else "pantograph_arc_none",
        "severity": severity,
        "confidence": round(min(0.95, 0.5 + len(runs) * 0.05 + min(max_hf_ratio / 100.0, 0.3)), 3),
        "degraded": False,
        "arc_events": len(runs),
        "arc_total_ms": round(total_arc_ms, 2),
        "event_durations_ms": [round(d, 2) for d in durations_ms[:10]],
        "max_hf_ratio": round(max_hf_ratio, 1),
        "arc_score": round(score, 1),
        "reasons": reasons,
        "fs": fs,
    }


# ---------- 合成演示波形（DEMO_ONLY） ----------

def synthetic_waveform(fs: int = FS_DEFAULT, seconds: float = 0.2, arc_events: int = 3, seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """合成 3kV DC 供电下的受电弓电压/电流：工频纹波 + 随机电弧事件（电压跌落+高频抖动+电流平顶）。"""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    voltage = 3000.0 + 8.0 * np.sin(2 * np.pi * 100 * t) + rng.normal(0, 1.2, n)
    current = 220.0 + 3.0 * np.sin(2 * np.pi * 100 * t + 0.7) + rng.normal(0, 0.8, n)
    arc_len = int(fs * 0.002)
    for k in range(arc_events):
        s = int(rng.integers(0, n - arc_len))
        voltage[s:s + arc_len] *= rng.uniform(0.55, 0.75)
        voltage[s:s + arc_len] += rng.normal(0, 60, arc_len)
        current[s:s + arc_len] = current[s:s + arc_len] * 0.25 + 40
    return voltage, current


# ---------- SDK 推理入口 ----------

ARC_DATA_DIR = os.environ.get(
    "RAILMIND_ARC_DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "pantograph", "arcing")),
)


def infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """payload 三选一：
    * {mendeley_record: {folder, phase, name}} —— Mendeley 电弧数据集真实记录（DOI 10.17632/74nz86wcgy）
    * {waveform_uri: .npy/.csv/.mat}
    * {synthetic: {arc_events, seconds, seed}} —— DEMO_ONLY 合成波形
    """
    fs = int(payload.get("fs", 0))
    if payload.get("mendeley_record"):
        rec = payload["mendeley_record"]
        voltage, current, fs_real = _load_mendeley_record(rec["folder"], rec["phase"], rec["name"])
        source = f"mendeley:{rec['folder']}/{rec['phase']}/{rec['name']}"
    elif payload.get("waveform_uri"):
        voltage, current = _load_waveform(payload["waveform_uri"])
        fs_real = fs or FS_DEFAULT
        source = "file"
    else:
        syn = payload.get("synthetic") or {}
        voltage, current = synthetic_waveform(
            fs=fs or FS_DEFAULT,
            seconds=float(syn.get("seconds", 0.2)),
            arc_events=int(syn.get("arc_events", 3)),
            seed=int(syn.get("seed", 7)),
        )
        fs_real = fs or FS_DEFAULT
        source = "synthetic(DEMO_ONLY)"
    diag = analyze_waveform(voltage, current, fs=fs_real)
    diag["source"] = source
    if source.startswith("synthetic"):
        diag["note"] = "演示波形（DEMO_ONLY），非实车数据"
    return {"diagnosis": diag, "evidence": [{"type": "waveform", "uri": payload.get("waveform_uri", source), "samples": int(len(voltage)), "fs": fs_real}]}


def _load_mendeley_record(folder: str, phase: str, name: str) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Mendeley 电弧数据集（DOI 10.17632/74nz86wcgy）：平铺 txt 布局 <root>/<公司>/<工况>/<记录名>_Vp.txt。"""
    p = os.path.join(ARC_DATA_DIR, folder, phase)
    x = np.loadtxt(os.path.join(p, f"{name}_x.txt"))
    voltage = np.loadtxt(os.path.join(p, f"{name}_Vp.txt"))
    ip_path = os.path.join(p, f"{name}_Ip.txt")
    current = np.loadtxt(ip_path) if os.path.exists(ip_path) else None
    fs = int(round((len(x) - 1) / (x[-1] - x[0]))) if x[-1] > x[0] else FS_DEFAULT
    return voltage, current, fs


def _load_waveform(uri: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if uri.endswith(".npy"):
        arr = np.load(uri, allow_pickle=True)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
        return arr.astype(np.float64).ravel(), None
    if uri.endswith(".csv"):
        import csv

        cols: List[List[float]] = []
        with open(uri, newline="", encoding="utf-8", errors="ignore") as fh:
            for row in csv.reader(fh):
                vals = []
                for cell in row:
                    try:
                        vals.append(float(cell))
                    except ValueError:
                        pass
                if vals:
                    cols.append(vals)
        n = min(len(c) for c in cols)
        data = np.array([c[:n] for c in cols])  # (cols, n)
        return data[0], (data[1] if len(data) > 1 else None)
    if uri.endswith(".mat"):
        import scipy.io as sio

        data = sio.loadmat(uri)
        for key in ("voltage", "Vp", "v"):
            if key in data:
                v = np.asarray(data[key]).ravel()
                i = None
                for k2 in ("current", "Ip", "i"):
                    if k2 in data:
                        i = np.asarray(data[k2]).ravel()
                        break
                return v, i
        first = [k for k in data if not k.startswith("__")]
        if first:
            return np.asarray(data[first[0]]).ravel(), None
    raise ValueError(f"E_INVALID_INPUT:不支持的波形文件 {uri}")
