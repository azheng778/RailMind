"""Cabin 演示缓存预计算：乘务员巡检视频 → 抽帧 → VLM 多帧分析 → demo_cache.json。

运行： D:/Anaconda3/envs/railmind/python.exe -X utf8 scripts/precompute_cabin.py [--refresh]
  默认缓存已存在且未加 --refresh 时直接退出（省 token）。

产物（railmind/capabilities/cabin_vlm/）：
  frames/w{1,2}_f{1..6}.jpg   证据帧（窗口内每秒 1 帧）
  demo_cache.json             每窗口 VLM 结构化结果 + 窗口元数据

方案 5.5：4-10s 窗口 / 每窗 6-8 帧 / 固定 JSON / 不输出确定性暴力结论。
"""

from __future__ import annotations

import json
import os
import sys

import cv2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

VIDEO = os.path.join(REPO, "web", "assets", "cabin_patrol.mp4")
OUT_DIR = os.path.join(REPO, "railmind", "capabilities", "cabin_vlm")
FRAME_DIR = os.path.join(OUT_DIR, "frames")
CACHE_PATH = os.path.join(OUT_DIR, "demo_cache.json")

# 演示视频有效段 0-11.4s（其后为素材片尾），两个窗口各 6 帧（1s 间隔）
WINDOWS = [
    {"index": 1, "start": 0.5, "end": 6.0, "n_frames": 6},
    {"index": 2, "start": 6.0, "end": 11.4, "n_frames": 6},
]
VIDEO_SECONDS = 11.4

PROMPT = (
    "你是高铁车厢智能巡检系统的视觉分析模块。以下 {n} 帧画面来自行驶中高铁车厢的巡检相机，按时间顺序排列。"
    "请只输出一个 JSON 对象（不要多余文字），字段："
    '{"scene": "normal|smoke|person_on_floor|aisle_blocked|door_obstruction|luggage_overhang|abnormal_gathering|violent_motion|unknown", '
    '"confidence": 0到1的小数, "summary": "一句话中文结论", '
    '"events": [{"type": "事件类型", "confidence": 0到1, "frame": 帧序号从1开始}]}. '
    "事件类型仅限：烟雾火光/人员倒地/通道堵塞/车门阻挡/行李架物品伸出/异常聚集/疑似剧烈动作。"
    "注意：不得输出'确认暴力事件'等确定性结论；画面正常则 scene=normal；无法判断则 scene=unknown。"
)


def extract_frames() -> None:
    os.makedirs(FRAME_DIR, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"E_OPEN_VIDEO: 无法打开 {VIDEO}")
    for win in WINDOWS:
        for i in range(win["n_frames"]):
            t = win["start"] + i * (win["end"] - win["start"]) / max(1, win["n_frames"] - 1)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"E_READ_FRAME: 窗口{win['index']} 帧{i} 读取失败")
            scale = min(1.0, 720.0 / max(frame.shape[:2]))
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            path = os.path.join(FRAME_DIR, f"w{win['index']}_f{i + 1}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    cap.release()
    print(f"[extract] {sum(w['n_frames'] for w in WINDOWS)} 帧已写入 {FRAME_DIR}")


def analyze_window(client, win) -> dict:
    from railmind.core.vlm import resize_jpeg_bytes  # noqa: PLC0415

    paths = [os.path.join(FRAME_DIR, f"w{win['index']}_f{i + 1}.jpg") for i in range(win["n_frames"])]
    images = [resize_jpeg_bytes(open(p, "rb").read(), max_side=720, quality=80) for p in paths]
    result = client.analyze_images_with_fallback(images, PROMPT.replace("{n}", str(win["n_frames"])))
    result["window_index"] = win["index"]
    result["start_s"] = win["start"]
    result["end_s"] = win["end"]
    result["frames"] = [f"/static/cabin_vlm/frames/{os.path.basename(p)}" for p in paths]
    return result


def main() -> None:
    if os.path.exists(CACHE_PATH) and "--refresh" not in sys.argv:
        print(f"[skip] 缓存已存在：{CACHE_PATH}（加 --refresh 重新分析）")
        return
    extract_frames()

    from railmind.core.vlm import VlmClient  # noqa: PLC0415

    client = VlmClient(timeout_s=120)
    if not client.configured:
        raise SystemExit("E_VLM_NOT_CONFIGURED: 未配置 RAILMIND_LLM_*，无法预计算")
    windows = [analyze_window(client, w) for w in WINDOWS]
    cache = {
        "video": "/static/web/assets/cabin_patrol.mp4",
        "video_seconds": VIDEO_SECONDS,
        "precomputed_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "windows": windows,
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=1)
    for w in windows:
        print(f"[vlm] 窗口{w['window_index']}: scene={w.get('scene')} conf={w.get('confidence')} degraded={w.get('degraded')}")
    print(f"[done] {CACHE_PATH}")


if __name__ == "__main__":
    main()
