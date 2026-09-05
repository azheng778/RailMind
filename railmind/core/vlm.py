"""共享 VLM 客户端：OpenAI 兼容视觉模型调用 + 固定 JSON 输出 + 降级路径。

配置来源（优先级）：构造参数 > 环境变量 > 仓库根目录 .env
    RAILMIND_LLM_BASE_URL / RAILMIND_LLM_API_KEY / RAILMIND_VLM_MODEL
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional


def _load_dotenv() -> None:
    """极简 .env 加载（不覆盖已有环境变量）。"""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_dotenv()


class VlmError(RuntimeError):
    pass


class VlmClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 60,
    ):
        self.base_url = (base_url or os.environ.get("RAILMIND_LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("RAILMIND_LLM_API_KEY", "")
        self.model = model or os.environ.get("RAILMIND_VLM_MODEL", "deepseek-v4-flash-vision-exp")
        self.timeout_s = timeout_s

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def analyze_image(self, image_bytes: bytes, prompt: str, mime: str = "jpeg") -> Dict[str, Any]:
        """图 + 提示词 → 解析后的 JSON dict；无法解析时返回 {"_raw": ...} 并标记 degraded。"""
        if not self.configured:
            raise VlmError("E_VLM_NOT_CONFIGURED:未配置 RAILMIND_LLM_BASE_URL/RAILMIND_LLM_API_KEY")
        import requests  # noqa: PLC0415

        b64 = base64.b64encode(image_bytes).decode()
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                ]}],
                "temperature": 0.1,
                "max_tokens": 500,
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
        return _parse_json_loose(content)

    def analyze_image_with_fallback(self, image_bytes: bytes, prompt: str, mime: str = "jpeg") -> Dict[str, Any]:
        """演示稳定性：远端失败时返回 degraded 结果，调用方据此走拒判路径。"""
        try:
            result = self.analyze_image(image_bytes, prompt, mime)
            result.setdefault("degraded", False)
            return result
        except Exception as exc:  # noqa: BLE001
            return {"degraded": True, "severity": "UNKNOWN", "confidence": 0.0,
                    "note": f"VLM不可用，转人工复核: {str(exc)[:160]}"}


def _parse_json_loose(text: str) -> Dict[str, Any]:
    """从模型回复中提取第一个 JSON 对象（容忍 ```json 围栏与前后缀文本）。"""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    raw = m.group(0) if m else text
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"_raw": text}
    except json.JSONDecodeError:
        return {"_raw": text}


def resize_jpeg_bytes(image_bytes: bytes, max_side: int = 900, quality: int = 85) -> bytes:
    """压图：控制 VLM 上传体积。依赖 cv2，失败则原样返回。"""
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes
        scale = min(1.0, max_side / max(img.shape[:2]))
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        ok, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return out.tobytes() if ok else image_bytes
    except Exception:  # noqa: BLE001
        return image_bytes
