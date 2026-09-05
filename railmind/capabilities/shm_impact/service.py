"""独立部署入口：把第 7 号能力包装成 REST 服务（外部热加载演示）。

用法：
    D:/Anaconda3/envs/railmind/python.exe -m railmind.capabilities.shm_impact.service --port 8901
然后在平台侧 POST /api/v1/capabilities/register 提交 capability.yaml
（invoke.protocol 改为 REST，base_url 指向 http://127.0.0.1:8901），
平台不重启即可发现并调用 —— 对应方案 2.2"外部能力热加载流程"。
"""

from __future__ import annotations

import argparse
import os

from railmind.capabilities.sdk import serve_capability
from railmind.capabilities.shm_impact.adapter import infer

DESCRIPTOR = os.path.join(os.path.dirname(__file__), "capability.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RailMind SHM 冲击定位能力服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("RAILMIND_SHM_PORT", "8901")))
    parser.add_argument("--model-version", default="1.0.0")
    args = parser.parse_args()
    serve_capability(DESCRIPTOR, infer, host=args.host, port=args.port, model_version=args.model_version)
