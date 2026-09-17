"""RAG 检索效果评测 CLI。

用法（railmind 环境）:
    python scripts/eval_rag.py                     # 默认输出 docs/RAG检索效果评测报告.md
    python scripts/eval_rag.py --top-k 5           # 自定义 Top-K
    python scripts/eval_rag.py --json out.json     # 同时输出 JSON 明细
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from railmind.core.rag_eval import RagEvaluator, report_markdown  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索效果评测")
    parser.add_argument("--top-k", type=int, default=3, help="检索 Top-K（默认 3，与生产配置一致）")
    parser.add_argument("--alpha", type=float, default=0.55, help="语义权重 α（默认 0.55）")
    parser.add_argument("--md", default="docs/RAG检索效果评测报告.md", help="Markdown 报告输出路径")
    parser.add_argument("--json", default=None, help="JSON 明细输出路径（可选）")
    args = parser.parse_args()

    from railmind.core.rag import RagClient
    client = RagClient(top_k=args.top_k)
    client.kb.alpha = args.alpha

    print(f"开始评测：Top-K={args.top_k}, α={args.alpha}, 用例=36 ...")
    report = RagEvaluator(client, top_k=args.top_k).run()

    # 控制台摘要
    sp = report["search_path"]
    rp = report["retrieve_path"]
    k = args.top_k
    print()
    print("=" * 62)
    print(f"{'指标':<16}{'search() 路径':>18}{'retrieve() 路径':>20}")
    print("-" * 62)
    for label, key in (
        (f"Hit@{k}", f"hit@{k}"), ("MRR", "mrr"),
        (f"Precision@{k}", f"precision@{k}"), (f"Recall@{k}", f"recall@{k}"),
        (f"NDCG@{k}", f"ndcg@{k}"),
        ("正确拒答率", "correct_refusal_rate"), ("误拒答率", "false_refusal_rate"),
        ("P95 时延(ms)", "latency_p95_ms"),
    ):
        print(f"{label:<16}{sp.get(key, '-'):>18}{rp.get(key, '-'):>20}")
    print("-" * 62)
    n_fail = len(report["failures"])
    print(f"失败用例：{n_fail} 条" if n_fail else "失败用例：0 条（全部通过）")

    # 写 Markdown 报告
    md_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.md)
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_markdown(report))
    print(f"\nMarkdown 报告已写入: {md_path}")

    # 可选 JSON 明细
    if args.json:
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.json)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"JSON 明细已写入: {json_path}")


if __name__ == "__main__":
    main()
