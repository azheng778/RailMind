"""RAG 检索效果评测框架 —— Hit@K / MRR / Precision@K / Recall@K / NDCG@K / 拒答准确率 / 时延。

组成：
  1. EvalCase        —— 单条评测用例（查询 + 期望命中的知识块 + 难度）
  2. GOLDEN_DATASET  —— 人工标注的黄金数据集（覆盖 5 领域 + 拒答 + 生产路径）
  3. RetrievalMetrics—— 检索质量指标计算（与实现无关，只看结果 ID 列表）
  4. RagEvaluator    —— 评测执行器（跑用例、聚合、分域统计、失败分析）
  5. 报告生成        —— Markdown / JSON 两种输出

用法：
    python scripts/eval_rag.py                # 运行并生成 docs/RAG检索效果评测报告.md
    from railmind.core.rag_eval import RagEvaluator
    report = RagEvaluator(client).run()
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from railmind.core.rag import RagClient

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

ChunkId = Tuple[str, str]  # (document_id, chapter) 唯一定位一个知识块


@dataclass
class EvalCase:
    """单条评测用例。"""

    case_id: str
    query: str                       # 自然语言查询（search 路径）
    keywords: Optional[List[str]]    # 关键词（retrieve 路径；None 表示不走该路径）
    asset_type: Optional[str]        # 元数据过滤
    relevant: Set[ChunkId]           # 期望命中的知识块集合
    should_refuse: bool = False      # 期望拒答（relevant 为空时必须为 True）
    difficulty: str = "easy"         # easy / medium / hard
    domain: str = "general"          # 所属领域（用于分域统计）
    note: str = ""                   # 用例说明


@dataclass
class CaseResult:
    """单条用例的执行结果。"""

    case: EvalCase
    search_ids: List[ChunkId] = field(default_factory=list)   # search() 返回
    search_refused: bool = False
    retrieve_ids: List[ChunkId] = field(default_factory=list) # retrieve() 返回
    retrieve_refused: bool = False
    search_ms: float = 0.0
    retrieve_ms: float = 0.0


# ---------------------------------------------------------------------------
# 检索质量指标（实现无关，输入为有序结果 ID 列表）
# ---------------------------------------------------------------------------


class RetrievalMetrics:
    """标准 IR 指标（二值相关性版本）。"""

    @staticmethod
    def hit_at_k(result_ids: List[ChunkId], relevant: Set[ChunkId], k: int) -> float:
        """Top-K 内是否命中至少一个相关知识块（1.0 / 0.0）。"""
        top = set(result_ids[:k])
        return 1.0 if top & relevant else 0.0

    @staticmethod
    def reciprocal_rank(result_ids: List[ChunkId], relevant: Set[ChunkId]) -> float:
        """第一个相关知识块的倒数排名（1/rank），未命中为 0。"""
        for rank, rid in enumerate(result_ids, start=1):
            if rid in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def precision_at_k(result_ids: List[ChunkId], relevant: Set[ChunkId], k: int) -> float:
        """Top-K 中相关知识块占比。"""
        if not result_ids:
            return 0.0
        top = result_ids[:k]
        hits = sum(1 for rid in top if rid in relevant)
        return hits / len(top) if top else 0.0

    @staticmethod
    def recall_at_k(result_ids: List[ChunkId], relevant: Set[ChunkId], k: int) -> float:
        """相关知识块被召回的比例。"""
        if not relevant:
            return 0.0
        top = set(result_ids[:k])
        return len(top & relevant) / len(relevant)

    @staticmethod
    def ndcg_at_k(result_ids: List[ChunkId], relevant: Set[ChunkId], k: int) -> float:
        """NDCG@K（二值相关性）：rel_i ∈ {0,1}，DCG/IDCG。"""
        def dcg(gains: List[float]) -> float:
            return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

        gains = [1.0 if rid in relevant else 0.0 for rid in result_ids[:k]]
        ideal = [1.0] * min(len(relevant), k)
        idcg = dcg(ideal)
        if idcg <= 0:
            return 0.0
        return dcg(gains) / idcg


# ---------------------------------------------------------------------------
# 黄金数据集（人工标注）
# ---------------------------------------------------------------------------
# 标注规则：
#   relevant = 与查询直接回答该问题的知识块（不含仅擦边的通用条目）；
#   拒答用例 relevant 为空且 should_refuse=True；
#   难度：easy=查询含原文术语 / medium=同义改写 / hard=口语化间接描述。
# ---------------------------------------------------------------------------


def _c(doc: str, chapter: str) -> ChunkId:
    return (doc, chapter)


GOLDEN_DATASET: List[EvalCase] = [
    # ==================== 复合材料结构 (composite_structure) ====================
    EvalCase(
        case_id="CS-01", query="复合材料受到冲击能量0.7焦耳以上怎么处置",
        keywords=["冲击能量分级", "处置"], asset_type="composite_structure",
        relevant={_c("DOC-DEMO-001", "冲击能量分级与处置")},
        difficulty="easy", domain="composite_structure", note="原文术语精确匹配",
    ),
    EvalCase(
        case_id="CS-02", query="同一时间窗内定位到两处冲击如何判读",
        keywords=["双点冲击"], asset_type="composite_structure",
        relevant={_c("DOC-DEMO-001", "双点冲击判读")},
        difficulty="easy", domain="composite_structure", note="双点冲击场景",
    ),
    EvalCase(
        case_id="CS-03", query="冲击监测传感器自检异常怎么办",
        keywords=["传感器", "自检"], asset_type="composite_structure",
        relevant={_c("DOC-DEMO-001", "传感器状态自检")},
        difficulty="easy", domain="composite_structure", note="传感器降级",
    ),
    EvalCase(
        case_id="CS-04", query="一个小时内出现好几次冲击但每次都不严重要紧吗",
        keywords=["冲击", "频次"], asset_type="composite_structure",
        relevant={_c("DOC-DEMO-001", "历史冲击趋势分析")},
        difficulty="hard", domain="composite_structure", note="口语化：高频轻冲击升级规则",
    ),
    EvalCase(
        case_id="CS-05", query="被撞了一下但是力量很小需要处理吗",
        keywords=["冲击", "处置"], asset_type="composite_structure",
        relevant={_c("DOC-DEMO-001", "冲击能量分级与处置")},
        difficulty="hard", domain="composite_structure", note="口语化改写轻微冲击",
    ),
    EvalCase(
        case_id="CS-06", query="传感器噪声基线不正常",
        keywords=["传感器", "自检"], asset_type="composite_structure",
        relevant={_c("DOC-DEMO-001", "传感器状态自检")},
        difficulty="medium", domain="composite_structure", note="噪声基线原文术语",
    ),

    # ==================== 受电弓 (pantograph) ====================
    EvalCase(
        case_id="PT-01", query="受电弓滑板磨耗到什么程度需要更换",
        keywords=["滑板磨耗", "处置"], asset_type="pantograph",
        relevant={_c("DOC-DEMO-002", "滑板磨耗分级")},
        difficulty="easy", domain="pantograph", note="磨耗分级核心规程",
    ),
    EvalCase(
        case_id="PT-02", query="电弧事件持续超过2秒怎么处理",
        keywords=["电弧", "处置"], asset_type="pantograph",
        relevant={_c("DOC-DEMO-002", "电弧异常处置")},
        difficulty="easy", domain="pantograph", note="2秒阈值原文",
    ),
    EvalCase(
        case_id="PT-03", query="累计燃弧时长20毫秒说明什么",
        keywords=["电弧", "处置"], asset_type="pantograph",
        relevant={_c("DOC-DEMO-002", "电弧信号判据")},
        difficulty="medium", domain="pantograph", note="燃弧判据术语",
    ),
    EvalCase(
        case_id="PT-04", query="滑板上发现裂纹怎么处理",
        keywords=["滑板磨耗", "处置"], asset_type="pantograph",
        relevant={_c("DOC-DEMO-002", "滑板裂纹检查")},
        difficulty="medium", domain="pantograph", note="裂纹检查章节",
    ),
    EvalCase(
        case_id="PT-05", query="受电弓老是打火该怎么处理",
        keywords=["电弧", "处置"], asset_type="pantograph",
        relevant={_c("DOC-DEMO-002", "电弧异常处置"), _c("DOC-DEMO-002", "电弧信号判据")},
        difficulty="hard", domain="pantograph", note="口语化：打火=电弧（依赖查询扩展）",
    ),
    EvalCase(
        case_id="PT-06", query="滑板两边磨损厚度不一样",
        keywords=["滑板磨耗"], asset_type="pantograph",
        relevant={_c("DOC-DEMO-002", "滑板裂纹检查")},
        difficulty="hard", domain="pantograph", note="口语化：偏磨=两侧厚度差",
    ),

    # ==================== 车底检查门 (underbody) ====================
    EvalCase(
        case_id="UB-01", query="车底检查门未完全闭合如何处置",
        keywords=["检查门", "处置", "锁闭"], asset_type="underbody",
        relevant={_c("DOC-DEMO-003", "检查门未闭合处置")},
        difficulty="easy", domain="underbody", note="未闭合核心规程",
    ),
    EvalCase(
        case_id="UB-02", query="门把手角度偏差超过75度怎么判断",
        keywords=["检查门", "处置"], asset_type="underbody",
        relevant={_c("DOC-DEMO-003", "门把手角度判据")},
        difficulty="easy", domain="underbody", note="75度阈值原文",
    ),
    EvalCase(
        case_id="UB-03", query="锁机构多久需要润滑保养一次",
        keywords=["检查门", "维护"], asset_type="underbody",
        relevant={_c("DOC-DEMO-003", "锁机构维护周期")},
        difficulty="medium", domain="underbody", note="30天维护周期",
    ),
    EvalCase(
        case_id="UB-04", query="盖板边缘检测出异常",
        keywords=["检查门", "处置"], asset_type="underbody",
        relevant={_c("DOC-DEMO-003", "门把手角度判据")},
        difficulty="medium", domain="underbody", note="盖板边缘在把手章节",
    ),
    EvalCase(
        case_id="UB-05", query="同一个门反复报警说没关好",
        keywords=["检查门", "处置", "锁闭"], asset_type="underbody",
        relevant={_c("DOC-DEMO-003", "检查门未闭合处置")},
        difficulty="hard", domain="underbody", note="口语化：重复告警→锁机构磨损",
    ),

    # ==================== 线路侧 (lineside) ====================
    EvalCase(
        case_id="LS-01", query="线路上发现大面积异物要拦停列车吗",
        keywords=["异物", "处置", "限界"], asset_type="lineside",
        relevant={_c("DOC-DEMO-004", "异物分级与处置")},
        difficulty="easy", domain="lineside", note="异物分级核心规程",
    ),
    EvalCase(
        case_id="LS-02", query="同一个影像点一小时内反复出现异物告警",
        keywords=["异物", "处置"], asset_type="lineside",
        relevant={_c("DOC-DEMO-004", "影像点复核要求")},
        difficulty="medium", domain="lineside", note="重复告警复核",
    ),
    EvalCase(
        case_id="LS-03", query="大风天气需要加强线路巡检吗",
        keywords=["异物", "处置"], asset_type="lineside",
        relevant={_c("DOC-DEMO-004", "恶劣天气加强巡检")},
        difficulty="hard", domain="lineside", note="口语化：大风→恶劣天气加密巡检",
    ),
    EvalCase(
        case_id="LS-04", query="扣件弹条断裂了需要限速吗",
        keywords=["扣件", "处置", "弹条"], asset_type="lineside",
        relevant={_c("DOC-DEMO-005", "扣件缺陷分级处置")},
        difficulty="easy", domain="lineside", note="弹条断裂核心规程",
    ),
    EvalCase(
        case_id="LS-05", query="弹条扣件的安装扭矩标准是多少",
        keywords=["扣件", "弹条"], asset_type="lineside",
        relevant={_c("DOC-DEMO-005", "扣件扭矩标准")},
        difficulty="easy", domain="lineside", note="扭矩数值查询",
    ),
    EvalCase(
        case_id="LS-06", query="更换完扣件之后还要复查吗",
        keywords=["扣件", "处置"], asset_type="lineside",
        relevant={_c("DOC-DEMO-005", "复检与验收")},
        difficulty="medium", domain="lineside", note="复检要求",
    ),
    EvalCase(
        case_id="LS-07", query="弹条位置歪了算缺陷吗",
        keywords=["扣件", "处置", "弹条"], asset_type="lineside",
        relevant={_c("DOC-DEMO-005", "扣件缺陷分级处置")},
        difficulty="hard", domain="lineside", note="口语化：歪了=移位/翻转",
    ),

    # ==================== 车厢巡检 (cabin) ====================
    EvalCase(
        case_id="CB-01", query="车厢内发现疑似烟雾怎么处置",
        keywords=["车内巡检", "处置"], asset_type="cabin",
        relevant={_c("DOC-DEMO-006", "车内事件分级与处置"), _c("DOC-DEMO-006", "烟雾与火灾应急预案")},
        difficulty="easy", domain="cabin", note="烟雾核心规程",
    ),
    EvalCase(
        case_id="CB-02", query="乘务员复核流程是怎么样的",
        keywords=["车内巡检", "处置"], asset_type="cabin",
        relevant={_c("DOC-DEMO-006", "乘务员复核流程")},
        difficulty="easy", domain="cabin", note="复核流程",
    ),
    EvalCase(
        case_id="CB-03", query="乘客躺地上多久算异常",
        keywords=["车内巡检", "处置"], asset_type="cabin",
        relevant={_c("DOC-DEMO-006", "通道堵塞评估标准")},
        difficulty="hard", domain="cabin", note="口语化：躺地上=人员倒地15秒",
    ),
    EvalCase(
        case_id="CB-04", query="通道被行李堵住了怎么判断严重程度",
        keywords=["车内巡检", "处置"], asset_type="cabin",
        relevant={_c("DOC-DEMO-006", "通道堵塞评估标准")},
        difficulty="medium", domain="cabin", note="通道堵塞30%宽度标准",
    ),
    EvalCase(
        case_id="CB-05", query="确认火灾之后系统应该做什么",
        keywords=["烟雾", "处置"], asset_type="cabin",
        relevant={_c("DOC-DEMO-006", "烟雾与火灾应急预案")},
        difficulty="medium", domain="cabin", note="火灾应急联动",
    ),
    EvalCase(
        case_id="CB-06", query="巡检识别置信度不够怎么办",
        keywords=["车内巡检", "处置"], asset_type="cabin",
        relevant={_c("DOC-DEMO-006", "乘务员复核流程")},
        difficulty="medium", domain="cabin", note="低置信度→人工复核",
    ),

    # ==================== 通用跨域 ====================
    EvalCase(
        case_id="GN-01", query="告警达到HIGH等级系统会自动做什么",
        keywords=["告警", "升级"], asset_type="pantograph",
        relevant={_c("DOC-DEMO-007", "告警升级与联动处置")},
        difficulty="easy", domain="general", note="通用告警升级（受电弓域）",
    ),

    # ==================== 拒答用例（should_refuse=True）====================
    EvalCase(
        case_id="RF-01", query="XYZQWE完全无关的查询词汇",
        keywords=["XYZQWE无关术语"], asset_type="pantograph",
        relevant=set(), should_refuse=True,
        difficulty="easy", domain="refusal", note="乱码词应拒答",
    ),
    EvalCase(
        case_id="RF-02", query="火车轮胎气压标准是多少",
        keywords=["轮胎", "气压"], asset_type="pantograph",
        relevant=set(), should_refuse=True,
        difficulty="medium", domain="refusal", note="知识库无轮胎内容（无气轮胎除外）",
    ),
    EvalCase(
        case_id="RF-03", query="食堂饭菜卫生管理规定",
        keywords=["食堂", "饭菜"], asset_type="cabin",
        relevant=set(), should_refuse=True,
        difficulty="easy", domain="refusal", note="完全域外话题",
    ),
    EvalCase(
        case_id="RF-04", query="牵引电机绕组绝缘阻值测量方法",
        keywords=["牵引电机", "绝缘"], asset_type="pantograph",
        relevant=set(), should_refuse=True,
        difficulty="hard", domain="refusal", note="相关但未入库的领域知识（不应虚构）",
    ),
    EvalCase(
        case_id="RF-05", query="车厢烟雾处置流程",
        keywords=["烟雾", "处置"], asset_type="pantograph",
        relevant=set(), should_refuse=True,
        difficulty="medium", domain="refusal", note="正确知识挂错域：pantograph 域查烟雾应拒答",
    ),
]


# ---------------------------------------------------------------------------
# 评测执行器
# ---------------------------------------------------------------------------


def _result_to_ids(basis: List[Dict[str, Any]]) -> List[ChunkId]:
    """从 RagResult.answer_basis 提取 (document_id, chapter) 列表。"""
    return [(c.get("document_id", ""), c.get("chapter", "")) for c in basis]


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = min(int(len(sv) * pct / 100.0), len(sv) - 1)
    return sv[idx]


class RagEvaluator:
    """RAG 检索效果评测器。

    对每个用例同时跑两条检索路径：
      search()   —— 自然语言路径（Chat-Agent 使用）
      retrieve() —— 关键词路径（Chief/Specialist Agent 使用，生产主路径）
    """

    def __init__(
        self,
        client: Optional[RagClient] = None,
        dataset: Optional[List[EvalCase]] = None,
        top_k: int = 3,
    ):
        self.client = client or RagClient(top_k=top_k)
        self.dataset = dataset or GOLDEN_DATASET
        self.top_k = top_k
        self.results: List[CaseResult] = []

    # ---- 执行 ----

    def run(self) -> Dict[str, Any]:
        """执行全部用例，返回完整评测报告字典。"""
        # 预热：首次检索会构建 TF-IDF 索引（~2s），不纳入时延统计
        self.client.search("预热查询", top_k=1)

        self.results = []
        for case in self.dataset:
            r = CaseResult(case=case)

            # 路径1：自然语言 search()
            if not case.should_refuse or case.query:
                t0 = time.perf_counter()
                res = self.client.search(
                    query_text=case.query, asset_type=case.asset_type, top_k=self.top_k
                )
                r.search_ms = (time.perf_counter() - t0) * 1000
                r.search_refused = res.refused
                r.search_ids = _result_to_ids(res.answer_basis)

            # 路径2：关键词 retrieve()（生产 Agent 路径）
            if case.keywords:
                t0 = time.perf_counter()
                res2 = self.client.retrieve(
                    asset_type=case.asset_type or "composite_structure",
                    keywords=case.keywords,
                )
                r.retrieve_ms = (time.perf_counter() - t0) * 1000
                r.retrieve_refused = res2.refused
                r.retrieve_ids = _result_to_ids(res2.answer_basis)

            self.results.append(r)

        return self.aggregate()

    # ---- 聚合 ----

    def _metric_block(self, picker, eligible: Optional[List[CaseResult]] = None) -> Dict[str, Any]:
        """计算一个检索路径的全部指标。

        picker: CaseResult -> (结果ID列表, 是否拒答, 时延ms)
        eligible: 参与该路径统计的用例（None = 全部）。
        """
        m = RetrievalMetrics
        pool = eligible if eligible is not None else self.results
        answerable = [r for r in pool if not r.case.should_refuse]
        refusal = [r for r in pool if r.case.should_refuse]

        hit, mrr, prec, rec, ndcg = [], [], [], [], []
        for r in answerable:
            ids, _refused, _ms = picker(r)
            rel = r.case.relevant
            hit.append(m.hit_at_k(ids, rel, self.top_k))
            mrr.append(m.reciprocal_rank(ids, rel))
            prec.append(m.precision_at_k(ids, rel, self.top_k))
            rec.append(m.recall_at_k(ids, rel, self.top_k))
            ndcg.append(m.ndcg_at_k(ids, rel, self.top_k))

        n = max(len(answerable), 1)
        block: Dict[str, Any] = {
            "answerable_cases": len(answerable),
            f"hit@{self.top_k}": round(sum(hit) / n, 4),
            "mrr": round(sum(mrr) / n, 4),
            f"precision@{self.top_k}": round(sum(prec) / n, 4),
            f"recall@{self.top_k}": round(sum(rec) / n, 4),
            f"ndcg@{self.top_k}": round(sum(ndcg) / n, 4),
        }

        # 拒答准确率（仅统计该路径实际执行的用例）
        if refusal:
            correct_refuse = sum(1 for r in refusal if picker(r)[1])
            block["refusal_cases"] = len(refusal)
            block["correct_refusal_rate"] = round(correct_refuse / len(refusal), 4)
        if answerable:
            false_refuse = sum(1 for r in answerable if picker(r)[1])
            block["false_refusal_rate"] = round(false_refuse / len(answerable), 4)

        # 时延
        lat = [picker(r)[2] for r in pool if picker(r)[2] > 0]
        if lat:
            block.update({
                "latency_mean_ms": round(sum(lat) / len(lat), 1),
                "latency_p50_ms": round(_percentile(lat, 50), 1),
                "latency_p95_ms": round(_percentile(lat, 95), 1),
                "latency_max_ms": round(max(lat), 1),
            })
        return block

    def aggregate(self) -> Dict[str, Any]:
        """聚合所有指标。"""
        # search 路径：全部用例都执行了 search()
        def pick_search(r: CaseResult):
            return r.search_ids, r.search_refused, r.search_ms

        # retrieve 路径：仅统计带 keywords 的用例（模拟 Agent 生产调用）
        def pick_retrieve(r: CaseResult):
            return r.retrieve_ids, r.retrieve_refused, r.retrieve_ms

        retrieve_eligible = [r for r in self.results if r.case.keywords]
        search_block = self._metric_block(pick_search)
        retrieve_block = self._metric_block(pick_retrieve, eligible=retrieve_eligible)

        # 分域统计（search 路径）
        by_domain: Dict[str, Dict[str, Any]] = {}
        for dom in sorted({r.case.domain for r in self.results}):
            sub = [r for r in self.results if r.case.domain == dom]
            ev = RagEvaluator.__new__(RagEvaluator)
            ev.results = sub
            ev.top_k = self.top_k
            ev.client = self.client
            by_domain[dom] = ev._metric_block(pick_search)

        # 分难度统计（search 路径，仅可答用例）
        by_difficulty: Dict[str, Any] = {}
        m = RetrievalMetrics
        for diff in ("easy", "medium", "hard"):
            sub = [r for r in self.results if r.case.difficulty == diff and not r.case.should_refuse]
            if not sub:
                continue
            hits = [m.hit_at_k(r.search_ids, r.case.relevant, self.top_k) for r in sub]
            mrrs = [m.reciprocal_rank(r.search_ids, r.case.relevant) for r in sub]
            by_difficulty[diff] = {
                "cases": len(sub),
                f"hit@{self.top_k}": round(sum(hits) / len(sub), 4),
                "mrr": round(sum(mrrs) / len(sub), 4),
            }

        # 失败用例（search 路径未命中）
        failures = []
        for r in self.results:
            if r.case.should_refuse:
                if not r.search_refused:
                    failures.append({
                        "case_id": r.case.case_id, "type": "应拒未拒",
                        "query": r.case.query, "returned": r.search_ids[:3],
                    })
            else:
                if r.search_refused:
                    failures.append({
                        "case_id": r.case.case_id, "type": "误拒答",
                        "query": r.case.query, "expected": sorted(map(str, r.case.relevant)),
                    })
                elif not (set(r.search_ids[: self.top_k]) & r.case.relevant):
                    failures.append({
                        "case_id": r.case.case_id, "type": "未命中",
                        "query": r.case.query,
                        "expected": sorted(map(str, r.case.relevant)),
                        "returned": r.search_ids[:3],
                    })

        return {
            "meta": {
                "total_cases": len(self.dataset),
                "answerable": sum(1 for c in self.dataset if not c.should_refuse),
                "refusal": sum(1 for c in self.dataset if c.should_refuse),
                "top_k": self.top_k,
                "kb_stats": self.client.stats(),
            },
            "search_path": search_block,
            "retrieve_path": retrieve_block,
            "by_domain_search": by_domain,
            "by_difficulty_search": by_difficulty,
            "failures": failures,
        }


# ---------------------------------------------------------------------------
# Markdown 报告生成
# ---------------------------------------------------------------------------


def report_markdown(report: Dict[str, Any]) -> str:
    """将评测报告字典渲染为 Markdown。"""
    meta = report["meta"]
    sp = report["search_path"]
    rp = report["retrieve_path"]
    k = meta["top_k"]
    lines: List[str] = []

    lines.append("# RAG 检索效果评测报告")
    lines.append("")
    lines.append(f"- 评测用例：**{meta['total_cases']}** 条（可答 {meta['answerable']} / 拒答 {meta['refusal']}）")
    lines.append(f"- 检索配置：Top-K = {k}，混合权重 α = 0.55（语义）+ 0.45（关键词）")
    kb = meta["kb_stats"]
    lines.append(f"- 知识库规模：{kb['documents']} 文档 / {kb['total_chunks']} 知识块 / 词表 {kb['vocab_size']} n-gram")
    lines.append("")

    lines.append("## 一、总体指标")
    lines.append("")
    lines.append("| 指标 | search() 自然语言路径 | retrieve() 关键词路径（生产 Agent 路径） |")
    lines.append("|------|----------------------|----------------------------------------|")
    rows = [
        (f"Hit@{k}", f"hit@{k}"), ("MRR", "mrr"),
        (f"Precision@{k}", f"precision@{k}"), (f"Recall@{k}", f"recall@{k}"),
        (f"NDCG@{k}", f"ndcg@{k}"),
    ]
    for label, key in rows:
        lines.append(f"| {label} | {sp.get(key, '-')} | {rp.get(key, '-')} |")
    lines.append("| 正确拒答率 | {} | {} |".format(sp.get("correct_refusal_rate", "-"), rp.get("correct_refusal_rate", "-")))
    lines.append("| 误拒答率 | {} | {} |".format(sp.get("false_refusal_rate", "-"), rp.get("false_refusal_rate", "-")))
    lines.append("")

    lines.append("## 二、时延")
    lines.append("")
    lines.append("| 指标 | search() | retrieve() |")
    lines.append("|------|----------|------------|")
    for label, key in (("平均", "latency_mean_ms"), ("P50", "latency_p50_ms"),
                       ("P95", "latency_p95_ms"), ("最大", "latency_max_ms")):
        lines.append(f"| {label} (ms) | {sp.get(key, '-')} | {rp.get(key, '-')} |")
    lines.append("")

    lines.append("## 三、分领域指标（search 路径）")
    lines.append("")
    lines.append("| 领域 | 用例数 | Hit@K | MRR | P@K | R@K | NDCG@K | 正确拒答率 |")
    lines.append("|------|--------|-------|-----|-----|-----|--------|------------|")
    for dom, blk in report["by_domain_search"].items():
        n_ans = blk.get("answerable_cases", 0) + blk.get("refusal_cases", 0)
        crr = blk.get("correct_refusal_rate", "-")
        lines.append(
            f"| {dom} | {n_ans} | {blk.get(f'hit@{k}', '-')} | {blk.get('mrr', '-')} | "
            f"{blk.get(f'precision@{k}', '-')} | {blk.get(f'recall@{k}', '-')} | "
            f"{blk.get(f'ndcg@{k}', '-')} | {crr} |"
        )
    lines.append("")

    lines.append("## 四、分难度指标（search 路径）")
    lines.append("")
    lines.append("| 难度 | 用例数 | Hit@K | MRR |")
    lines.append("|------|--------|-------|-----|")
    for diff, blk in report["by_difficulty_search"].items():
        lines.append(f"| {diff} | {blk['cases']} | {blk.get(f'hit@{k}', '-')} | {blk.get('mrr', '-')} |")
    lines.append("")

    lines.append("## 五、失败用例分析")
    lines.append("")
    failures = report["failures"]
    if not failures:
        lines.append("✅ 全部用例通过，无失败。")
    else:
        lines.append(f"共 {len(failures)} 条失败用例：")
        lines.append("")
        lines.append("| 用例 | 类型 | 查询 | 期望 | 实际返回 |")
        lines.append("|------|------|------|------|----------|")
        for f in failures:
            exp = "; ".join(map(str, f.get("expected", []))) or "（应拒答）"
            ret = "; ".join(map(str, f.get("returned", []))) or "（拒答/空）"
            lines.append(f"| {f['case_id']} | {f['type']} | {f['query'][:30]} | {exp[:50]} | {ret[:50]} |")
    lines.append("")

    lines.append("## 六、结论与建议")
    lines.append("")
    hit_val = sp.get(f"hit@{k}", 0)
    lines.append(_conclusions(report, hit_val, k))
    return "\n".join(lines)


def _conclusions(report: Dict[str, Any], hit_val: float, k: int) -> str:
    """根据指标生成结论文本。"""
    sp = report["search_path"]
    rp = report["retrieve_path"]
    parts: List[str] = []
    if hit_val >= 0.9:
        parts.append(f"- 检索质量优秀：Hit@{k} = {hit_val}，绝大多数查询能召回相关知识块。")
    elif hit_val >= 0.75:
        parts.append(f"- 检索质量良好：Hit@{k} = {hit_val}，主要查询场景可用，口语化长尾仍有提升空间。")
    else:
        parts.append(f"- 检索质量待改进：Hit@{k} = {hit_val}，建议扩充同义词表或升级嵌入模型。")

    mrr = sp.get("mrr", 0)
    if mrr >= 0.7:
        parts.append(f"- 排序质量优秀：MRR = {mrr}，首条结果多为相关知识块。")
    else:
        parts.append(f"- 排序质量一般：MRR = {mrr}，相关结果平均排在第 {1/max(mrr, 0.01):.1f} 位。")

    crr = sp.get("correct_refusal_rate")
    if crr is not None:
        parts.append(f"- 拒答机制正确率 {crr}（知识库外查询 {'可靠拦截' if crr >= 0.8 else '存在漏拦，需调高语义门槛'}）。")
    frr = sp.get("false_refusal_rate")
    if frr is not None and frr > 0:
        parts.append(f"- ⚠ 存在误拒答（{frr}）：可答查询被错误拒绝，建议检查同义词覆盖。")

    r_hit = rp.get(f"hit@{k}", 0)
    r_crr = rp.get("correct_refusal_rate")
    parts.append(
        f"- 生产关键词路径（retrieve，Chief/专业 Agent 使用）：Hit@{k} = {r_hit}，"
        f"MRR = {rp.get('mrr', '-')}，正确拒答率 = {r_crr}；"
        f"低于自然语言路径属预期（关键词路径依赖精确术语，生产调用方固定传入领域词表）。"
    )

    lat = sp.get("latency_p95_ms")
    if lat:
        parts.append(f"- 时延 P95 = {lat}ms（{'满足实时编排要求' if lat < 500 else '偏高，建议索引常驻/预构建'}）。")

    n_fail = len(report["failures"])
    if n_fail:
        parts.append(f"- 失败用例 {n_fail} 条，明细见上表；优先修复『误拒答』与『应拒未拒』两类（安全相关）。")
    else:
        parts.append("- 无失败用例。")
    return "\n".join(parts)
