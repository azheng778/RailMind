"""RAG 混合检索引擎 —— 方案第 7 章的完整实现。

架构：
  知识库分层（KBChunk → KnowledgeBase → RagClient）
  混合检索（元数据过滤 + TF-IDF 语义召回 + 关键词/BM25 召回 + 重排序）
  版本生命周期（DRAFT → REVIEWING → ACTIVE → OBSOLETE）
  反馈记录与统计

向后兼容：
  RagClient.retrieve(asset_type, keywords, ...) 签名不变，旧调用方无需修改。
  新增 RagClient.search(query_text, ...) 支持自然语言检索。
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, vstack

NO_ANSWER = "当前知识库中未找到足够依据，建议人工查询相关规程。"

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

VERSION_DRAFT = "DRAFT"
VERSION_REVIEWING = "REVIEWING"
VERSION_ACTIVE = "ACTIVE"
VERSION_OBSOLETE = "OBSOLETE"
VERSION_REJECTED = "REJECTED"

_VALID_STATUSES = {VERSION_DRAFT, VERSION_REVIEWING, VERSION_ACTIVE, VERSION_OBSOLETE, VERSION_REJECTED}

# 默认混合检索权重
_DEFAULT_ALPHA = 0.55  # 语义得分权重 (1 - alpha 为关键词得分权重)
_DEFAULT_TOP_K = 3
_MAX_FEATURES = 12000


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class KBChunk:
    """知识库单块条目。"""

    document_id: str
    title: str
    chapter: str
    page: int
    text: str
    asset_type: str
    fault_type: List[str]
    version: str = "1.0"
    effective_status: str = VERSION_ACTIVE
    permission: str = "INTERNAL"
    tags: List[str] = field(default_factory=list)  # 额外标签,如 ["urgent", "限速"]
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = self.created_at
        if self.effective_status not in _VALID_STATUSES:
            raise ValueError(f"无效状态: {self.effective_status}")

    def citation(self, max_quote: int = 120) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "chapter": self.chapter,
            "page": self.page,
            "version": self.version,
            "status": self.effective_status,
            "quote": self.text[:max_quote],
            "tags": self.tags,
        }

    @property
    def searchable_text(self) -> str:
        """拼接所有可检索文本字段。"""
        return f"{self.title} {self.chapter} {self.text}"


@dataclass
class RagResult:
    """检索结果。"""
    answer_basis: List[Dict[str, Any]] = field(default_factory=list)
    refused: bool = False
    query_time_ms: float = 0.0

    @property
    def refused_text(self) -> str:
        return NO_ANSWER


# ---------------------------------------------------------------------------
# 查询扩展（中文铁路领域术语）
# ---------------------------------------------------------------------------

# 领域同义/相关词映射（用于查询扩展）
_QUERY_EXPANSION_MAP: Dict[str, List[str]] = {
    # 复合材料
    "冲击": ["冲击", "撞击", "碰撞"],
    "撞": ["冲击", "撞击", "碰撞"],
    "碰撞": ["冲击", "撞击"],
    "复合材料": ["复合材料", "复材", "碳纤维"],
    "能量": ["能量", "焦耳", "J"],
    "敲击检测": ["敲击检测", "锤击", "无损检测", "NDT"],
    "双点冲击": ["双点冲击", "双冲击", "多点冲击"],
    # 受电弓
    "受电弓": ["受电弓", "弓网", "集电弓"],
    "滑板": ["滑板", "碳滑板", "受电弓滑板"],
    "磨耗": ["磨耗", "磨损", "磨耗等级"],
    "电弧": ["电弧", "燃弧", "火花", "拉弧"],
    "弓网": ["弓网", "接触网", "接触线"],
    # 车底
    "检查门": ["检查门", "盖板", "包覆板"],
    "锁闭": ["锁闭", "锁紧", "闭合"],
    "把手": ["把手", "手柄", "门把手"],
    # 线路侧
    "异物": ["异物", "漂浮物", "侵入物"],
    "限界": ["限界", "界限", "安全距离"],
    "扣件": ["扣件", "弹条", "扣压"],
    "弹条": ["弹条", "扣件弹条", "弹性扣件"],
    # 车厢
    "烟雾": ["烟雾", "烟", "烟气", "火光"],
    "火灾": ["火灾", "火警", "火情"],
    "通道堵塞": ["通道堵塞", "通道阻塞", "通道拥堵"],
    "倒地": ["倒地", "摔倒", "卧倒"],
    # 通用
    "处置": ["处置", "处理", "应对", "措施"],
    "检修": ["检修", "维修", "维护", "保养"],
    "复核": ["复核", "复查", "确认", "检查"],
    "工单": ["工单", "检修工单", "告警工单"],
}


def _expand_query(text: str) -> str:
    """对中文查询做同义扩展（展开为一个更长更丰富的检索串）。"""
    expanded_terms = []
    for term, syns in _QUERY_EXPANSION_MAP.items():
        if term in text:
            expanded_terms.extend(syns)
    if expanded_terms:
        text = text + " " + " ".join(expanded_terms)
    return text


# ---------------------------------------------------------------------------
# KnowledgeBase —— 知识库（向量索引 + 版本管理）
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """知识库：管理 KBChunk 集合 + TF-IDF 向量索引 + 混合检索。"""

    def __init__(self, chunks: Optional[List[KBChunk]] = None, top_k: int = _DEFAULT_TOP_K):
        self._chunks: List[KBChunk] = list(chunks) if chunks else []
        self._vectorizer: Any = None  # TfidfVectorizer
        self._tfidf_matrix: Optional[csr_matrix] = None
        self._index_built: bool = False
        self._index_dirty: bool = True
        self._query_log: List[Dict[str, Any]] = []
        self._feedback_log: List[Dict[str, Any]] = []
        self.top_k = top_k
        self.alpha = _DEFAULT_ALPHA  # 语义权重

    # ---- 索引管理 ----

    def _ensure_index(self) -> None:
        """延迟重建 TF-IDF 索引。"""
        if not self._index_dirty and self._index_built:
            return
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """全量重建 TF-IDF 向量索引。"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [c.searchable_text for c in self._chunks]
        if not texts:
            self._vectorizer = None
            self._tfidf_matrix = None
            self._index_built = True
            self._index_dirty = False
            return

        # 使用字符 n-gram（对中文友好） + 子线性 TF（类 BM25）
        self._vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 4),
            sublinear_tf=True,
            max_features=_MAX_FEATURES,
            dtype=np.float32,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        self._index_built = True
        self._index_dirty = False

    def _invalidate_index(self) -> None:
        """标记索引为脏（增删改后调用）。"""
        self._index_dirty = True

    # ---- CRUD ----

    @property
    def chunks(self) -> List[KBChunk]:
        return self._chunks

    def add(self, chunk: KBChunk) -> None:
        """添加一个知识块。"""
        self._chunks.append(chunk)
        self._invalidate_index()

    def add_all(self, chunks: List[KBChunk]) -> None:
        """批量添加知识块。"""
        self._chunks.extend(chunks)
        self._invalidate_index()

    def remove(self, document_id: str) -> int:
        """按 document_id 删除，返回删除数量。"""
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.document_id != document_id]
        removed = before - len(self._chunks)
        if removed:
            self._invalidate_index()
        return removed

    def update(self, document_id: str, **kwargs: Any) -> bool:
        """更新指定文档的字段。返回是否找到并更新。"""
        for c in self._chunks:
            if c.document_id == document_id:
                for k, v in kwargs.items():
                    if hasattr(c, k):
                        setattr(c, k, v)
                c.updated_at = time.time()
                self._invalidate_index()
                return True
        return False

    def get(self, document_id: str) -> Optional[KBChunk]:
        """按 document_id 获取条目。"""
        for c in self._chunks:
            if c.document_id == document_id:
                return c
        return None

    def filter(
        self,
        asset_type: Optional[str] = None,
        fault_type: Optional[str] = None,
        status: str = VERSION_ACTIVE,
        permission: str = "INTERNAL",
    ) -> List[int]:
        """返回满足过滤条件的 chunk 索引列表。"""
        indices: List[int] = []
        for i, c in enumerate(self._chunks):
            if c.effective_status != status:
                continue
            if asset_type and c.asset_type != asset_type:
                continue
            if fault_type and fault_type not in c.fault_type:
                continue
            indices.append(i)
        return indices

    # ---- 语义检索（TF-IDF 余弦相似度） ----

    def _semantic_scores(self, query: str, indices: List[int]) -> np.ndarray:
        """计算查询与指定索引块的余弦相似度得分。"""
        self._ensure_index()
        if self._vectorizer is None or self._tfidf_matrix is None or not indices:
            return np.array([])
        q_vec = self._vectorizer.transform([query])
        # 计算余弦相似度（对归一化 TF-IDF 向量 = 点积）
        sub_matrix = self._tfidf_matrix[indices]
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(q_vec, sub_matrix).flatten()
        return sims

    # ---- 关键词得分（TF-IDF 词级匹配 + 重叠加权） ----

    def _keyword_scores(
        self, keywords: List[str], indices: List[int]
    ) -> np.ndarray:
        """对每个关键词做重叠计数，并赋予 IDF 加权。"""
        if not indices:
            return np.array([])
        scores = np.zeros(len(indices), dtype=np.float32)
        for kw in keywords:
            if not kw:
                continue
            kw_lower = kw.lower()
            for j, idx in enumerate(indices):
                chunk = self._chunks[idx]
                text = chunk.searchable_text.lower()
                count = text.count(kw_lower)
                if count > 0:
                    # IDF 加权：稀有词得分更高
                    _idf_weight = 1.0
                    if self._vectorizer is not None and kw in self._vectorizer.vocabulary_:
                        # 从 vectorizer 的 IDF 中提取权重
                        pass  # 保持简单，用 count
                    # 标题/章节命中加权
                    if kw_lower in chunk.title.lower():
                        count += 2
                    if kw_lower in chunk.chapter.lower():
                        count += 1
                    scores[j] += count
        # 归一化
        max_s = scores.max()
        if max_s > 0:
            scores = scores / max_s
        return scores

    # ---- BM25 风格得分 ----

    def _bm25_scores(self, query: str, indices: List[int]) -> np.ndarray:
        """简化的 BM25 得分（使用 TF-IDF 向量的点积近似）。"""
        self._ensure_index()
        if self._vectorizer is None or self._tfidf_matrix is None or not indices:
            return np.array([])
        q_vec = self._vectorizer.transform([query])
        sub_matrix = self._tfidf_matrix[indices]
        # BM25 近似：对每个查询词取 TF-IDF 加权匹配
        # 更精确的做法是逐词计算 BM25，但这里用 TF-IDF 点积作为近似
        dot = (q_vec @ sub_matrix.T).toarray().flatten()
        if dot.max() > 0:
            dot = dot / dot.max()
        return dot

    # ---- 混合检索 ----

    def hybrid_search(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        asset_type: Optional[str] = None,
        fault_type: Optional[str] = None,
        status: str = VERSION_ACTIVE,
        permission: str = "INTERNAL",
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Tuple[float, KBChunk]]:
        """混合检索：元数据过滤 + 语义召回 + 关键词/B̂M25 召回 + 重排序。

        Returns:
            List[(combined_score, chunk)] 按得分降序排列。
        """
        top_k = top_k or self.top_k
        alpha = self.alpha if alpha is None else alpha

        # 1. 元数据过滤
        indices = self.filter(
            asset_type=asset_type,
            fault_type=fault_type,
            status=status,
            permission=permission,
        )
        if not indices:
            return []

        # 2. 查询扩展
        expanded_query = _expand_query(query)

        # 3. 语义得分（TF-IDF 余弦相似度）
        semantic = self._semantic_scores(expanded_query, indices)

        # 4. BM25/关键词得分
        if keywords:
            keyword_sc = self._keyword_scores(keywords, indices)
        else:
            # 从查询中提取关键词（按空格/逗号分割，或直接用查询做 BM25）
            keyword_sc = self._bm25_scores(expanded_query, indices)

        # 5. 混合加权 + 拒答门槛（标定数据见 docs/RAG检索效果评测报告.md）
        # 语义得分为最终拒答依据：
        #   - 关键词路径：关键词全零 → 0.15；有匹配 → 0.08（防单个泛化词如"处置"击穿拒答，
        #     标定：真实用例最低 0.128，跨域误配 0.04）
        #   - 自然语言路径：BM25 字符 n-gram 恒非零不可作证据 → 绝对下限 0.06
        #     （标定：拒答查询 ≤0.05，真实查询 ≥0.07）
        max_kw = float(keyword_sc.max()) if len(keyword_sc) > 0 else 0.0
        kw_all_zero = max_kw < 1e-9
        if keywords:
            sem_floor = 0.15 if kw_all_zero else 0.08
        else:
            sem_floor = 0.15 if kw_all_zero else 0.06

        scored: List[Tuple[float, int]] = []
        for j, idx in enumerate(indices):
            sem = float(semantic[j]) if len(semantic) > j else 0.0
            kw = float(keyword_sc[j]) if len(keyword_sc) > j else 0.0
            if sem < sem_floor:
                continue  # 语义不相关一律拒答（防泛化词/字符偶然命中）
            combined = alpha * sem + (1.0 - alpha) * kw
            if combined > 1e-9:
                scored.append((combined, idx))

        # 6. 排序取 top_k
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [(score, self._chunks[idx]) for score, idx in scored[:top_k]]

    # ---- 持久化 ----

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 序列化的字典。"""
        return {
            "chunks": [
                {
                    "document_id": c.document_id,
                    "title": c.title,
                    "chapter": c.chapter,
                    "page": c.page,
                    "text": c.text,
                    "asset_type": c.asset_type,
                    "fault_type": c.fault_type,
                    "version": c.version,
                    "effective_status": c.effective_status,
                    "permission": c.permission,
                    "tags": c.tags,
                    "created_at": c.created_at,
                    "updated_at": c.updated_at,
                }
                for c in self._chunks
            ],
            "query_log": self._query_log[-500:],
            "feedback_log": self._feedback_log[-500:],
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典恢复知识库。"""
        self._chunks = [KBChunk(**c) for c in data.get("chunks", [])]
        self._query_log = data.get("query_log", [])
        self._feedback_log = data.get("feedback_log", [])
        self._invalidate_index()

    def save(self, path: str) -> None:
        """保存到 JSON 文件。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "KnowledgeBase":
        """从 JSON 文件加载。"""
        kb = cls()
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                kb.from_dict(json.load(f))
        return kb

    # ---- 统计 ----

    def stats(self) -> Dict[str, Any]:
        """知识库统计。"""
        self._ensure_index()
        active = [c for c in self._chunks if c.effective_status == VERSION_ACTIVE]
        docs = set(c.document_id for c in self._chunks)
        types = set(c.asset_type for c in self._chunks)
        return {
            "documents": len(docs),
            "total_chunks": len(self._chunks),
            "active_chunks": len(active),
            "asset_types": sorted(types),
            "queries_logged": len(self._query_log),
            "feedback_count": len(self._feedback_log),
            "index_built": self._index_built,
            "vocab_size": len(self._vectorizer.vocabulary_) if self._vectorizer else 0,
        }

    # ---- 版本管理 ----

    def set_status(self, document_id: str, new_status: str) -> bool:
        """更新文档生命周期状态。"""
        if new_status not in _VALID_STATUSES:
            raise ValueError(f"无效状态: {new_status}")
        return self.update(document_id, effective_status=new_status)

    def list_by_status(self, status: str) -> List[KBChunk]:
        """按状态列出条目。"""
        return [c for c in self._chunks if c.effective_status == status]

    # ---- 反馈 ----

    def record_feedback(
        self,
        query: str,
        document_id: str,
        rating: int,
        comment: str = "",
    ) -> None:
        """记录用户对检索结果的反馈。rating: 1-5。"""
        self._feedback_log.append({
            "ts": time.time(),
            "query": query,
            "document_id": document_id,
            "rating": max(1, min(5, rating)),
            "comment": comment,
        })


# ---------------------------------------------------------------------------
# 内置默认知识库
# ---------------------------------------------------------------------------


def _build_default_kb() -> List[KBChunk]:
    """构建覆盖全部 5 个领域的丰富默认知识条目。"""
    now = time.time()
    return [
        # ========== DOC-DEMO-001: 复合材料结构 ==========
        KBChunk(
            document_id="DOC-DEMO-001",
            title="复合材料结构冲击检测作业指引（试行版）",
            chapter="冲击能量分级与处置",
            page=4,
            text="复合材料结构受到冲击后，应按冲击能量分级处置：能量低于0.35J视为轻微冲击，记录观察；"
                 "0.35J至0.70J之间为重点关注冲击，应在下一停靠站进行目视复核；"
                 "0.70J及以上为显著冲击，须立即安排人工敲击检测或无损检测复核，并评估是否限制运行。",
            asset_type="composite_structure",
            fault_type=["impact", "energy_low", "energy_medium", "energy_high"],
            tags=["核心规程"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-001",
            title="复合材料结构冲击检测作业指引（试行版）",
            chapter="双点冲击判读",
            page=6,
            text="当监测系统在同一时刻窗内定位到两处冲击时，应分别核对两处坐标与能量等级；"
                 "两点能量均超过0.50J时按显著冲击处置流程处理，并检查两处之间的结构连通区域；"
                 "若其中一点能量超过0.70J，即使另一点能量较低也升级为显著冲击。",
            asset_type="composite_structure",
            fault_type=["impact", "double_impact", "energy_high"],
            tags=["多冲击", "特殊判读"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-001",
            title="复合材料结构冲击检测作业指引（试行版）",
            chapter="传感器状态自检",
            page=2,
            text="冲击监测系统在每次发车前应完成传感器自检：各通道噪声基线正常、灵敏度在标定范围内；"
                 "任一通道自检异常时系统应降级为观察模式，并通知检修人员检查传感器线路与耦合状态。",
            asset_type="composite_structure",
            fault_type=["sensor_check", "degraded"],
            tags=["系统自检"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-001",
            title="复合材料结构冲击检测作业指引（试行版）",
            chapter="历史冲击趋势分析",
            page=10,
            text="同一监测区域1小时内累计冲击频次超过3次时，即使单次能量均低于0.35J，"
                 "也应升级为重点关注并分析冲击来源；同一区域24小时内累计超过10次时，"
                 "应安排结构专项检查。",
            asset_type="composite_structure",
            fault_type=["impact", "high_frequency", "trend"],
            tags=["趋势分析"],
            created_at=now,
        ),
        # ========== DOC-DEMO-002: 受电弓 ==========
        KBChunk(
            document_id="DOC-DEMO-002",
            title="受电弓检修规程（试行版）",
            chapter="滑板磨耗分级",
            page=12,
            text="受电弓滑板磨耗等级分为正常(NORMAL)、轻度磨耗(LIGHT)、明显磨耗(HEAVY)三级；"
                 "轻度磨耗时记录并加密观测，明显磨耗时应在最近具备检修条件的停靠站安排更换；"
                 "若同时出现电弧频次升高，应提升处置优先级并在下次入库检修前加密观测频次。",
            asset_type="pantograph",
            fault_type=["strip_wear", "wear_light", "wear_heavy"],
            tags=["核心规程", "磨耗"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-002",
            title="受电弓检修规程（试行版）",
            chapter="电弧异常处置",
            page=15,
            text="电弧事件持续时长超过2秒或单位时间频次超过阈值时，应复核弓网接触状态，"
                 "并结合滑板磨耗等级综合判断是否需要降弓或限速运行；"
                 "累计燃弧时长超过20毫秒或事件评分达到显著水平时，应复核弓网接触状态与滑板接触面。",
            asset_type="pantograph",
            fault_type=["arc", "arc_severe"],
            tags=["核心规程", "电弧"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-002",
            title="受电弓检修规程（试行版）",
            chapter="电弧信号判据",
            page=18,
            text="弓网电弧分析应统计单位时间内电弧事件次数与累计燃弧时长："
                 "累计燃弧时长超过20毫秒或事件评分达到显著水平时，应复核弓网接触状态与滑板接触面；"
                 "燃弧事件伴随电流平顶特征明显时，优先检查滑板与接触线贴合状态。",
            asset_type="pantograph",
            fault_type=["arc", "arc_signal", "arc_severe"],
            tags=["电弧", "信号分析"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-002",
            title="受电弓检修规程（试行版）",
            chapter="滑板裂纹检查",
            page=14,
            text="滑板除磨耗检查外还应检查是否存在裂纹、掉块或偏磨："
                 "裂纹长度超过15mm或掉块面积超过50mm²时应立即更换滑板；"
                 "偏磨导致两侧厚度差超过3mm时应在下次入库前调整弓头平衡。",
            asset_type="pantograph",
            fault_type=["strip_wear", "crack", "chipping", "uneven_wear"],
            tags=["裂纹", "偏磨"],
            created_at=now,
        ),
        # ========== DOC-DEMO-003: 车底检查门 ==========
        KBChunk(
            document_id="DOC-DEMO-003",
            title="车底检查门检修说明（试行版）",
            chapter="检查门未闭合处置",
            page=3,
            text="车底检查门未完全闭合时，进站后应由地勤人员现场复核并重新锁闭；"
                 "同一检查门重复出现未闭合告警时，应检查锁机构磨损情况并生成检修工单。",
            asset_type="underbody",
            fault_type=["door_not_closed", "lock_wear"],
            tags=["核心规程"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-003",
            title="车底检查门检修说明（试行版）",
            chapter="门把手角度判据",
            page=5,
            text="检查门把手与门板垂直方向偏差角超过75度视为把手异常开启，须现场确认锁机构是否失效；"
                 "角度在50至75度之间为重点关注，应在下一停靠站目视复核；"
                 "盖板中心区域边缘检测异常时，应检查包覆板是否缺失或错位。",
            asset_type="underbody",
            fault_type=["door_handle_angle", "board_abnormal", "door_handle_warning"],
            tags=["核心规程"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-003",
            title="车底检查门检修说明（试行版）",
            chapter="锁机构维护周期",
            page=7,
            text="车底检查门锁机构应每30天进行一次润滑维护，每90天进行一次功能测试；"
                 "运行里程超过20万公里或使用超过1年应更换锁机构弹簧组件。",
            asset_type="underbody",
            fault_type=["maintenance", "lock_maintenance"],
            tags=["维护周期"],
            created_at=now,
        ),
        # ========== DOC-DEMO-004: 线路侧异物 ==========
        KBChunk(
            document_id="DOC-DEMO-004",
            title="线路侧异物处置办法（试行版）",
            chapter="异物分级与处置",
            page=7,
            text="线路侧异物按与限界的距离和尺寸分级处置：面积占比超过限界阈值5%的漂浮物或大型异物，"
                 "应立即通知工务与调度，评估是否拦停后续列车；"
                 "面积占比1%~5%的中型异物应安排工务现场清理复核；"
                 "面积占比低于1%的小型异物记录并持续观察；"
                 "未检出异物时按正常记录处理，影像点持续观察。",
            asset_type="lineside",
            fault_type=["foreign_object", "limit_boundary", "fod_large", "fod_medium", "fod_small"],
            tags=["核心规程"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-004",
            title="线路侧异物处置办法（试行版）",
            chapter="影像点复核要求",
            page=9,
            text="同一影像点1小时内重复出现异物告警的，应在清理完成后核查异物来源（风揭垃圾、货物散落等），"
                 "并对相邻影像点开展一次补充巡检。",
            asset_type="lineside",
            fault_type=["foreign_object", "repeat_alarm", "fod_recurring"],
            tags=["影像点"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-004",
            title="线路侧异物处置办法（试行版）",
            chapter="恶劣天气加强巡检",
            page=11,
            text="大风（风速≥15m/s）、暴雨、暴雪等恶劣天气条件下，"
                 "应自动加密线路侧影像巡检频次至正常间隔的2倍；"
                 "台风预警生效期间应安排一次全线影像巡检。",
            asset_type="lineside",
            fault_type=["foreign_object", "extreme_weather", "fod_weather"],
            tags=["恶劣天气", "巡检策略"],
            created_at=now,
        ),
        # ========== DOC-DEMO-005: 钢轨扣件 ==========
        KBChunk(
            document_id="DOC-DEMO-005",
            title="钢轨扣件检修规程（试行版）",
            chapter="扣件缺陷分级处置",
            page=11,
            text="扣件弹条断裂、缺失时扣压功能失效，判定为显著缺陷(HIGH)，应立即通知工务安排更换，"
                 "并评估该区段是否限速通过；弹条移位、翻转、变形为重点缺陷(WARNING)，应列入最近天窗点检计划；"
                 "弹条轻微松动为关注级(OBSERVE)，记录并在下次天窗点复核；"
                 "未检出缺陷的扣件按正常状态记录。",
            asset_type="lineside",
            fault_type=["fastener_defect", "broken", "missing", "displaced", "deformed", "loose"],
            tags=["核心规程", "扣件"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-005",
            title="钢轨扣件检修规程（试行版）",
            chapter="复检与验收",
            page=14,
            text="扣件更换完成后，应在下一巡检周期对更换位置进行图像复检，确认弹条状态恢复正常；"
                 "同一里程重复出现同类缺陷时，应检查轨枕与垫板状态。",
            asset_type="lineside",
            fault_type=["fastener_defect", "recheck", "recurring"],
            tags=["扣件", "复检"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-005",
            title="钢轨扣件检修规程（试行版）",
            chapter="扣件扭矩标准",
            page=8,
            text="弹条扣件安装扭矩标准值为150~180N·m；巡检中发现弹条明显松动时应使用扭矩扳手复核，"
                 "低于120N·m应按松动缺陷记录并纳入天窗点紧固计划。",
            asset_type="lineside",
            fault_type=["fastener_defect", "torque", "loose"],
            tags=["扭矩标准"],
            created_at=now,
        ),
        # ========== DOC-DEMO-006: 车厢巡检 ==========
        KBChunk(
            document_id="DOC-DEMO-006",
            title="车内异常巡检处置指引（试行版）",
            chapter="车内事件分级与处置",
            page=5,
            text="车内巡检发现的异常按方案分级处置："
                 "疑似烟雾或火光单窗口高置信即升级(HIGH)，立即通知乘务员现场确认并按火灾预案处置；"
                 "人员持续倒地、通道持续堵塞需连续两个窗口结论一致方可升级告警(WARNING)；"
                 "异常聚集与疑似剧烈动作仅记录观察(OBSERVE)并由乘务员复核，不得输出确定性结论。",
            asset_type="cabin",
            fault_type=["cabin_patrol", "smoke", "fire", "aisle_blocked", "person_down", "crowd"],
            tags=["核心规程", "车厢"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-006",
            title="车内异常巡检处置指引（试行版）",
            chapter="乘务员复核流程",
            page=8,
            text="车厢巡检VLM窗口结论置信度不足或相邻窗口不一致时，应转乘务员人工复核原始视频；"
                 "复核结果回填平台后关闭或升级对应事件，形成处置闭环。",
            asset_type="cabin",
            fault_type=["cabin_patrol", "human_review", "low_confidence"],
            tags=["复核流程"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-006",
            title="车内异常巡检处置指引（试行版）",
            chapter="烟雾与火灾应急预案",
            page=10,
            text="车厢内确认烟雾或火光后，系统应：1)立即向乘务员终端推送告警及视频截图；"
                 "2)通知司机评估就近停靠或疏散；"
                 "3)开启该车厢通风排烟系统（如可用）；"
                 "4)记录事件时间线用于事后分析。",
            asset_type="cabin",
            fault_type=["cabin_patrol", "smoke", "fire", "emergency"],
            tags=["应急预案"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-006",
            title="车内异常巡检处置指引（试行版）",
            chapter="通道堵塞评估标准",
            page=7,
            text="通道堵塞指行李、手推车或其他物品占用通道超过30%宽度且持续时间超过30秒；"
                 "人员倒地指同一位置有人员处于躺卧姿态超过15秒；"
                 "两类事件均需连续两个巡检窗口确认方可升级告警。",
            asset_type="cabin",
            fault_type=["cabin_patrol", "aisle_blocked", "person_down"],
            tags=["评估标准"],
            created_at=now,
        ),
        # ========== 补充跨领域安全通用条目 ==========
        KBChunk(
            document_id="DOC-DEMO-007",
            title="列车运行安全通用规程（试行版）",
            chapter="告警升级与联动处置",
            page=3,
            text="任何领域产生HIGH等级告警时，系统应自动通知司机与调度中心，"
                 "并在30秒内生成处置工单草稿；"
                 "WARNING等级告警应在下一停靠站完成人工复核；"
                 "OBSERVE等级告警记录事件日志，纳入日常数据分析。",
            asset_type="composite_structure",
            fault_type=["general", "alert_escalation"],
            tags=["通用", "安全"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-007",
            title="列车运行安全通用规程（试行版）",
            chapter="告警升级与联动处置",
            page=3,
            text="任何领域产生HIGH等级告警时，系统应自动通知司机与调度中心，"
                 "并在30秒内生成处置工单草稿；"
                 "WARNING等级告警应在下一停靠站完成人工复核；"
                 "OBSERVE等级告警记录事件日志，纳入日常数据分析。",
            asset_type="pantograph",
            fault_type=["general", "alert_escalation"],
            tags=["通用", "安全"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-007",
            title="列车运行安全通用规程（试行版）",
            chapter="告警升级与联动处置",
            page=3,
            text="任何领域产生HIGH等级告警时，系统应自动通知司机与调度中心，"
                 "并在30秒内生成处置工单草稿；"
                 "WARNING等级告警应在下一停靠站完成人工复核；"
                 "OBSERVE等级告警记录事件日志，纳入日常数据分析。",
            asset_type="underbody",
            fault_type=["general", "alert_escalation"],
            tags=["通用", "安全"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-007",
            title="列车运行安全通用规程（试行版）",
            chapter="告警升级与联动处置",
            page=3,
            text="任何领域产生HIGH等级告警时，系统应自动通知司机与调度中心，"
                 "并在30秒内生成处置工单草稿；"
                 "WARNING等级告警应在下一停靠站完成人工复核；"
                 "OBSERVE等级告警记录事件日志，纳入日常数据分析。",
            asset_type="lineside",
            fault_type=["general", "alert_escalation"],
            tags=["通用", "安全"],
            created_at=now,
        ),
        KBChunk(
            document_id="DOC-DEMO-007",
            title="列车运行安全通用规程（试行版）",
            chapter="告警升级与联动处置",
            page=3,
            text="任何领域产生HIGH等级告警时，系统应自动通知司机与调度中心，"
                 "并在30秒内生成处置工单草稿；"
                 "WARNING等级告警应在下一停靠站完成人工复核；"
                 "OBSERVE等级告警记录事件日志，纳入日常数据分析。",
            asset_type="cabin",
            fault_type=["general", "alert_escalation"],
            tags=["通用", "安全"],
            created_at=now,
        ),
    ]


# ---------------------------------------------------------------------------
# RagClient —— 对外 Facade（保持向后兼容）
# ---------------------------------------------------------------------------


class RagClient:
    """RAG 混合检索客户端。

    使用方式（向后兼容）:
        client = RagClient()
        result = client.retrieve(asset_type="pantograph", keywords=["电弧", "处置"])
        if not result.refused:
            for citation in result.answer_basis:
                print(citation)

    新增能力:
        result = client.search("受电弓滑板磨耗严重应该怎么处理", asset_type="pantograph")
        client.record_feedback(query="...", document_id="...", rating=5)
        client.add_chunk(new_chunk)
        stats = client.stats()
    """

    def __init__(
        self,
        kb: Optional[List[KBChunk]] = None,
        top_k: int = _DEFAULT_TOP_K,
        alpha: float = _DEFAULT_ALPHA,
        persist_path: Optional[str] = None,
    ):
        """初始化。

        Args:
            kb: 初始知识块列表。None 则使用内置默认知识库。
            top_k: 默认返回 top K 条结果。
            alpha: 语义检索权重（0=纯关键词, 1=纯语义）。
            persist_path: 持久化文件路径。提供后自动加载已有数据。
        """
        self._persist_path = persist_path
        if persist_path and os.path.isfile(persist_path):
            self.kb = KnowledgeBase.load(persist_path)
            if kb:
                self.kb.add_all(kb)
        else:
            self.kb = KnowledgeBase(chunks=kb or _build_default_kb())
        self.kb.top_k = top_k
        self.kb.alpha = alpha
        self.query_log: List[Dict[str, Any]] = self.kb._query_log  # 向后兼容

    # ---- 核心检索（向后兼容） ----

    def retrieve(
        self,
        asset_type: str,
        keywords: List[str],
        fault_type: Optional[str] = None,
        permission: str = "INTERNAL",
    ) -> RagResult:
        """向后兼容的检索接口：asset_type 过滤 + 关键词混合检索。

        Args:
            asset_type: 资产类型（如 "pantograph"）。
            keywords: 关键词列表（如 ["电弧", "处置"]）。
            fault_type: 可选的故障类型过滤。
            permission: 权限等级。

        Returns:
            RagResult（refused=True 表示无结果）。
        """
        t0 = time.time()
        query = " ".join(keywords)
        scored = self.kb.hybrid_search(
            query=query,
            keywords=keywords,
            asset_type=asset_type,
            fault_type=fault_type,
            permission=permission,
        )
        elapsed = (time.time() - t0) * 1000

        self.query_log.append({
            "ts": time.time(),
            "method": "retrieve",
            "asset_type": asset_type,
            "keywords": keywords,
            "fault_type": fault_type,
            "hits": len(scored),
            "time_ms": round(elapsed, 1),
        })

        if not scored:
            return RagResult(answer_basis=[], refused=True, query_time_ms=elapsed)
        citations = [chunk.citation() for _, chunk in scored]
        return RagResult(answer_basis=citations, refused=False, query_time_ms=elapsed)

    # ---- 自然语言检索（新增） ----

    def search(
        self,
        query_text: str,
        asset_type: Optional[str] = None,
        fault_type: Optional[str] = None,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> RagResult:
        """自然语言查询检索。

        Args:
            query_text: 自然语言查询（如 "受电弓出现电弧应该怎么处理"）。
            asset_type: 可选的资产类型过滤。
            fault_type: 可选的故障类型过滤。
            top_k: 返回条数。
            alpha: 语义权重，None 使用默认值。

        Returns:
            RagResult。
        """
        t0 = time.time()
        scored = self.kb.hybrid_search(
            query=query_text,
            keywords=None,
            asset_type=asset_type,
            fault_type=fault_type,
            top_k=top_k,
            alpha=alpha,
        )
        elapsed = (time.time() - t0) * 1000

        self.query_log.append({
            "ts": time.time(),
            "method": "search",
            "query": query_text[:100],
            "asset_type": asset_type,
            "hits": len(scored),
            "time_ms": round(elapsed, 1),
        })

        if not scored:
            return RagResult(answer_basis=[], refused=True, query_time_ms=elapsed)
        citations = [chunk.citation() for _, chunk in scored]
        return RagResult(answer_basis=citations, refused=False, query_time_ms=elapsed)

    # ---- 知识库管理 ----

    def add_chunk(self, chunk: KBChunk) -> None:
        """添加一条知识。"""
        self.kb.add(chunk)
        self._maybe_persist()

    def add_chunks(self, chunks: List[KBChunk]) -> None:
        """批量添加知识。"""
        self.kb.add_all(chunks)
        self._maybe_persist()

    def remove_chunk(self, document_id: str) -> int:
        """删除知识条目。返回删除数量。"""
        n = self.kb.remove(document_id)
        if n:
            self._maybe_persist()
        return n

    def update_chunk(self, document_id: str, **kwargs: Any) -> bool:
        """更新知识条目字段。"""
        ok = self.kb.update(document_id, **kwargs)
        if ok:
            self._maybe_persist()
        return ok

    def get_chunk(self, document_id: str) -> Optional[KBChunk]:
        """获取单条知识详情。"""
        return self.kb.get(document_id)

    def list_chunks(self, status: Optional[str] = None) -> List[KBChunk]:
        """列出知识条目，可按状态过滤。"""
        if status:
            return self.kb.list_by_status(status)
        return self.kb.chunks

    # ---- 版本管理 ----

    def set_chunk_status(self, document_id: str, new_status: str) -> bool:
        """更新文档生命周期状态。"""
        ok = self.kb.set_status(document_id, new_status)
        if ok:
            self._maybe_persist()
        return ok

    # ---- 反馈 ----

    def record_feedback(
        self,
        query: str,
        document_id: str,
        rating: int,
        comment: str = "",
    ) -> None:
        """记录用户反馈。"""
        self.kb.record_feedback(query, document_id, rating, comment)
        self._maybe_persist()

    # ---- 统计 ----

    def stats(self) -> Dict[str, Any]:
        """知识库统计。"""
        return self.kb.stats()

    def rebuild_index(self) -> None:
        """强制重建向量索引。"""
        self.kb._rebuild_index()

    # ---- 持久化 ----

    def _maybe_persist(self) -> None:
        if self._persist_path:
            self.kb.save(self._persist_path)

    def persist(self, path: Optional[str] = None) -> None:
        """显式保存知识库到文件。"""
        self.kb.save(path or self._persist_path or "data/knowledge_base.json")

    # ---- 查询日志 ----

    def recent_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """最近查询记录。"""
        return self.query_log[-limit:]

    def recent_feedback(self, limit: int = 20) -> List[Dict[str, Any]]:
        """最近反馈记录。"""
        return self.kb._feedback_log[-limit:]