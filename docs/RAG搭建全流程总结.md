# RailMind 列车检修 RAG 知识库搭建全流程总结

## 1. 概述

RailMind 的 RAG（Retrieval-Augmented Generation）知识库为列车智能运维平台提供**可追溯、可拒答、可反馈、可评测**的检修知识检索能力。RAG 覆盖 5 个专业领域（复合材料结构、受电弓、车底检查门、线路侧、车厢巡检），集成到 Chief-Agent 编排链路和专业 Agent 的处置结论中，确保每一条处置建议都有规程依据可查。

> **RAG 只负责知识检索、解释和建议，不负责直接判定安全风险等级。** 风险等级完全由确定性 RiskEngine 计算。

---

## 2. 搭建全流程

### 2.1 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG 混合检索引擎                          │
│                                                             │
│  用户查询 / 诊断事件                                          │
│       ↓                                                     │
│  ① 元数据过滤 (asset_type / fault_type / status)              │
│       ↓                                                     │
│  ② 查询扩展 (领域同义/相关词映射)                               │
│       ↓                                                     │
│  ③ 语义召回 (TF-IDF 字符 n-gram 余弦相似度)                    │
│       ↓                                                     │
│  ④ 关键词/BM25 召回 (TF-IDF 加权词级匹配)                      │
│       ↓                                                     │
│  ⑤ 混合加权排序 (α·语义 + (1-α)·关键词)                       │
│       ↓                                                     │
│  ⑥ 拒答门槛（语义下限标定，见评测报告）                          │
│       ↓                                                     │
│  ⑦ Top-K 返回 + 引用元数据                                    │
│       ↓                                                     │
│  Agent 生成带引用依据的处置建议                                 │
└─────────────────────────────────────────────────────────────┘
```

**核心设计原则：**
- **混合检索**：语义相似度 + 关键词精确匹配，避免纯向量检索丢失专业术语
- **元数据过滤**：检索前按 asset_type / fault_type / 状态过滤，减少无关噪音
- **查询扩展**：中文铁路领域同义词映射（如"电弧"→"燃弧,火花,拉弧"）
- **拒答机制**：无可靠依据时必须拒绝，不得虚构规程条款（有评测验证：正确拒答率 1.0）
- **引用可追溯**：每条结果包含文档 ID、标题、章节、页码、版本

### 2.2 文件结构

```
railmind/core/
├── rag.py          # RAG 混合检索引擎（核心）
│   ├── KBChunk     # 知识块数据类
│   ├── KnowledgeBase # 知识库管理 + TF-IDF 向量索引 + 混合检索
│   ├── RagClient   # 对外 Facade（向后兼容）
│   ├── RagResult   # 检索结果
│   └── _build_default_kb() # 内置默认知识库（26条，7文档，5领域）
│
├── kb_builder.py   # 文档处理与知识库构建管线
│   ├── DocumentProcessor # 解析→切分→元数据标注→KBChunk列表
│   ├── parse_text()      # 支持 MD/TXT/HTML 格式
│   ├── chunk_sections()  # 按章节结构化切分
│   └── import_directory()# 批量导入目录文档
│
├── rag_eval.py     # 检索效果评测框架（36条黄金数据集 + IR 指标）
│
└── ... (chief.py, chat.py 等调用方不变)

railmind/apps/api.py  # 7 个 KB 相关 API 端点
scripts/eval_rag.py   # 评测 CLI 入口
```

### 2.3 数据流

```
外部文档 (MD/TXT/HTML)
    │
    ▼
DocumentProcessor.process_file()
    │ 解析 → 分块 → 标注元数据
    ▼
List[KBChunk]
    │
    ▼
RagClient.add_chunks()
    │ 索引标记脏 → 下次检索时自动重建 TF-IDF
    ▼
KnowledgeBase._rebuild_index()
    │ sklearn TfidfVectorizer (char n-gram 1-4)
    ▼
TF-IDF 稀疏矩阵 (csr_matrix)
    │
    ▼
Chief-Agent / Specialist-Agent / Chat-Agent 调用
    │ retrieve(asset_type, keywords) 或 search(query_text)
    ▼
知识引用嵌入处置结论 / 工单草稿
```

### 2.4 检索流程详解

**接口1：** `RagClient.retrieve(asset_type, keywords, fault_type, permission)`（向后兼容，生产 Agent 路径）

1. 元数据过滤：`asset_type` + `fault_type`(可选) + `effective_status=ACTIVE`
2. 查询扩展：将 keywords join 后做同义词扩展
3. 语义得分：TF-IDF 字符 n-gram 向量化后计算余弦相似度
4. 关键词得分：计算每个关键词在标题/章节/文本中的加权出现次数（标题+2，章节+1）
5. 混合排序：`score = α·semantic + (1-α)·keyword`（默认 α=0.55）
6. 拒答门槛：语义得分 < 0.08 一律拒答（防泛化词击穿）
7. Top-K 返回（默认 top_k=3，可配置）

**接口2：** `RagClient.search(query_text, asset_type, top_k, alpha)`（自然语言检索，Chat 路径）

1. 元数据过滤（可选）
2. 查询扩展
3. 语义得分 + BM25 近似得分
4. 拒答门槛：语义得分 < 0.06 一律拒答（BM25 字符 n-gram 恒非零，不可作匹配证据）
5. 混合排序 + Top-K

**拒答条件：**
- 元数据过滤后无匹配块
- 语义得分低于门槛（关键词路径 0.08 / 自然语言路径 0.06，标定依据见评测报告）
- 返回：`RagResult(refused=True, answer_basis=[], refused_text=NO_ANSWER)`

### 2.5 知识版本生命周期

```
DRAFT → REVIEWING → ACTIVE → OBSOLETE
                          ↓
                       REJECTED
```

- **DRAFT**：正在编辑中
- **REVIEWING**：待审核发布
- **ACTIVE**：参与默认检索（知识库载入时）
- **OBSOLETE**：旧版本保留但不参与检索（历史工单可回溯）
- **REJECTED**：审核未通过

### 2.6 反馈闭环

```
用户查询 → 检索结果 → 用户评分(1-5) + 评语
                          ↓
                  RagClient.record_feedback()
                          ↓
                  _feedback_log 持久化到 JSON
                          ↓
                  用于后续检索质量分析
```

### 2.7 检索效果评测闭环

```
黄金数据集 (36条: 31可答 + 5拒答, 3级难度)
        ↓
scripts/eval_rag.py 运行双路径评测
        ↓
Hit@K / MRR / P@K / R@K / NDCG@K / 拒答率 / 时延
        ↓
docs/RAG检索效果评测报告.md + JSON 明细
        ↓
失败用例分析 → 修复门槛/同义词 → 回归评测
```

> 评测搭建与三轮修复的完整过程记录见 **[RAG评测优化记录.md](RAG评测优化记录.md)**（含每轮指标对比与门槛标定数据）。

---

## 3. 技术栈

| 层级 | 技术选型 | 版本 | 用途 |
|------|---------|------|------|
| **编程语言** | Python 3.9+ | 3.9.25 | 核心引擎开发 |
| **向量化** | scikit-learn TfidfVectorizer | 1.6.1 | 中文文本 TF-IDF 嵌入（字符 n-gram） |
| **数值计算** | NumPy / SciPy | 2.0.2 / 1.13.1 | 稀疏矩阵运算、余弦相似度计算 |
| **深度学习（备用）** | PyTorch | 2.8.0+cpu | 环境已有，供后续升级为 BERT 类嵌入 |
| **Web 框架** | FastAPI + Uvicorn | 0.110+ | REST API 服务 |
| **数据验证** | Pydantic | 2.6+ | API 请求/响应模型 |
| **持久化** | JSON 文件 | - | 知识库快照存储 |
| **单元测试** | pytest | 8.0+ | 全链路集成测试 |
| **评测框架** | 自研 rag_eval.py | - | 标准 IR 指标（零外部依赖） |
| **版本管理** | Git | - | 知识库文档的 changelog |

**为什么不直接用向量数据库（Qdrant/FAISS）？**

当前为原型演示阶段，内置 26 条知识条目（覆盖全领域），scikit-learn 的 TF-IDF 向量化足以满足需求，且检索时延 P95 < 2ms。接口设计为可替换模式：

```python
# 当前：sklearn 内存索引
KnowledgeBase._rebuild_index()  # TfidfVectorizer

# 未来：Qdrant + PostgreSQL（方案 7.5/7.6）
# 接口不变，替换 _rebuild_index() 与 _semantic_scores() 实现
# PostgreSQL: kb_documents / kb_chunks / kb_entities / kb_query_logs / kb_feedback
# Qdrant: 文本向量 + 元数据
# MinIO: 原始文件存储
```

---

## 4. 遇到的问题及解决方案

### 4.1 中文分词与嵌入

**问题：** 中文文本没有天然分隔符，常规英文 Tokenizer 不适用。项目环境未安装 jieba 分词库。

**解决方案：**
- 使用 scikit-learn 的 TfidfVectorizer 配置 `analyzer='char', ngram_range=(1, 4)`，即字符级 1-4-gram
- 这天然适用于中文：单字、双字词、三/四字短语都能被捕获
- `sublinear_tf=True` 模拟 BM25 的子线性词频压制

```python
self._vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 4),
    sublinear_tf=True,
    max_features=12000,
    dtype=np.float32,
)
```

### 4.2 拒答不生效：ASCII 查询字符偶然命中中文 n-gram

**问题：** 无意义查询（如 `"UNKNOWN_TERM_XYZ"`）的 TF-IDF 向量有少量非零元素（11/5869），因为单个 ASCII 大写字母作为 1-gram 偶然匹配到知识库中的类似字符（如 "J"、"mm" 等单位符号），导致余弦相似度非零（最高 0.104），拒绝失败。

**解决方案：**
在混合打分循环中增加语义门槛：

```python
# 当关键词得分为全零时，抬高语义门槛
max_kw = keyword_sc.max()
kw_all_zero = max_kw < 1e-9
sem_floor = 0.15 if kw_all_zero else 0.0

for j, idx in enumerate(indices):
    sem = float(semantic[j]) if len(semantic) > j else 0.0
    kw = float(keyword_sc[j]) if len(keyword_sc) > j else 0.0
    if kw < 1e-9 and sem < sem_floor:
        continue  # 低于门槛，排除
    combined = alpha * sem + (1.0 - alpha) * kw
    if combined > 1e-9:
        scored.append((combined, idx))
```

**效果：** 无意义 ASCII 查询顺利拒答（`refused=True`），真实中文查询（如 "受电弓出现严重磨耗"）语义得分 0.3-0.6，正常通过。

### 4.3 search 路径拒答完全失效（评测发现）

**问题：** 首轮评测显示 search 路径正确拒答率 0.0——BM25 字符 n-gram 点积按最大值归一化后恒非零（总有人是 1.0），`kw_all_zero` 永远为 False，语义门槛从未生效；语义得分仅 0.01~0.05 的无关查询照常返回结果。

**解决方案：**
- 自然语言路径（keywords=None）对语义得分设**绝对下限 0.06**
- 标定依据：拒答查询 top-1 语义得分全部 ≤ 0.0494；真实查询最低 0.0714、中位数 0.2336
- 修复后 search 路径正确拒答率 0.0 → 1.0，且 31 条可答用例无一误拒

### 4.4 关键词路径被泛化词击穿（评测发现）

**问题：** 跨域拒答用例 RF-05（在 pantograph 域查"烟雾处置"）凭泛化词"处置"命中所有条目——关键词路径原逻辑"关键词>0 即跳过语义门槛"，单个泛化词就绕过了拒答机制。

**解决方案：**
- 关键词路径也设语义下限 0.08
- 标定依据：真实关键词用例最低 0.1284（CB-01），跨域误配仅 0.0396，3 倍安全边际
- 修复后关键词路径正确拒答率 0.8 → 1.0，E2E 6/6 无回归

### 4.5 向后兼容性：kb_list / kb_stats 端点数据结构变更

**问题：** `RagClient.kb` 从 `List[KBChunk]` 重构为 `KnowledgeBase` 对象，旧 `kb_list` 和 `kb_stats` API 端点直接访问 `pf.rag.kb` 作为列表，导致 AttributeError。

**解决方案：**
- `pf.rag.kb` → `pf.rag.kb.chunks`（访问原始列表）
- 旧 `kb_stats` 手动计算 → `pf.rag.stats()`（统一统计入口）
- 旧 `kb_list` → `pf.rag.list_chunks(status, asset_type)`（支持过滤）

### 4.6 document_id 重复导致验证警告

**问题：** 当多个知识块来自同一文档时，`DocumentProcessor.validate_chunks()` 报告 document_id 重复。

**分析：** 这是**预期行为**——同一文档（如 "DOC-DEMO-002 受电弓检修规程"）被切分为多个章节块，每个块的 document_id 相同但 chapter 不同。重复 document_id 本身不是错误，而是结构化切分的自然结果。

**解决方案：** 将 validate_chunks 中的重复检查降级为 WARNING 级别（非 ERROR），并在文档中说明这是预期设计。

### 4.7 长文档的切分与重叠策略

**问题：** 纯文本文件没有标题结构，单段可能超过 1000 字，需要自动切分；且无换行文本在 rfind 找不到断点时存在死循环风险。

**解决方案：**
```python
CHUNK_TARGET_CHARS = 600    # 目标块大小
CHUNK_MAX_CHARS = 1000      # 超过则强制分割
CHUNK_OVERLAP_CHARS = 80    # 相邻块重叠（防止切在关键句中间）
CHUNK_MIN_CHARS = 200       # 过短则与前块合并
```
切分时优先在句号处断开，次选换行处，确保语义完整；增加 `end <= start` 安全防护与 `next_start` 推进检测。

### 4.8 环境依赖冲突

**问题：** 默认 base Python 环境缺少 scikit-learn（requirements.txt 声明但 base 未安装）。项目使用 railmind conda 环境。

**解决方案：** 确认使用 `C:\Users\Qun\.conda\envs\railmind\python.exe` 运行，该环境已预装 sklearn 1.6.1、numpy 2.0.2、scipy 1.13.1、torch 2.8.0。

---

## 5. 知识库现状

| 指标 | 数值 |
|------|------|
| 文档数 | 7 |
| 总知识块数 | 26 |
| 活跃块数 (ACTIVE) | 26 |
| 覆盖领域 | 5（composite_structure / pantograph / underbody / lineside / cabin） |
| 词表大小 | 5,869 个字符 n-gram |
| 检索方式 | 混合检索（α=0.55 语义 + 0.45 关键词） |
| 默认 Top-K | 3 |
| 查询扩展词 | ~20 组领域同义映射 |
| 检索时延 P95 | 1.7ms（search）/ 1.1ms（retrieve） |
| Hit@3 | 1.0（search）/ 0.9032（retrieve） |
| 正确拒答率 | 1.0（双路径） |

---

## 6. API 端点汇总

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/v1/kb/stats` | 知识库统计（文档数/块数/查询数/反馈数） |
| GET | `/api/v1/kb/list` | 知识条目列表（支持 status/asset_type 过滤） |
| POST | `/api/v1/kb/search` | 自然语言知识检索 |
| POST | `/api/v1/kb/feedback` | 记录用户反馈（评分 1-5） |
| POST | `/api/v1/kb/rebuild` | 强制重建 TF-IDF 索引 |
| GET | `/api/v1/kb/queries` | 最近查询日志 |
| GET | `/api/v1/kb/feedback` | 最近反馈记录 |

---

## 7. 生产化升级路径（未来规划）

### 7.1 近期可做

| 改进项 | 方案 | 优先级 |
|--------|------|--------|
| 中文 Embedding 替换 | 使用 sentence-transformers / BERT 中文模型替换字符 n-gram | 高 |
| BM25 精确实现 | 用 rank_bm25 库替换 TF-IDF 近似 | 中 |
| 重排序模型 | 引入 cross-encoder 对 Top-K 结果重排序 | 中 |
| 持久化升级 | JSON → SQLite/PostgreSQL | 高 |
| 评测扩充 | 黄金数据集扩至 100+ 条；LLM-as-Judge 忠实度评测 | 中 |

### 7.2 中远期规划（方案 7.5-7.6）

```
PostgreSQL: kb_documents / kb_chunks / kb_entities / kb_relations / kb_query_logs / kb_feedback
Qdrant: 文本向量 + 元数据（用于语义搜索）
MinIO: 原始文件存储（PDF/DOCX）
混合检索流程：元数据过滤 → 向量召回 → BM25 召回 → 重排序 → 答案生成
```

---

## 8. 快速验证

```python
from railmind.core.rag import RagClient

client = RagClient()

# 关键词检索（向后兼容，生产 Agent 路径）
result = client.retrieve(asset_type="pantograph", keywords=["电弧", "处置"])
for c in result.answer_basis:
    print(f"[{c['document_id']}] {c['chapter']} 页码={c['page']}")

# 自然语言检索（Chat 路径）
result = client.search("弹条断裂要不要限速", asset_type="lineside")
if not result.refused:
    for c in result.answer_basis:
        print(f"引用: {c['title']} → {c['chapter']}")

# 统计
print(client.stats())

# 反馈
client.record_feedback(query="电弧处置", document_id="DOC-DEMO-002", rating=5)

# 持久化
client.persist("kb_snapshot.json")
```

```bash
# 检索效果评测（railmind 环境）
python scripts/eval_rag.py                # 生成 docs/RAG检索效果评测报告.md
python scripts/eval_rag.py --top-k 5      # 自定义 Top-K
python scripts/eval_rag.py --json docs/rag_eval_detail.json  # 输出 JSON 明细
```