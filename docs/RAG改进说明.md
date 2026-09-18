# RAG 知识库改进说明

本文档记录 RailMind 项目 RAG（检索增强生成）知识库从原型桩到混合检索引擎的完整改进，以及检索效果评测体系。

---

## 一、改进背景

原 RAG 实现（`railmind/core/rag.py`）是一个最小可用桩，采用简单的**关键词重叠打分**，仅内置 14 条知识条目，不具备语义理解能力、动态管理功能和反馈闭环，与方案设计文档中规划的完整 RAG 架构差距较大。

---

## 二、改进总览

| 维度 | 改进前 | 改进后 |
|------|--------|--------|
| 检索方式 | 关键词字符串包含计数 | TF-IDF 语义相似度 + BM25 关键词加权混合 |
| 中文处理 | 无（简单 in 操作） | 字符 n-gram (1-4) 向量化，天然适配中文 |
| 查询扩展 | 无 | 20+ 组铁路领域同义词自动扩展 |
| 知识库大小 | 14 条 / 6 文档 | 26 条 / 7 文档 |
| 知识库管理 | 固定不可变 | 动态 CRUD（增/删/改/查） |
| 版本管理 | 无 | DRAFT→REVIEWING→ACTIVE→OBSOLETE 生命周期 |
| 用户反馈 | 无 | 评分(1-5) + 评语记录与查询 |
| 持久化 | 无 | JSON 文件快照（保存/加载） |
| 文档处理 | 无 | 文档解析→结构化切分→元数据标注管线 |
| API 端点 | 2 个（stats / list） | 7 个（+search / feedback / rebuild / queries / feedback_log） |
| 拒答保护 | 关键词不命中 | 语义门槛 + 关键词零分双重保障 |
| 检索效果评测 | **无** | 36 条黄金数据集 + 标准 IR 指标 + 自动报告 |
| E2E 测试 | 6 通过 | 6 通过（完全向后兼容） |

---

## 三、详细改进说明

### 3.1 混合检索引擎（`railmind/core/rag.py` 全部重写）

**旧架构：**
```
用户关键词 → 按 asset_type 过滤 → 遍历所有块统计关键词出现次数 → 排序取 Top-2
```

**新架构：**
```
用户查询 / 诊断事件
    ↓
① 元数据过滤 (asset_type / fault_type / effective_status)
    ↓
② 查询扩展（铁路领域同义词映射）
    ↓
③ 语义召回（TF-IDF 字符 n-gram → 余弦相似度）
    ↓
④ 关键词召回（TF-IDF 加权词级匹配）
    ↓
⑤ 混合排序 (α·语义 + (1-α)·关键词)，默认 α=0.55
    ↓
⑥ 拒答门槛（语义下限，见 9.2 节标定）
    ↓
⑦ Top-K 返回 + 完整引用元数据
```

**涉及的代码文件：** `railmind/core/rag.py`（全部 341 行重写）

**关键类：**
- `KBChunk` — 知识块数据类（新增 `tags`、`created_at`、`updated_at` 字段）
- `KnowledgeBase` — 知识库引擎（向量索引管理、CRUD、版本管理、混合检索、持久化）
- `RagClient` — 对外 Facade（向后兼容 `retrieve()`，新增 `search()` / `add_chunk()` / `record_feedback()` 等）
- `RagResult` — 检索结果（新增 `query_time_ms`）

### 3.2 文档处理管线（`railmind/core/kb_builder.py` 新增）

**新增能力：** 支持将外部文档自动解析、切分、标注后注入知识库。

```
外部文件 (TXT/MD/HTML)
    ↓ DocumentProcessor.process_file()
① 格式检测与文件读取（支持 UTF-8/GBK）
② 文本清洗（统一换行、压缩空白）
③ 结构化解析（MD 标题 / HTML 标签 / 纯文本段落）
④ 长内容切分（600 字目标 / 80 字重叠 / 句号优先断点）
⑤ 元数据标注（asset_type / fault_type / 版本 / 权限 / 标签）
⑥ 冲突校验（document_id 重复 / 文本过短 / 缺少 asset_type）
    ↓
List[KBChunk] → RagClient.add_chunks()
```

**涉及的代码文件：** `railmind/core/kb_builder.py`（新增 364 行）

**关键函数：**
- `DocumentProcessor.process_text()` — 处理文本内容
- `DocumentProcessor.process_file()` — 处理文档文件
- `parse_text()` — 按格式解析（md / html / txt）
- `chunk_sections()` — 结构化切分
- `import_directory()` — 批量导入目录

### 3.3 领域知识条目扩展

**旧知识库：** 14 条 / 6 文档，部分领域只有 1-2 条

**新知识库：** 26 条 / 7 文档

| 文档 ID | 标题 | 章节数 | 覆盖故障类型 |
|---------|------|--------|------------|
| DOC-DEMO-001 | 复合材料结构冲击检测作业指引 | 4 | impact, energy_low/medium/high, double_impact, sensor_check, high_frequency, trend, degraded |
| DOC-DEMO-002 | 受电弓检修规程 | 4 | strip_wear, wear_light/heavy, arc/arc_severe, arc_signal, crack, chipping, uneven_wear |
| DOC-DEMO-003 | 车底检查门检修说明 | 3 | door_not_closed, lock_wear, door_handle_angle, board_abnormal, lock_maintenance |
| DOC-DEMO-004 | 线路侧异物处置办法 | 3 | foreign_object, limit_boundary, fod_large/medium/small, repeat_alarm, extreme_weather |
| DOC-DEMO-005 | 钢轨扣件检修规程 | 3 | fastener_defect, broken, missing, displaced, deformed, loose, torque, recheck |
| DOC-DEMO-006 | 车内异常巡检处置指引 | 4 | cabin_patrol, smoke, fire, aisle_blocked, person_down, crowd, human_review, emergency |
| DOC-DEMO-007 | 列车运行安全通用规程（新增） | 1 | general, alert_escalation（跨 5 领域各一条） |

**新增的关键条目：**
- 传感器自检与降级模式（复合材料）
- 滑板裂纹与偏磨检查（受电弓）
- 锁机构维护周期（车底检查门）
- 恶劣天气加强巡检（线路侧异物）
- 扣件扭矩标准（钢轨扣件）
- 烟雾火灾联动应急预案（车厢）
- 通道堵塞定量评估标准（车厢）
- 跨领域告警升级联动规程（通用安全，覆盖 5 领域）

### 3.4 API 端点增强

**旧端点：**
```
GET  /api/v1/kb/stats    # 基础统计（documents, chunks, queries）
GET  /api/v1/kb/list     # 列出所有块
```

**新端点：**
```
GET  /api/v1/kb/stats     # 增强统计（+active_chunks, +asset_types, +feedback_count, +vocab_size）
GET  /api/v1/kb/list      # 增强（支持 status / asset_type 过滤，+fault_type, +tags）
POST /api/v1/kb/search    # 新增：自然语言检索
POST /api/v1/kb/feedback  # 新增：记录用户反馈
POST /api/v1/kb/rebuild   # 新增：强制重建 TF-IDF 索引
GET  /api/v1/kb/queries   # 新增：最近查询日志
GET  /api/v1/kb/feedback  # 新增：最近反馈记录
```

**涉及的代码文件：** `railmind/apps/api.py`（新增 7 个端点，替换 2 个旧端点）

### 3.5 E2E 测试

全部 6 个既有集成测试通过，零修改：

```
tests/test_e2e.py::test_full_pipeline_creates_workorder      PASSED
tests/test_e2e.py::test_workorder_confirm_flow               PASSED
tests/test_e2e.py::test_offline_capability_refuses           PASSED
tests/test_e2e.py::test_mode_switch_to_observe_disables_dispatch  PASSED
tests/test_e2e.py::test_events_recorded_for_history          PASSED
tests/test_e2e.py::test_summary_is_valid_json                PASSED
```

---

## 四、修复的关键问题

### 问题 1：ASCII 偶然命中中文 n-gram 导致拒答失效

**现象：** 无意义查询 `"UNKNOWN_TERM_XYZ"` 返回 `refused=False`，因为单字符 1-gram（N, O, W, T 等）偶然匹配到知识库中的单位符号。

**修复：** 关键词得分为全零时，语义得分需 > 0.15 才算有效匹配。

### 问题 2：旧 kb_list 端点引用 `pf.rag.kb` 作为列表

**现象：** API 加载时报 AttributeError，因为 `pf.rag.kb` 从 `List[KBChunk]` 变为 `KnowledgeBase`。

**修复：** 统一通过 `pf.rag.stats()` / `pf.rag.list_chunks()` 访问，删除残留的旧端点定义。

### 问题 3：长文本切分循环可能无法推进

**现象：** 无换行符的纯文本中，`rfind` 找不到断点时 `end` 不变化，导致死循环风险。

**修复：** 增加 `end <= start` 安全防护和 `next_start` 推进检测。

---

## 五、技术栈变更

| 组件 | 之前 | 之后 |
|------|------|------|
| 向量化引擎 | 无 | scikit-learn 1.6.1 (TfidfVectorizer) |
| 数值计算 | 无 | numpy 2.0.2, scipy 1.13.1 |
| 相似度计算 | 无 | sklearn.metrics.pairwise.cosine_similarity |
| 文档解析 | 无 | 内置正则 + 字符串处理（零外部依赖） |
| 评测框架 | 无 | 自研 IR 指标库（零外部依赖） |
| 依赖变化 | requirements.txt 不变 | 利用环境已有 sklearn/scipy（requirements.txt 早有声明） |

---

## 六、涉及的代码文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `railmind/core/rag.py` | 全部重写 | 341 行，混合检索引擎 |
| `railmind/core/kb_builder.py` | 新增 | 364 行，文档处理管线 |
| `railmind/core/rag_eval.py` | 新增 | 评测框架 + 36 条黄金数据集 |
| `scripts/eval_rag.py` | 新增 | 评测 CLI 入口 |
| `railmind/apps/api.py` | 修改 | 新增 7 个 KB 端点，替换 2 个旧端点 |
| `docs/RAG搭建全流程总结.md` | 新增 | 完整搭建流程与问题记录 |
| `docs/RAG改进说明.md` | 新增 | 本文 |
| `docs/RAG检索效果评测报告.md` | 自动生成 | 评测报告（含具体指标数值） |
| `docs/rag_eval_detail.json` | 自动生成 | 评测 JSON 明细 |

**未修改（零影响）：**
- `railmind/core/chief.py` — 接口不变，直接受益
- `railmind/core/chat.py` — 接口不变，直接受益
- `railmind/agents/*.py` — 接口不变，直接受益
- `railmind/platform.py` — 接口不变，直接受益
- `tests/test_e2e.py` — 全部通过，无需修改

---

## 七、与方案设计的差距分析

| 方案要求（第 7 章） | 当前状态 | 差距 |
|--------------------|---------|------|
| PostgreSQL 结构化表 | ❌ 未实现 | JSON 文件替代 |
| Qdrant 向量数据库 | ❌ 未实现 | sklearn 内存 TF-IDF 替代 |
| MinIO 原文件存储 | ❌ 未实现 | 直接读取本地文件 |
| BM25 精确实现 | ⚠️ 近似 | 用 TF-IDF `sublinear_tf=True` 近似 |
| 重排序模型 | ❌ 未实现 | 简单混合排序替代 |
| 嵌入模型可替换 | ⚠️ 接口支持 | TfidfVectorizer 参数化 |
| 管理页面 | ❌ 未实现 | 仅 API 端点 |
| 版本管理 | ✅ 已实现 | DRAFT→REVIEWING→ACTIVE→OBSOLETE |
| 拒答行为 | ✅ 已实现（有评测验证） | 语义门槛标定 + 拒答用例 100% 通过 |
| 引用可追溯 | ✅ 已实现 | 文档 ID + 章节 + 页码 + 版本 |

---

## 八、检索效果评测（新增）

原项目**无任何 RAG 评测**。本次新增评测框架 `railmind/core/rag_eval.py`（标准 IR 指标 + 36 条人工标注黄金数据集）与 CLI 入口 `scripts/eval_rag.py`，报告输出至 `docs/RAG检索效果评测报告.md`。

> 完整的评测搭建与三轮修复过程（含每轮指标数值与门槛标定数据）见 **[RAG评测优化记录.md](RAG评测优化记录.md)**。

### 8.1 评测结论（Top-K=3，α=0.55）

| 指标 | search() 自然语言路径 | retrieve() 关键词路径（生产 Agent 路径） |
|------|----------------------|----------------------------------------|
| Hit@3 | **1.0** | 0.9032 |
| MRR | **0.9677** | 0.6774 |
| NDCG@3 | **0.9762** | 0.7233 |
| 正确拒答率 | **1.0** | **1.0** |
| 误拒答率 | 0.0 | 0.0 |
| P95 时延 | 1.7ms | 1.1ms |

36/36 用例通过（31 可答 + 5 拒答），E2E 回归 6/6。分难度：easy Hit@3=1.0 / medium Hit@3=1.0 / hard Hit@3=1.0（MRR 0.875）。

### 8.2 评测驱动修复的两个真实缺陷

1. **search 路径拒答完全失效**（首轮评测正确拒答率 0.0）：BM25 字符 n-gram 点积归一化后恒非零，语义门槛从未生效——拒答查询语义得分 ≤0.05 仍返回结果。修复：自然语言路径设语义绝对下限 0.06（标定：真实查询 ≥0.07，拒答查询 ≤0.05）。
2. **关键词路径被泛化词击穿**：跨域查询凭单个泛化词"处置"即可命中任意领域条目（关键词>0 即跳过门槛）。修复：关键词路径语义下限 0.08（标定：真实用例最低 0.128，跨域误配 0.04）。修复后两路径正确拒答率均达 1.0，且 E2E 无回归。

### 8.3 评测体系构成

- **指标**：Hit@K / MRR / Precision@K / Recall@K / NDCG@K / 正确拒答率 / 误拒答率 / 时延（mean/P50/P95/max）
- **维度**：双路径（search/retrieve）× 分领域（6）× 分难度（easy/medium/hard）× 失败用例明细
- **黄金数据集**：36 条，覆盖 5 领域 + 跨域通用 + 拒答（含跨域误配、域外知识、乱码词），难度分三级（easy=原文术语 / medium=同义改写 / hard=口语化间接描述）
- **复跑方式**：`python scripts/eval_rag.py [--top-k 3] [--alpha 0.55] [--json docs/rag_eval_detail.json]`

---

## 九、未来优化方向

1. **替换 Embedding**：字符 n-gram → sentence-transformers 中文模型 → BERT
2. **精确 BM25**：引入 `rank_bm25` 库替换 TF-IDF 近似
3. **重排序层**：cross-encoder 对 Top-20 结果重排序
4. **持久化升级**：JSON → SQLite → PostgreSQL
5. **增量索引**：当前全量重建 → HNSW/IVF 增量索引
6. **管理前端**：文档上传、切分预览、检索测试界面
7. **评测扩充**：黄金数据集扩至 100+ 条；引入 LLM-as-Judge 对生成答案做忠实度评测
