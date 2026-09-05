"""RAG 知识检索桩 —— 方案第 7 章的最小可用版。

内置固定知识条目（试行版），提供 元数据过滤 + 关键词打分 的混合检索；
检索不到可靠依据时按方案 7.8 固定拒答。接口稳定，后续可平滑替换为
PostgreSQL 元数据过滤 + Qdrant 向量召回 + BM25 的完整实现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

NO_ANSWER = "当前知识库中未找到足够依据，建议人工查询相关规程。"


@dataclass
class KBChunk:
    document_id: str
    title: str
    chapter: str
    page: int
    text: str
    asset_type: str
    fault_type: List[str]
    version: str = "1.0"
    effective_status: str = "ACTIVE"
    permission: str = "INTERNAL"

    def citation(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "chapter": self.chapter,
            "page": self.page,
            "version": self.version,
            "quote": self.text[:120],
        }


def _default_kb() -> List[KBChunk]:
    return [
        KBChunk(
            document_id="DOC-DEMO-001",
            title="复合材料结构冲击检测作业指引（试行版）",
            chapter="冲击能量分级与处置",
            page=4,
            text="复合材料结构受到冲击后，应按冲击能量分级处置：能量低于0.35J视为轻微冲击，记录观察；"
                 "0.35J至0.70J之间为重点关注冲击，应在下一停靠站进行目视复核；"
                 "0.70J及以上为显著冲击，须立即安排人工敲击检测或无损检测复核，并评估是否限制运行。",
            asset_type="composite_structure",
            fault_type=["impact", "energy_high"],
        ),
        KBChunk(
            document_id="DOC-DEMO-001",
            title="复合材料结构冲击检测作业指引（试行版）",
            chapter="双点冲击判读",
            page=6,
            text="当监测系统在同一时刻窗内定位到两处冲击时，应分别核对两处坐标与能量等级；"
                 "两点能量均超过0.50J时按显著冲击处置流程处理，并检查两处之间的结构连通区域。",
            asset_type="composite_structure",
            fault_type=["impact", "double_impact"],
        ),
        KBChunk(
            document_id="DOC-DEMO-002",
            title="受电弓检修规程（试行版）",
            chapter="滑板磨耗",
            page=12,
            text="受电弓滑板磨耗等级达到明显磨耗时，应在最近具备检修条件的停靠站安排更换；"
                 "若同时出现电弧频次升高，应提升处置优先级并在下次入库检修前加密观测频次。",
            asset_type="pantograph",
            fault_type=["strip_wear", "arc"],
        ),
        KBChunk(
            document_id="DOC-DEMO-002",
            title="受电弓检修规程（试行版）",
            chapter="电弧异常",
            page=15,
            text="电弧事件持续时长超过2秒或单位时间频次超过阈值时，应复核弓网接触状态，"
                 "并结合滑板磨耗等级综合判断是否需要降弓或限速运行。",
            asset_type="pantograph",
            fault_type=["arc"],
        ),
        KBChunk(
            document_id="DOC-DEMO-003",
            title="车底检查门检修说明（试行版）",
            chapter="检查门未闭合处置",
            page=3,
            text="车底检查门未完全闭合时，进站后应由地勤人员现场复核并重新锁闭；"
                 "同一检查门重复出现未闭合告警时，应检查锁机构磨损情况并生成检修工单。",
            asset_type="underbody",
            fault_type=["door_not_closed"],
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
            fault_type=["door_handle_angle", "board_abnormal"],
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
            fault_type=["arc", "arc_signal"],
        ),
        KBChunk(
            document_id="DOC-DEMO-004",
            title="线路侧异物处置办法（试行版）",
            chapter="异物分级与处置",
            page=7,
            text="线路侧异物按与限界的距离和尺寸分级处置：面积占比超过限界阈值的漂浮物、大型异物，"
                 "应立即通知工务与调度，评估是否拦停后续列车；小型异物应安排工务现场清理复核；"
                 "未检出异物时按正常记录处理，影像点持续观察。",
            asset_type="lineside",
            fault_type=["foreign_object", "limit_boundary"],
        ),
        KBChunk(
            document_id="DOC-DEMO-004",
            title="线路侧异物处置办法（试行版）",
            chapter="影像点复核要求",
            page=9,
            text="同一影像点1小时内重复出现异物告警的，应在清理完成后核查异物来源（风揭垃圾、货物散落等），"
                 "并对相邻影像点开展一次补充巡检。",
            asset_type="lineside",
            fault_type=["foreign_object", "repeat_alarm"],
        ),
        KBChunk(
            document_id="DOC-DEMO-005",
            title="钢轨扣件检修规程（试行版）",
            chapter="扣件缺陷分级处置",
            page=11,
            text="扣件弹条断裂、缺失时扣压功能失效，判定为显著缺陷，应立即通知工务安排更换，"
                 "并评估该区段是否限速通过；弹条移位、翻转、变形为重点缺陷，应列入最近天窗点检计划；"
                 "未检出缺陷的扣件按正常状态记录。",
            asset_type="lineside",
            fault_type=["fastener_defect", "broken", "missing"],
        ),
        KBChunk(
            document_id="DOC-DEMO-005",
            title="钢轨扣件检修规程（试行版）",
            chapter="复检与验收",
            page=14,
            text="扣件更换完成后，应在下一巡检周期对更换位置进行图像复检，确认弹条状态恢复正常；"
                 "同一里程重复出现同类缺陷时，应检查轨枕与垫板状态。",
            asset_type="lineside",
            fault_type=["fastener_defect", "recheck"],
        ),
        KBChunk(
            document_id="DOC-DEMO-006",
            title="车内异常巡检处置指引（试行版）",
            chapter="车内事件分级与处置",
            page=5,
            text="车内巡检发现的异常按方案分级处置：疑似烟雾或火光单窗口高置信即升级，立即通知乘务员现场确认并按火灾预案处置；"
                 "人员持续倒地、通道持续堵塞需连续两个窗口结论一致方可升级告警；"
                 "异常聚集与疑似剧烈动作仅记录观察并由乘务员复核，不得输出确定性结论。",
            asset_type="cabin",
            fault_type=["cabin_patrol", "smoke", "aisle_blocked"],
        ),
        KBChunk(
            document_id="DOC-DEMO-006",
            title="车内异常巡检处置指引（试行版）",
            chapter="乘务员复核流程",
            page=8,
            text="车厢巡检VLM窗口结论置信度不足或相邻窗口不一致时，应转乘务员人工复核原始视频；"
                 "复核结果回填平台后关闭或升级对应事件，形成处置闭环。",
            asset_type="cabin",
            fault_type=["cabin_patrol", "human_review"],
        ),
    ]


@dataclass
class RagResult:
    answer_basis: List[Dict[str, Any]] = field(default_factory=list)
    refused: bool = False

    @property
    def refused_text(self) -> str:
        return NO_ANSWER


class RagClient:
    """混合检索桩：asset_type/fault_type 过滤 + 关键词重合打分。"""

    def __init__(self, kb: Optional[List[KBChunk]] = None, top_k: int = 2):
        self.kb = kb or _default_kb()
        self.top_k = top_k
        self.query_log: List[Dict[str, Any]] = []

    def retrieve(
        self,
        asset_type: str,
        keywords: List[str],
        fault_type: Optional[str] = None,
        permission: str = "INTERNAL",
    ) -> RagResult:
        scored: List[tuple] = []
        for chunk in self.kb:
            if chunk.effective_status != "ACTIVE":
                continue
            if chunk.asset_type != asset_type:
                continue
            if fault_type and fault_type not in chunk.fault_type:
                continue
            text = chunk.title + chunk.chapter + chunk.text
            overlap = sum(1 for kw in keywords if kw and kw in text)
            if overlap == 0:
                continue
            scored.append((overlap, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits = [chunk for _, chunk in scored[: self.top_k]]
        self.query_log.append(
            {"ts": time.time(), "asset_type": asset_type, "keywords": keywords, "fault_type": fault_type, "hits": len(hits)}
        )
        if not hits:
            return RagResult(answer_basis=[], refused=True)
        return RagResult(answer_basis=[h.citation() for h in hits], refused=False)
