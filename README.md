# RailMind · 列车智能运维智能体平台（原型 V0.3）

全空间、动静结合的列车智能运维平台原型。内核采用 **pi 式极简 Agent 设计**（agent = while 循环 + 工具注册表 + 事件流）：
**Chief-Agent 由远端 LLM 驱动**（DeepSeek V4 Flash Vision Exp，opencode zen，工具调用自主编排），确定性风险引擎与工单闭环由平台保证，
网络故障自动落回本地脚本（演示稳定性）。前端为按参考图实现的深色大屏 + **运维对话助手**。

## 七个能力（能力中心实时可视）

| 能力 ID | 领域 | 模型/数据 | 指标 |
| --- | --- | --- | --- |
| internal.shm.impact_locator | composite_structure | EnhancedLambWaveNet（残差+PAN+CBAM，657K 参数）· 团队复赛数据 | RMSE 40.99mm / 能量准确率 90.5% |
| internal.underbody.door_pose | underbody | YOLOv8-pose ONNX（board 4 点 + handle 6 点）· 团队自研 | 关门 0°/开门 85-91° 三路径验证 |
| internal.panto.wear_vlm | pantograph | LLM 视觉评估（strip 检测引导裁剪 → 固定 JSON） | 低置信度自动拒判 |
| internal.panto.arc_signal | pantograph | 可解释信号链 · Mendeley 电弧数据集（CC BY 4.0） | **19/19 真实电弧记录检出 100%** |
| internal.lineside.fod | lineside | **YOLO26n** · RailFOD23（4 类异物，CC BY 4.0） | mAP50 0.699（30ep 从零训练） |
| internal.lineside.fastener | lineside | **YOLO26n** · RFDD 扣件缺陷数据集（6 类，CC BY 4.0） | **mAP50 0.975**（断裂/缺失→HIGH） |
| internal.cabin.patrol_vlm | cabin | 乘务员巡检视频 → 双窗口各 6 帧 → VLM 多帧固定 JSON（DeepSeek V4 Flash Vision） | 连续两窗口一致性校验；预计算缓存回放 + 实时重分析双模式 |

另有受电弓部件检测层（YOLO26n，contact_point/mast/strip，mAP50 0.768）服务磨耗评估裁剪。

## 快速开始

环境：`D:/Anaconda3/envs/railmind`（Python 3.9，torch 2.2.2 + CUDA、ultralytics 8.4.140（YOLO26）、fastapi 等）

```bat
cd /d D:\workspace\RailMind
set PYTHON=D:\Anaconda3\envs\railmind\python.exe

:: 测试（26 项，本地大脑，秒级）
%PYTHON% -m pytest tests -q

:: 平台服务（大屏 + API + 对话助手）：浏览器打开 http://127.0.0.1:8900/
%PYTHON% -X utf8 -m railmind.apps.api

:: 离线端到端演示（三领域，不依赖 LLM）
%PYTHON% -X utf8 -m demo.run_demo
```

LLM 大脑配置（`.env`，**勿提交公开仓库**）：

```
RAILMIND_LLM_BASE_URL=https://api.deepseek.com
RAILMIND_LLM_API_KEY=sk-***
RAILMIND_LLM_MODEL=deepseek-v4-flash-vision-exp
RAILMIND_LLM_REASONING=none    # 关闭深度思考，编排延迟 153s → 20s
```

**演示模式**：综合态势页「▶ 一键演示 · 评委模式」按 48s 时间线编排 6 项能力真实事件（后端 Chief 全链路），
车速/里程/传动健康度实时模拟，车内巡检走 VLM 预计算缓存回放（`scripts/precompute_cabin.py --refresh` 重新生成）。

## 架构

```text
前端大屏（web/index.html）           运维对话助手（/api/v1/chat）
        │ POST /api/v1/events              │
        ▼                                  ▼
Chief-Agent（LLM 大脑，自主编排 6 步；Echo 降级兜底 + 闭环保障）
   find_capability → invoke_capability → consult_specialist
   → assess_risk（确定性引擎）→ retrieve_kb（RAG 桩）→ draft_workorder
        │                                    │
能力注册中心（心跳/TTL/主备/模式）    4 个领域专业 Agent（确定性分级）
        ↓
5 个能力（进程内直调 / REST 热加载）→ 统一诊断事件 → 证据存档
```

- **大脑分层**：Chief = LLM（决定"何时做什么"）；工具参数全部由会话注入（数据不经 LLM 转手，杜绝转录幻觉）；
  专业 Agent = 本地确定性分级（规程阈值计算）；LLM 漏掉的编排步骤由平台自动补齐（工单闭环硬保证）。
- **风险等级绝不由 LLM 生成**（方案 6.3）：RiskEngine 按严重等级/置信度/频次/多源支持/关键部件等因子加分，理由可解释。

## API 摘要

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/healthz` | 健康检查（含 brain 类型） |
| GET | `/api/v1/capabilities` | 能力中心 |
| POST | `/api/v1/capabilities/register` | 外部能力热注册 |
| POST | `/api/v1/capabilities/{id}/mode` | 模式切换 PRIMARY/STANDBY/AUXILIARY/OBSERVE/DISABLED |
| POST | `/api/v1/events` | 事件入口 → Chief 全链路 |
| POST | `/api/v1/events/demo/{kind}` | 一键演示事件（shm_impact/door_normal/door_open/panto_wear/panto_arc/lineside_fod/fastener_defect/cabin_patrol/cabin_live） |
| POST | `/api/v1/chat` | 运维对话助手（LLM 大脑） |
| GET/POST | `/api/v1/workorders...` | 工单查询与人工确认/驳回/关闭 |

## 外部能力热加载（方案 2.2）

```bat
:: 终端1：能力独立 REST 服务
%PYTHON% -m railmind.capabilities.shm_impact.service --port 8901
:: 终端2：POST /api/v1/capabilities/register 提交 capability.yaml + base_url —— 平台不重启即可调用
```

## 数据集（详见 datasets/DATA_MANIFEST.md）

- **Mendeley 电弧**（DOI 10.17632/74nz86wcgy，CC BY 4.0）：19 条真实电弧记录，信号链检出率 100%
- **RailFOD23** 轨道异物 + **受电弓部件检测**（均 CC BY 4.0，Roboflow 导出）：YOLO26n 训练于 RTX 3080（scripts/train_yolo26.py）
- 团队数据：Lamb 波复赛数据集、检查门把手原图 + YOLOv8-pose 权重

## ⚠️ 权重提醒

冲击检测正确权重（epoch 184, RMSE 40.99mm / 能量准确率 90.5%）只保留两份：平台自带
`railmind/capabilities/shm_impact/weights/best_enhanced_model.pth`（Git 跟踪，运行时加载）与
`第二届轨道交通比赛/提交材料/semi-final/`（离线备份）。SHM 演示样本抽取在 `datasets/shm_samples/`
（7 条 1.00J），完整复赛数据集位置见 `datasets/DATA_MANIFEST.md`；历次整理已删除根目录 `semi-final/`、
`完整项目备份/` 及 epoch 0/2 欠训练残留权重。

> 公开/模拟数据验证不等同于真实高速动车上线认证（方案 1.4）。
