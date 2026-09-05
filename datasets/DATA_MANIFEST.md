# RailMind 数据清单（方案 4.3）

> 状态定义见开发方案 4.1：AVAILABLE / VERIFIED / PENDING / APPLICATION_REQUIRED / DEMO_ONLY / INTERNAL
>
> 原则：Git 只存清单与脚本；大数据/权重放本地 `datasets/` 与能力包 `weights/`（参赛打包时一并提供）。

## 已入库

### pantograph_arcing_v1 — 受电弓电弧波形数据集

- **status**: VERIFIED
- **source**: Mendeley Data，DOI `10.17632/74nz86wcgy`（*Pantograph Arcing in DC Railway Systems*）
- **license**: CC BY 4.0（引用须注明作者与 DOI）
- **storage**: `datasets/pantograph/arcing/{MetroMadrid,Trenitalia}/{traction,braking}/`
- **内容**: 19 条真实电弧事件记录；每条含 `_x`(时间轴) `_Vp`(弓上电压) `_Ip`(电流) `_Vf`(滤波电容电压) `.mat/.tif`；采样率 50kHz，每条 3~4 秒窗口
- **验证**: 信号链（滑动窗口突变 + 高频能量 + 事件统计）对 19/19 条全部检出（100%），severity 分布 HIGH 14 / WARNING 2 / OBSERVE 1 / NORMAL 2（弱电弧事件评分低属合理）
- **服务能力**: `internal.panto.arc_signal`（运行中场景）

### pantograph_samples_v1 — 受电弓样本图

- **status**: AVAILABLE（图片来自 GitHub 公开仓库 share2code99/* 与 sriniuketh 的演示素材，仅作演示输入，不用于训练）
- **storage**: `datasets/pantograph/samples/`（video_frame_*.jpg 为真实受电弓照片）
- **服务能力**: `internal.panto.wear_vlm` 的演示输入

### door_handle_raw_v1 — 检查门把手原图（团队数据）

- **status**: INTERNAL
- **storage**: `pose_detect/*.jpg`（4 张轨旁相机原图，含开启/关闭状态）+ `pose_detect/model_pose_detect/best.onnx`（YOLOv8-pose 权重）
- **服务能力**: `internal.underbody.door_pose`；检测结果图存 `data/door_evidence/`

### shm_lambwave_v1 — 复合材料 Lamb 波数据集（团队复赛数据）

- **status**: INTERNAL
- **storage**: 完整数据集 `第二届轨道交通比赛/提交材料/semi-final/Composite Material_Semi-Final_Dataset/`（105 训练 + 45 测试 .mat，5000×8）；演示样本 `datasets/shm_samples/`（7 条 1.00J）；权重 `railmind/capabilities/shm_impact/weights/best_enhanced_model.pth`（epoch184 版本）
- **服务能力**: `internal.shm.impact_locator`

### cabin_patrol_v1 — 乘务员车厢巡检视频（演示素材）

- **status**: DEMO_ONLY
- **source**: 抖音公开素材 @华联田雨（抖音号 1290332714），仅作平台功能演示，不用于训练
- **storage**: `web/assets/cabin_patrol.mp4`（14.6s，有效分析段 0–11.4s）；证据帧 `railmind/capabilities/cabin_vlm/frames/`；VLM 预计算结果 `railmind/capabilities/cabin_vlm/demo_cache.json`
- **服务能力**: `internal.cabin.patrol_vlm`（方案 5.5 车内巡检：双窗口 × 6 帧 VLM 固定 JSON + 一致性校验）
- **⚠️ 合规提醒**: 素材含可识别乘客面部（方案 4.4 数据安全）；**参赛报告/PPT/演示视频中使用须人脸打码或替换为可商用素材，并注明来源**

### 下载与转换脚本

- `scripts/`（规划）：数据下载、清单校验、哈希登记脚本随能力一并提供

## 待入库（已购买，等待文件到位）

| 数据集 | 规格 | 状态 | 计划接入 |
| --- | --- | --- | --- |
| 轨道交通受电弓目标检测数据集 | 2676 张 YOLO，3 类（contact_point/mast/strip），含 YOLOv8s 100epoch 权重 mAP@0.5=0.921 | PENDING（下载中） | Panto 检测层：检测 strip → 裁剪 → VLM 磨耗评估；检测结果图供前端展示 |
| 轨道交通轨道异物检测数据集 | 2541 张 YOLO，4 类（鸟草/漂浮物/气球/塑料袋），含 v8s 权重 mAP@0.5=0.924 | PENDING（下载中） | LineSide-Agent：线路侧异物检测能力（方案 5.6） |
| 高铁铁轨紧固件缺陷检测数据集 | 1350 张全景 YOLO，6 类扣件状态 | PENDING（下载中） | LineSide 扩展：扣件缺陷检测（时间富余再做） |

> 购买数据集为网上整理资源，仅用于比赛演示与测试；参赛材料中注明来源与仅研究用途，不作为核心模块唯一依赖（方案 4.1 原则）。

## ✅ 已到位并完成训练（2026-09-05 更新）

| 数据集 | 实际目录 | 状态 | 产出 |
| --- | --- | --- | --- |
| 受电弓目标检测（3 类 contact_point/mast/strip，train 2275/val 265/test 136） | `高体受电弓设备目标检测数据集 YOLO格式/`（Roboflow pantograph-9bhnb v6，CC BY 4.0） | VERIFIED | **YOLO26n 自训 30ep：mAP50 0.768**（runs/panto_y26n），权重已入 `railmind/capabilities/panto/weights/`，服务 strip 检测引导裁剪 |
| 轨道异物检测（4 类 niaocao/piaofuwu/qiqiu/suliaodai，train 1779/val 508/test 254） | `RailFOD23.v1i.yolov8/`（Roboflow adeemvlm/railfod23 v1，CC BY 4.0） | VERIFIED | **YOLO26n 自训 30ep：mAP50 0.699**（runs/fod_y26n），权重已入 `railmind/capabilities/lineside_fod/weights/`，服务 `internal.lineside.fod` |
| 高铁铁轨紧固件缺陷检测（1250 训练 + 100 测试，6 类：弹条变形/断裂/缺失/翻转/移位 + 正常扣件） | `E:/BaiduNetdiskDownload/铁轨紧固件损坏检测数据集 YOLO格式/`（RFDD，Science Data Bank CSTR:149.11.sciencedb.msdc.00071，CC BY 4.0，原始 13.4GB 不入 Git） | VERIFIED | **YOLO26n 自训 30ep：mAP50 0.975 / mAP50-95 0.796**（runs/fastener_y26n），权重已入 `railmind/capabilities/fastener/weights/`，服务 `internal.lineside.fastener`；演示样本 `railmind/capabilities/fastener/samples/` |

- 训练脚本：`scripts/train_yolo26.py`（ultralytics 8.4.140，YOLO26n，imgsz 640，batch 16，RTX 3080，两集串行约 40 分钟）
- 卖家宣传的 mAP（0.92x）基于 v8s + 100 epoch；本项目为突出"最新 YOLO26"叙事用 26n 短训，指标见上，参赛材料如实标注训练配置

## 演示数据（DEMO_ONLY）

- `railmind/capabilities/panto/arc.py::synthetic_waveform`：合成电弧波形，仅在无真实数据时降级演示
- 前端模拟事件按钮生成的车辆/场景上下文
