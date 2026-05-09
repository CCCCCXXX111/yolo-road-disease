# DAOLU.py — 道路病害检测 YOLO11 全流程管线

**日期:** 2026-05-09
**数据集:** RDD2022 (4 类道路病害)
**框架:** ultralytics YOLO11

## 概述

单文件 CLI 工具，覆盖道路病害目标检测的完整生命周期：训练、推理、模型导出、数据预处理。
默认使用 `yolo11s.pt` 预训练权重，RDD2022 数据集（已划分 train/val）。

## 架构

```
DAOLU.py (单文件)
  ├── 全局配置 (CLASS_NAMES, SEVERITY_*, DATASET_YAML, DEFAULT_MODEL)
  ├── train    → cmd_train()
  ├── predict  → cmd_predict() → predict_image() / predict_video()
  ├── export   → cmd_export()
  ├── preprocess → pre_split() / pre_validate() / pre_count() / pre_analyze()
  └── main()   → argparse 子命令分发
```

## 模块详述

### 1. train — 训练

- **模型:** 默认 `yolo11s.pt`，可选 n/s/m/l/x
- **数据:** 默认 `data/rdd2022/dataset.yaml`
- **默认超参:** epochs=100, imgsz=640, batch=16, lr=0.001, patience=15
- **学习率:** cos_lr=True（余弦退火）
- **数据增强:** HSV 抖动、旋转±10°、平移0.1、缩放0.5、水平翻转0.5、Mosaic 1.0、Mixup 0.1
- **Mosaic 关闭:** 最后 10 epoch
- **可视化:** TensorBoard（ultralytics 内置），`tensorboard --logdir outputs`
- **输出:**
  - 模型权重: `outputs/road_disease/weights/best.pt`, `last.pt`
  - 配置快照: `outputs/road_disease/train_config.json`
  - 打印 mAP@50 和 mAP@50-95

### 2. predict — 推理

- **输入类型分发:**
  - 数字串 → 摄像头实时检测
  - 图片 (.jpg/.png/.bmp/.tif) → 图片检测
  - 视频 (.mp4/.avi/.mov/.mkv) → 视频检测
- **严重程度评估:** 检测框面积 / 图片面积
  - < 2% → 轻度（绿框）
  - 2%-8% → 中度（黄框）
  - > 8% → 重度（红框）
- **标注格式:** `"类别名 置信度 [严重程度]"`
- **默认参数:** conf=0.3, 输出到 `outputs/results/`
- **视频输出:** mp4v 编码，保持原帧率

### 3. export — 模型导出

- **支持格式:** onnx（默认）、tensorrt、openvino、tflite、ncnn、coreml
- **可选:** --half (FP16)、--int8 (量化)、--dynamic (动态尺寸)
- **默认源:** `outputs/road_disease/weights/best.pt`

### 4. preprocess — 数据预处理

| 子命令 | 功能 | 关键参数 |
|--------|------|---------|
| split | 划分训练/验证集 | --ratio 0.8, 输出到 data/images/{train,val} 和 data/labels/{train,val} |
| validate | 校验 YOLO 标注格式 | --num-classes 4, 检查字段数/类别ID/坐标范围 |
| count | 类别分布统计 | 输出每类数量和占比 |
| analyze | 图片尺寸分析 | 输出 min/max/mean 和建议 imgsz |

## 类别定义

| ID | 中文 | 英文 |
|----|------|------|
| 0 | 横向裂缝 | Transverse Crack |
| 1 | 纵向裂缝 | Longitudinal Crack |
| 2 | 龟裂 | Alligator Crack |
| 3 | 坑槽 | Pothole |

## 依赖

- ultralytics (YOLO11)
- opencv-python
- numpy

## 实现决策

- **扁平函数式** 而非类封装：单文件脚本，argparse + 独立函数足够清晰
- **TensorBoard 内置**：ultralytics 自带 TensorBoard 回调，无需额外集成
- **数据增强固定**：作为合理默认值内嵌，不暴露 CLI 参数以降低复杂度
- **保留严重程度评估**：沿用面积比启发式方法，对实际应用有参考价值
