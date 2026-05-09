"""
道路病害目标检测 — YOLO11 全流程训练管线
RDD2022 数据集 | 4 类病害：横向裂缝、纵向裂缝、龟裂、坑槽

用法:
  python DAOLU.py train     --epochs 100 --device 0
  python DAOLU.py predict   --source road.jpg --conf 0.3
  python DAOLU.py export    --format onnx
  python DAOLU.py preprocess count --label-dir data/rdd2022/labels/train
"""

import argparse
import shutil
import random
import json
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# 配置
# ============================================================

CLASS_NAMES = {
    0: "横向裂缝 (Transverse Crack)",
    1: "纵向裂缝 (Longitudinal Crack)",
    2: "龟裂 (Alligator Crack)",
    3: "坑槽 (Pothole)",
}

SEVERITY_NAMES = {0: "轻度", 1: "中度", 2: "重度"}
SEVERITY_COLORS = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}

DATASET_YAML = "data/rdd2022/dataset.yaml"
DEFAULT_MODEL = "yolo11s.pt"


# ============================================================
# 训练
# ============================================================

def cmd_train(args):
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        device=args.device,
        patience=args.patience,
        resume=args.resume,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # 保存
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        save_period=10,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n最佳模型: {best_pt}")
        metrics = results.results_dict
        print(f"mAP@50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"mAP@50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")

    config_path = Path(results.save_dir) / "train_config.json"
    config_path.write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ============================================================
# 推理
# ============================================================

def estimate_severity(box_area, img_area):
    raise NotImplementedError


def predict_image(model, img_path, conf, save_dir):
    raise NotImplementedError


def predict_video(model, video_path, conf, save_dir):
    raise NotImplementedError


def cmd_predict(args):
    raise NotImplementedError


# ============================================================
# 模型导出
# ============================================================

def cmd_export(args):
    raise NotImplementedError


# ============================================================
# 数据预处理
# ============================================================

def pre_split(args):
    raise NotImplementedError


def pre_validate(args):
    raise NotImplementedError


def pre_count(args):
    raise NotImplementedError


def pre_analyze(args):
    raise NotImplementedError


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="YOLO11 道路病害检测全流程")
    sub = parser.add_subparsers(dest="command")

    # ---- train ----
    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--model", default=DEFAULT_MODEL,
                         choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
                                  "yolo11l.pt", "yolo11x.pt"])
    p_train.add_argument("--data", default=DATASET_YAML)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=0.001)
    p_train.add_argument("--device", default="0")
    p_train.add_argument("--patience", type=int, default=15)
    p_train.add_argument("--resume", action="store_true")
    cos_lr_group = p_train.add_mutually_exclusive_group()
    cos_lr_group.add_argument("--cos-lr", action="store_true", dest="cos_lr", default=True,
                              help="余弦退火学习率 (default: True)")
    cos_lr_group.add_argument("--no-cos-lr", action="store_false", dest="cos_lr",
                              help="禁用余弦退火")
    p_train.add_argument("--close-mosaic", type=int, default=10)
    p_train.add_argument("--project", default="outputs")
    p_train.add_argument("--name", default="road_disease")

    # ---- predict ----
    p_pred = sub.add_parser("predict", help="推理检测")
    p_pred.add_argument("--model", default="outputs/road_disease/weights/best.pt")
    p_pred.add_argument("--source", required=True)
    p_pred.add_argument("--conf", type=float, default=0.3)
    p_pred.add_argument("--save-dir", default="outputs/results")
    p_pred.add_argument("--device", default="0")

    # ---- export ----
    p_export = sub.add_parser("export", help="导出模型")
    p_export.add_argument("--model", default="outputs/road_disease/weights/best.pt")
    p_export.add_argument("--format", default="onnx",
                          choices=["onnx", "tensorrt", "openvino",
                                   "tflite", "ncnn", "coreml"])
    p_export.add_argument("--imgsz", type=int, default=640)
    p_export.add_argument("--half", action="store_true")
    p_export.add_argument("--int8", action="store_true")
    p_export.add_argument("--dynamic", action="store_true")
    p_export.add_argument("--workspace", type=float, default=4.0)

    # ---- preprocess ----
    p_pre = sub.add_parser("preprocess", help="数据预处理")
    pre_sub = p_pre.add_subparsers(dest="pre_cmd")

    p_split = pre_sub.add_parser("split", help="划分训练/验证集")
    p_split.add_argument("--img-dir", required=True)
    p_split.add_argument("--label-dir", required=True)
    p_split.add_argument("--train-img", default="data/images/train")
    p_split.add_argument("--val-img", default="data/images/val")
    p_split.add_argument("--train-label", default="data/labels/train")
    p_split.add_argument("--val-label", default="data/labels/val")
    p_split.add_argument("--ratio", type=float, default=0.8)

    p_val = pre_sub.add_parser("validate", help="校验标注格式")
    p_val.add_argument("--label-dir", required=True)
    p_val.add_argument("--num-classes", type=int, default=4)

    p_cnt = pre_sub.add_parser("count", help="类别统计")
    p_cnt.add_argument("--label-dir", required=True)

    p_ana = pre_sub.add_parser("analyze", help="图片尺寸分析")
    p_ana.add_argument("--img-dir", required=True)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "preprocess":
        if args.pre_cmd == "split":
            pre_split(args)
        elif args.pre_cmd == "validate":
            pre_validate(args)
        elif args.pre_cmd == "count":
            pre_count(args)
        elif args.pre_cmd == "analyze":
            pre_analyze(args)
        else:
            p_pre.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
