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
import torch
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
DEFAULT_MODEL = "yolo11l.pt"


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
        amp=args.amp,
        seed=42,
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
        label_smoothing=0.1,
        # 验证 & 可视化
        val=True,
        plots=True,
        # 保存
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        save_period=10,
        workers=4,
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

    torch.cuda.empty_cache()


# ============================================================
# 推理
# ============================================================

def estimate_severity(box_area, img_area, cls_id):
    ratio = box_area / max(img_area, 1)
    if cls_id in (0, 1):  # 裂缝类：细长但面积小，采用更敏感阈值
        if ratio < 0.005:
            return 0
        elif ratio < 0.03:
            return 1
        else:
            return 2
    else:  # 龟裂、坑槽：面积比判定
        if ratio < 0.02:
            return 0
        elif ratio < 0.08:
            return 1
        else:
            return 2


def draw_detections(img, results, img_area):
    """在图像上绘制检测框和标签（原地修改）"""
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        conf_val = box.conf[0].item()
        box_area = (x2 - x1) * (y2 - y1)
        sev_id = estimate_severity(box_area, img_area, cls_id)
        color = SEVERITY_COLORS[sev_id]
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf_val:.2f} [{SEVERITY_NAMES[sev_id]}]"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def predict_image(model, img_path, conf, save_dir):
    results = model(img_path, conf=conf, verbose=False)[0]
    img = results.orig_img
    if img is None:
        print(f"无法读取图片: {img_path}")
        return
    h, w = img.shape[:2]
    draw_detections(img, results, h * w)
    out_path = Path(save_dir) / f"pred_{Path(img_path).name}"
    cv2.imwrite(str(out_path), img)
    print(f"结果已保存: {out_path}")


def predict_video(model, video_path, conf, save_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_path = Path(save_dir) / f"pred_{Path(video_path).name}"
    ext = Path(video_path).suffix.lower()
    fourcc_map = {".mp4": "avc1", ".avi": "XVID", ".mov": "avc1", ".mkv": "avc1"}
    fourcc = cv2.VideoWriter_fourcc(*fourcc_map.get(ext, "mp4v"))
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    for results in model(video_path, stream=True, conf=conf, verbose=False):
        frame = results.orig_img
        draw_detections(frame, results, h * w)
        writer.write(frame)
        frame_count += 1

    writer.release()
    print(f"视频处理完成: {frame_count} 帧 → {out_path}")


def cmd_predict(args):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.to(args.device)

    source = args.source
    if source.isdigit():
        cam_id = int(source)
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"无法打开摄像头 {cam_id}")
            return
        print(f"摄像头 {cam_id} 已打开，按 'q' 退出")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=args.conf, verbose=False)[0]
            draw_detections(frame, results, frame.shape[0] * frame.shape[1])
            cv2.imshow("道路病害检测 — Road Disease Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    elif Path(source).exists():
        ext = Path(source).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            predict_image(model, source, args.conf, args.save_dir)
        elif ext in (".mp4", ".avi", ".mov", ".mkv"):
            predict_video(model, source, args.conf, args.save_dir)
        else:
            print(f"不支持的格式: {ext}")
    else:
        print(f"来源不可用: {source}")


# ============================================================
# 模型导出
# ============================================================

def cmd_export(args):
    model = YOLO(args.model)
    kwargs = {}
    for name in ["format", "imgsz", "half", "int8", "dynamic", "workspace"]:
        val = getattr(args, name, None)
        if val is not None and val is not False:
            kwargs[name] = val
    exported_path = model.export(**kwargs)
    print(f"模型已导出: {exported_path}")
    torch.cuda.empty_cache()


# ============================================================
# 数据预处理
# ============================================================

def pre_split(args):
    """划分训练/验证集"""
    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    train_img_dir = Path(args.train_img)
    val_img_dir = Path(args.val_img)
    train_lbl_dir = Path(args.train_label)
    val_lbl_dir = Path(args.val_label)

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in exts])
    random.seed(args.seed)
    random.shuffle(images)

    n_train = int(len(images) * args.ratio)
    for files, dst_img, dst_lbl in [
        (images[:n_train], train_img_dir, train_lbl_dir),
        (images[n_train:], val_img_dir, val_lbl_dir),
    ]:
        for img_file in files:
            shutil.copy2(img_file, dst_img / img_file.name)
            lbl_file = label_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                shutil.copy2(lbl_file, dst_lbl / lbl_file.name)

    print(f"训练集: {n_train} 张 | 验证集: {len(images) - n_train} 张")


def pre_validate(args):
    """校验 YOLO 标注格式"""
    label_dir = Path(args.label_dir)
    txt_files = list(label_dir.glob("*.txt"))
    issues = []
    for tf in txt_files:
        with open(tf) as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    issues.append(f"{tf.name}:L{i} 字段数={len(parts)} 期望5")
                    continue
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                if cls_id < 0 or cls_id >= args.num_classes:
                    issues.append(f"{tf.name}:L{i} 类别ID非法 cls={cls_id}")
                if any(c < 0 or c > 1 for c in coords):
                    issues.append(f"{tf.name}:L{i} 坐标超出 [0,1] 范围")

    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for x in issues[:30]:
            print(f"  - {x}")
    else:
        print(f"校验通过: {len(txt_files)} 个标注文件格式正确")


def pre_count(args):
    """类别分布统计"""
    label_dir = Path(args.label_dir)
    counter = Counter()
    for tf in label_dir.glob("*.txt"):
        with open(tf) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counter[int(parts[0])] += 1
    total = sum(counter.values())
    print("\n类别分布统计:")
    for cid in sorted(counter):
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        pct = counter[cid] / total * 100 if total else 0
        print(f"  [{cid}] {name}: {counter[cid]} ({pct:.1f}%)")
    print(f"  总计: {total}")


def pre_analyze(args):
    """图片尺寸分析"""
    from PIL import Image as PILImage

    img_dir = Path(args.img_dir)
    images = list(img_dir.glob("*"))
    if not images:
        print("未找到图片")
        return
    ws, hs = [], []
    for f in images:
        try:
            with PILImage.open(f) as im:
                w, h = im.size
        except Exception:
            continue
        hs.append(h)
        ws.append(w)
    print(f"图片数量: {len(images)}")
    print(f"尺寸范围: {min(ws)}x{min(hs)} ~ {max(ws)}x{max(hs)}")
    print(f"平均尺寸: {np.mean(ws):.0f}x{np.mean(hs):.0f}")
    print(f"建议 imgsz: {int(max(max(ws), max(hs)))}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="YOLO11 道路病害检测全流程")
    sub = parser.add_subparsers(dest="command")

    # ---- train ----
    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--model", default=DEFAULT_MODEL)
    p_train.add_argument("--data", default=DATASET_YAML)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=28)
    p_train.add_argument("--lr", type=float, default=0.001)
    p_train.add_argument("--device", default="0")
    p_train.add_argument("--patience", type=int, default=15)
    p_train.add_argument("--resume", action="store_true")
    cos_lr_group = p_train.add_mutually_exclusive_group()
    cos_lr_group.add_argument("--cos-lr", action="store_true", dest="cos_lr", default=True,
                              help="余弦退火学习率 (default: True)")
    cos_lr_group.add_argument("--no-cos-lr", action="store_false", dest="cos_lr",
                              help="禁用余弦退火")
    amp_group = p_train.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", action="store_true", dest="amp", default=True,
                           help="自动混合精度训练 (default: True)")
    amp_group.add_argument("--no-amp", action="store_false", dest="amp",
                           help="禁用混合精度")
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
    p_split.add_argument("--seed", type=int, default=42)

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
