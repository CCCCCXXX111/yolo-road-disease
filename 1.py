"""道路病害目标检测 — 基于 YOLO11 的全流程工具

用法:
  python 1.py train    --model yolo11n.pt --epochs 100
  python 1.py predict  --source road.jpg
  python 1.py export   --format onnx
  python 1.py preprocess split  --img-dir ./images --label-dir ./labels
"""

import argparse
import shutil
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# 全局配置
# ============================================================

CLASS_NAMES = {
    0: "横向裂缝",
    1: "纵向裂缝",
    2: "龟裂",
    3: "坑槽",
    4: "修补",
    5: "车辙",
    6: "沉陷",
    7: "松散",
}

SEVERITY_COLORS = {
    0: (0, 255, 0),     # 绿色：轻度
    1: (0, 255, 255),   # 黄色：中度
    2: (0, 0, 255),     # 红色：重度
}

DATASET_YAML = "data/dataset.yaml"
DEFAULT_MODEL = "yolo11n.pt"


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
        project="outputs",
        name="road_disease",
        exist_ok=True,
        save=True,
        save_period=10,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n最佳模型: {best_pt}")
        print(f"mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")


# ============================================================
# 推理
# ============================================================

def estimate_severity(box_area, img_area):
    ratio = box_area / img_area
    if ratio < 0.02:
        return 0, "轻度"
    elif ratio < 0.08:
        return 1, "中度"
    else:
        return 2, "重度"


def predict_image(model, img_path, conf, save_dir):
    results = model(img_path, conf=conf)[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取: {img_path}")
        return
    h, w = img.shape[:2]
    img_area = h * w

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        conf_val = box.conf[0].item()
        box_area = (x2 - x1) * (y2 - y1)
        sev_id, sev_label = estimate_severity(box_area, img_area)
        color = SEVERITY_COLORS[sev_id]
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf_val:.2f} [{sev_label}]"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = Path(save_dir) / f"result_{Path(img_path).name}"
    cv2.imwrite(str(out_path), img)
    print(f"结果已保存: {out_path}")


def predict_video(model, video_path, conf, save_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(save_dir) / f"result_{Path(video_path).name}"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf, verbose=False)[0]
        frame_area = h * w
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf_val = box.conf[0].item()
            box_area = (x2 - x1) * (y2 - y1)
            sev_id, sev_label = estimate_severity(box_area, frame_area)
            color = SEVERITY_COLORS[sev_id]
            label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf_val:.2f} [{sev_label}]"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"视频完成: {frame_count} 帧 → {out_path}")


def cmd_predict(args):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.to(args.device)

    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        print("按 'q' 退出")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=args.conf, verbose=False)[0]
            cv2.imshow("Road Disease Detection", results.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        ext = Path(args.source).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            predict_image(model, args.source, args.conf, args.save_dir)
        elif ext in (".mp4", ".avi", ".mov", ".mkv"):
            predict_video(model, args.source, args.conf, args.save_dir)
        else:
            print(f"不支持的格式: {ext}")


# ============================================================
# 模型导出
# ============================================================

def cmd_export(args):
    model = YOLO(args.model)
    kwargs = {
        k: v for k, v in [
            ("format", args.format),
            ("imgsz", args.imgsz),
            ("half", args.half),
            ("int8", args.int8),
            ("dynamic", args.dynamic),
            ("workspace", args.workspace),
        ] if v is not None and v is not False
    }
    path = model.export(**kwargs)
    print(f"模型已导出: {path}")


# ============================================================
# 数据预处理
# ============================================================

def pre_split(args):
    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    train_img = Path(args.train_img)
    val_img = Path(args.val_img)
    train_lbl = Path(args.train_label)
    val_lbl = Path(args.val_label)

    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in img_dir.iterdir()
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    random.shuffle(images)

    n_train = int(len(images) * args.ratio)
    for files, dst_img, dst_lbl in [
        (images[:n_train], train_img, train_lbl),
        (images[n_train:], val_img, val_lbl),
    ]:
        for img_file in files:
            shutil.copy2(img_file, dst_img / img_file.name)
            lbl_file = label_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                shutil.copy2(lbl_file, dst_lbl / lbl_file.name)

    print(f"训练集 {n_train} 张, 验证集 {len(images) - n_train} 张")


def pre_validate(args):
    label_dir = Path(args.label_dir)
    txt_files = list(label_dir.glob("*.txt"))
    issues = []
    for tf in txt_files:
        with open(tf) as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    issues.append(f"{tf.name}:L{i + 1} 字段数={len(parts)}")
                    continue
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                if cls_id < 0 or cls_id >= args.num_classes:
                    issues.append(f"{tf.name}:L{i + 1} 类别ID非法 cls={cls_id}")
                if any(c < 0 or c > 1 for c in coords):
                    issues.append(f"{tf.name}:L{i + 1} 坐标超出[0,1]")

    if issues:
        print(f"{len(issues)} 个问题:")
        for x in issues[:20]:
            print(f"  - {x}")
    else:
        print(f"检查通过: {len(txt_files)} 个文件均正常")


def pre_count(args):
    label_dir = Path(args.label_dir)
    counter = Counter()
    for tf in label_dir.glob("*.txt"):
        with open(tf) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counter[int(parts[0])] += 1
    print("\n类别统计:")
    for cid in range(8):
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        print(f"  {cid} ({name}): {counter.get(cid, 0)}")
    print(f"  总计: {sum(counter.values())}")


def pre_analyze(args):
    img_dir = Path(args.img_dir)
    images = list(img_dir.glob("*"))
    if not images:
        print("未找到图片")
        return
    ws, hs = [], []
    for f in images:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        hs.append(h)
        ws.append(w)
    print(f"共 {len(images)} 张, 尺寸 {min(ws)}x{min(hs)} ~ {max(ws)}x{max(hs)}, "
          f"平均 {np.mean(ws):.0f}x{np.mean(hs):.0f}, "
          f"建议 imgsz={int(max(max(ws), max(hs)))}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="YOLO11 道路病害检测")
    sub = parser.add_subparsers(dest="command")

    # ---- train ----
    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--model", default=DEFAULT_MODEL,
                         choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])
    p_train.add_argument("--data", default=DATASET_YAML)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=0.001)
    p_train.add_argument("--device", default="0")
    p_train.add_argument("--patience", type=int, default=15)
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--cos-lr", action="store_true", default=True)
    p_train.add_argument("--close-mosaic", type=int, default=10)

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
                          choices=["onnx", "tensorrt", "openvino", "tflite", "ncnn", "coreml"])
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

    p_val = pre_sub.add_parser("validate", help="检查标注格式")
    p_val.add_argument("--label-dir", required=True)
    p_val.add_argument("--num-classes", type=int, default=8)

    p_cnt = pre_sub.add_parser("count", help="类别统计")
    p_cnt.add_argument("--label-dir", required=True)

    p_ana = pre_sub.add_parser("analyze", help="图片信息分析")
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
