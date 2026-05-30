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
import csv
import os
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SEVERITY_NAMES = {0: "轻度", 1: "中度", 2: "重度"}
SEVERITY_COLORS = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}

# 严重程度阈值: (轻度上限, 中度上限)，超过则为重度
# 基于检测框面积 / 图片面积的比例，启发式估算，无相机标定时仅供实验参考
SEVERITY_THRESHOLDS = {
    # 裂缝类 (class 0,1): 细长型，阈值更敏感
    0: (0.005, 0.03),
    1: (0.005, 0.03),
    # 龟裂、坑槽 (class 2,3): 面状，阈值宽松
    2: (0.02, 0.08),
    3: (0.02, 0.08),
}

DATASET_YAML = "data/rdd2022/dataset.yaml"
DEFAULT_MODEL = "yolo11l.pt"
DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"

# 分层置信度阈值: 裂缝类特征弱需低阈值提高召回，龟裂易误检需适当收紧
PER_CLASS_CONF = {
    0: 0.15,   # 横向裂缝 — 细长目标特征弱，低阈值
    1: 0.15,   # 纵向裂缝 — 同上
    2: 0.25,   # 龟裂 — 易误检背景纹理，收紧
    3: 0.20,   # 坑槽 — 中等
}


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
        seed=args.seed,
        warmup_epochs=5,
        # 数据增强 — 针对裂缝检测优化:
        #   mosaic/mixup 降低以保留裂缝线状结构不被截断
        #   hsv_v 扩大到 0.6 覆盖隧道暗光→正午强光变化
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.6,
        degrees=15.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
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
        workers=args.workers,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt" if results.save_dir else None
    if best_pt and best_pt.exists():
        print(f"\n最佳模型: {best_pt}")
        metrics = results.results_dict
        print(f"mAP@50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"mAP@50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")

    if results.save_dir:
        config_path = Path(results.save_dir) / "train_config.json"
        config_path.write_text(
            json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # 保存完整训练指标到 CSV
        metrics = results.results_dict
        if metrics:
            csv_path = Path(results.save_dir) / "metrics.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in metrics.items():
                    writer.writerow([k, v])
            print(f"指标已保存: {csv_path}")

    torch.cuda.empty_cache()


# ============================================================
# 推理
# ============================================================

def estimate_severity(box_area, img_area, cls_id):
    """基于框面积比的启发式严重程度估计。

    注意: 此方法依赖框面积/图片面积比例，无相机标定或尺度参考，
    同一病害在不同拍摄距离下会得到完全不同的严重等级。
    仅适用于固定拍摄条件的定性比较，不可用于精确评估。
    """
    ratio = box_area / max(img_area, 1)
    low, mid = SEVERITY_THRESHOLDS.get(cls_id, (0.01, 0.05))
    if ratio < low:
        return 0
    elif ratio < mid:
        return 1
    else:
        return 2


def filter_boxes_by_class(boxes, per_class_conf):
    """按类特定置信度阈值过滤检测框。

    boxes: ultralytics Results.boxes 对象或 None
    per_class_conf: {class_id: conf_threshold} 字典
    返回: 过滤后的 boxes（原地修改属性），以及保留的索引列表
    """
    if boxes is None or len(boxes) == 0:
        return boxes, []
    keep_idx = []
    for i, (cls_id, conf) in enumerate(zip(
        boxes.cls.cpu().int().tolist(),
        boxes.conf.cpu().float().tolist(),
    )):
        threshold = per_class_conf.get(cls_id, 0.25)
        if conf >= threshold:
            keep_idx.append(i)
    if keep_idx:
        boxes.cls = boxes.cls[keep_idx]
        boxes.conf = boxes.conf[keep_idx]
        boxes.xyxy = boxes.xyxy[keep_idx]
        if boxes.xywh is not None:
            boxes.xywh = boxes.xywh[keep_idx]
        if boxes.xywhn is not None:
            boxes.xywhn = boxes.xywhn[keep_idx]
    else:
        boxes.cls = boxes.cls[:0]
        boxes.conf = boxes.conf[:0]
        boxes.xyxy = boxes.xyxy[:0]
    return boxes, keep_idx


def draw_detections(img, results, img_area, per_class_conf=None):
    """在图像上绘制检测框和标签（原地修改）"""
    boxes = results.boxes
    if per_class_conf:
        boxes, keep_idx = filter_boxes_by_class(boxes, per_class_conf)
    for box in boxes:
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


def save_detection_results(results, img_path, save_dir, img_w, img_h,
                           save_txt, save_json, save_csv, per_class_conf=None):
    """保存检测结果为 TXT (YOLO 格式)、JSON 和/或 CSV"""
    stem = Path(img_path).stem
    boxes = results.boxes
    if per_class_conf:
        boxes, _ = filter_boxes_by_class(boxes, per_class_conf)
    if boxes is None or len(boxes) == 0:
        return

    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
            "confidence": round(conf, 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "bbox_norm": [round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)],
        })

    if save_txt:
        txt_path = Path(save_dir) / f"{stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for d in detections:
                cx, cy, bw, bh = d["bbox_norm"]
                f.write(f"{d['class_id']} {cx} {cy} {bw} {bh} {d['confidence']}\n")

    if save_json:
        json_path = Path(save_dir) / f"{stem}.json"
        json_path.write_text(
            json.dumps({"file": Path(img_path).name, "size": [img_w, img_h],
                        "detections": detections, "count": len(detections)},
                       indent=2, ensure_ascii=False),
            encoding="utf-8")

    if save_csv:
        csv_path = Path(save_dir) / f"{stem}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = ["class_id", "class_name", "confidence",
                          "x1", "y1", "x2", "y2",
                          "cx_norm", "cy_norm", "w_norm", "h_norm"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in detections:
                writer.writerow({
                    "class_id": d["class_id"],
                    "class_name": d["class_name"],
                    "confidence": d["confidence"],
                    "x1": d["bbox"][0], "y1": d["bbox"][1],
                    "x2": d["bbox"][2], "y2": d["bbox"][3],
                    "cx_norm": d["bbox_norm"][0],
                    "cy_norm": d["bbox_norm"][1],
                    "w_norm": d["bbox_norm"][2],
                    "h_norm": d["bbox_norm"][3],
                })


def predict_image_tiled(model, img_path, conf, save_dir, tile_size=640, overlap=0.3,
                        save_txt=False, save_json=False, save_csv=False, tta=False,
                        per_class_conf=None):
    """滑动窗口推理，适用于分辨率远大于 imgsz 的道路检测图像。

    将大图切分为重叠瓦片，逐片推理后在原图坐标系合并结果，
    对重叠区域做 NMS 去重。overlap=0.3 减少裂缝在瓦片边界被截断。
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return
    h, w = img.shape[:2]
    stride = int(tile_size * (1 - overlap))

    all_boxes = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            tile = img[y:y2, x:x2]
            results = model(tile, conf=conf, verbose=False, augment=tta)[0]
            if results.boxes is not None:
                for box in results.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    all_boxes.append({
                        "xyxy": [bx1 + x, by1 + y, bx2 + x, by2 + y],
                        "cls": int(box.cls[0].item()),
                        "conf": float(box.conf[0].item()),
                    })

    if not all_boxes:
        print(f"未检测到病害: {img_path}")
        out_path = Path(save_dir) / f"pred_{Path(img_path).name}"
        cv2.imwrite(str(out_path), img)
        return

    # 瓦片合并 NMS（IoU=0.3，因为边界截断框 IoU 仅 0.2-0.4）
    boxes_tensor = torch.tensor([b["xyxy"] for b in all_boxes])
    scores_tensor = torch.tensor([b["conf"] for b in all_boxes])
    cls_tensor = torch.tensor([b["cls"] for b in all_boxes])
    keep = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, 0.3)

    # 在原图上绘制保留的检测
    for idx in keep:
        b = all_boxes[int(idx)]
        cls_id = b["cls"]
        if per_class_conf:
            threshold = per_class_conf.get(cls_id, 0.25)
            if b["conf"] < threshold:
                continue
        x1, y1, x2, y2 = map(int, b["xyxy"])
        sev_id = estimate_severity((x2 - x1) * (y2 - y1), w * h, cls_id)
        color = SEVERITY_COLORS[sev_id]
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {b['conf']:.2f} [{SEVERITY_NAMES[sev_id]}]"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = Path(save_dir) / f"pred_{Path(img_path).name}"
    cv2.imwrite(str(out_path), img)
    print(f"瓦片推理完成: {len(all_boxes)} 个候选 → {len(keep)} 个 (NMS后) → {out_path}")

    if save_csv:
        stem = Path(img_path).stem
        csv_path = Path(save_dir) / f"{stem}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = ["class_id", "class_name", "confidence",
                          "x1", "y1", "x2", "y2",
                          "cx_norm", "cy_norm", "w_norm", "h_norm"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx in keep:
                b = all_boxes[int(idx)]
                cls_id = b["cls"]
                if per_class_conf:
                    if b["conf"] < per_class_conf.get(cls_id, 0.25):
                        continue
                x1, y1, x2, y2 = b["xyxy"]
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                writer.writerow({
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    "confidence": round(b["conf"], 4),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                    "cx_norm": round(cx, 6), "cy_norm": round(cy, 6),
                    "w_norm": round(bw, 6), "h_norm": round(bh, 6),
                })


def predict_image(model, img_path, conf, save_dir, save_txt=False, save_json=False,
                  save_csv=False, tta=False, per_class_conf=None):
    """单图推理。conf 为基准阈值（应设为 min(per_class_conf)），
    实际每类用 per_class_conf 进一步过滤。"""
    results = model(img_path, conf=conf, verbose=False, augment=tta)[0]
    img = results.orig_img
    if img is None:
        print(f"无法读取图片: {img_path}")
        return
    h, w = img.shape[:2]
    draw_detections(img, results, h * w, per_class_conf)
    out_path = Path(save_dir) / f"pred_{Path(img_path).name}"
    cv2.imwrite(str(out_path), img)
    print(f"结果已保存: {out_path}")
    save_detection_results(results, img_path, save_dir, w, h,
                          save_txt, save_json, save_csv, per_class_conf)


def predict_video(model, video_path, conf, save_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        fps = 30.0
    if w <= 0 or h <= 0:
        print(f"无法获取视频尺寸: {video_path}")
        return

    out_path = Path(save_dir) / f"pred_{Path(video_path).name}"
    fourcc_list = ["avc1", "mp4v", "XVID", "MJPG"]
    writer = None
    for fc in fourcc_list:
        fourcc = cv2.VideoWriter_fourcc(*fc)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if writer.isOpened():
            break
        writer.release()

    if writer is None or not writer.isOpened():
        print(f"无法创建输出视频文件，所有编码器均失败: {out_path}")
        return

    frame_count = 0
    try:
        for results in model(video_path, stream=True, conf=conf, verbose=False):
            frame = results.orig_img
            draw_detections(frame, results, h * w)
            writer.write(frame)
            frame_count += 1
    finally:
        writer.release()
    print(f"视频处理完成: {frame_count} 帧 → {out_path}")


def cmd_predict(args):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.to(args.device)

    # 分层阈值: 类特定过滤，裂缝低阈值提高召回
    use_class_conf = not args.no_class_conf
    per_class_conf = PER_CLASS_CONF if use_class_conf else None
    # 模型基准 conf 取用户值和类阈值最小值，确保不漏过低置信度裂缝
    min_class_conf = min(PER_CLASS_CONF.values()) if use_class_conf else args.conf
    base_conf = min(args.conf, min_class_conf)

    source = args.source
    if source.isdigit():
        cam_id = int(source)
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"无法打开摄像头 {cam_id}")
            return
        has_display = os.name == "nt" or os.environ.get("DISPLAY", "")
        print(f"摄像头 {cam_id} 已打开，按 'q' 退出")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=base_conf, verbose=False)[0]
                draw_detections(frame, results, frame.shape[0] * frame.shape[1],
                               per_class_conf)
                if has_display:
                    cv2.imshow("道路病害检测 — Road Disease Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    elif Path(source).exists():
        src_path = Path(source)
        if src_path.is_dir():
            print(f"来源是目录，请提供图片或视频文件路径，或将目录中的文件作为参数传入")
            return
        ext = src_path.suffix.lower()
        if ext in IMAGE_EXTS:
            if args.tile > 0:
                predict_image_tiled(model, source, base_conf, args.save_dir,
                                   tile_size=args.tile, save_txt=args.save_txt,
                                   save_json=args.save_json, save_csv=args.save_csv,
                                   tta=args.tta,
                                   per_class_conf=per_class_conf)
            else:
                predict_image(model, source, base_conf, args.save_dir,
                             args.save_txt, args.save_json, args.save_csv,
                             args.tta, per_class_conf)
        elif ext in (".mp4", ".avi", ".mov", ".mkv"):
            predict_video(model, source, base_conf, args.save_dir)
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

    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
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
        try:
            with open(tf, encoding="utf-8") as f:
                lines = f.readlines()
        except (OSError, UnicodeDecodeError):
            issues.append(f"{tf.name}: 无法读取文件")
            continue
        for i, line in enumerate(lines, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                issues.append(f"{tf.name}:L{i} 字段数={len(parts)} 期望5")
                continue
            try:
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                issues.append(f"{tf.name}:L{i} 含非数字字段")
                continue
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
        try:
            with open(tf, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            counter[int(parts[0])] += 1
                        except ValueError:
                            pass
        except (OSError, UnicodeDecodeError):
            pass
    total = sum(counter.values())
    print("\n类别分布统计:")
    for cid in sorted(counter):
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        pct = counter[cid] / total * 100 if total else 0
        print(f"  [{cid}] {name}: {counter[cid]} ({pct:.1f}%)")
    print(f"  总计: {total}")


def pre_analyze(args):
    """图片尺寸分析"""
    try:
        from PIL import Image as PILImage
    except ImportError:
        print("pre_analyze 需要 Pillow，请安装: pip install Pillow")
        return

    img_dir = Path(args.img_dir)
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    if not images:
        print("未找到图片文件")
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

    if not ws:
        print("未能成功读取任何图片")
        return

    print(f"图片数量: {len(images)}")
    print(f"尺寸范围: {min(ws)}x{min(hs)} ~ {max(ws)}x{max(hs)}")
    print(f"平均尺寸: {np.mean(ws):.0f}x{np.mean(hs):.0f}")
    print(f"建议 imgsz: {int(max(max(ws), max(hs)))}")


def cmd_validate(args):
    """在验证/测试集上评估模型"""
    model = YOLO(args.model)
    metrics = model.val(data=args.data, split=args.split, device=args.device,
                        imgsz=args.imgsz, batch=args.batch)
    print(f"\n验证结果 ({args.split}):")
    print(f"  mAP@50: {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    if hasattr(metrics.box, "ap_class_index"):
        for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap):
            name = CLASS_NAMES.get(int(i), f"class_{i}")
            print(f"  {name} AP@50-95: {ap:.4f}")
    torch.cuda.empty_cache()


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
    p_train.add_argument("--lr", type=float, default=0.0005,
                        help="学习率 (default: 0.0005，COCO微调用小lr)")
    p_train.add_argument("--device", default=DEFAULT_DEVICE)
    p_train.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)))
    p_train.add_argument("--seed", type=int, default=42)
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
    p_train.add_argument("--close-mosaic", type=int, default=15,
                        help="最后 N 个 epoch 关闭 mosaic (default: 15)")
    p_train.add_argument("--project", default="outputs")
    p_train.add_argument("--name", default="road_disease")

    # ---- predict ----
    p_pred = sub.add_parser("predict", help="推理检测")
    p_pred.add_argument("--model", default="outputs/road_disease/weights/best.pt")
    p_pred.add_argument("--source", required=True)
    p_pred.add_argument("--conf", type=float, default=0.3)
    p_pred.add_argument("--save-dir", default="outputs/results")
    p_pred.add_argument("--save-txt", action="store_true", help="保存 YOLO 格式检测结果")
    p_pred.add_argument("--save-json", action="store_true", help="保存 JSON 格式检测结果")
    p_pred.add_argument("--save-csv", action="store_true", help="保存 CSV 格式检测结果")
    p_pred.add_argument("--no-class-conf", action="store_true",
                       help="禁用分层置信度阈值，统一使用 --conf")
    p_pred.add_argument("--tta", action="store_true", help="测试时增强 (augment=True)")
    p_pred.add_argument("--tile", type=int, default=0, metavar="SIZE",
                       help="滑动窗口推理，将大图切分为 SIZE×SIZE 瓦片")
    p_pred.add_argument("--device", default=DEFAULT_DEVICE)

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

    # ---- validate ----
    p_val = sub.add_parser("validate", help="验证/评估模型")
    p_val.add_argument("--model", default="outputs/road_disease/weights/best.pt")
    p_val.add_argument("--data", default=DATASET_YAML)
    p_val.add_argument("--split", default="val", choices=["val", "test"])
    p_val.add_argument("--imgsz", type=int, default=640)
    p_val.add_argument("--batch", type=int, default=16)
    p_val.add_argument("--device", default=DEFAULT_DEVICE)

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
    elif args.command == "validate":
        cmd_validate(args)
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
