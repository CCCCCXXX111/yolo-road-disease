"""道路病害检测 — 推理脚本"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 类别中文名映射
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


def estimate_severity(box_area: float, img_area: float) -> tuple[int, str]:
    """根据目标占图像面积比例估算病害严重程度"""
    ratio = box_area / img_area
    if ratio < 0.02:
        return 0, "轻度"
    elif ratio < 0.08:
        return 1, "中度"
    else:
        return 2, "重度"


def process_image(model: YOLO, image_path: str, conf: float, save_dir: str):
    """处理单张图片并保存结果"""
    results = model(image_path, conf=conf)[0]
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
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

    out_path = Path(save_dir) / f"result_{Path(image_path).name}"
    cv2.imwrite(str(out_path), img)
    print(f"结果已保存: {out_path}")


def process_video(model: YOLO, video_path: str, conf: float, save_dir: str):
    """处理视频并保存结果"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(save_dir) / f"result_{Path(video_path).name}"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h),
    )

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
    print(f"视频处理完成，共 {frame_count} 帧，输出: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO11 道路病害检测推理")
    parser.add_argument("--model", default="outputs/road_disease/weights/best.pt",
                        help="模型权重路径")
    parser.add_argument("--source", required=True, help="输入：图片/视频路径 或 摄像头编号(0)")
    parser.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--save-dir", default="outputs/results", help="输出目录")
    parser.add_argument("--device", default="0", help="推理设备")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.to(args.device)

    if args.source.isdigit():
        # 摄像头实时检测
        cap = cv2.VideoCapture(int(args.source))
        print("按 'q' 退出实时检测")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=args.conf, verbose=False)[0]
            annotated = results.plot()
            cv2.imshow("Road Disease Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        ext = Path(args.source).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            process_image(model, args.source, args.conf, args.save_dir)
        elif ext in (".mp4", ".avi", ".mov", ".mkv"):
            process_video(model, args.source, args.conf, args.save_dir)
        else:
            print(f"不支持的格式: {ext}")


if __name__ == "__main__":
    main()
