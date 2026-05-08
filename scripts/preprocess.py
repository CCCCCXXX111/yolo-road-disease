"""道路病害检测 — 数据预处理工具

支持功能：
  - 数据集划分（train/val）
  - 标注文件检查与修复
  - 类别标签统计
  - 图片统计与异常检测
"""

import argparse
import shutil
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

CLASS_NAMES = {
    0: "横向裂缝", 1: "纵向裂缝", 2: "龟裂", 3: "坑槽",
    4: "修补", 5: "车辙", 6: "沉陷", 7: "松散",
}


def split_dataset(img_dir: str, label_dir: str, train_dir: str, val_dir: str,
                  train_label_dir: str, val_label_dir: str, ratio: float):
    """按比例划分训练集和验证集"""
    img_path = Path(img_dir)
    label_path = Path(label_dir)
    train_img = Path(train_dir)
    val_img = Path(val_dir)
    train_lbl = Path(train_label_dir)
    val_lbl = Path(val_label_dir)

    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in img_path.iterdir()
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    random.shuffle(images)

    split_idx = int(len(images) * ratio)
    train_files = images[:split_idx]
    val_files = images[split_idx:]

    for files, dst_img, dst_lbl in [
        (train_files, train_img, train_lbl),
        (val_files, val_img, val_lbl),
    ]:
        for img_file in files:
            label_file = label_path / f"{img_file.stem}.txt"
            shutil.copy2(img_file, dst_img / img_file.name)
            if label_file.exists():
                shutil.copy2(label_file, dst_lbl / label_file.name)

    print(f"数据集划分完成: 训练集 {len(train_files)} 张, 验证集 {len(val_files)} 张")


def validate_labels(label_dir: str, num_classes: int = 8):
    """检查标注文件格式是否正确，修复异常值"""
    label_path = Path(label_dir)
    txt_files = list(label_path.glob("*.txt"))
    issues = []

    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    issues.append(f"{txt_file.name}:L{i + 1} 字段数={len(parts)}")
                    continue

                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                if cls_id < 0 or cls_id >= num_classes:
                    issues.append(f"{txt_file.name}:L{i + 1} 类别ID无效 cls={cls_id}")
                if any(c < 0 or c > 1 for c in coords):
                    issues.append(f"{txt_file.name}:L{i + 1} 坐标超出[0,1]范围: {coords}")

    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for issue in issues[:20]:  # 最多显示20个
            print(f"  - {issue}")
    else:
        print(f"检查通过: {len(txt_files)} 个标注文件均正常")
    return issues


def count_labels(label_dir: str):
    """统计各类别数量"""
    label_path = Path(label_dir)
    counter = Counter()

    for txt_file in label_path.glob("*.txt"):
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counter[int(parts[0])] += 1

    print("\n类别统计:")
    for cls_id in range(8):
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        print(f"  {cls_id} ({name}): {counter.get(cls_id, 0)} 个实例")
    print(f"  总计: {sum(counter.values())} 个实例")


def analyze_images(img_dir: str):
    """分析图片尺寸分布和异常"""
    img_path = Path(img_dir)
    images = list(img_path.glob("*"))
    if not images:
        print("未找到图片文件")
        return

    widths, heights = [], []
    zero_files = []

    for img_file in images:
        img = cv2.imread(str(img_file))
        if img is None:
            zero_files.append(img_file.name)
            continue
        h, w = img.shape[:2]
        heights.append(h)
        widths.append(w)

    print(f"\n图片分析 (共 {len(images)} 张):")
    if zero_files:
        print(f"  损坏/无法读取: {len(zero_files)} 个 ({zero_files[:5]}...)")
    if widths:
        print(f"  尺寸范围: {min(widths)}x{min(heights)} ~ {max(widths)}x{max(heights)}")
        print(f"  平均尺寸: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
        print(f"  建议 imgsz: {int(max(max(widths), max(heights)))}")


def main():
    parser = argparse.ArgumentParser(description="道路病害数据预处理")
    sub = parser.add_subparsers(dest="command")

    p_split = sub.add_parser("split", help="划分训练/验证集")
    p_split.add_argument("--img-dir", required=True, help="原始图片目录")
    p_split.add_argument("--label-dir", required=True, help="原始标注目录")
    p_split.add_argument("--train-img", default="data/images/train")
    p_split.add_argument("--val-img", default="data/images/val")
    p_split.add_argument("--train-label", default="data/labels/train")
    p_split.add_argument("--val-label", default="data/labels/val")
    p_split.add_argument("--ratio", type=float, default=0.8, help="训练集比例")

    p_validate = sub.add_parser("validate", help="检查标注文件格式")
    p_validate.add_argument("--label-dir", required=True)
    p_validate.add_argument("--num-classes", type=int, default=8)

    p_count = sub.add_parser("count", help="统计各类别数量")
    p_count.add_argument("--label-dir", required=True)

    p_analyze = sub.add_parser("analyze", help="分析图片信息")
    p_analyze.add_argument("--img-dir", required=True)

    args = parser.parse_args()

    if args.command == "split":
        split_dataset(args.img_dir, args.label_dir,
                      args.train_img, args.val_img,
                      args.train_label, args.val_label, args.ratio)
    elif args.command == "validate":
        validate_labels(args.label_dir, args.num_classes)
    elif args.command == "count":
        count_labels(args.label_dir)
    elif args.command == "analyze":
        analyze_images(args.img_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
