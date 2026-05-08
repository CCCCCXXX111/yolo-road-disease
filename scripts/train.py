"""道路病害检测 — 训练脚本"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO11 道路病害检测训练")
    parser.add_argument("--model", default="yolo11n.pt",
                        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
                        help="预训练模型权重")
    parser.add_argument("--data", default="data/dataset.yaml", help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--device", default="0", help="训练设备：0(GPU) / cpu")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    parser.add_argument("--resume", action="store_true", help="从中断点恢复训练")
    parser.add_argument("--cos-lr", action="store_true", default=True, help="余弦学习率衰减")
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="最后 N 轮关闭 mosaic 增强，提升精度")
    args = parser.parse_args()

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
        degrees=10.0,         # 小角度旋转（道路图像方向相对固定）
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # 保存设置
        project="outputs",
        name="road_disease",
        exist_ok=True,
        save=True,
        save_period=10,       # 每 10 个 epoch 保存一次
    )

    # 导出最佳模型
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n最佳模型已保存至: {best_pt}")
        print(f"验证集 mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")


if __name__ == "__main__":
    main()
