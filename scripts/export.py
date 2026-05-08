"""道路病害检测 — 模型导出脚本"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO11 模型导出")
    parser.add_argument("--model", default="outputs/road_disease/weights/best.pt",
                        help="训练好的模型权重路径")
    parser.add_argument("--format", default="onnx",
                        choices=["onnx", "tensorrt", "openvino", "tflite", "ncnn", "coreml"],
                        help="导出格式")
    parser.add_argument("--imgsz", type=int, default=640, help="导出输入尺寸")
    parser.add_argument("--half", action="store_true",
                        help="FP16 量化（仅 TensorRT / ONNX）")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 量化（仅 TensorRT）")
    parser.add_argument("--dynamic", action="store_true",
                        help="动态 batch 维度（仅 ONNX / TensorRT）")
    parser.add_argument("--workspace", type=float, default=4.0,
                        help="TensorRT 最大工作空间(GB)")
    args = parser.parse_args()

    model = YOLO(args.model)

    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "workspace": args.workspace,
    }

    if args.format in ("onnx", "tensorrt"):
        export_kwargs["half"] = args.half
        export_kwargs["dynamic"] = args.dynamic

    if args.format == "tensorrt":
        export_kwargs["int8"] = args.int8

    path = model.export(**{k: v for k, v in export_kwargs.items() if v is not None})
    print(f"模型已导出至: {path}")


if __name__ == "__main__":
    main()
