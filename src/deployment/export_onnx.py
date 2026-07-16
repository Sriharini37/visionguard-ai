"""
export_onnx.py — Export a trained YOLOv8 PyTorch model to ONNX for downstream
TensorRT conversion.

Usage:
    python src/deployment/export_onnx.py --weights runs/detect/best.pt \
        --imgsz 640 --opset 17
"""
import argparse

from ultralytics import YOLO


def export(weights: str, imgsz: int, opset: int, dynamic: bool, simplify: bool):
    model = YOLO(weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
    )
    print(f"Exported ONNX model to {onnx_path}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true", default=True)
    args = parser.parse_args()

    export(args.weights, args.imgsz, args.opset, args.dynamic, args.simplify)


if __name__ == "__main__":
    main()
