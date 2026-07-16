"""
build_tensorrt.py — Convert an ONNX model into a TensorRT engine with FP16 or
INT8 quantization, for deployment on NVIDIA Jetson (Orin Nano / Xavier NX).

Must be run ON the Jetson (or a machine with matching TensorRT + CUDA
versions) — TensorRT engines are hardware/version-locked.

INT8 mode requires a small representative calibration dataset (a folder of
~200-500 typical production images) to compute per-layer quantization scales.

Usage:
    python src/deployment/build_tensorrt.py --onnx runs/detect/best.onnx \
        --precision int8 --calib_dir data/processed/calibration_images
"""
import argparse
import glob
import os

import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator that feeds a batch of representative production images."""

    def __init__(self, calib_dir: str, imgsz: int, batch_size: int, cache_file: str):
        super().__init__()
        import cv2
        self.cv2 = cv2
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.image_paths = glob.glob(os.path.join(calib_dir, "*"))
        self.index = 0
        self.device_input = None

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        import pycuda.driver as cuda
        if self.index + self.batch_size > len(self.image_paths):
            return None

        batch_imgs = []
        for p in self.image_paths[self.index: self.index + self.batch_size]:
            img = self.cv2.imread(p)
            img = self.cv2.resize(img, (self.imgsz, self.imgsz))
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            batch_imgs.append(img)
        batch = np.ascontiguousarray(np.stack(batch_imgs))

        if self.device_input is None:
            self.device_input = cuda.mem_alloc(batch.nbytes)
        cuda.memcpy_htod(self.device_input, batch)
        self.index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine(onnx_path: str, precision: str, calib_dir: str, imgsz: int,
                  out_path: str, workspace_gb: int = 4):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("Warning: platform reports no fast FP16 support; proceeding anyway.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # fallback for unsupported layers
        config.int8_calibrator = EntropyCalibrator(
            calib_dir, imgsz, batch_size=8, cache_file=out_path + ".calibration_cache"
        )

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    with open(out_path, "wb") as f:
        f.write(engine_bytes)
    print(f"Saved TensorRT engine ({precision}) to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--calib_dir", type=str, default=None,
                         help="Required for --precision int8")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_path = args.out or args.onnx.replace(".onnx", f"_{args.precision}.engine")

    if args.precision == "int8" and not args.calib_dir:
        raise ValueError("--calib_dir is required for INT8 calibration")

    build_engine(args.onnx, args.precision, args.calib_dir, args.imgsz, out_path)


if __name__ == "__main__":
    main()
