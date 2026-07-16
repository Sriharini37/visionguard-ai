"""
jetson_inference.py — Real-time inference loop on NVIDIA Jetson using a
pre-built TensorRT engine. Reads from a CSI/USB camera or video file, runs
detection, and prints/logs FPS + detections. Feeds the Streamlit dashboard
via a shared results queue/file in the full deployment.

Usage (on Jetson):
    python src/deployment/jetson_inference.py --engine runs/detect/best_int8.engine \
        --source /dev/video0
"""
import argparse
import time

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTEngine:
    def __init__(self, engine_path: str):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self.host_inputs, self.host_outputs = [], []
        self.device_inputs, self.device_outputs = [], []

        for binding in self.engine:
            shape = self.engine.get_tensor_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

    def infer(self, preprocessed_frame: np.ndarray):
        np.copyto(self.host_inputs[0], preprocessed_frame.ravel())
        cuda.memcpy_htod_async(self.device_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for host_out, device_out in zip(self.host_outputs, self.device_outputs):
            cuda.memcpy_dtoh_async(host_out, device_out, self.stream)
        self.stream.synchronize()
        return self.host_outputs


def preprocess(frame: np.ndarray, imgsz: int) -> np.ndarray:
    resized = cv2.resize(frame, (imgsz, imgsz))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw)


def run(engine_path: str, source: str, imgsz: int, conf_thresh: float):
    engine = TRTEngine(engine_path)
    cap = cv2.VideoCapture(0 if source == "0" else source)

    fps_history = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        input_tensor = preprocess(frame, imgsz)
        outputs = engine.infer(input_tensor)
        latency_ms = (time.time() - t0) * 1000
        fps_history.append(1000.0 / latency_ms if latency_ms > 0 else 0)

        # NOTE: outputs[0] is the raw YOLO output tensor; decode boxes/scores
        # here with your model's specific output layout (see Ultralytics
        # export docs for the exact tensor shape used).
        print(f"Latency: {latency_ms:.1f} ms | Rolling FPS: {np.mean(fps_history[-30:]):.1f}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--source", type=str, default="0",
                         help="Camera index (e.g. 0) or video file path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    run(args.engine, args.source, args.imgsz, args.conf)


if __name__ == "__main__":
    main()
