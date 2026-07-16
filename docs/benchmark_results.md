# Benchmark Results

> **Fill this in with your own numbers after training.** The tables below are
> the exact structure to report — do not copy placeholder numbers from this
> template onto a resume. Recruiters and interviewers who know the space will
> ask you to walk through how a number was measured; an unearned number is a
> bigger risk than a modest, well-explained real one.

## 1. Detection performance (YOLOv8, MVTec AD test split)

| Defect class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| (fill in per class) | | | | |
| **Overall** | | | | |

Training command used: `python src/detection/train_yolo.py --config configs/detection.yaml`
Hardware: (e.g. 1x RTX 4090, or Colab T4/A100)
Epochs / wall-clock time: 

## 2. Self-supervised pretraining impact

| Init | Linear-probe accuracy | Downstream detection mAP@0.5 |
|---|---|---|
| Random init | | |
| ImageNet pretrained | | |
| SimCLR pretrained (this project) | | |

## 3. Few-shot rare-defect classification

N-way K-shot setting used: ___-way ___-shot

| Metric | Value |
|---|---|
| Eval accuracy | |
| Confusion matrix | (attach image/CSV) |

## 4. Continual learning (catastrophic forgetting check)

| | Old-class accuracy before new task | Old-class accuracy after new task | Forgetting |
|---|---|---|---|
| EWC | | | |
| Replay | | | |
| EWC + Replay | | | |
| Naive fine-tune (baseline, expected to forget badly) | | | |

Target: <3% forgetting for the continual-learning methods vs. the naive baseline.

## 5. Explainability

- Mean Grad-CAM / ground-truth-mask IoU across test set: ___
- Qualitative examples: (attach 3-5 overlay images per defect class)

## 6. Edge deployment (measure ON the actual Jetson device)

| Format | Precision | Latency (ms) | FPS | Model size (MB) | Accuracy vs. PyTorch baseline |
|---|---|---|---|---|---|
| PyTorch (.pt) | FP32 | | | | — |
| ONNX | FP32 | | | | |
| TensorRT | FP16 | | | | |
| TensorRT | INT8 | | | | |

Device used: (e.g. Jetson Orin Nano 8GB, JetPack version)
Calibration set size (INT8): ___ images

## 7. Business-impact estimate

- Manual baseline inspection rate: ___ items/sec, ___% defect miss rate (cite source or your own measurement)
- VisionGuard AI measured throughput: ___ items/sec
- Estimated inspection-time reduction: ___%
- Estimated false-negative reduction: ___% (requires a labeled validation set with known ground truth)

---

### How to fill this in without access to real factory hardware/data

1. Train and evaluate the detection + few-shot + self-supervised + continual
   learning sections on Colab/Kaggle with a free GPU, using MVTec AD (public,
   free to download for research use) — sections 1-5 are fully achievable this way.
2. For section 6, either (a) borrow/rent a Jetson Orin Nano for a weekend
   (~$250 device, widely used in academic project reviews), or (b) report
   ONNX Runtime CPU/GPU latency honestly and clearly label it as "cloud GPU
   benchmark; Jetson figures pending hardware access" — this is a defensible,
   honest position in an interview.
3. For section 7, be explicit that manual-baseline numbers are drawn from
   published industry sources (cite them) unless you ran your own comparison study.
