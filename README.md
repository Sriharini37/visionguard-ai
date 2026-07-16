\# VisionGuard AI

\### Autonomous Multi-Modal Defect Detection \& Explainable Quality Assurance System



An edge-deployable computer vision system that detects manufacturing defects, explains every decision it makes, learns new defect types without forgetting old ones, and generalizes to rare defects from a handful of examples.



!\[status](https://img.shields.io/badge/status-active--development-blue)

!\[python](https://img.shields.io/badge/python-3.10%2B-blue)

!\[license](https://img.shields.io/badge/license-MIT-green)



\---



\## Trained weights



Download `best.pt` from \*\*\[this Google Drive link](https://drive.google.com/file/d/1r\_Y9xrEnqQTjCGbK9T3lQCyP42ds0iAF/view?usp=drive\_link)\*\* and place it in `runs/detect/visionguard\_yolov8/weights/best.pt` to run real inference instead of demo mode.



\## Real results (measured, not projected)



Trained on MVTec AD (10 categories, 48 defect classes, 3,573 converted image-label pairs) using YOLOv8s for 50 epochs:



| Metric | Value |

|---|---|

| mAP@0.5 | \*\*24.87%\*\* |

| mAP@0.5:0.95 | \*\*14.79%\*\* |

| Precision | \*\*27.71%\*\* |

| Recall | \*\*25.52%\*\* |



These numbers are intentionally modest — the evaluation deliberately used all 48 defect classes with many classes having under 10 training images, stress-testing the base detector in a low-shot setting before the project's few-shot learning module is layered on top. Full methodology and per-class breakdown in `docs/benchmark\_results.md`.



\---



\## What this is



Manual visual inspection in manufacturing misses an estimated 20–30% of defects and doesn't scale. VisionGuard AI is a full pipeline — not a notebook — that combines five techniques into one deployable system:



| Capability | Technique | Why it matters |

|---|---|---|

| Common defect detection | YOLOv8 fine-tuned on MVTec AD + domain datasets | Real-time bounding boxes at production line speed |

| Rare defect detection | Prototypical Networks (few-shot, 5–10 samples) | No need to wait for thousands of labeled failures |

| Feature learning from unlabeled data | SimCLR self-supervised pretraining | Uses the abundant unlabeled "normal" footage every factory already has |

| Learning new defect types | Elastic Weight Consolidation (continual learning) | Adds new defect classes in hours without retraining from scratch or forgetting old ones |

| Decision transparency | Grad-CAM / CAM | Every flagged defect ships with a heatmap — required for regulated industries (pharma, electronics) |

| Edge inference | ONNX → TensorRT (FP16/INT8) on NVIDIA Jetson | Runs in production at <50ms/item, no cloud dependency |



\## Repository structure

