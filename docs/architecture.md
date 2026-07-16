# VisionGuard AI — Architecture

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VISIONGUARD AI PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│  [Camera/Edge Sensor] → [Preprocessing] → [Detection Core]           │
│         │                                        │                   │
│         │                          ┌─────────────┼─────────────┐     │
│         │                          ▼             ▼             ▼     │
│         │                    YOLOv8-based   Few-Shot     Self-Sup.   │
│         │                    Detector       Module       Encoder     │
│         │                    (common        (rare        (SimCLR on  │
│         │                    defects)        defects)     unlabeled) │
│         │                          └─────────────┬─────────────┘     │
│         │                                        ▼                   │
│         │                            [Fusion / Decision Layer]       │
│         │                          ┌─────────────┼─────────────┐     │
│         │                          ▼                           ▼     │
│         │                  [Grad-CAM / XAI]          [Continual      │
│         │                   Explainability            Learning       │
│         │                   Module]                   Module         │
│         │                          └─────────────┬─────────────┘     │
│         │                                        ▼                   │
│         │                          [TensorRT Optimized Inference]    │
│         │                                        ▼                   │
│         │                        [NVIDIA Jetson Edge Deployment]     │
│         └───────────────────────────►  [Streamlit Dashboard / API]   │
└─────────────────────────────────────────────────────────────────────┘
```

## Module responsibilities

1. **Detection core (`src/detection`)** — YOLOv8 fine-tuned on bounding-box-converted
   MVTec AD masks plus domain-specific datasets (PCB, ELPV, textile). Handles the
   common-defect majority case at production-line frame rates.

2. **Self-supervised encoder (`src/self_supervised`)** — SimCLR pretraining on
   large volumes of unlabeled "normal" production footage. The resulting encoder
   initializes both the few-shot module and (optionally) the detector backbone,
   improving performance when labeled data is scarce.

3. **Few-shot module (`src/few_shot`)** — Prototypical Networks classify rare
   defect types from as few as 5 labeled examples by computing class prototypes
   in embedding space and doing nearest-prototype classification.

4. **Continual learning (`src/continual_learning`)** — When a new defect type is
   identified in production, EWC penalizes drift on parameters important to
   previously learned classes (measured via Fisher information), and/or a replay
   buffer interleaves old-task samples. Target: <3% accuracy drop on old classes.

5. **Explainability (`src/explainability`)** — Grad-CAM overlays for every flagged
   defect, plus an IoU metric between the heatmap and MVTec AD's ground-truth
   pixel masks, quantifying whether the model is looking at the right region.

6. **Edge deployment (`src/deployment`)** — PyTorch → ONNX → TensorRT (FP16/INT8)
   for real-time inference on Jetson Orin Nano / Xavier NX.

7. **Serving layer (`dashboard/`, `api/`)** — Streamlit dashboard for live
   visual monitoring and shift reports; FastAPI layer for MES/SCADA integration.

## Data flow: adding a new defect type in production

1. QA flags a handful of newly-seen defect images (say 5–10).
2. They're added as a new class folder under `data/few_shot_support_sets/`.
3. The few-shot module immediately classifies future occurrences via
   nearest-prototype (no retraining needed for day-one coverage).
4. Once enough examples accumulate (e.g. 50+), `src/continual_learning/ewc.py`
   folds the new class into the main detector without forgetting old ones.

## "Proprietary" dataset framing

If real factory data isn't available, this is documented and built honestly as:
self-collected phone-camera images of a physical proxy (e.g. textile swatches,
PCB samples), hand-annotated with Roboflow/LabelImg, stored under
`data/raw/proprietary_demo/`. This is a legitimate and common way to demonstrate
"custom dataset collection and annotation" as a skill — it should be described
accurately in interviews as a self-collected demo dataset, not implied to be
real production data from an employer.

## Alerting integration point

The dashboard simulates alerts in-app. For a real deployment, wire
`dashboard/app.py`'s alert block to a webhook, e.g.:

```python
import requests
requests.post(SLACK_WEBHOOK_URL, json={"text": f"Defect rate {rate:.1%} exceeds threshold"})
```
