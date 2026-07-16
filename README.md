# VisionGuard AI
### Autonomous Multi-Modal Defect Detection & Explainable Quality Assurance System

An edge-deployable computer vision system that detects manufacturing defects, explains every decision it makes, learns new defect types without forgetting old ones, and generalizes to rare defects from a handful of examples.

![status](https://img.shields.io/badge/status-active--development-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## What this is

Manual visual inspection in manufacturing misses an estimated 20–30% of defects and doesn't scale. VisionGuard AI is a full pipeline — not a notebook — that combines five techniques into one deployable system:

| Capability | Technique | Why it matters |
|---|---|---|
| Common defect detection | YOLOv8 fine-tuned on MVTec AD + domain datasets | Real-time bounding boxes at production line speed |
| Rare defect detection | Prototypical Networks (few-shot, 5–10 samples) | No need to wait for thousands of labeled failures |
| Feature learning from unlabeled data | SimCLR self-supervised pretraining | Uses the abundant unlabeled "normal" footage every factory already has |
| Learning new defect types | Elastic Weight Consolidation (continual learning) | Adds new defect classes in hours without retraining from scratch or forgetting old ones |
| Decision transparency | Grad-CAM / CAM | Every flagged defect ships with a heatmap — required for regulated industries (pharma, electronics) |
| Edge inference | ONNX → TensorRT (FP16/INT8) on NVIDIA Jetson | Runs in production at <50ms/item, no cloud dependency |

## Repository structure

```
visionguard-ai/
├── README.md
├── requirements.txt
├── configs/                    # YAML configs per experiment/module
├── data/
│   ├── raw/
│   ├── processed/
│   └── few_shot_support_sets/
├── src/
│   ├── detection/               # YOLOv8 training & inference
│   ├── self_supervised/         # SimCLR pretraining
│   ├── few_shot/                 # Prototypical networks
│   ├── continual_learning/       # EWC + replay buffer
│   ├── explainability/           # Grad-CAM utilities
│   └── deployment/
│       ├── export_onnx.py
│       ├── build_tensorrt.py
│       └── jetson_inference.py
├── dashboard/                    # Streamlit app (runs standalone in demo mode)
├── api/                          # FastAPI model-serving layer
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.deploy
├── tests/
└── docs/
    ├── architecture.md
    └── benchmark_results.md
```

## ⚡ Fastest path: one command

```bash
cd visionguard-ai
./quickstart.sh          # Mac/Linux
quickstart.bat           # Windows (double-click or run in cmd)
```

This creates a virtual environment, installs dependencies, generates a small
synthetic dataset, starts the FastAPI backend, and opens the Streamlit
dashboard — all in one step. No GPU, no dataset download, no manual config.

## Quickstart (demo mode, no GPU required, manual steps)

The dashboard ships with a **demo mode** that runs on any laptop with no trained weights and no GPU, so you can show the full UX in an interview immediately.

```bash
git clone <your-repo-url>
cd visionguard-ai
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

In a second terminal, also start the API so "Real model" mode has something to connect to:
```bash
source venv/bin/activate
uvicorn api.main:app --port 8000
```

## Quickstart (real training, requires GPU — Colab/Kaggle/local CUDA)

```bash
# 1. Fine-tune the YOLOv8 detector
python src/detection/train_yolo.py --config configs/detection.yaml

# 2. Self-supervised pretraining on unlabeled production images
python src/self_supervised/simclr_pretrain.py --config configs/simclr.yaml

# 3. Few-shot module on rare defect support sets
python src/few_shot/prototypical_network.py --config configs/few_shot.yaml

# 4. Grad-CAM sanity check
python src/explainability/gradcam_utils.py --weights runs/detect/best.pt --image sample.jpg

# 5. Continual learning — add a new defect class without forgetting old ones
python src/continual_learning/ewc.py --config configs/continual.yaml

# 6. Export for edge deployment
python src/deployment/export_onnx.py --weights runs/detect/best.pt
python src/deployment/build_tensorrt.py --onnx runs/detect/best.onnx --precision int8

# 7. Run on a Jetson device
python src/deployment/jetson_inference.py --engine runs/detect/best.engine
```

## API serving

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Datasets

| Dataset | Use |
|---|---|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Primary benchmark (15 categories, industrial objects & textures) |
| [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) | Steel surface defects, domain diversity |
| [Kaggle PCB Defect Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects) | Electronics use case |
| [ELPV Solar Panel Dataset](https://github.com/zae-bayern/elpv-dataset) | Solar panel use case |
| Self-collected | Textile/apparel swatches shot on phone camera, hand-annotated with Roboflow — documented in `docs/architecture.md` |

## Evaluation metrics reported

- Detection: mAP@0.5, mAP@0.5:0.95, precision/recall/F1 per class
- Few-shot: N-way K-shot accuracy, confusion matrix on rare classes
- Self-supervised: linear-probe accuracy vs. random init
- Continual learning: backward-transfer / forgetting metric
- Explainability: Grad-CAM–vs–ground-truth-mask IoU (MVTec AD provides masks)
- Edge: latency (ms), FPS, model size, accuracy delta after INT8 quantization

Full methodology and results in `docs/benchmark_results.md`.

## Honest scope note

Training the full pipeline to the headline numbers (99%+ mAP, 45 FPS on Jetson) requires: (1) a GPU for several days of training, (2) the datasets above downloaded locally, and (3) physical Jetson hardware for the edge benchmark. This repo gives you real, working code for every stage plus a live demo UI — plug in a GPU and the datasets to produce your own numbers, then report those (not placeholder numbers) on your resume and in interviews.
