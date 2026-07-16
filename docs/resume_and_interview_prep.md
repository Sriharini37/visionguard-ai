# Resume, Portfolio & Interview Prep

## Resume header
`VisionGuard AI — Edge-Deployed Explainable Defect Detection System | PyTorch, YOLOv8, TensorRT, Streamlit, FastAPI`

## Resume bullets (customize the bolded numbers to your actual measured results — see docs/benchmark_results.md)

- Engineered an end-to-end computer vision pipeline combining YOLOv8 object detection, SimCLR self-supervised pretraining, and Prototypical Network few-shot learning to detect manufacturing defects across 4 industrial domains (electronics, pharma, textile, solar), achieving **[X]%** mAP@0.5 on the MVTec AD benchmark.
- Designed a continual learning module (Elastic Weight Consolidation + experience replay) enabling the system to learn new defect categories with **<[X]%** accuracy degradation on previously learned classes, removing the need for full model retraining.
- Implemented Grad-CAM explainability with quantitative heatmap-to-ground-truth-mask IoU evaluation, producing auditable visual justification for every inspection decision to support QA compliance workflows.
- Optimized inference for edge deployment via ONNX export and TensorRT INT8 quantization, improving throughput from **[X]** FPS to **[Y]** FPS on NVIDIA Jetson [model] with **<[X]%** accuracy loss.
- Built a real-time Streamlit dashboard with live defect analytics, downloadable shift logs, and alerting, plus a FastAPI serving layer for MES/SCADA integration.
- Containerized training and inference pipelines with Docker and tracked experiments with MLflow across **[N]** training runs for reproducible model development.

## Skills line
`Computer Vision, Deep Learning, PyTorch, YOLOv8, Self-Supervised Learning (SimCLR), Few-Shot Learning, Continual Learning (EWC), Explainable AI (Grad-CAM), ONNX, TensorRT, NVIDIA Jetson, Edge AI, MLOps (MLflow, Docker, DVC), Streamlit, FastAPI, OpenCV`

## LinkedIn / portfolio one-liner
"Built an edge-AI quality inspection system combining few-shot, self-supervised, and continual learning with built-in explainability — deployed with ONNX/TensorRT for real-time inference on NVIDIA Jetson."

## Interview questions to be ready for

1. **"Walk me through what happens when a new, never-seen defect type appears."**
   Answer with the actual flow: few-shot module classifies it from a handful of
   examples on day one → once enough examples accumulate, EWC/replay folds it
   into the main detector without forgetting old classes. Reference your actual
   forgetting-percentage number from `docs/benchmark_results.md`.

2. **"Why Grad-CAM and not just a confidence score?"**
   Confidence alone doesn't tell a QA auditor *why* something was flagged, which
   matters in regulated industries (pharma, aerospace-adjacent electronics).
   Show a real heatmap example and your IoU-vs-ground-truth-mask number.

3. **"How did you validate the TensorRT model didn't lose meaningful accuracy?"**
   Cite your actual before/after INT8 quantization accuracy delta — be ready to
   explain what a calibration set is and why its size/representativeness matters.

4. **"How would this scale to 50 production lines?"**
   Discuss: centralized MLflow experiment tracking, versioned model registry,
   per-line edge devices pulling the latest validated TensorRT engine, and a
   canary/rollback strategy for model updates.

5. **"What was your biggest technical challenge?"**
   Good honest answers: class imbalance in defect data (rare defects vastly
   outnumbered by "good" samples), or catastrophic forgetting during continual
   learning — describe how you measured it and what actually worked vs. didn't.

6. **"Is this running on real production data?"**
   Be honest. If you used MVTec AD + a self-collected demo dataset, say so
   plainly — it's still a legitimate, well-engineered system. Overclaiming
   "deployed in a real factory" when you haven't is the fastest way to lose
   credibility in a technical interview.

## What NOT to do
- Don't put the 99%+ / 45 FPS headline numbers from the original planning doc
  on your resume unless you've actually measured them yourself.
- Don't claim "proprietary company data" if it's a self-collected demo dataset —
  call it what it is; it's still a real, valuable skill to demonstrate.
