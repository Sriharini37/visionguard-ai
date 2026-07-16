"""
train_yolo.py — Fine-tune YOLOv8 on industrial defect data (MVTec AD + custom classes).

Usage:
    python src/detection/train_yolo.py --config configs/detection.yaml

Expects a YOLO-format dataset described by data.yaml_path (see
src/detection/prepare_mvtec.py for converting MVTec AD's anomaly masks
into YOLO bounding-box labels).
"""
import argparse
import yaml
from pathlib import Path

import mlflow
from ultralytics import YOLO


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(cfg: dict):
    model = YOLO(cfg["model"]["backbone"])

    mlflow.set_experiment(cfg["logging"]["mlflow_experiment"])
    with mlflow.start_run():
        mlflow.log_params({
            "backbone": cfg["model"]["backbone"],
            "epochs": cfg["train"]["epochs"],
            "batch_size": cfg["train"]["batch_size"],
            "img_size": cfg["data"]["img_size"],
            "optimizer": cfg["train"]["optimizer"],
            "lr0": cfg["train"]["lr0"],
        })

        results = model.train(
            data=cfg["data"]["yaml_path"],
            imgsz=cfg["data"]["img_size"],
            epochs=cfg["train"]["epochs"],
            batch=cfg["train"]["batch_size"],
            optimizer=cfg["train"]["optimizer"],
            lr0=cfg["train"]["lr0"],
            patience=cfg["train"]["patience"],
            device=cfg["train"]["device"],
            project=cfg["train"]["project"],
            name=cfg["train"]["name"],
            mosaic=cfg["augmentation"]["mosaic"],
            mixup=cfg["augmentation"]["mixup"],
            degrees=cfg["augmentation"]["degrees"],
            translate=cfg["augmentation"]["translate"],
            scale=cfg["augmentation"]["scale"],
            fliplr=cfg["augmentation"]["fliplr"],
            hsv_h=cfg["augmentation"]["hsv_h"],
            hsv_s=cfg["augmentation"]["hsv_s"],
            hsv_v=cfg["augmentation"]["hsv_v"],
        )

        # Validate and log final metrics
        metrics = model.val()
        mlflow.log_metrics({
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        })

        best_weights = Path(cfg["train"]["project"]) / cfg["train"]["name"] / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
