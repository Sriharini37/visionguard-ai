"""
prepare_mvtec.py — Convert MVTec AD ground-truth anomaly masks into YOLO-format
bounding box labels, and emit a data.yaml for training.

MVTec AD layout (per category, e.g. "bottle"):
    bottle/train/good/*.png
    bottle/test/{defect_type}/*.png
    bottle/ground_truth/{defect_type}/*_mask.png

Usage:
    python src/detection/prepare_mvtec.py \
        --mvtec_root /path/to/mvtec_ad \
        --categories bottle cable capsule \
        --out_dir data/processed/mvtec_yolo
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def mask_to_yolo_boxes(mask_path: Path, class_id: int):
    """Find connected components in a binary mask and convert to normalized
    YOLO boxes: class_id, x_center, y_center, w, h (all in [0,1])."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    h_img, w_img = mask.shape
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 25:  # discard noise specks
            continue
        xc = (x + w / 2) / w_img
        yc = (y + h / 2) / h_img
        boxes.append((class_id, xc, yc, w / w_img, h / h_img))
    return boxes


def process_category(mvtec_root: Path, category: str, out_dir: Path, class_map: dict):
    cat_root = mvtec_root / category
    gt_root = cat_root / "ground_truth"

    for split, img_subdirs in [("train", ["good"]), ("test", None)]:
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        test_root = cat_root / split
        if not test_root.exists():
            continue

        subdirs = img_subdirs if img_subdirs else [d.name for d in test_root.iterdir() if d.is_dir()]

        for defect_type in subdirs:
            src_dir = test_root / defect_type
            if not src_dir.exists():
                continue
            for img_path in src_dir.glob("*.png"):
                dst_name = f"{category}_{defect_type}_{img_path.stem}"
                shutil.copy(img_path, img_out / f"{dst_name}.png")

                label_lines = []
                if defect_type != "good":
                    class_key = f"{category}_{defect_type}"
                    class_id = class_map.setdefault(class_key, len(class_map))
                    mask_path = gt_root / defect_type / f"{img_path.stem}_mask.png"
                    if mask_path.exists():
                        label_lines = [
                            " ".join(map(str, box))
                            for box in mask_to_yolo_boxes(mask_path, class_id)
                        ]

                with open(lbl_out / f"{dst_name}.txt", "w") as f:
                    f.write("\n".join(label_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec_root", type=str, required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    mvtec_root = Path(args.mvtec_root)
    out_dir = Path(args.out_dir)
    class_map = {}

    for category in args.categories:
        process_category(mvtec_root, category, out_dir, class_map)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/test",
        "names": {v: k for k, v in class_map.items()},
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"Wrote {len(class_map)} defect classes to {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
