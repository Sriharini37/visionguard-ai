"""
generate_synthetic_demo_data.py — Generates a small synthetic labeled dataset
(good + defect images with YOLO-format labels) so the full training pipeline
can be smoke-tested end-to-end without downloading MVTec AD first.

This is NOT a substitute for real training data — it exists so you can verify
`train_yolo.py`, the few-shot episode sampler, and the Grad-CAM script all run
correctly before pointing them at real datasets.

Usage:
    python src/detection/generate_synthetic_demo_data.py --out_dir data/processed/synthetic_demo --n_images 200
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEFECT_CLASSES = ["scratch", "dent", "crack", "discoloration"]


def make_image(rng, has_defect: bool):
    img = Image.new("RGB", (416, 416), color=(230, 230, 235))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([60, 60, 356, 356], radius=20, fill=(200, 205, 215),
                            outline=(150, 150, 160), width=3)

    label_lines = []
    if has_defect:
        class_id = rng.integers(0, len(DEFECT_CLASSES))
        x1, y1 = rng.integers(90, 250), rng.integers(90, 250)
        w, h = rng.integers(25, 70), rng.integers(25, 70)
        x2, y2 = x1 + w, y1 + h
        draw.ellipse([x1, y1, x2, y2], fill=(120, 40, 40))

        xc, yc = (x1 + x2) / 2 / 416, (y1 + y2) / 2 / 416
        nw, nh = w / 416, h / 416
        label_lines.append(f"{class_id} {xc:.4f} {yc:.4f} {nw:.4f} {nh:.4f}")

    noise = rng.normal(0, 6, (416, 416, 3))
    arr = np.clip(np.array(img).astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr), label_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_images", type=int, default=200)
    parser.add_argument("--defect_rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)

    for split, n in [("train", int(args.n_images * 0.8)), ("val", int(args.n_images * 0.2))]:
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            has_defect = rng.random() < args.defect_rate
            img, labels = make_image(rng, has_defect)
            img.save(img_dir / f"{split}_{i:04d}.png")
            with open(lbl_dir / f"{split}_{i:04d}.txt", "w") as f:
                f.write("\n".join(labels))

    yaml_content = f"""path: {out_dir.resolve()}
train: images/train
val: images/val
names:
  0: scratch
  1: dent
  2: crack
  3: discoloration
"""
    with open(out_dir / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Generated synthetic dataset at {out_dir} — ready for a smoke-test training run:")
    print(f"  python src/detection/train_yolo.py --config configs/detection.yaml "
          f"(after pointing configs/detection.yaml's data.yaml_path at {out_dir / 'data.yaml'})")


if __name__ == "__main__":
    main()
