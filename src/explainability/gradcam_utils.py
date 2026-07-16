"""
gradcam_utils.py — Grad-CAM explainability overlays for defect detections.

Produces a heatmap showing which pixels drove the model's decision, overlaid
on the original image. Used in the Streamlit dashboard and for computing
heatmap-vs-ground-truth-mask IoU against MVTec AD's pixel-level masks.

Usage:
    python src/explainability/gradcam_utils.py --weights runs/detect/best.pt --image sample.jpg
"""
import argparse

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO


def get_gradcam_overlay(model_path: str, image_path: str, target_layer_index: int = -2):
    """Returns (original_bgr, overlay_bgr, heatmap_float) for a YOLOv8 model."""
    yolo = YOLO(model_path)
    torch_model = yolo.model
    torch_model.eval()

    # Target the last conv block before the detection head — good default
    # for CAM-style visualization on YOLOv8's backbone.
    target_layers = [list(torch_model.model.children())[target_layer_index]]

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_float = np.float32(img_resized) / 255.0

    input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)

    cam = GradCAM(model=torch_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    overlay = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    return img_resized, overlay, grayscale_cam


def heatmap_mask_iou(grayscale_cam: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """IoU between the thresholded Grad-CAM activation and a ground-truth
    binary defect mask (as provided by MVTec AD) — a proxy for whether the
    model is 'looking at' the actual defect region."""
    cam_bin = (grayscale_cam >= threshold).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    if gt_bin.shape != cam_bin.shape:
        gt_bin = cv2.resize(gt_bin, (cam_bin.shape[1], cam_bin.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    intersection = np.logical_and(cam_bin, gt_bin).sum()
    union = np.logical_or(cam_bin, gt_bin).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, default="gradcam_overlay.png")
    args = parser.parse_args()

    _, overlay, _ = get_gradcam_overlay(args.weights, args.image)
    cv2.imwrite(args.out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM overlay to {args.out}")


if __name__ == "__main__":
    main()
