"""
Basic unit tests — run with: pytest tests/
These test the pure-logic pieces that don't require GPU, trained weights,
or external datasets, so they run in any CI pipeline.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explainability.gradcam_utils import heatmap_mask_iou


def test_heatmap_mask_iou_perfect_overlap():
    cam = np.ones((10, 10), dtype=np.float32)
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    assert heatmap_mask_iou(cam, mask, threshold=0.5) == 1.0


def test_heatmap_mask_iou_no_overlap():
    cam = np.zeros((10, 10), dtype=np.float32)
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    assert heatmap_mask_iou(cam, mask, threshold=0.5) == 0.0


def test_heatmap_mask_iou_partial_overlap():
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[:5, :] = 1.0
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:8, :] = 255
    iou = heatmap_mask_iou(cam, mask, threshold=0.5)
    assert 0.0 < iou < 1.0


def test_prototypical_loss_shapes():
    import torch
    from src.few_shot.prototypical_network import compute_prototypes, prototypical_loss

    n_way, k_shot, q_query, dim = 3, 5, 4, 16
    support_emb = torch.randn(n_way * k_shot, dim)
    support_labels = torch.arange(n_way).repeat_interleave(k_shot)
    query_emb = torch.randn(n_way * q_query, dim)
    query_labels = torch.arange(n_way).repeat_interleave(q_query)

    prototypes = compute_prototypes(support_emb, support_labels, n_way)
    assert prototypes.shape == (n_way, dim)

    loss, acc = prototypical_loss(support_emb, support_labels, query_emb, query_labels, n_way)
    assert loss.item() >= 0
    assert 0.0 <= acc.item() <= 1.0


def test_replay_buffer_capacity():
    from src.continual_learning.ewc import ReplayBuffer

    buf = ReplayBuffer(capacity=5)
    for i in range(20):
        buf.add((i, i))
    assert len(buf.buffer) == 5
