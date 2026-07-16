"""
ewc.py — Elastic Weight Consolidation for adding new defect classes without
catastrophic forgetting of previously learned ones.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks" (PNAS, 2017).

Usage:
    python src/continual_learning/ewc.py --config configs/continual.yaml
"""
import argparse
import copy
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml


class EWC:
    """Computes and stores the Fisher information matrix for a trained model
    on a given task, then penalizes drift on important parameters when
    training on a new task."""

    def __init__(self, model: nn.Module, old_task_loader: DataLoader, device: str):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(old_task_loader)

    def _compute_fisher(self, loader: DataLoader):
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        n_batches = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            n_batches += 1

        for n in fisher:
            fisher[n] /= max(n_batches, 1)
        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss


class ReplayBuffer:
    """Simple reservoir-sampling replay buffer of old-task examples, used
    alongside or instead of EWC."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.seen = 0

    def add(self, sample):
        self.seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            import random
            idx = random.randint(0, self.seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = sample

    def sample(self, batch_size: int):
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def train_new_task(cfg: dict, model, new_loader, old_loader_eval, device):
    ewc = EWC(model, old_loader_eval, device) if cfg["method"] in ("ewc", "ewc_replay") else None
    replay = ReplayBuffer(cfg["replay_buffer_size"]) if cfg["method"] in ("replay", "ewc_replay") else None

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    acc_before = evaluate(model, old_loader_eval, device) if cfg["eval_old_tasks"] else None

    model.train()
    for epoch in range(cfg["epochs"]):
        epoch_loss = 0.0
        for x, y in new_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            if ewc is not None:
                loss = loss + cfg["ewc_lambda"] * ewc.penalty(model)

            if replay is not None and len(replay.buffer) > 0:
                rx = torch.stack([s[0] for s in replay.sample(x.size(0))]).to(device)
                ry = torch.tensor([s[1] for s in replay.sample(x.size(0))]).to(device)
                loss = loss + criterion(model(rx), ry)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if replay is not None:
                for xi, yi in zip(x, y):
                    replay.add((xi.cpu(), yi.item()))

        print(f"Epoch {epoch + 1}/{cfg['epochs']}  loss={epoch_loss / len(new_loader):.4f}")

    acc_after = evaluate(model, old_loader_eval, device) if cfg["eval_old_tasks"] else None
    return model, acc_before, acc_after


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NOTE: plug in your actual classifier head + dataloaders for the
    # old-task eval set and the new-task training set here. This module is
    # dataset-agnostic by design so it can sit on top of either the YOLOv8
    # backbone's classification head or the few-shot encoder.
    raise SystemExit(
        "Wire up `model`, `new_loader`, and `old_loader_eval` for your dataset "
        "before running — see train_new_task() docstring."
    )


if __name__ == "__main__":
    main()
