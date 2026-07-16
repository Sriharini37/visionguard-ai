"""
simclr_pretrain.py — Self-supervised pretraining (SimCLR) on unlabeled
production-line images. Produces an encoder checkpoint that improves
downstream few-shot and supervised detection performance.

Usage:
    python src/self_supervised/simclr_pretrain.py --config configs/simclr.yaml
"""
import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import yaml


class SimCLRTransform:
    """Two correlated augmented views per image, per Chen et al. 2020."""

    def __init__(self, img_size: int):
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * img_size) | 1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self, backbone_name: str, proj_dim: int):
        super().__init__()
        backbone_fn = getattr(torchvision.models, backbone_name)
        backbone = backbone_fn(weights=None)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = ProjectionHead(in_dim, proj_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy loss (SimCLR)."""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature
    sim.fill_diagonal_(-1e9)

    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)

    return nn.functional.cross_entropy(sim, targets)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ImageFolder(cfg["unlabeled_data_dir"], transform=SimCLRTransform(cfg["img_size"]))
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                         num_workers=4, drop_last=True)

    model = SimCLRModel(cfg["backbone"], cfg["projection_dim"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                 momentum=0.9, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    mlflow.set_experiment(cfg["mlflow_experiment"])
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params(cfg)
        for epoch in range(cfg["epochs"]):
            model.train()
            epoch_loss = 0.0
            for (x1, x2), _ in loader:
                x1, x2 = x1.to(device), x2.to(device)
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = nt_xent_loss(z1, z2, cfg["temperature"])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(loader)
            mlflow.log_metric("nt_xent_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch + 1}/{cfg['epochs']}  loss={avg_loss:.4f}")

        torch.save(model.encoder.state_dict(), ckpt_dir / "best_encoder.pt")
        mlflow.log_artifact(str(ckpt_dir / "best_encoder.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
