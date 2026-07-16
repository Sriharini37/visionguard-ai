"""
prototypical_network.py — Few-shot rare-defect classification via Prototypical
Networks (Snell et al., 2017). Handles the case where a defect type has only
5-10 labeled examples (e.g. a rare solder-bridge or hairline crack).

Usage:
    python src/few_shot/prototypical_network.py --config configs/few_shot.yaml
"""
import argparse
import random
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import yaml


class Encoder(nn.Module):
    """Embedding backbone — optionally initialized from a SimCLR checkpoint."""

    def __init__(self, backbone_name: str, pretrained_encoder_path: str | None):
        super().__init__()
        backbone_fn = getattr(torchvision.models, backbone_name)
        backbone = backbone_fn(weights="DEFAULT" if pretrained_encoder_path is None else None)
        self.out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if pretrained_encoder_path:
            state = torch.load(pretrained_encoder_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)

    def forward(self, x):
        return self.backbone(x)


class EpisodeSampler:
    """Samples N-way K-shot Q-query episodes from a folder-per-class dataset."""

    def __init__(self, root: str, img_size: int):
        self.root = Path(root)
        self.classes = [d for d in self.root.iterdir() if d.is_dir()]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def sample_episode(self, n_way: int, k_shot: int, q_query: int):
        chosen_classes = random.sample(self.classes, min(n_way, len(self.classes)))
        support, query = [], []
        for label, cls_dir in enumerate(chosen_classes):
            imgs = list(cls_dir.glob("*.*"))
            random.shuffle(imgs)
            s_imgs = imgs[:k_shot]
            q_imgs = imgs[k_shot:k_shot + q_query]
            support += [(self._load(p), label) for p in s_imgs]
            query += [(self._load(p), label) for p in q_imgs]
        return support, query

    def _load(self, path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.transform(img)


def compute_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, n_way: int):
    return torch.stack([embeddings[labels == c].mean(dim=0) for c in range(n_way)])


def prototypical_loss(support_emb, support_labels, query_emb, query_labels, n_way):
    prototypes = compute_prototypes(support_emb, support_labels, n_way)
    dists = torch.cdist(query_emb, prototypes)  # euclidean distance
    log_p_y = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p_y, query_labels)
    acc = (log_p_y.argmax(dim=1) == query_labels).float().mean()
    return loss, acc


def run_episode(model, sampler, n_way, k_shot, q_query, device):
    support, query = sampler.sample_episode(n_way, k_shot, q_query)
    s_x = torch.stack([s[0] for s in support]).to(device)
    s_y = torch.tensor([s[1] for s in support]).to(device)
    q_x = torch.stack([q[0] for q in query]).to(device)
    q_y = torch.tensor([q[1] for q in query]).to(device)

    s_emb = model(s_x)
    q_emb = model(q_x)
    return prototypical_loss(s_emb, s_y, q_emb, q_y, n_way)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Encoder(cfg["embedding_backbone"], cfg.get("pretrained_encoder_path")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sampler = EpisodeSampler(cfg["support_set_dir"], cfg["img_size"])

    mlflow.set_experiment(cfg["mlflow_experiment"])
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params(cfg)
        model.train()
        for ep in range(cfg["episodes_train"]):
            loss, acc = run_episode(model, sampler, cfg["n_way"], cfg["k_shot"],
                                     cfg["q_query"], device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (ep + 1) % 100 == 0:
                mlflow.log_metrics({"train_loss": loss.item(), "train_acc": acc.item()}, step=ep)
                print(f"Episode {ep + 1}/{cfg['episodes_train']}  "
                      f"loss={loss.item():.4f}  acc={acc.item():.4f}")

        # Held-out evaluation
        model.eval()
        accs = []
        with torch.no_grad():
            for _ in range(cfg["episodes_eval"]):
                _, acc = run_episode(model, sampler, cfg["n_way"], cfg["k_shot"],
                                      cfg["q_query"], device)
                accs.append(acc.item())
        mean_acc = sum(accs) / len(accs)
        mlflow.log_metric(f"{cfg['n_way']}way_{cfg['k_shot']}shot_eval_acc", mean_acc)
        print(f"{cfg['n_way']}-way {cfg['k_shot']}-shot eval accuracy: {mean_acc:.4f}")

        torch.save(model.state_dict(), ckpt_dir / "prototypical_encoder.pt")
        mlflow.log_artifact(str(ckpt_dir / "prototypical_encoder.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
