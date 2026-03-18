"""
Train a multi-label damage model for /api/v1/predict/image.

Output: saves a PyTorch model to `ig2/ml/weights/crack_cnn.pt` by default.

Dataset format: CSV with columns:
  image_path, crack, pothole, wear, water, deformation

Targets are expected as 0-100 scores (will be scaled to 0-1).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


logger = logging.getLogger("infraguard.train_damage_cnn")
logging.basicConfig(level=logging.INFO)


@dataclass
class Row:
    image_path: str
    y: List[float]  # 5 targets in [0, 1]


def _load_rows(csv_path: Path) -> List[Row]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = ["image_path", "crack", "pothole", "wear", "water", "deformation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    rows: List[Row] = []
    for _, r in df.iterrows():
        img = str(r["image_path"])
        y = [float(r[c]) / 100.0 for c in required[1:]]
        rows.append(Row(image_path=img, y=y))
    return rows


def _split(rows: List[Row], val_ratio: float, seed: int) -> Tuple[List[Row], List[Row]]:
    import random

    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio)) if len(rows) > 1 else 0
    return rows[n_val:], rows[:n_val]


class DamageDataset:
    def __init__(self, rows: List[Row], train: bool):
        from PIL import Image
        from torchvision import transforms
        import torch

        self.rows = rows
        self.Image = Image
        self.torch = torch
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)) if train else transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10) if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        pil = self.Image.open(r.image_path).convert("RGB")
        x = self.transform(pil)
        y = self.torch.tensor(r.y, dtype=self.torch.float32)
        return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="ig2/data/damage_labels.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="ig2/ml/weights/crack_cnn.pt")
    ap.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = _load_rows(csv_path)
    if len(rows) < 5:
        logger.warning("Very small dataset (%d rows). Expect poor accuracy.", len(rows))

    train_rows, val_rows = _split(rows, args.val_ratio, args.seed)
    logger.info("Rows: train=%d val=%d", len(train_rows), len(val_rows))

    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if args.backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 5)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 5)

    model = model.to(device)

    train_ds = DamageDataset(train_rows, train=True)
    val_ds = DamageDataset(val_rows, train=False) if val_rows else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_ds else None

    # BCE with soft targets in [0,1] works well for "percentage-like" labels.
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    best_val = None
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())

        train_loss = loss_sum / max(1, len(train_loader))

        val_loss = None
        if val_loader:
            model.eval()
            v_sum = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    v_sum += float(criterion(logits, yb).item())
            val_loss = v_sum / max(1, len(val_loader))

        logger.info("Epoch %d/%d | train_loss=%.4f | val_loss=%s", epoch, args.epochs, train_loss, f"{val_loss:.4f}" if val_loss is not None else "n/a")

        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            torch.save(model, str(out_path))
            logger.info("✅ Saved best model to %s", out_path)

    if best_val is None:
        torch.save(model, str(out_path))
        logger.info("✅ Saved model to %s", out_path)


if __name__ == "__main__":
    main()

