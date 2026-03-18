"""
Train an infrastructure-type classifier for /api/v1/predict/image.

Output: saves to `ig2/ml/weights/infra_type_resnet.pt` by default.

Dataset format (ImageFolder):
ig2/data/infra_type/
  train/
    road/
    bridge/
    pipeline/
    building/
    manhole/
  val/
    (same)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


logger = logging.getLogger("infraguard.train_infra_type")
logging.basicConfig(level=logging.INFO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="ig2/data/infra_type")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--output", type=str, default="ig2/ml/weights/infra_type_resnet.pt")
    ap.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    args = ap.parse_args()

    base = Path(args.data)
    train_dir = base / "train"
    val_dir = base / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(f"Expected {train_dir} and {val_dir} to exist.")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    classes = train_ds.classes
    logger.info("Classes: %s", classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(classes))
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(classes))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

        acc = correct / max(1, total)
        logger.info("Epoch %d/%d | val_acc=%.3f", epoch, args.epochs, acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model, str(out_path))
            logger.info("✅ Saved best model to %s", out_path)

    logger.info("Best val_acc=%.3f", best_acc)


if __name__ == "__main__":
    main()

