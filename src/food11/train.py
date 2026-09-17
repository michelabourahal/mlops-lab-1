from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

# mlflow prints an emoji-laden run summary on Windows terminals whose stdout defaults to a
# legacy code page (e.g. cp1252), which raises UnicodeEncodeError; force utf-8 to avoid it.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "food11"


def dataset_dir(name: str) -> Path:
    folder = "food11_processed" if name == "processed" else "food11_processed_mini"
    return DATA_ROOT / folder


def build_dataloaders(data_dir: Path, batch_size: int):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = ImageFolder(data_dir / "training", transform=transform)
    val_ds = ImageFolder(data_dir / "validation", transform=transform)
    test_ds = ImageFolder(data_dir / "evaluation", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ResNet18 classifier on Food-11")
    parser.add_argument("--dataset", choices=["processed", "mini"], default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = dataset_dir(args.dataset)

    train_loader, val_loader, test_loader, classes = build_dataloaders(data_dir, args.batch_size)
    model = build_model(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "model": "resnet18",
                "num_classes": len(classes),
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "test_samples": len(test_loader.dataset),
            }
        )

        for epoch in range(args.epochs):
            train_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(
                f"epoch {epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
            )

        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(f"test_accuracy={test_accuracy:.4f}")

        input_example, _ = next(iter(test_loader))
        mlflow.pytorch.log_model(model, "model", input_example=input_example.numpy())


if __name__ == "__main__":
    main()
