"""Fine-tune ResNet18 and train it as a fixed feature extractor."""
import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from src.common import device_for_run, evaluate, fit, save_history, save_plots, seed_everything


DATA_URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"


def ensure_dataset(data_dir: Path) -> Path:
    root = data_dir / "hymenoptera_data"
    if (root / "train").is_dir() and (root / "val").is_dir():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "hymenoptera_data.zip"
    print(f"Downloading Hymenoptera dataset from {DATA_URL}", flush=True)
    temporary_archive = archive.with_suffix(".zip.part")
    try:
        urllib.request.urlretrieve(DATA_URL, temporary_archive)
        temporary_archive.replace(archive)
    except Exception:
        temporary_archive.unlink(missing_ok=True)
        raise
    with zipfile.ZipFile(archive) as zipped:
        base = data_dir.resolve()
        for member in zipped.infolist():
            target = (data_dir / member.filename).resolve()
            if base not in target.parents and target != base:
                raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
        zipped.extractall(data_dir)
    archive.unlink()
    return root


def make_model(weights, classes: int, fixed_features: bool) -> nn.Module:
    model = models.resnet18(weights=weights)
    if fixed_features:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, classes)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    root = ensure_dataset(args.data)
    mean, std = (.485, .456, .406), (.229, .224, .225)
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    train = datasets.ImageFolder(root / "train", transform=train_transform)
    test = datasets.ImageFolder(root / "val", transform=test_transform)
    if train.classes != test.classes:
        raise ValueError(f"Train/validation class mismatch: {train.classes} != {test.classes}")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = device_for_run()
    print(f"Device: {device}; classes: {train.classes}")
    results = {}
    for name, fixed in (("fine_tune", False), ("feature_extractor", True)):
        print(f"\nTraining {name}")
        model = make_model(models.ResNet18_Weights.DEFAULT, len(train.classes), fixed)
        trainable = [p for p in model.parameters() if p.requires_grad]
        history = fit(model, train_loader, test_loader, args.epochs, args.lr, device,
                      Path(f"outputs/checkpoints/hymenoptera_{name}.pt"), trainable_parameters=trainable)
        metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, collect_predictions=True)
        prefix = f"hymenoptera_{name}"
        save_history(history, Path(f"reports/{prefix}_history.json"))
        save_plots(history, Path("outputs/figures"), prefix, metrics["targets"], metrics["predictions"], train.classes)
        results[name] = {"test_accuracy": metrics["accuracy"], "test_loss": metrics["loss"],
                         "trainable_parameters": sum(p.numel() for p in trainable)}
        print(f"Final test accuracy: {metrics['accuracy']:.4%}")
    Path("reports/hymenoptera_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nComparison: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
