"""Shared training and evaluation utilities for EN3160 Assignment 3."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for_run() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, collect_predictions: bool = False) -> dict[str, Any]:
    model.eval()
    total_loss = total_correct = total = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_loss += criterion(logits, targets).item() * targets.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += predictions.eq(targets).sum().item()
        total += targets.size(0)
        if collect_predictions:
            targets_all.extend(targets.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
    result: dict[str, Any] = {"loss": total_loss / total, "accuracy": total_correct / total}
    if collect_predictions:
        result["targets"] = targets_all
        result["predictions"] = predictions_all
    return result


def fit(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
        epochs: int, learning_rate: float, device: torch.device,
        checkpoint: Path | None = None, momentum: float = 0.9,
        trainable_parameters=None) -> dict[str, list[float]]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    parameters = trainable_parameters if trainable_parameters is not None else model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)
    history: dict[str, list[float]] = {f"{split}_{metric}": [] for split in ("train", "test") for metric in ("loss", "accuracy")}
    for epoch in range(epochs):
        model.train()
        # Frozen convolutional backbones must also keep BatchNorm running
        # statistics fixed; requires_grad=False alone does not freeze buffers.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                parameter.requires_grad for parameter in module.parameters(recurse=False)
            ):
                module.eval()
        # These train accuracies are an online average over minibatches in
        # training mode; dropout and random image augmentation remain active.
        total_loss = total_correct = total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_correct += logits.argmax(dim=1).eq(targets).sum().item()
            total += targets.size(0)
        train_metrics = {"loss": total_loss / total, "accuracy": total_correct / total}
        # Test metrics are measured after the epoch in evaluation mode.
        test_metrics = evaluate(model, test_loader, criterion, device)
        for metric in ("loss", "accuracy"):
            history[f"train_{metric}"].append(train_metrics[metric])
            history[f"test_{metric}"].append(test_metrics[metric])
        print(f"Epoch {epoch + 1:02d}/{epochs}: train loss {train_metrics['loss']:.4f}, "
              f"accuracy {train_metrics['accuracy']:.4%}; test loss {test_metrics['loss']:.4f}, "
              f"accuracy {test_metrics['accuracy']:.4%}", flush=True)
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "history": history}, checkpoint)
    return history


def save_history(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def save_plots(history: dict[str, list[float]], out_dir: Path, prefix: str,
               targets: list[int] | None = None, predictions: list[int] | None = None,
               class_names: list[str] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ("train", "test"):
        axes[0].plot(epochs, history[f"{split}_loss"], marker="o", label=split.title())
        axes[1].plot(epochs, [100 * v for v in history[f"{split}_accuracy"]], marker="o", label=split.title())
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_learning_curves.png", dpi=160)
    plt.close(fig)
    if targets is not None and predictions is not None:
        labels = list(range(len(class_names))) if class_names else None
        matrix = confusion_matrix(targets, predictions, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=ax)
        names = class_names or [str(i) for i in range(matrix.shape[0])]
        ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names,
               yticklabels=names, xlabel="Predicted label", ylabel="True label", title=f"{prefix}: test confusion matrix")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        threshold = matrix.max() / 2
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > threshold else "black", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=220)
        plt.close(fig)
