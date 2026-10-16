"""Train the manually specified 100-unit sigmoid network on CIFAR-10."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.common import device_for_run, fit, save_history, seed_everything


class DenseNetCIFAR(nn.Module):
    def __init__(self, input_dim: int = 3072, hidden_dim: int = 100, num_classes: int = 10):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        return self.output(torch.sigmoid(self.hidden(x)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,) * 3, (.5,) * 3)])
    train = datasets.CIFAR10(args.data, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(args.data, train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = device_for_run()
    print(f"Device: {device}; classes: {train.classes}")
    history = fit(DenseNetCIFAR(), train_loader, test_loader, args.epochs, args.lr, device,
                  Path("outputs/checkpoints/cifar10_dense.pt"))
    save_history(history, Path("reports/cifar10_dense_history.json"))
    print(f"Final train accuracy: {history['train_accuracy'][-1]:.4%}")
    print(f"Final test accuracy:  {history['test_accuracy'][-1]:.4%}")


if __name__ == "__main__":
    main()
