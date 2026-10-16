"""Train both MNIST networks, save learning curves and test confusion matrices."""
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.common import device_for_run, evaluate, fit, save_history, save_plots, seed_everything
from src.mnist.models import LeNet5, SmallCNN
import torch.nn as nn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    train = datasets.MNIST(args.data, train=True, download=True, transform=transform)
    test = datasets.MNIST(args.data, train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = device_for_run()
    print(f"Device: {device}")
    for name, model in (("lenet5", LeNet5()), ("small_cnn", SmallCNN())):
        print(f"\nTraining {name}")
        checkpoint = Path(f"outputs/checkpoints/mnist_{name}.pt")
        history = fit(model, train_loader, test_loader, args.epochs, args.lr, device, checkpoint)
        metrics = evaluate(model.to(device), test_loader, nn.CrossEntropyLoss(), device, collect_predictions=True)
        prefix = f"mnist_{name}"
        save_history(history, Path(f"reports/{prefix}_history.json"))
        save_plots(history, Path("outputs/figures"), prefix, metrics["targets"], metrics["predictions"],
                   [str(i) for i in range(10)])
        print(f"Final test accuracy: {metrics['accuracy']:.4%}")


if __name__ == "__main__":
    main()
