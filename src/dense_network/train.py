"""Train CIFAR-10 dense classifiers with explicit manual backpropagation."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.common import save_history, seed_everything


class ManualDenseCIFAR:
    """One-layer Listing 1 baseline or 100-unit sigmoid network."""

    def __init__(self, device: torch.device, hidden_dim: int | None = 100):
        self.device = device
        self.hidden_dim = hidden_dim
        bound = (6 / (3072 + (hidden_dim or 10))) ** 0.5
        self.W1 = torch.empty(3072, hidden_dim or 10, device=device).uniform_(-bound, bound)
        self.b1 = torch.zeros(hidden_dim or 10, device=device)
        if hidden_dim:
            bound2 = (6 / (hidden_dim + 10)) ** 0.5
            self.W2 = torch.empty(hidden_dim, 10, device=device).uniform_(-bound2, bound2)
            self.b2 = torch.zeros(10, device=device)
            self.parameters = [self.W1, self.b1, self.W2, self.b2]
        else:
            self.W2 = self.b2 = None
            self.parameters = [self.W1, self.b1]

    def forward(self, x: torch.Tensor):
        x = x.reshape(x.size(0), -1)
        if self.hidden_dim:
            h = torch.sigmoid(x @ self.W1 + self.b1)
            return h @ self.W2 + self.b2, (x, h)
        return x @ self.W1 + self.b1, (x, None)

    def train_batch(self, x, labels, velocity, learning_rate, momentum):
        logits, (x, h) = self.forward(x)
        # dL/dlogits = (softmax(logits) - one_hot(labels)) / batch_size
        dlogits = (logits.softmax(dim=1) - F.one_hot(labels, 10).to(logits.dtype)) / len(labels)
        if h is None:
            gradients = [x.T @ dlogits, dlogits.sum(dim=0)]
        else:
            dW2 = h.T @ dlogits
            db2 = dlogits.sum(dim=0)
            dh = dlogits @ self.W2.T
            dz = dh * h * (1 - h)
            gradients = [x.T @ dz, dz.sum(dim=0), dW2, db2]
        for parameter, gradient, speed in zip(self.parameters, gradients, velocity):
            speed.mul_(momentum).add_(gradient)
            parameter.add_(speed, alpha=-learning_rate)
        return F.cross_entropy(logits, labels).item(), logits.argmax(1).eq(labels).sum().item()


def make_loaders(data_dir: Path, batch_size: int, seed: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def evaluate(model, loader):
    loss_sum = correct = count = 0
    for x, labels in loader:
        x, labels = x.to(model.device), labels.to(model.device)
        logits, _ = model.forward(x)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += logits.argmax(1).eq(labels).sum().item()
        count += len(labels)
    return {"loss": loss_sum / count, "accuracy": correct / count}


def train_model(data_dir, device, epochs, batch_size, learning_rate, momentum, seed, hidden_dim):
    seed_everything(seed)
    model = ManualDenseCIFAR(device, hidden_dim)
    train_loader, test_loader = make_loaders(data_dir, batch_size, seed)
    velocity = [torch.zeros_like(parameter) for parameter in model.parameters]
    history = {key: [] for key in ("train_loss", "train_accuracy", "test_loss", "test_accuracy")}
    for epoch in range(epochs):
        loss_sum = correct = count = 0
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            loss, batch_correct = model.train_batch(x, labels, velocity, learning_rate, momentum)
            loss_sum += loss * len(labels)
            correct += batch_correct
            count += len(labels)
        history["train_loss"].append(loss_sum / count)
        history["train_accuracy"].append(correct / count)
        print(f"{hidden_dim or 'linear'} epoch {epoch + 1}/{epochs}: "
              f"loss={history['train_loss'][-1]:.4f}, "
              f"train accuracy={history['train_accuracy'][-1]:.2%}", flush=True)
    train_metrics = evaluate(model, make_loaders(data_dir, batch_size, seed)[0])
    test_metrics = evaluate(model, test_loader)
    history["train_accuracy"][-1] = train_metrics["accuracy"]
    history["test_loss"].append(test_metrics["loss"])
    history["test_accuracy"].append(test_metrics["accuracy"])
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; batch size: {args.batch_size}; lr: {args.lr}; momentum: {args.momentum}")

    all_results = {}
    for name, hidden_dim in (("linear_baseline", None), ("manual_sigmoid", 100)):
        history = train_model(args.data, device, args.epochs, args.batch_size,
                              args.lr, args.momentum, args.seed, hidden_dim)
        all_results[name] = history
        print(f"{name}: train={history['train_accuracy'][-1]:.2%}, "
              f"test={history['test_accuracy'][-1]:.2%}, "
              f"test loss={history['test_loss'][-1]:.4f}")

    save_history(all_results["linear_baseline"], Path("reports/cifar10_linear_baseline_history.json"))
    # Keep the assignment's result filename pointed at the required manual model.
    save_history(all_results["manual_sigmoid"], Path("reports/cifar10_dense_history.json"))
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for key, label in (("linear_baseline", "Listing 1 linear baseline"),
                       ("manual_sigmoid", "100-unit sigmoid network")):
        ax.plot(range(1, args.epochs + 1), all_results[key]["train_loss"], marker="o", label=label)
    ax.set(xlabel="Epoch", ylabel="Mean cross-entropy", title="CIFAR-10 training loss")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/figures/cifar10_manual_training_loss.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
