# EN3160 Assignment 3: Neural Networks

This repository contains the assignment notebook, runnable experiment scripts, saved result histories, figures, and the four-page submission PDF.

## Setup

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The scripts download CIFAR-10, MNIST, Hymenoptera, and pretrained ResNet-18 weights when needed. Dataset files and model checkpoints are kept out of Git.

## Run the experiments

Run from the repository root:

```powershell
python -m src.dense_network.train
python -m src.mnist.train
python -m src.transfer_learning.train
```

Each script runs for 10 epochs by default. Use `--help` to see optional settings. Q1 uses explicit softmax cross-entropy backpropagation and compares the required 100-unit sigmoid network with a one-layer baseline. Q2 and Q3 use the shared PyTorch training utilities. Histories are saved under `reports/`; plots and confusion matrices are saved under `outputs/figures/`.

The notebook at `notebooks/assignment_03_initial_parts.ipynb` contains the assignment questions, important code, saved results, comparisons, and result figures. Its default `RUN_TRAINING = False` setting loads saved results without training again; set it to `True` to rerun.

## Submission

The four-page PDF is `230563H_a02.pdf`. The assignment brief is `en3160_assignment_03.pdf`.
