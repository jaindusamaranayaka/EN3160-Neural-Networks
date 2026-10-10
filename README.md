# EN3160 Assignment 3: Neural Networks

Starter structure for the EN3160 neural-networks assignment. The assignment work
has not been implemented yet.

## Assignment outline

1. **Dense network (CIFAR-10):** Extend the provided manually implemented
   network with a 100-node sigmoid hidden layer, use cross-entropy loss, and
   report training and test accuracy after 10 epochs.
2. **CNNs (MNIST):** Train a LeNet-5 and a three-layer CNN with batch
   normalization and dropout for 10 epochs. For both models, record and plot
   per-epoch training/test accuracy and loss, and display a test-set confusion
   matrix.
3. **Transfer learning (Hymenoptera):** Use ImageNet-1K-pretrained ResNet18
   both by fine-tuning and as a fixed feature extractor, then report the results.

## Project structure

```text
data/
  raw/                    Downloaded datasets (not committed)
  processed/              Optional prepared data (not committed)
notebooks/                Exploration and assignment notebooks
src/
  dense_network/          Part 1: manually implemented dense network
  mnist/                  Part 2: LeNet-5 and small CNN
  transfer_learning/      Part 3: ResNet18 experiments
outputs/
  figures/                Training curves and confusion matrices
  checkpoints/             Saved model weights (not committed)
reports/                   Final report source and exported PDF
```

## Setup

Use Python 3.10 or newer in a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Datasets and pretrained weights should be downloaded by the relevant
`torchvision` dataset/model utilities when the experiments are implemented.
Keep downloaded data in `data/` and generated figures in `outputs/figures/`.

## Results and submission

Results are **pending**; this repository currently contains only the project
scaffold. The assignment asks for a PDF named `your_index_a02.pdf`, including
your name and index number, important code, image results, comparisons, and
interpretation/discussion. Replace the filename with the required personal
details when preparing the final submission.

GitHub profile: [jaindusamaranayaka](https://github.com/jaindusamaranayaka)

Make regular commits as the experiments and report are developed, as requested
by the assignment.
