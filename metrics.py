import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from ivon import IVON

from typing import Tuple, Dict


def evaluate_all_probs(model: nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       loader: DataLoader, 
                       device: torch.device, 
                       test_samples=0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs through a dataloader and aggregates the model outputs for all datapoints, including probabilities, true labels and logits
    The argument test_samples specifies how to handle the MC sampling in the case of IVON.

    test_samples = 0 → IVON@mean        (deterministic mean weights)
    test_samples = 1 → single posterior sample
    test_samples > 1 → MC posterior averaging
    Deterministic optimizers are handled automatically.

    :returns probs: tensor of shape (N, n_categories) holding probabilities for all N elements in the loader.
    :returns labels: integer tensor of shape (N,) holding true labels.
    :returns logits: tensor of shape (N, n_categories) holding the output logits.
    """

    model.eval()
    all_probs, all_labels, all_logits = [], [], []
    ivon_mode = (type(optimizer) is IVON)

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            # CASE A — IVON@mean (deterministic)
            if ivon_mode and test_samples == 0:
                logits = model(X)
                probs = F.softmax(logits, dim=1)

            # CASE B — IVON with posterior sampling
            elif ivon_mode and test_samples >= 1:

                if test_samples == 1:
                    # single posterior sample
                    with optimizer.sampled_params(train=False):
                        logits = model(X)
                        probs = F.softmax(logits, dim=1)

                else:
                    # MC averaging
                    num_classes = model.fc.out_features if hasattr(model, "fc") else 10
                    mc_probs = torch.zeros(test_samples, X.size(0), num_classes, device=device)

                    for s in range(test_samples):
                        with optimizer.sampled_params(train=False):
                            mc_probs[s] = F.softmax(model(X), dim=1)

                    probs = mc_probs.mean(0)
                    logits = probs.log()  # logits not meaningful under MC, but needed for uniform API

            # CASE C — Deterministic optimizers (SGD/AdamW/etc.)
            else:
                logits = model(X)
                probs = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    return torch.cat(all_probs), torch.cat(all_labels), torch.cat(all_logits)


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins=15) -> float:
    """
    Get the overall ECE score given predicted probabilities and true labels, using 'n_bins' bins
    """
    confidences, preds = probs.max(dim=1)
    correctness = preds.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences >= low) & (confidences < high)
        if not mask.any():
            continue

        bin_conf = confidences[mask].mean()
        bin_acc = correctness[mask].float().mean()
        ece += mask.float().mean() * (bin_conf - bin_acc).abs()

    return float(ece)


def compute_brier(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the overall brier score given a set of probability estimates [N, n_classes] and true labels [N]
    :argument probs: tensor of probabilities [N, n_classes]
    :argument labels: integer tensor of true labels [N]
    :returns brier_score: float for the average brier score over all predictions
    """
    one_hot = F.one_hot(labels, probs.size(1)).float()
    brier = ((probs - one_hot) ** 2).mean().item()
    return brier


def plot_calibration_curve(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15, label_name: str = "Model"):
    """
    Plots a reliability diagram (calibration curve) for a model.

    :argument probs: tensor [N, n_classes] with predicted probabilities
    :argument labels: tensor [N] with true labels
    """
    from sklearn.calibration import calibration_curve

    # Convert to numpy
    probs_np = probs.numpy()
    labels_np = labels.numpy()

    # Predicted confidence = max probability per sample
    confidences = np.max(probs_np, axis=1)

    # Predicted class
    predictions = np.argmax(probs_np, axis=1)

    # Correctness mask
    correctness = (predictions == labels_np).astype(int)

    # Compute calibration curve (fraction of positives vs confidence)
    frac_pos, mean_conf = calibration_curve(
        correctness,
        confidences,
        n_bins=n_bins,
        strategy='uniform'
    )

    # Plot
    plt.plot(mean_conf, frac_pos, marker="o", label=label_name)
    plt.plot([0, 1], [0, 1], "--", color="gray")  # perfect calibration
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Fraction of correct predictions")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.grid(True)

