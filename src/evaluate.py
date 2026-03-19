"""
thoravis/src/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Evaluation utilities for ThoraVis:
  - Per-class AUC-ROC with confidence intervals
  - Precision / Recall / F1 at a given threshold
  - ROC curve plots
  - Confusion matrix grid
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay,
)
from tqdm import tqdm
from typing import Optional, Dict, Tuple

from src.dataset import PATHOLOGY_LABELS, NUM_CLASSES


# ─── Inference pass ────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model,
    dataloader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on dataloader and collect all predictions + ground truths.

    Returns
    -------
    all_probs  : (N, 14) float32  — sigmoid probabilities
    all_labels : (N, 14) float32  — binary ground truth
    """
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        logits = model(images)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.vstack(all_labels)


# ─── AUC-ROC ──────────────────────────────────────────────────────────────────

def compute_auc_table(
    probs: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute per-class AUC-ROC, skipping classes with no positive examples.

    Returns dict: {pathology_label: auc_value}
    """
    results = {}
    for i, label in enumerate(PATHOLOGY_LABELS):
        if labels[:, i].sum() == 0:
            continue
        auc = roc_auc_score(labels[:, i], probs[:, i])
        results[label] = round(auc, 4)
    return results


def print_auc_table(auc_dict: Dict[str, float]):
    macro = np.mean(list(auc_dict.values()))
    print("\n── Per-class AUC-ROC ─────────────────────────────────────────")
    print(f"  {'Pathology':<26}  AUC-ROC   Bar")
    print(f"  {'─'*26}  {'─'*7}   {'─'*20}")
    for label, auc in sorted(auc_dict.items(), key=lambda x: -x[1]):
        bar = "█" * int(auc * 20)
        print(f"  {label:<26}  {auc:.4f}    {bar}")
    print(f"\n  {'Macro AUC':<26}  {macro:.4f}")


# ─── ROC Curves ───────────────────────────────────────────────────────────────

def plot_roc_curves(
    probs: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    top_k: int = 8,
):
    """
    Plot ROC curves for the top-k most prevalent pathologies.
    """
    # Select top-k by number of positive examples
    counts   = labels.sum(axis=0)
    top_idxs = np.argsort(counts)[::-1][:top_k]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle("ThoraVis — ROC Curves (Top-8 Pathologies)", fontsize=13)
    axes = axes.flatten()

    for plot_i, cls_idx in enumerate(top_idxs):
        label_name = PATHOLOGY_LABELS[cls_idx]
        if labels[:, cls_idx].sum() == 0:
            axes[plot_i].axis("off")
            continue

        fpr, tpr, _ = roc_curve(labels[:, cls_idx], probs[:, cls_idx])
        auc = roc_auc_score(labels[:, cls_idx], probs[:, cls_idx])

        ax = axes[plot_i]
        ax.plot(fpr, tpr, color="#E87040", linewidth=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
        ax.fill_between(fpr, tpr, alpha=0.10, color="#E87040")
        ax.set_title(label_name, fontsize=9)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curves saved → {save_path}")
    plt.show()


# ─── Precision / Recall / F1 ──────────────────────────────────────────────────

def compute_pr_f1(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1 per class at a fixed threshold."""
    preds = (probs >= threshold).astype(int)

    results = {}
    for i, label in enumerate(PATHOLOGY_LABELS):
        if labels[:, i].sum() == 0:
            continue
        p, r, f1, _ = precision_recall_fscore_support(
            labels[:, i], preds[:, i], average="binary", zero_division=0
        )
        results[label] = {
            "precision": round(float(p),  4),
            "recall":    round(float(r),  4),
            "f1":        round(float(f1), 4),
        }
    return results


# ─── Training history plot ────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: Optional[str] = None):
    """Plot loss curves and macro AUC from trainer.history dict."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("ThoraVis — Training History", fontsize=13)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # AUC
    ax2.plot(epochs, history["val_auc_macro"], label="Val Macro AUC",
             color="#E87040", marker="o")
    ax2.axhline(0.80, color="gray", linestyle="--", linewidth=0.8, label="AUC=0.80")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("Validation Macro AUC")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0.4, 1.0])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history plot saved → {save_path}")
    plt.show()
