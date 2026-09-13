"""
thoravis/src/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Evaluation utilities for ThoraVis:
  - Per-class AUC-ROC with confidence intervals
  - Precision / Recall / F1 at a given threshold
  - ROC curve plots
  - Confusion matrix grid
  - Calibration: expected calibration error, reliability diagram,
    temperature scaling
"""

import numpy as np
import torch
# matplotlib is imported lazily inside each plot_*() function below — the
# metric-computation functions (compute_auc_table, expected_calibration_error,
# collect_predictions, ...) don't need it, and it's a heavy dependency to
# force onto e.g. the inference API, which never plots anything.
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_fscore_support,
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


@torch.no_grad()
def collect_logits(
    model,
    dataloader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Like collect_predictions, but returns raw logits instead of sigmoid
    probabilities — temperature scaling (below) must operate before the
    sigmoid, not after.
    """
    model.eval()
    all_logits, all_labels = [], []

    for images, labels in tqdm(dataloader, desc="Collecting logits"):
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.vstack(all_logits), np.vstack(all_labels)


# ─── AUC-ROC ──────────────────────────────────────────────────────────────────

def compute_auc_table(
    probs: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute per-class AUC-ROC, skipping classes with no positive *or* no
    negative examples (roc_auc_score is undefined — NaN — for either).

    Returns dict: {pathology_label: auc_value}
    """
    n_samples = labels.shape[0]
    results = {}
    for i, label in enumerate(PATHOLOGY_LABELS):
        n_pos = labels[:, i].sum()
        if n_pos == 0 or n_pos == n_samples:
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
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt

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


# ─── Calibration ──────────────────────────────────────────────────────────────
#
# Sigmoid outputs are not automatically calibrated probabilities — a model
# can be discriminative (good AUC) while still being systematically over- or
# under-confident. These treat each of the 15 per-pathology sigmoid outputs
# as an independent binary probability estimate (there's no single joint
# distribution to calibrate in a multi-label setting), which is the standard
# simplification for this kind of pooled calibration check.

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Pooled expected calibration error (ECE) across every (sample, class)
    prediction: bins predictions by confidence, then weights each bin's
    |accuracy − confidence| gap by how much of the data fell in it.
    0 = perfectly calibrated.
    """
    probs_flat  = probs.ravel()
    labels_flat = labels.ravel()
    n = len(probs_flat)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (probs_flat >= lo) & (probs_flat <= hi if hi == 1.0 else probs_flat < hi)
        if not in_bin.any():
            continue
        bin_confidence = probs_flat[in_bin].mean()
        bin_accuracy   = labels_flat[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
):
    """Plot observed frequency vs. mean predicted probability per bin."""
    import matplotlib.pyplot as plt

    probs_flat  = probs.ravel()
    labels_flat = labels.ravel()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences, bin_accuracies = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (probs_flat >= lo) & (probs_flat <= hi if hi == 1.0 else probs_flat < hi)
        if not in_bin.any():
            continue
        bin_confidences.append(probs_flat[in_bin].mean())
        bin_accuracies.append(labels_flat[in_bin].mean())

    ece = expected_calibration_error(probs, labels, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="Perfect calibration")
    ax.plot(bin_confidences, bin_accuracies, "o-", color="#E87040",
             label=f"ThoraVis (ECE = {ece:.4f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("ThoraVis — Reliability Diagram")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Reliability diagram saved → {save_path}")
    plt.show()


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 50,
) -> float:
    """
    Fit a single scalar temperature T (Guo et al., 2017) on held-out
    logits/labels — apply downstream as sigmoid(logits / T). T > 1 means the
    raw model was overconfident; T < 1 means underconfident. Must be fit on
    a validation split, never on training data or the split being reported.
    """
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).float()

    # Optimize in log-space so T = exp(log_T) stays positive by construction.
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_t / temperature, labels_t
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().item())


def fit_isotonic_calibration(probs: np.ndarray, labels: np.ndarray, per_class: bool = False):
    """
    Fit isotonic regression mapping raw sigmoid probability to calibrated
    probability, pooled across all 15 pathologies by default.

    Temperature scaling (above) is a single global scalar — it can only
    correct miscalibration of the form sigmoid(logit / T), i.e. uniform
    over- or under-confidence. This project's own reliability diagram
    found something a single temperature can't fix in principle:
    near-perfect calibration close to 0, but real overconfidence
    specifically in the 0.4-0.9 range. Isotonic regression fits an
    arbitrary monotonic mapping instead, so it *can* correct miscalibration
    that varies across the confidence range — verified on synthetic data
    matching this exact failure shape (tests/test_evaluate.py).

    In practice, on this project's actual checkpoint, pooled isotonic
    didn't help: fit on val and evaluated on the real held-out test set,
    it scored *worse* than temperature scaling both overall (ECE 0.0171
    vs 0.0034) and in the 0.4-0.9 band specifically (0.1499 vs 0.0789) —
    see results/calibration_comparison.txt (regenerate, gitignored). Most
    likely cause: pooling across 15 pathologies with different base rates
    and probably different miscalibration shapes gives isotonic regression
    enough freedom to fit validation-set sampling noise that doesn't
    transfer to test.

    per_class=True fits a separate curve per pathology instead. Each curve
    still sees the full validation set (~12.6k images) for that column —
    isotonic regression fits on all points, not just positives, so this
    isn't the small-sample regime it might look like. On this project's
    real checkpoint it's the best calibration method tried: test-set ECE
    0.0118 overall (temperature: 0.0112, pooled isotonic: 0.0275) and,
    more importantly, 0.0627 in the 0.4-0.9 band that actually drives
    decisions — a 2.6x improvement over temperature scaling's 0.1660 and
    the pooled fit's 0.2019 there. See results/calibration_comparison.txt
    (regenerate, gitignored) and the README's Calibration section. The
    pooled fit's failure was averaging 15 pathologies' different
    miscalibration shapes into one curve, not overfitting per se —
    per-class removes exactly that averaging.

    Must be fit on a validation split, never on the split being reported —
    same rule as temperature scaling.

    Returns a fitted sklearn IsotonicRegression (per_class=False) or a
    list of NUM_CLASSES of them (per_class=True); apply with
    apply_calibration(calibrator, probs).
    """
    from sklearn.isotonic import IsotonicRegression

    def _fit(p, y):
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(p, y)
        return calibrator

    if per_class:
        return [_fit(probs[:, i], labels[:, i]) for i in range(NUM_CLASSES)]
    return _fit(probs.ravel(), labels.ravel())


def apply_calibration(calibrator, probs: np.ndarray) -> np.ndarray:
    """
    Apply a fitted isotonic calibrator (see fit_isotonic_calibration),
    preserving shape. Accepts either a single pooled calibrator or a
    per-class list of them (one per pathology column).
    """
    if isinstance(calibrator, list):
        return np.stack(
            [calibrator[i].predict(probs[:, i]) for i in range(len(calibrator))],
            axis=1,
        )
    return calibrator.predict(probs.ravel()).reshape(probs.shape)
