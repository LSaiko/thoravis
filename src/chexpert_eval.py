"""
thoravis/src/chexpert_eval.py
─────────────────────────────────────────────────────────────────────────────
Cross-dataset validation: run the NIH ChestX-ray14 checkpoint against
CheXpert Plus's held-out validation split (234 images), the first
external-dataset check in this project (see MODEL_CARD.md recommendation #2).

CheXpert's 14 pathology labels and NIH's 15 overlap on 8: the mapping below.
CheXpert-only labels (Enlarged Cardiomediastinum, Lung Opacity, Lung Lesion,
Pleural Other, Fracture, Support Devices) have no NIH equivalent and are
skipped — this model was never trained to predict them. NIH-only labels
(Infiltration, Mass, Nodule, Fibrosis, Pleural_Thickening, Hernia, Emphysema)
can't be scored here since CheXpert doesn't annotate them.

CheXpert labels are -1 (uncertain), 0 (negative), 1 (positive), or blank
(not mentioned). Only rows with a definite 0/1 for a given pathology are
used when scoring that pathology — uncertain/blank rows are excluded from
that pathology's AUC, not counted as negative.

Requires the images and label CSV to already be on disk locally — see
src/chexpert_download.py for how to pull CheXpert Plus's validation split
via the Redivis API (needs your own Redivis access to the dataset).

Usage
-----
    python -m src.chexpert_eval \
        --labels-csv data/chexpert_valid/labels.csv \
        --image-root data/chexpert_valid/PNG_valid \
        --checkpoint models/best_thoravis.pt
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.dataset import PATHOLOGY_LABELS, NUM_CLASSES
from src.model import ThoraVisClassifier
from src.preprocessing import XRayPreprocessor

CHEXPERT_TO_NIH_LABEL: Dict[str, str] = {
    "No Finding": "No Finding",
    "Cardiomegaly": "Cardiomegaly",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Pneumonia": "Pneumonia",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Effusion",
}


def compute_cross_dataset_auc(
    probs: np.ndarray,
    labels_df: pd.DataFrame,
    label_map: Dict[str, str] = CHEXPERT_TO_NIH_LABEL,
) -> Dict[str, float]:
    """
    Per-pathology AUC-ROC for the CheXpert-labeled subset of images, scored
    against this model's matching output head. `probs` rows must align with
    `labels_df` rows (same order, same length).
    """
    results = {}
    for chexpert_col, nih_label in label_map.items():
        if chexpert_col not in labels_df.columns:
            continue
        col = pd.to_numeric(labels_df[chexpert_col], errors="coerce").to_numpy()
        mask = (col == 0.0) | (col == 1.0)
        if mask.sum() == 0:
            continue
        y_true = col[mask]
        if len(np.unique(y_true)) < 2:
            continue
        class_idx = PATHOLOGY_LABELS.index(nih_label)
        y_score = probs[mask, class_idx]
        results[nih_label] = round(float(roc_auc_score(y_true, y_score)), 4)
    return results


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _resolve_image_path(image_root: Path, raw_path: str) -> Optional[Path]:
    """CheXpert's Path column is typically 'valid/patientXXXXX/studyN/viewN.jpg' —
    strip the split-name prefix Redivis's per-split file index doesn't repeat,
    and try alternate extensions since e.g. PNG_valid re-encodes .jpg as .png."""
    p = Path(raw_path)
    candidates = [p, Path(*p.parts[1:])] if len(p.parts) > 1 else [p]
    for candidate in candidates:
        for ext in (candidate.suffix,) + _IMAGE_EXTENSIONS:
            target = image_root / candidate.with_suffix(ext)
            if target.exists():
                return target
    return None


@torch.no_grad()
def collect_chexpert_predictions(
    df: pd.DataFrame,
    image_root: Path,
    model: torch.nn.Module,
    preprocessor: XRayPreprocessor,
    device: torch.device,
    path_column: str = "Path",
):
    """Runs the model over every image in `df` that's found under `image_root`.
    Returns (probs, matched_df) — matched_df is df restricted to the rows that
    actually resolved to a file, in the same order as probs."""
    model.eval()
    probs_list, kept_idx = [], []
    for i, raw_path in enumerate(df[path_column]):
        img_path = _resolve_image_path(image_root, raw_path)
        if img_path is None:
            continue
        pil_img = Image.open(img_path).convert("RGB")
        tensor = preprocessor.preprocess_pil(pil_img).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(tensor)).cpu().numpy()[0]
        probs_list.append(prob)
        kept_idx.append(i)

    if not probs_list:
        sample_paths = df[path_column].head(3).tolist()
        raise SystemExit(
            f"No images under {image_root} matched any of {len(df)} label rows.\n"
            f"Example expected paths: {sample_paths}\n"
            f"Check --path-column and that --image-root points at the downloaded PNG_valid folder."
        )

    return np.stack(probs_list), df.iloc[kept_idx].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ThoraVis on CheXpert Plus's validation split")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", default="models/best_thoravis.pt")
    parser.add_argument("--path-column", default="Path")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = ThoraVisClassifier(num_classes=NUM_CLASSES)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    preprocessor = XRayPreprocessor(augment=False)
    df = pd.read_csv(args.labels_csv)

    probs, matched_df = collect_chexpert_predictions(
        df, Path(args.image_root), model, preprocessor, device, args.path_column
    )
    print(f"Matched {len(matched_df)}/{len(df)} labeled rows to local images.")

    auc_table = compute_cross_dataset_auc(probs, matched_df)

    lines = ["ThoraVis on CheXpert Plus (external validation set)", "=" * 55]
    lines.append(f"Checkpoint: {args.checkpoint}")
    lines.append(f"Images evaluated: {len(matched_df)}")
    lines.append("")
    lines.append("Per-pathology AUC-ROC (8 labels shared with NIH ChestX-ray14):")
    for label, auc in sorted(auc_table.items(), key=lambda x: -x[1]):
        lines.append(f"  {label:<24}{auc:.4f}")
    if auc_table:
        lines.append("")
        lines.append(f"  Macro AUC (shared labels)  {sum(auc_table.values())/len(auc_table):.4f}")

    out = "\n".join(lines)
    print("\n" + out)
    with open("results/chexpert_cross_validation.txt", "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
