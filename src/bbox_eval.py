"""
thoravis/src/bbox_eval.py
─────────────────────────────────────────────────────────────────────────────
Grad-CAM vs. radiologist ground truth: how often does the model's attention
peak actually fall inside the box a radiologist drew for that finding? NIH
released 984 hand-annotated boxes across 8 pathologies (BBox_List_2017.csv)
— a real, checkable measure of whether Grad-CAM reflects genuine
localization or just pixels that happen to correlate with the label.

Metric: "pointing game" accuracy (Zhang et al., 2016) — the heatmap's
argmax point counts as a hit if it falls inside the ground-truth box.
Standard in the weakly-supervised localization literature for exactly this
check, and threshold-free (no IoU cutoff to tune). Reported alongside a
trivial "always guess image center" baseline, since chest X-ray anatomy is
roughly centered and Grad-CAM heatmaps often peak near center regardless of
class — a model score that doesn't clear that baseline isn't localizing.

CLI usage
---------
    python -m src.bbox_eval --checkpoint models/best_thoravis.pt --image-root <dir with the extracted PNGs>
"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.dataset import PATHOLOGY_LABELS
from src.gradcam import GradCAMViT
from src.model import ThoraVisClassifier
from src.preprocessing import XRayPreprocessor

HF_REPO = "alkzar90/NIH-Chest-X-ray-dataset"
BBOX_CSV_REPO_PATH = "data/BBox_List_2017.csv"
ORIGINAL_IMAGE_SIZE = 1024  # NIH boxes are in this coordinate space
HEATMAP_SIZE = 224          # Grad-CAM output resolution (see gradcam.py)

# BBox_List_2017.csv uses "Infiltrate"; PATHOLOGY_LABELS (matching the NIH
# ChestX-ray14 classification taxonomy) uses "Infiltration" for the same
# finding — everything else already matches by name.
BBOX_LABEL_TO_PATHOLOGY = {"Infiltrate": "Infiltration"}


def load_bbox_annotations() -> pd.DataFrame:
    """Download (if needed) and parse NIH's 984 hand-annotated bounding boxes."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, BBOX_CSV_REPO_PATH, repo_type="dataset")
    # The CSV ships with a trailing comma on its header row (a stray
    # "Bbox [x,y,w,h]," artifact), which pandas otherwise turns into three
    # empty Unnamed columns — usecols sidesteps that instead of relying on
    # column-name string matching against something that isn't valid CSV.
    df = pd.read_csv(path, usecols=[0, 1, 2, 3, 4, 5])
    df.columns = ["image_index", "label", "x", "y", "w", "h"]
    df["label"] = df["label"].replace(BBOX_LABEL_TO_PATHOLOGY)
    return df


def find_images(image_names, search_root: str) -> Dict[str, str]:
    """Map filename -> full path by walking an already-extracted image tree."""
    wanted = set(image_names)
    found = {}
    for dirpath, _, filenames in os.walk(search_root):
        for f in filenames:
            if f in wanted and f not in found:
                found[f] = os.path.join(dirpath, f)
        if len(found) == len(wanted):
            break
    return found


def _box_to_heatmap_space(box_xywh: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box_xywh
    scale = HEATMAP_SIZE / ORIGINAL_IMAGE_SIZE
    return x * scale, y * scale, (x + w) * scale, (y + h) * scale


def pointing_game_hit(heatmap: np.ndarray, box_xywh: Tuple[float, float, float, float]) -> bool:
    """True if `heatmap`'s argmax point falls inside `box_xywh` (original-image-space x, y, w, h)."""
    x1, y1, x2, y2 = _box_to_heatmap_space(box_xywh)
    peak_row, peak_col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return (x1 <= peak_col <= x2) and (y1 <= peak_row <= y2)


def center_baseline_hit(box_xywh: Tuple[float, float, float, float], heatmap_size: int = HEATMAP_SIZE) -> bool:
    """True if the exact image center falls inside `box_xywh` — the trivial baseline."""
    x1, y1, x2, y2 = _box_to_heatmap_space(box_xywh)
    center = heatmap_size / 2
    return (x1 <= center <= x2) and (y1 <= center <= y2)


def evaluate_localization(
    checkpoint_path: str,
    image_root: str,
    device: str = "cpu",
    limit: Optional[int] = None,
) -> dict:
    """Run Grad-CAM against every BBox_List_2017 annotation found under `image_root`."""
    device = torch.device(device)
    model = ThoraVisClassifier().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    annotations = load_bbox_annotations()
    if limit:
        annotations = annotations.head(limit)

    image_paths = find_images(annotations["image_index"].unique(), image_root)
    preprocessor = XRayPreprocessor(augment=False)
    cam = GradCAMViT(model, device=device)

    model_hits, center_hits, totals = {}, {}, {}
    missing = 0

    try:
        for _, row in annotations.iterrows():
            path = image_paths.get(row["image_index"])
            if path is None:
                missing += 1
                continue
            label = row["label"]
            if label not in PATHOLOGY_LABELS:
                continue  # shouldn't happen after the rename map, but stay safe

            box = (row["x"], row["y"], row["w"], row["h"])
            image_tensor = preprocessor.preprocess_pil(Image.open(path)).unsqueeze(0).to(device)
            heatmap = cam.generate(image_tensor, target_class=PATHOLOGY_LABELS.index(label))

            totals[label] = totals.get(label, 0) + 1
            model_hits[label] = model_hits.get(label, 0) + int(pointing_game_hit(heatmap, box))
            center_hits[label] = center_hits.get(label, 0) + int(center_baseline_hit(box))
    finally:
        cam.remove_hooks()

    per_label = {
        label: {
            "model_accuracy": model_hits[label] / totals[label],
            "center_baseline_accuracy": center_hits[label] / totals[label],
            "n": totals[label],
        }
        for label in totals
    }
    n_total = sum(totals.values())

    return {
        "per_label": per_label,
        "overall_model_accuracy": sum(model_hits.values()) / n_total,
        "overall_center_baseline_accuracy": sum(center_hits.values()) / n_total,
        "n_evaluated": n_total,
        "n_missing_images": missing,
    }


def print_results(results: dict) -> None:
    print(f"\nEvaluated {results['n_evaluated']} images "
          f"({results['n_missing_images']} referenced but not found under --image-root)\n")
    print(f"  {'Pathology':<16s} {'Model':>8s} {'Center':>8s}   n")
    print(f"  {'-'*16} {'-'*8} {'-'*8}   {'-'*4}")
    for label, stats in sorted(results["per_label"].items(), key=lambda kv: -kv[1]["model_accuracy"]):
        print(f"  {label:<16s} {stats['model_accuracy']:>8.3f} {stats['center_baseline_accuracy']:>8.3f}   {stats['n']}")
    print(f"  {'-'*16} {'-'*8} {'-'*8}")
    print(f"  {'Overall':<16s} {results['overall_model_accuracy']:>8.3f} {results['overall_center_baseline_accuracy']:>8.3f}")
    print("\n(Model = Grad-CAM pointing-game accuracy. Center = trivial "
          "'always guess image center' baseline — the model should clear this.)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Grad-CAM against radiologist bounding boxes")
    parser.add_argument("--checkpoint", default="models/best_thoravis.pt")
    parser.add_argument("--image-root", required=True, help="Root dir to search for the ~880 annotated images")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    results = evaluate_localization(args.checkpoint, args.image_root, args.device, args.limit)
    print_results(results)


if __name__ == "__main__":
    main()
