"""
thoravis/src/predict.py
─────────────────────────────────────────────────────────────────────────────
Single-image inference CLI — loads a trained checkpoint and prints per-
pathology probabilities for one chest X-ray.

CLI usage
---------
    python src/predict.py --image path/to/xray.png --checkpoint models/best_thoravis.pt
"""

import argparse
import os
from typing import Dict

import torch
from PIL import Image

from src.dataset import PATHOLOGY_LABELS
from src.model import ThoraVisClassifier
from src.preprocessing import XRayPreprocessor


def load_model(checkpoint_path: str, device: torch.device) -> ThoraVisClassifier:
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(
            f"No checkpoint at '{checkpoint_path}'. Train one first:\n"
            f"  python src/train.py --subset 5000 --epochs 5"
        )
    model = ThoraVisClassifier().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_image(
    model: ThoraVisClassifier,
    image_path: str,
    device: torch.device,
) -> Dict[str, float]:
    preprocessor = XRayPreprocessor(augment=False)
    pil_img = Image.open(image_path)
    image_tensor = preprocessor.preprocess_pil(pil_img).unsqueeze(0).to(device)

    probs = model.predict_proba(image_tensor)[0]
    return {label: float(p) for label, p in zip(PATHOLOGY_LABELS, probs)}


def main():
    parser = argparse.ArgumentParser(description="ThoraVis single-image inference")
    parser.add_argument("--image", required=True, help="Path to a chest X-ray image")
    parser.add_argument("--checkpoint", default="models/best_thoravis.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(args.checkpoint, device)
    probs = predict_image(model, args.image, device)

    print(f"\nThoraVis predictions — {args.image}\n")
    for label, p in sorted(probs.items(), key=lambda item: -item[1]):
        marker = "x" if p >= args.threshold else " "
        print(f"  [{marker}] {label:<22s} {p:.3f}")
    print()


if __name__ == "__main__":
    main()
