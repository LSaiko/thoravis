"""
thoravis/src/export.py
─────────────────────────────────────────────────────────────────────────────
Export a trained checkpoint to TorchScript — a single self-contained file
that loads and runs with plain `torch`, no `transformers`/HuggingFace Hub
access needed at inference time. Lighter to deploy than the full checkpoint.

CLI usage
---------
    python -m src.export --checkpoint models/best_thoravis.pt --out models/thoravis_traced.pt
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from typing import Optional

import torch

from src.model import ThoraVisClassifier


def export_torchscript(
    checkpoint_path: str,
    out_path: str,
    device: str = "cpu",
    pretrained_ckpt: Optional[str] = None,
) -> float:
    """
    Trace `checkpoint_path`'s model and save it to `out_path`.

    `pretrained_ckpt` overrides the ViT backbone the checkpoint was trained
    with (default: ThoraVisClassifier's own default, ViT-B/16) — needed if a
    checkpoint was trained against a non-default backbone.

    Returns the max absolute difference between the eager and traced
    model's output on a dummy batch — tracing a model with any real control
    flow can silently diverge, so this is checked, not assumed.
    """
    device = torch.device(device)
    kwargs = {"pretrained_ckpt": pretrained_ckpt} if pretrained_ckpt else {}
    model = ThoraVisClassifier(**kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # ponytail: torch.jit.trace/load are flagged unsupported on Python 3.14+
    # by PyTorch itself (still works today, tested below) — if this starts
    # breaking on a newer PyTorch/Python pairing, migrate to torch.export
    # instead, which is what the deprecation warning points to.
    example = torch.randn(1, 3, 224, 224, device=device)
    traced = torch.jit.trace(model, example)

    with torch.no_grad():
        eager_out  = model(example)
        traced_out = traced(example)
    max_diff = (eager_out - traced_out).abs().max().item()
    if max_diff > 1e-4:
        raise RuntimeError(
            f"TorchScript trace diverges from the eager model "
            f"(max abs diff {max_diff:.2e}) — do not ship this export."
        )

    traced.save(out_path)
    return max_diff


def main():
    parser = argparse.ArgumentParser(description="Export a ThoraVis checkpoint to TorchScript")
    parser.add_argument("--checkpoint", default="models/best_thoravis.pt")
    parser.add_argument("--out",        default="models/thoravis_traced.pt")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--vit-ckpt",   default=None,
                         help="override the ViT backbone (only if the checkpoint wasn't trained with the default)")
    args = parser.parse_args()

    max_diff = export_torchscript(args.checkpoint, args.out, args.device, pretrained_ckpt=args.vit_ckpt)
    print(f"Traced model saved to {args.out} (max eager/traced diff: {max_diff:.2e})")


if __name__ == "__main__":
    main()
