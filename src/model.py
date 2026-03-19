"""
thoravis/src/model.py
─────────────────────────────────────────────────────────────────────────────
ThoraVis model: HuggingFace ViT-B/16 backbone + custom multi-label head.

Architecture
------------
  ViT-B/16  (google/vit-base-patch16-224-in21k)
    └── CLS token  [hidden=768]
        └── LayerNorm
            └── Linear(768 → 256)
                └── GELU
                    └── Dropout(0.3)
                        └── Linear(256 → 14)   ← raw logits per pathology
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from typing import Optional, Dict
from src.dataset import NUM_CLASSES, PATHOLOGY_LABELS


# ─── Model ────────────────────────────────────────────────────────────────────

class ThoraVisClassifier(nn.Module):
    """
    Multi-label thoracic pathology classifier.

    Uses the CLS token from a pretrained ViT-B/16 backbone and
    attaches a 2-layer MLP classification head.

    Parameters
    ----------
    num_classes    : number of output labels (14 for NIH ChestX-ray14)
    pretrained_ckpt: HuggingFace model ID
    dropout        : dropout rate in classification head
    freeze_backbone: freeze ViT weights for first N steps (warm-up)
    """

    VIT_CKPT = "google/vit-base-patch16-224-in21k"

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained_ckpt: str = VIT_CKPT,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────────────
        print(f"Loading ViT backbone: {pretrained_ckpt}")
        self.backbone = ViTModel.from_pretrained(pretrained_ckpt)
        self.hidden_size = self.backbone.config.hidden_size  # 768

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("  [backbone frozen — only head will train initially]")

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._init_head()

    def _init_head(self):
        """Xavier uniform init for the classification head layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def unfreeze_backbone(self):
        """Unfreeze backbone after warm-up phase."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen — all parameters now trainable.")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args
        ----
        pixel_values : (B, 3, 224, 224) normalized float tensor

        Returns
        -------
        logits : (B, num_classes) — raw scores (apply sigmoid for probabilities)
        """
        # Extract CLS token representation from ViT
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]   # (B, 768)

        logits = self.classifier(cls_token)               # (B, 14)
        return logits

    def predict_proba(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Sigmoid-activated probabilities for each class."""
        with torch.no_grad():
            logits = self.forward(pixel_values)
        return torch.sigmoid(logits)

    def predict_dict(
        self,
        pixel_values: torch.Tensor,
        threshold: float = 0.5,
    ) -> list[Dict[str, float]]:
        """
        Return a list of dicts mapping pathology name → probability,
        filtered to predictions above `threshold`.

        Useful for single-image inference.
        """
        probs = self.predict_proba(pixel_values)  # (B, 14)
        results = []
        for sample_probs in probs:
            pred_dict = {
                label: float(prob)
                for label, prob in zip(PATHOLOGY_LABELS, sample_probs)
                if float(prob) >= threshold
            }
            results.append(pred_dict)
        return results

    def count_parameters(self) -> Dict[str, int]:
        total  = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "total":     total,
            "trainable": total - frozen,
            "frozen":    frozen,
        }


# ─── Loss function ────────────────────────────────────────────────────────────

class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy with class weights to handle label imbalance.

    Wraps `nn.BCEWithLogitsLoss` with a per-class weight tensor.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=class_weights,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)
