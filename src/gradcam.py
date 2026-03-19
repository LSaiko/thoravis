"""
thoravis/src/gradcam.py
─────────────────────────────────────────────────────────────────────────────
Gradient-weighted Class Activation Maps (Grad-CAM) for ThoraVisClassifier.

Grad-CAM computes gradients of a target class score w.r.t. the final
attention / feature layer, then produces a spatial heatmap showing which
image regions most influenced the prediction.

Reference: Selvaraju et al. (2017) https://arxiv.org/abs/1610.02391
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.dataset import PATHOLOGY_LABELS
from src.preprocessing import overlay_heatmap


class GradCAMViT:
    """
    Grad-CAM for ViT models (hooks into the last transformer encoder layer).

    For ViT the attention tokens have shape (B, num_patches+1, hidden).
    We reshape the gradient-weighted token features back to a 2D spatial map.

    Parameters
    ----------
    model       : ThoraVisClassifier instance
    layer_name  : dot-path to target layer (default: last encoder layer)
    device      : torch.device
    """

    def __init__(
        self,
        model,
        device: Optional[torch.device] = None,
    ):
        self.model  = model
        self.device = device or next(model.parameters()).device

        # Target: last ViT encoder layer's output
        self._target_layer = model.backbone.encoder.layer[-1]

        self._gradients   = None
        self._activations = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # output shape: (B, num_patches+1, hidden_size)
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        fh = self._target_layer.register_forward_hook(forward_hook)
        bh = self._target_layer.register_full_backward_hook(backward_hook)
        self._hooks = [fh, bh]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.

        Args
        ----
        image_tensor : (1, 3, 224, 224) preprocessed tensor
        target_class : index into PATHOLOGY_LABELS

        Returns
        -------
        heatmap : (H, W) float32 array in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        image_tensor = image_tensor.to(self.device)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Forward pass
        logits = self.model(image_tensor)           # (1, 14)

        # Backward on target class
        target_score = logits[0, target_class]
        target_score.backward()

        # Gradients: (1, num_patches+1, hidden)
        grads = self._gradients[0]              # (num_patches+1, hidden)
        acts  = self._activations[0]            # (num_patches+1, hidden)

        # Discard CLS token (index 0), keep patch tokens
        grads = grads[1:, :]   # (num_patches, hidden)
        acts  = acts[1:, :]    # (num_patches, hidden)

        # Global average pooling over hidden dimension → (num_patches,)
        weights = grads.mean(dim=-1)

        # Weighted sum of activations → (num_patches,)
        cam = (weights.unsqueeze(-1) * acts).sum(dim=-1)
        cam = F.relu(cam)

        # Reshape flat patch sequence back to 2D spatial grid
        # ViT-B/16 with 224×224 input → 14×14 patches
        n_patches = cam.shape[0]
        grid_size = int(n_patches ** 0.5)
        cam_2d = cam.reshape(grid_size, grid_size).cpu().numpy()

        # Upsample to 224×224
        cam_2d = cv2.resize(cam_2d, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Normalize to [0, 1]
        cam_min, cam_max = cam_2d.min(), cam_2d.max()
        if cam_max > cam_min:
            cam_2d = (cam_2d - cam_min) / (cam_max - cam_min)
        else:
            cam_2d = np.zeros_like(cam_2d)

        return cam_2d.astype(np.float32)

    def visualize(
        self,
        original_pil_image,
        image_tensor: torch.Tensor,
        top_k: int = 3,
        save_path: Optional[str] = None,
    ):
        """
        Generate Grad-CAM for top-k predicted pathologies and display them.

        Args
        ----
        original_pil_image : PIL.Image (original, before preprocessing)
        image_tensor       : (1, 3, 224, 224) preprocessed tensor
        top_k              : number of top predictions to visualize
        save_path          : optional path to save figure
        """
        # Get top-k predictions
        with torch.no_grad():
            probs = torch.sigmoid(self.model(image_tensor.unsqueeze(0).to(self.device)))
        top_indices = probs[0].argsort(descending=True)[:top_k].tolist()

        original_np = np.array(original_pil_image.resize((224, 224)))
        if original_np.ndim == 2:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)

        fig = plt.figure(figsize=(5 * (top_k + 1), 4))
        fig.suptitle("ThoraVis — Grad-CAM Pathology Attention Maps", fontsize=13)
        gs = gridspec.GridSpec(1, top_k + 1, figure=fig)

        # Original
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(original_np, cmap="gray")
        ax0.set_title("Original X-ray", fontsize=9)
        ax0.axis("off")

        # Grad-CAM per top prediction
        for plot_i, cls_idx in enumerate(top_indices):
            heatmap = self.generate(image_tensor, target_class=cls_idx)
            blended = overlay_heatmap(original_np, heatmap, alpha=0.45)

            prob = float(probs[0, cls_idx])
            label = PATHOLOGY_LABELS[cls_idx]

            ax = fig.add_subplot(gs[plot_i + 1])
            ax.imshow(blended)
            ax.set_title(f"{label}\n(p={prob:.2f})", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Grad-CAM saved → {save_path}")
        plt.show()
