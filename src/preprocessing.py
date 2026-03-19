"""
thoravis/src/preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Clinical-grade OpenCV preprocessing for chest X-ray images.

Techniques applied:
  1. Resize  – standardize to (224, 224) for ViT input
  2. CLAHE   – Contrast Limited Adaptive Histogram Equalization
               enhances local contrast without blowing out bone detail
  3. Bilateral filter – edge-preserving noise reduction
  4. Sobel edge map   – optional diagnostic overlay / feature augmentation
  5. ImageNet normalization – required for pretrained ViT backbone
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─── Constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TARGET_SIZE   = (224, 224)


class XRayPreprocessor:
    """
    Full OpenCV preprocessing pipeline for chest X-rays.

    Parameters
    ----------
    target_size   : (H, W) to resize images to
    clahe_clip    : CLAHE clip limit  (higher → more contrast boost)
    clahe_grid    : CLAHE tile grid size
    bilateral_d   : bilateral filter diameter
    bilateral_sc  : bilateral sigma color
    bilateral_ss  : bilateral sigma space
    augment       : apply random augmentations (training only)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = TARGET_SIZE,
        clahe_clip: float = 3.0,
        clahe_grid: int = 8,
        bilateral_d: int = 9,
        bilateral_sc: int = 75,
        bilateral_ss: int = 75,
        augment: bool = False,
    ):
        self.target_size  = target_size
        self.clahe        = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid)
        )
        self.bilateral_d  = bilateral_d
        self.bilateral_sc = bilateral_sc
        self.bilateral_ss = bilateral_ss
        self.augment      = augment

        # PyTorch tensor transforms (applied after OpenCV steps)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        if augment:
            self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.aug = None

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Full pipeline: numpy uint8 image → normalized float32 tensor.

        Args
        ----
        image : H×W grayscale or H×W×3 BGR/RGB uint8 array

        Returns
        -------
        torch.Tensor of shape (3, H, W)
        """
        img = self._ensure_gray(image)
        img = self._resize(img)
        img = self._clahe(img)
        img = self._bilateral(img)
        img_rgb = self._to_rgb(img)          # ViT expects 3-channel input
        pil_img = Image.fromarray(img_rgb)

        if self.aug is not None:
            pil_img = self.aug(pil_img)

        return self.to_tensor(pil_img)

    def preprocess_pil(self, pil_image: Image.Image) -> torch.Tensor:
        """Convenience wrapper accepting PIL.Image input."""
        np_img = np.array(pil_image)
        return self.preprocess(np_img)

    # ── Individual steps (public for unit-testing / visualization) ────────────

    def _ensure_gray(self, img: np.ndarray) -> np.ndarray:
        """Convert to single-channel grayscale if needed."""
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img.astype(np.uint8)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)

    def _clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE for local contrast normalisation."""
        return self.clahe.apply(img)

    def _bilateral(self, img: np.ndarray) -> np.ndarray:
        """Edge-preserving noise reduction."""
        return cv2.bilateralFilter(
            img, self.bilateral_d, self.bilateral_sc, self.bilateral_ss
        )

    def _to_rgb(self, gray: np.ndarray) -> np.ndarray:
        """Stack grayscale into 3-channel RGB (required by ViT ImageNet weights)."""
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def sobel_edge_map(self, img: np.ndarray) -> np.ndarray:
        """
        Compute Sobel edge map — useful as a diagnostic overlay and
        demonstrates OpenCV gradient-based operations.

        Returns uint8 edge magnitude image.
        """
        gray = self._ensure_gray(img)
        gray = self._resize(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── Visualization ─────────────────────────────────────────────────────────

    def visualize_pipeline(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the step-by-step preprocessing pipeline for a single X-ray.
        Great for README / notebook showcase.
        """
        gray     = self._ensure_gray(image)
        resized  = self._resize(gray)
        clahe_d  = self._clahe(resized)
        bilateral_d = self._bilateral(clahe_d)
        edges    = self.sobel_edge_map(image)

        fig = plt.figure(figsize=(18, 4))
        fig.suptitle("ThoraVis — OpenCV Preprocessing Pipeline", fontsize=14, y=1.02)
        gs = gridspec.GridSpec(1, 5, figure=fig)

        stages = [
            (resized,     "1. Resized (224×224)", "gray"),
            (clahe_d,     "2. CLAHE Enhancement",  "gray"),
            (bilateral_d, "3. Bilateral Filter",   "gray"),
            (edges,       "4. Sobel Edge Map",      "hot"),
            (cv2.cvtColor(bilateral_d, cv2.COLOR_GRAY2RGB),
                          "5. RGB (ViT Input)",     None),
        ]

        for i, (arr, title, cmap) in enumerate(stages):
            ax = fig.add_subplot(gs[i])
            if cmap:
                ax.imshow(arr, cmap=cmap)
            else:
                ax.imshow(arr)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Pipeline visualization saved → {save_path}")
        plt.show()


# ─── Grad-CAM overlay helper ──────────────────────────────────────────────────

def overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto the original X-ray using OpenCV.

    Args
    ----
    original_img : H×W×3 uint8 RGB image
    heatmap      : H×W float32 in [0, 1]
    alpha        : heatmap opacity

    Returns
    -------
    H×W×3 uint8 blended image
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match original
    h, w = original_img.shape[:2]
    heatmap_color = cv2.resize(heatmap_color, (w, h))

    blended = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return blended
