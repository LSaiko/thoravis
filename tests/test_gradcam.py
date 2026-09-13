"""Contract tests for src/gradcam.py — previously had zero coverage, which is
how two real bugs went unnoticed: a transformers version bump
(ViTModel.encoder.layer -> .layers), and a deeper one this file's earlier
version couldn't have caught even with full coverage — its own assertions
only checked heatmap bounds ([0, 1]), which an all-zero heatmap also
satisfies. Hooking the LAST transformer layer is mathematically broken for
a CLS-token-only classifier: the model's final LayerNorm operates per-token,
so the last layer's *patch*-token outputs have zero gradient path to the
CLS logit — not small, exactly zero. Grad-CAM never produced a real
heatmap until the default moved to the second-to-last layer."""

import unittest

import numpy as np
import torch

from src.dataset import NUM_CLASSES
from src.gradcam import GradCAMViT
from src.model import ThoraVisClassifier
from tests._helpers import make_tiny_vit_checkpoint


class TestGradCAMViT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ckpt = make_tiny_vit_checkpoint(image_size=224)
        cls.model = ThoraVisClassifier(num_classes=NUM_CLASSES, pretrained_ckpt=ckpt)

    def test_locates_second_to_last_layer_on_current_transformers_layout(self):
        # Regression: transformers 5.x exposes ViTModel.layers directly,
        # not .encoder.layer like older releases this code was written for.
        cam = GradCAMViT(self.model)
        self.assertIs(cam._target_layer, self.model.backbone.layers[-2])
        cam.remove_hooks()

    def test_generate_returns_normalized_nondegenerate_heatmap(self):
        cam = GradCAMViT(self.model)
        image_tensor = torch.randn(1, 3, 224, 224)

        heatmap = cam.generate(image_tensor, target_class=0)

        self.assertEqual(heatmap.shape, (224, 224))
        self.assertEqual(heatmap.dtype, np.float32)
        self.assertGreaterEqual(heatmap.min(), 0.0)
        self.assertLessEqual(heatmap.max(), 1.0)
        # An all-zero (or otherwise constant) heatmap satisfies the bounds
        # above too — this is the check that actually catches a degenerate
        # CAM, which the original version of this test didn't have.
        self.assertGreater(heatmap.max(), heatmap.min())
        cam.remove_hooks()

    def test_hooking_the_last_layer_gives_a_degenerate_heatmap(self):
        # Documents *why* the default is -2, not -1: with a CLS-token-only
        # head, the last layer's patch outputs never reach the loss.
        cam = GradCAMViT(self.model, layer_index=-1)
        heatmap = cam.generate(torch.randn(1, 3, 224, 224), target_class=0)
        self.assertEqual(heatmap.max(), heatmap.min())
        cam.remove_hooks()

    def test_last_vit_layer_rejects_unknown_backbone_shape(self):
        class NotAViT:
            pass

        with self.assertRaises(AttributeError):
            GradCAMViT._last_vit_layer(NotAViT())


if __name__ == "__main__":
    unittest.main()
