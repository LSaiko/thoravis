"""Contract tests for src/gradcam.py — previously had zero coverage, which is
how a transformers version bump (ViTModel.encoder.layer -> .layers) went
unnoticed until the API's /predict/gradcam endpoint hit it."""

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

    def test_locates_last_layer_on_current_transformers_layout(self):
        # Regression: transformers 5.x exposes ViTModel.layers directly,
        # not .encoder.layer like older releases this code was written for.
        cam = GradCAMViT(self.model)
        self.assertIs(cam._target_layer, self.model.backbone.layers[-1])
        cam.remove_hooks()

    def test_generate_returns_normalized_224x224_heatmap(self):
        cam = GradCAMViT(self.model)
        image_tensor = torch.randn(1, 3, 224, 224)

        heatmap = cam.generate(image_tensor, target_class=0)

        self.assertEqual(heatmap.shape, (224, 224))
        self.assertEqual(heatmap.dtype, np.float32)
        self.assertGreaterEqual(heatmap.min(), 0.0)
        self.assertLessEqual(heatmap.max(), 1.0)
        cam.remove_hooks()

    def test_last_vit_layer_rejects_unknown_backbone_shape(self):
        class NotAViT:
            pass

        with self.assertRaises(AttributeError):
            GradCAMViT._last_vit_layer(NotAViT())


if __name__ == "__main__":
    unittest.main()
