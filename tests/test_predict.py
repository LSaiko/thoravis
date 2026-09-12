"""Contract tests for src/predict.py."""

import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from src.dataset import NUM_CLASSES, PATHOLOGY_LABELS
from src.model import ThoraVisClassifier
from src.predict import load_model, predict_image
from tests._helpers import make_tiny_vit_checkpoint


class TestLoadModel(unittest.TestCase):
    def test_missing_checkpoint_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            load_model("models/does_not_exist.pt", torch.device("cpu"))


class TestPredictImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # XRayPreprocessor always resizes to 224x224 (predict.py doesn't
        # expose target_size), so the test backbone must accept that shape.
        tiny_ckpt = make_tiny_vit_checkpoint(image_size=224)
        cls.model = ThoraVisClassifier(
            num_classes=NUM_CLASSES, pretrained_ckpt=tiny_ckpt
        ).eval()

        img_size = 224
        arr = np.random.randint(0, 255, (img_size, img_size), dtype=np.uint8)
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.image_path = os.path.join(cls.tmpdir.name, "sample.png")
        Image.fromarray(arr).save(cls.image_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_returns_probability_per_pathology(self):
        probs = predict_image(self.model, self.image_path, torch.device("cpu"))
        self.assertEqual(set(probs.keys()), set(PATHOLOGY_LABELS))
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs.values()))


if __name__ == "__main__":
    unittest.main()
