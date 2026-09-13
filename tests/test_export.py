"""Contract tests for src/export.py's TorchScript export."""

import os
import tempfile
import unittest

import torch

from src.dataset import NUM_CLASSES
from src.export import export_torchscript
from src.model import ThoraVisClassifier
from tests._helpers import make_tiny_vit_checkpoint


class TestExportTorchscript(unittest.TestCase):
    def test_traces_and_saves_with_matching_output(self):
        tiny_ckpt = make_tiny_vit_checkpoint(image_size=224)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")
            out_path = os.path.join(tmp_dir, "traced.pt")

            model = ThoraVisClassifier(num_classes=NUM_CLASSES, pretrained_ckpt=tiny_ckpt)
            torch.save({"epoch": 1, "state_dict": model.state_dict()}, checkpoint_path)

            max_diff = export_torchscript(
                checkpoint_path, out_path, device="cpu", pretrained_ckpt=tiny_ckpt
            )

            self.assertLess(max_diff, 1e-4)
            self.assertTrue(os.path.isfile(out_path))

            traced = torch.jit.load(out_path)
            output = traced(torch.randn(2, 3, 224, 224))
            self.assertEqual(tuple(output.shape), (2, NUM_CLASSES))


if __name__ == "__main__":
    unittest.main()
