"""Forward-pass contract tests for src/model.py.

Uses a small locally-built ViT checkpoint (see tests/_helpers.py) instead of
the real ViT-B/16, so the suite runs in milliseconds, offline, while still
exercising the real ThoraVisClassifier code path end to end at the
pipeline's actual 224x224 resolution.
"""

import unittest

import torch

from src.dataset import NUM_CLASSES
from src.model import ThoraVisClassifier, WeightedBCELoss
from tests._helpers import make_tiny_vit_checkpoint

TINY_VIT_CKPT = make_tiny_vit_checkpoint()


class TestThoraVisClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ThoraVisClassifier(
            num_classes=NUM_CLASSES, pretrained_ckpt=TINY_VIT_CKPT
        )
        cls.img_size = cls.model.backbone.config.image_size

    def _random_batch(self, n=2):
        return torch.randn(n, 3, self.img_size, self.img_size)

    def test_forward_output_shape(self):
        logits = self.model(self._random_batch(2))
        self.assertEqual(tuple(logits.shape), (2, NUM_CLASSES))

    def test_predict_proba_is_valid_probability(self):
        probs = self.model.predict_proba(self._random_batch(2))
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

    def test_predict_proba_sets_eval_mode(self):
        self.model.train()
        self.model.predict_proba(self._random_batch(1))
        self.assertFalse(self.model.training)

    def test_predict_dict_respects_threshold(self):
        out = self.model.predict_dict(self._random_batch(1), threshold=1.1)
        self.assertEqual(out, [{}])

    def test_predict_with_uncertainty_shapes_and_bounds(self):
        mean, std = self.model.predict_with_uncertainty(self._random_batch(2), n_samples=10)
        self.assertEqual(tuple(mean.shape), (2, NUM_CLASSES))
        self.assertEqual(tuple(std.shape), (2, NUM_CLASSES))
        self.assertTrue(torch.all(mean >= 0) and torch.all(mean <= 1))
        self.assertTrue(torch.all(std >= 0))

    def test_predict_with_uncertainty_sets_eval_mode(self):
        self.model.predict_with_uncertainty(self._random_batch(1), n_samples=5)
        self.assertFalse(self.model.training)

    def test_predict_with_uncertainty_single_sample_has_zero_std(self):
        # Degenerate but correct: one MC sample can't disagree with itself.
        _, std = self.model.predict_with_uncertainty(self._random_batch(1), n_samples=1)
        self.assertTrue(torch.all(std == 0))

    def test_predict_with_uncertainty_dropout_produces_real_variance(self):
        # The classification head's Dropout(0.3) should make repeated
        # samples disagree — a std of exactly zero here would mean MC
        # dropout silently isn't perturbing anything.
        _, std = self.model.predict_with_uncertainty(self._random_batch(4), n_samples=25)
        self.assertTrue(torch.any(std > 0.01))

    def test_count_parameters_totals_are_consistent(self):
        info = self.model.count_parameters()
        self.assertEqual(info["trainable"] + info["frozen"], info["total"])

    def test_freeze_backbone_then_unfreeze(self):
        model = ThoraVisClassifier(
            num_classes=NUM_CLASSES, pretrained_ckpt=TINY_VIT_CKPT, freeze_backbone=True
        )
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        model.unfreeze_backbone()
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))


class TestWeightedBCELoss(unittest.TestCase):
    def test_matches_bce_with_logits_pos_weight(self):
        weights = torch.tensor([1.0, 2.0, 0.5, 3.0])
        criterion = WeightedBCELoss(class_weights=weights)

        logits = torch.randn(4, 4)
        targets = torch.randint(0, 2, (4, 4)).float()

        actual = criterion(logits, targets)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=weights
        )
        self.assertAlmostEqual(actual.item(), expected.item(), places=5)


if __name__ == "__main__":
    unittest.main()
