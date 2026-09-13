"""Regression tests for src/evaluate.py's AUC computation and calibration."""

import unittest

import numpy as np

from src.dataset import NUM_CLASSES, PATHOLOGY_LABELS
from src.evaluate import (
    compute_auc_table,
    expected_calibration_error,
    fit_temperature,
)


class TestComputeAucTable(unittest.TestCase):
    def test_skips_classes_with_no_positives(self):
        labels = np.zeros((10, NUM_CLASSES))
        labels[:, 0] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        probs = np.random.rand(10, NUM_CLASSES)

        result = compute_auc_table(probs, labels)
        self.assertIn(PATHOLOGY_LABELS[0], result)
        self.assertNotIn(PATHOLOGY_LABELS[1], result)  # all-zero column

    def test_skips_classes_with_no_negatives(self):
        # A column of all ones makes roc_auc_score undefined (NaN); it must
        # be skipped, not silently poison a downstream macro average.
        labels = np.zeros((10, NUM_CLASSES))
        labels[:, 0] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        labels[:, 1] = 1.0
        probs = np.random.rand(10, NUM_CLASSES)

        result = compute_auc_table(probs, labels)
        self.assertIn(PATHOLOGY_LABELS[0], result)
        self.assertNotIn(PATHOLOGY_LABELS[1], result)  # all-one column
        self.assertTrue(all(not np.isnan(v) for v in result.values()))


class TestExpectedCalibrationError(unittest.TestCase):
    def test_near_zero_for_well_calibrated_probabilities(self):
        rng = np.random.default_rng(0)
        probs = np.linspace(0.05, 0.95, 10_000)
        labels = (rng.random(10_000) < probs).astype(float)

        self.assertLess(expected_calibration_error(probs, labels), 0.02)

    def test_one_for_maximally_overconfident_wrong_predictions(self):
        probs = np.ones(500)
        labels = np.zeros(500)
        self.assertAlmostEqual(expected_calibration_error(probs, labels), 1.0)


class TestFitTemperature(unittest.TestCase):
    def test_recovers_known_overconfidence_scale(self):
        rng = np.random.default_rng(1)
        true_logits = rng.standard_normal((5000, 1)) * 1.5
        true_probs = 1 / (1 + np.exp(-true_logits))
        labels = (rng.random((5000, 1)) < true_probs).astype(np.float32)

        scale = 2.5
        observed_logits = (true_logits * scale).astype(np.float32)

        fitted_t = fit_temperature(observed_logits, labels)
        self.assertAlmostEqual(fitted_t, scale, delta=0.5)


if __name__ == "__main__":
    unittest.main()
