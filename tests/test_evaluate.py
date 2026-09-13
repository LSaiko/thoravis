"""Regression tests for src/evaluate.py's AUC computation and calibration."""

import unittest

import numpy as np

from src.dataset import NUM_CLASSES, PATHOLOGY_LABELS
from src.evaluate import (
    apply_calibration,
    compute_auc_table,
    expected_calibration_error,
    fit_isotonic_calibration,
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


class TestIsotonicCalibration(unittest.TestCase):
    def test_corrects_a_non_uniform_distortion_temperature_cannot(self):
        # A "hump" in the middle of the range that isn't expressible as
        # sigmoid(logit / T) for any single T — exactly the shape this
        # project's own reliability diagram found (fine near 0, real
        # overconfidence at 0.4-0.9).
        rng = np.random.default_rng(2)
        n = 20_000
        true_prob = rng.random(n)
        raw_prob = np.clip(true_prob + 0.18 * np.sin(np.pi * true_prob), 0, 1)
        labels = (rng.random(n) < true_prob).astype(float)

        ece_before = expected_calibration_error(raw_prob, labels)

        calibrator = fit_isotonic_calibration(raw_prob, labels)
        calibrated = apply_calibration(calibrator, raw_prob)
        ece_after = expected_calibration_error(calibrated, labels)

        self.assertLess(ece_after, ece_before / 5)

    def test_apply_calibration_preserves_input_shape(self):
        rng = np.random.default_rng(3)
        probs = rng.random((50, NUM_CLASSES))
        labels = (rng.random((50, NUM_CLASSES)) < 0.5).astype(float)

        calibrator = fit_isotonic_calibration(probs, labels)
        calibrated = apply_calibration(calibrator, probs)

        self.assertEqual(calibrated.shape, probs.shape)
        self.assertTrue(np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0))

    def test_per_class_fits_one_curve_per_pathology(self):
        rng = np.random.default_rng(4)
        probs = rng.random((200, NUM_CLASSES))
        labels = (rng.random((200, NUM_CLASSES)) < 0.5).astype(float)

        calibrators = fit_isotonic_calibration(probs, labels, per_class=True)
        self.assertEqual(len(calibrators), NUM_CLASSES)

        calibrated = apply_calibration(calibrators, probs)
        self.assertEqual(calibrated.shape, probs.shape)
        self.assertTrue(np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0))


if __name__ == "__main__":
    unittest.main()
