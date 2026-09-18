"""Offline tests for src/chexpert_eval.py's label-mapping and scoring logic.
No network access or real images — compute_cross_dataset_auc is a pure
function over probs + a labels DataFrame."""

import unittest

import numpy as np
import pandas as pd

from src.dataset import PATHOLOGY_LABELS, NUM_CLASSES
from src.chexpert_eval import compute_cross_dataset_auc


def _probs_for(label: str, values: list) -> np.ndarray:
    """(N, NUM_CLASSES) probs array with `values` placed in `label`'s column."""
    probs = np.full((len(values), NUM_CLASSES), 0.5)
    probs[:, PATHOLOGY_LABELS.index(label)] = values
    return probs


class TestComputeCrossDatasetAUC(unittest.TestCase):
    def test_perfect_separation_gives_auc_one(self):
        probs = _probs_for("Cardiomegaly", [0.1, 0.2, 0.8, 0.9])
        labels_df = pd.DataFrame({"Cardiomegaly": [0, 0, 1, 1]})

        result = compute_cross_dataset_auc(probs, labels_df)
        self.assertAlmostEqual(result["Cardiomegaly"], 1.0)

    def test_uncertain_and_blank_rows_are_excluded_not_treated_as_negative(self):
        # If -1/blank were treated as negative, this would tank the AUC;
        # excluding them should still show perfect separation on the
        # unambiguous rows.
        probs = _probs_for("Edema", [0.1, 0.9, 0.5, 0.5])
        labels_df = pd.DataFrame({"Edema": [0, 1, -1, np.nan]})

        result = compute_cross_dataset_auc(probs, labels_df)
        self.assertAlmostEqual(result["Edema"], 1.0)

    def test_skips_pathology_with_no_definite_labels(self):
        probs = _probs_for("Pneumonia", [0.5, 0.5])
        labels_df = pd.DataFrame({"Pneumonia": [-1, np.nan]})

        result = compute_cross_dataset_auc(probs, labels_df)
        self.assertNotIn("Pneumonia", result)

    def test_ignores_chexpert_only_columns_with_no_nih_equivalent(self):
        probs = _probs_for("No Finding", [0.1, 0.9])
        labels_df = pd.DataFrame({
            "No Finding": [0, 1],
            "Support Devices": [1, 0],  # no NIH equivalent — must be ignored
        })

        result = compute_cross_dataset_auc(probs, labels_df)
        self.assertIn("No Finding", result)
        self.assertEqual(len(result), 1)

    def test_only_maps_the_eight_shared_pathologies(self):
        from src.chexpert_eval import CHEXPERT_TO_NIH_LABEL
        self.assertEqual(len(CHEXPERT_TO_NIH_LABEL), 8)
        self.assertEqual(CHEXPERT_TO_NIH_LABEL["Pleural Effusion"], "Effusion")
        for nih_label in CHEXPERT_TO_NIH_LABEL.values():
            self.assertIn(nih_label, PATHOLOGY_LABELS)


if __name__ == "__main__":
    unittest.main()
