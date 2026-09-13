"""Label-vector and class-weight contract tests for src/dataset.py.

ChestXrayDataset.__init__ always calls HuggingFace's load_dataset(), so
instantiating it normally would pull the ~40GB NIH corpus. These tests
build the object via __new__ and inject a stub `.ds` / `.preprocessor`
instead, exercising the real __getitem__ / class_weights() logic without
touching the network.
"""

import unittest

import numpy as np
import torch

from src.dataset import ChestXrayDataset, NUM_CLASSES, set_global_seed, _train_val_indices


class _StubPreprocessor:
    """Skips OpenCV/ViT preprocessing so these tests isolate label logic."""

    def preprocess_pil(self, pil_image):
        return torch.zeros(3, 4, 4)


def _make_dataset(samples):
    ds = ChestXrayDataset.__new__(ChestXrayDataset)
    ds.ds = samples
    ds.preprocessor = _StubPreprocessor()
    return ds


class TestChestXrayDatasetLabels(unittest.TestCase):
    def test_getitem_builds_multihot_vector(self):
        ds = _make_dataset([{"image": None, "labels": [0, 3, 14]}])
        _, label_vec = ds[0]

        expected = torch.zeros(NUM_CLASSES)
        expected[[0, 3, 14]] = 1.0
        self.assertEqual(tuple(label_vec.shape), (NUM_CLASSES,))
        self.assertTrue(torch.equal(label_vec, expected))

    def test_getitem_ignores_out_of_range_label_indices(self):
        ds = _make_dataset([{"image": None, "labels": [2, 999, -1]}])
        _, label_vec = ds[0]

        expected = torch.zeros(NUM_CLASSES)
        expected[2] = 1.0
        self.assertTrue(torch.equal(label_vec, expected))

    def test_getitem_no_finding_gives_empty_vector(self):
        ds = _make_dataset([{"image": None, "labels": []}])
        _, label_vec = ds[0]
        self.assertTrue(torch.equal(label_vec, torch.zeros(NUM_CLASSES)))

    def test_len_matches_underlying_dataset(self):
        ds = _make_dataset([{"image": None, "labels": []}] * 5)
        self.assertEqual(len(ds), 5)

    def test_class_weights_inverse_frequency(self):
        samples = [
            {"labels": [0]},
            {"labels": [0]},
            {"labels": [1]},
            {"labels": []},
        ]
        ds = _make_dataset(samples)
        weights = ds.class_weights()

        counts = torch.zeros(NUM_CLASSES)
        counts[0] = 2
        counts[1] = 1
        expected = len(samples) / (counts.clamp(min=1) * NUM_CLASSES)

        self.assertTrue(torch.allclose(weights, expected))


class TestTrainValIndices(unittest.TestCase):
    def test_train_and_val_never_overlap_with_test_reserve(self):
        train_idx, val_idx = _train_val_indices(1000, val_split=0.15, test_reserve=100)
        self.assertEqual(set(train_idx) & set(val_idx), set())
        self.assertEqual(len(val_idx), 150)
        self.assertEqual(len(train_idx), 750)

    def test_full_dataset_mode_still_splits_disjointly(self):
        # Regression: get_dataloaders() used to fall through to
        # indices=None for BOTH train and val when subset_size was None,
        # making them the exact same, fully-overlapping dataset.
        train_idx, val_idx = _train_val_indices(112_120, val_split=0.15, test_reserve=0)
        self.assertEqual(set(train_idx) & set(val_idx), set())
        self.assertGreater(len(train_idx), 0)
        self.assertGreater(len(val_idx), 0)
        self.assertEqual(len(train_idx) + len(val_idx), 112_120)


class TestPatientGroupedSplit(unittest.TestCase):
    def test_no_patient_appears_in_both_train_and_val(self):
        # 10 patients, 1-4 images each, 30 rows total.
        rng = np.random.default_rng(0)
        patient_ids = np.repeat(np.arange(10), rng.integers(1, 5, size=10))
        total = len(patient_ids)

        train_idx, val_idx = _train_val_indices(
            total, val_split=0.2, test_reserve=0, patient_ids=patient_ids, seed=1,
        )

        train_patients = set(patient_ids[train_idx])
        val_patients = set(patient_ids[val_idx])
        self.assertEqual(train_patients & val_patients, set())
        self.assertEqual(set(train_idx) | set(val_idx), set(range(total)))

    def test_respects_test_reserve_with_patient_grouping(self):
        patient_ids = np.repeat(np.arange(20), 5)  # 20 patients x 5 images = 100 rows
        total = len(patient_ids)

        train_idx, val_idx = _train_val_indices(
            total, val_split=0.2, test_reserve=20, patient_ids=patient_ids, seed=1,
        )

        # The reserved tail (last 20 indices) must never appear in train/val.
        reserved = set(range(total - 20, total))
        self.assertEqual((set(train_idx) | set(val_idx)) & reserved, set())
        self.assertEqual(set(train_idx) & set(val_idx), set())

    def test_falls_back_to_index_range_when_patient_ids_is_none(self):
        # Same call as the pre-patient-grouping tests above, with the new
        # parameters at their defaults — behavior must be unchanged.
        train_idx, val_idx = _train_val_indices(1000, val_split=0.15, test_reserve=100)
        self.assertEqual(set(train_idx) & set(val_idx), set())
        self.assertEqual(len(val_idx), 150)
        self.assertEqual(len(train_idx), 750)


class TestSetGlobalSeed(unittest.TestCase):
    def test_same_seed_reproduces_torch_and_numpy_draws(self):
        import numpy as np

        set_global_seed(123)
        a_torch, a_numpy = torch.rand(5), np.random.rand(5)

        set_global_seed(123)
        b_torch, b_numpy = torch.rand(5), np.random.rand(5)

        self.assertTrue(torch.equal(a_torch, b_torch))
        self.assertTrue(np.array_equal(a_numpy, b_numpy))


if __name__ == "__main__":
    unittest.main()
