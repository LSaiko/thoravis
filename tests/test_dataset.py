"""Label-vector and class-weight contract tests for src/dataset.py.

ChestXrayDataset.__init__ always calls HuggingFace's load_dataset(), so
instantiating it normally would pull the ~40GB NIH corpus. These tests
build the object via __new__ and inject a stub `.ds` / `.preprocessor`
instead, exercising the real __getitem__ / class_weights() logic without
touching the network.
"""

import unittest

import torch

from src.dataset import ChestXrayDataset, NUM_CLASSES, set_global_seed


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
