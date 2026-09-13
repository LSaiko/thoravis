"""Shape/dtype contract tests for src/preprocessing.py."""

import pickle
import unittest

import numpy as np
import torch
from PIL import Image

from src.preprocessing import XRayPreprocessor


class TestXRayPreprocessor(unittest.TestCase):
    def setUp(self):
        self.pre = XRayPreprocessor(augment=False)

    def test_preprocess_grayscale_input(self):
        img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        tensor = self.pre.preprocess(img)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_preprocess_rgb_input(self):
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        tensor = self.pre.preprocess(img)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))

    def test_preprocess_pil_wrapper(self):
        pil_img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        tensor = self.pre.preprocess_pil(pil_img)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))

    def test_preprocess_with_augmentation_still_valid(self):
        pre = XRayPreprocessor(augment=True)
        img = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        tensor = pre.preprocess(img)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_sobel_edge_map_shape_and_range(self):
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        edges = self.pre.sobel_edge_map(img)
        self.assertEqual(edges.shape, (224, 224))
        self.assertEqual(edges.dtype, np.uint8)
        self.assertGreaterEqual(int(edges.min()), 0)
        self.assertLessEqual(int(edges.max()), 255)

    def test_picklable_for_dataloader_workers(self):
        # DataLoader(num_workers>0) must pickle the whole Dataset — and thus
        # this preprocessor — to hand it to worker processes. A raw
        # cv2.CLAHE instance stored as state breaks that (regression: was
        # stored directly in __init__, is now built on demand in _clahe()).
        restored = pickle.loads(pickle.dumps(self.pre))
        img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        tensor = restored.preprocess(img)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))


if __name__ == "__main__":
    unittest.main()
