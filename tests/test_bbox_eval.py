"""Unit tests for the pure logic in src/bbox_eval.py.

evaluate_localization() itself needs a real checkpoint, real images, and
network access to fetch BBox_List_2017.csv — it's validated by an actual
run against real data (see README), not a unit test. What's tested here is
the metric math and file-lookup logic that a bug could silently corrupt.
"""

import os
import tempfile
import unittest

import numpy as np

from src.bbox_eval import (
    BBOX_LABEL_TO_PATHOLOGY,
    center_baseline_hit,
    find_images,
    pointing_game_hit,
)
from src.dataset import PATHOLOGY_LABELS


class TestPointingGameHit(unittest.TestCase):
    def test_peak_inside_box_is_a_hit(self):
        heatmap = np.zeros((224, 224), dtype=np.float32)
        heatmap[100, 100] = 1.0  # peak at (row=100, col=100) in heatmap space
        # In original 1024-space that's (col*1024/224, row*1024/224) ~= (457, 457)
        box = (400, 400, 200, 200)  # x, y, w, h -> covers original (400-600, 400-600)
        self.assertTrue(pointing_game_hit(heatmap, box))

    def test_peak_outside_box_is_a_miss(self):
        heatmap = np.zeros((224, 224), dtype=np.float32)
        heatmap[10, 10] = 1.0  # top-left corner
        box = (600, 600, 200, 200)  # bottom-right region, far from the peak
        self.assertFalse(pointing_game_hit(heatmap, box))

    def test_peak_on_box_edge_counts_as_hit(self):
        heatmap = np.zeros((224, 224), dtype=np.float32)
        heatmap[50, 50] = 1.0
        # heatmap (50,50) -> original space (50*1024/224, 50*1024/224) ~= (228.57, 228.57).
        # Box left edge starts just before that point, so the peak lands
        # right at/just inside the boundary — exercises the inclusive <=
        # comparison without depending on exact float equality.
        box = (228.5, 228.5, 100, 100)
        self.assertTrue(pointing_game_hit(heatmap, box))


class TestCenterBaselineHit(unittest.TestCase):
    def test_box_covering_center_is_a_hit(self):
        box = (400, 400, 224, 224)  # covers original (400-624, 400-624), center=(512,512) inside
        self.assertTrue(center_baseline_hit(box))

    def test_box_far_from_center_is_a_miss(self):
        box = (0, 0, 100, 100)  # top-left corner, nowhere near center
        self.assertFalse(center_baseline_hit(box))


class TestFindImages(unittest.TestCase):
    def test_locates_files_nested_in_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmp_root:
            batch_dir = os.path.join(tmp_root, "batch_003", "images")
            os.makedirs(batch_dir)
            target_path = os.path.join(batch_dir, "00013118_008.png")
            open(target_path, "wb").close()

            found = find_images(["00013118_008.png", "00099999_999.png"], tmp_root)

            self.assertEqual(found["00013118_008.png"], target_path)
            self.assertNotIn("00099999_999.png", found)


class TestLabelMapping(unittest.TestCase):
    def test_bbox_infiltrate_maps_to_project_infiltration_label(self):
        self.assertEqual(BBOX_LABEL_TO_PATHOLOGY["Infiltrate"], "Infiltration")
        self.assertIn("Infiltration", PATHOLOGY_LABELS)
        self.assertNotIn("Infiltrate", PATHOLOGY_LABELS)


if __name__ == "__main__":
    unittest.main()
