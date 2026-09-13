"""Integration tests for the FastAPI service in src/api.py.

src/api.py reads its checkpoint/backbone paths from environment variables
at import time, so both must be set *before* `from src.api import app` runs
below — that's why this module builds its tiny fixtures ahead of the import
instead of in setUpClass like the other test modules.
"""

import io
import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from src.dataset import NUM_CLASSES, PATHOLOGY_LABELS
from src.model import ThoraVisClassifier
from tests._helpers import make_tiny_vit_checkpoint

_tmpdir = tempfile.TemporaryDirectory()
_TINY_VIT_CKPT = make_tiny_vit_checkpoint(image_size=224)
_tiny_model = ThoraVisClassifier(num_classes=NUM_CLASSES, pretrained_ckpt=_TINY_VIT_CKPT)
_CHECKPOINT_PATH = os.path.join(_tmpdir.name, "tiny_checkpoint.pt")
torch.save({"epoch": 1, "state_dict": _tiny_model.state_dict(), "val_auc": 0.5}, _CHECKPOINT_PATH)

os.environ["THORAVIS_CHECKPOINT"] = _CHECKPOINT_PATH
os.environ["THORAVIS_VIT_CKPT"] = _TINY_VIT_CKPT

from fastapi.testclient import TestClient  # noqa: E402
from src.api import app  # noqa: E402


def _fake_xray_bytes() -> bytes:
    arr = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class TestThoraVisAPI(unittest.TestCase):
    def test_health_reports_loaded_checkpoint(self):
        with TestClient(app) as client:
            resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["checkpoint"], _CHECKPOINT_PATH)
        self.assertEqual(body["checkpoint_epoch"], 1)

    def test_predict_returns_probability_per_pathology(self):
        with TestClient(app) as client:
            resp = client.post(
                "/predict",
                files={"file": ("xray.png", _fake_xray_bytes(), "image/png")},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(set(body["predictions"].keys()), set(PATHOLOGY_LABELS))
        self.assertTrue(all(0.0 <= p <= 1.0 for p in body["predictions"].values()))
        self.assertEqual(body["threshold"], 0.5)
        self.assertNotIn("uncertainty", body)  # default n_samples=1: no MC dropout overhead

    def test_predict_with_n_samples_adds_uncertainty(self):
        with TestClient(app) as client:
            resp = client.post(
                "/predict?n_samples=5",
                files={"file": ("xray.png", _fake_xray_bytes(), "image/png")},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(set(body["uncertainty"].keys()), set(PATHOLOGY_LABELS))
        self.assertTrue(all(s >= 0.0 for s in body["uncertainty"].values()))

    def test_predict_rejects_unreadable_file(self):
        with TestClient(app) as client:
            resp = client.post(
                "/predict",
                files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
            )
        self.assertEqual(resp.status_code, 400)

    def test_predict_gradcam_returns_png(self):
        with TestClient(app) as client:
            resp = client.post(
                "/predict/gradcam?label=Cardiomegaly",
                files={"file": ("xray.png", _fake_xray_bytes(), "image/png")},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/png")
        self.assertTrue(resp.content.startswith(b"\x89PNG"))

    def test_predict_gradcam_rejects_unknown_label(self):
        with TestClient(app) as client:
            resp = client.post(
                "/predict/gradcam?label=NotAPathology",
                files={"file": ("xray.png", _fake_xray_bytes(), "image/png")},
            )
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
