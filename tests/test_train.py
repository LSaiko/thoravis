"""Contract tests for the epoch routines in src/train.py.

ThoraVisTrainer.__init__ always calls get_dataloaders() (network + the full
HF dataset), so these tests build a trainer via __new__ and inject a tiny
local model plus fake in-memory loaders instead, exercising the real
_train_epoch()/_val_epoch() logic — including the AMP code path — without
touching the network.
"""

import unittest

import torch

from src.dataset import NUM_CLASSES
from src.model import ThoraVisClassifier, WeightedBCELoss
from src.train import ThoraVisTrainer
from tests._helpers import make_tiny_vit_checkpoint


def _fake_batches(n_batches, batch_size, img_size):
    return [
        (
            torch.randn(batch_size, 3, img_size, img_size),
            torch.randint(0, 2, (batch_size, NUM_CLASSES)).float(),
        )
        for _ in range(n_batches)
    ]


def _make_trainer(device: torch.device, use_amp: bool) -> ThoraVisTrainer:
    img_size = 32
    ckpt = make_tiny_vit_checkpoint(image_size=img_size)
    model = ThoraVisClassifier(num_classes=NUM_CLASSES, pretrained_ckpt=ckpt).to(device)

    trainer = ThoraVisTrainer.__new__(ThoraVisTrainer)
    trainer.device = device
    trainer.use_amp = use_amp and device.type == "cuda"
    trainer.scaler = torch.amp.GradScaler(device=device.type, enabled=trainer.use_amp)
    trainer.model = model
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer.criterion = WeightedBCELoss()
    trainer.train_loader = _fake_batches(2, batch_size=2, img_size=img_size)
    trainer.val_loader = _fake_batches(2, batch_size=2, img_size=img_size)
    return trainer


class TestEpochRoutinesCPU(unittest.TestCase):
    def test_train_epoch_returns_float_loss_and_updates_weights(self):
        trainer = _make_trainer(torch.device("cpu"), use_amp=False)
        before = next(trainer.model.parameters()).clone()

        train_loss = trainer._train_epoch(epoch=0)

        self.assertIsInstance(train_loss, float)
        after = next(trainer.model.parameters())
        self.assertFalse(torch.equal(before, after))

    def test_val_epoch_returns_loss_auc_and_per_class_dict(self):
        trainer = _make_trainer(torch.device("cpu"), use_amp=False)
        val_loss, macro_auc, per_class = trainer._val_epoch(epoch=0)

        self.assertIsInstance(val_loss, float)
        self.assertGreaterEqual(macro_auc, 0.0)
        self.assertLessEqual(macro_auc, 1.0)
        self.assertIsInstance(per_class, dict)


@unittest.skipUnless(torch.cuda.is_available(), "AMP only engages on CUDA")
class TestEpochRoutinesCUDA(unittest.TestCase):
    def test_train_epoch_runs_under_amp(self):
        trainer = _make_trainer(torch.device("cuda"), use_amp=True)
        self.assertTrue(trainer.use_amp)

        train_loss = trainer._train_epoch(epoch=0)
        self.assertIsInstance(train_loss, float)

    def test_val_epoch_runs_under_amp(self):
        trainer = _make_trainer(torch.device("cuda"), use_amp=True)
        val_loss, macro_auc, _ = trainer._val_epoch(epoch=0)
        self.assertIsInstance(val_loss, float)
        self.assertGreaterEqual(macro_auc, 0.0)


if __name__ == "__main__":
    unittest.main()
