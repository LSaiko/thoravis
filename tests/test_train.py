"""Contract tests for the epoch routines in src/train.py.

ThoraVisTrainer.__init__ always calls get_dataloaders() (network + the full
HF dataset), so these tests build a trainer via __new__ and inject a tiny
local model plus fake in-memory loaders instead, exercising the real
_train_epoch()/_val_epoch() logic — including the AMP code path — without
touching the network.
"""

import tempfile
import unittest
from unittest.mock import patch

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


def _make_full_trainer(tmp_checkpoint_dir, epochs, early_stopping_patience):
    """Enough of a real trainer for train() itself to run — not just the
    epoch routines above — with _val_epoch mocked to a fixed AUC sequence
    so early stopping can be tested deterministically."""
    device = torch.device("cpu")
    img_size = 32
    ckpt = make_tiny_vit_checkpoint(image_size=img_size)
    model = ThoraVisClassifier(num_classes=NUM_CLASSES, pretrained_ckpt=ckpt).to(device)

    trainer = ThoraVisTrainer.__new__(ThoraVisTrainer)
    trainer.device = device
    trainer.use_amp = False
    trainer.scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    trainer.model = model
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer.lr = 1e-4
    trainer.criterion = WeightedBCELoss()
    trainer.train_loader = _fake_batches(2, batch_size=2, img_size=img_size)
    trainer.val_loader = _fake_batches(2, batch_size=2, img_size=img_size)
    trainer.epochs = epochs
    trainer.warmup_epochs = 0
    trainer.checkpoint_dir = tmp_checkpoint_dir
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=max(1, epochs)
    )
    trainer.history = {"train_loss": [], "val_loss": [], "val_auc_macro": [], "val_auc_per_class": []}
    trainer.best_val_auc = 0.0
    trainer.epochs_since_improvement = 0
    trainer.early_stopping_patience = early_stopping_patience
    return trainer


class TestEarlyStopping(unittest.TestCase):
    def test_stops_after_patience_epochs_without_improvement(self):
        # Improves for 2 epochs, then plateaus/declines for the rest.
        auc_sequence = [0.60, 0.70, 0.65, 0.64, 0.63]

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _make_full_trainer(tmp_dir, epochs=5, early_stopping_patience=2)
            with patch.object(
                trainer, "_val_epoch",
                side_effect=[(0.5, auc, {}) for auc in auc_sequence],
            ):
                trainer.train()

        # epoch0=0.60 (saved), epoch1=0.70 (saved, counter reset),
        # epoch2=0.65 (counter=1), epoch3=0.64 (counter=2 -> stop after this)
        self.assertEqual(len(trainer.history["val_auc_macro"]), 4)
        self.assertEqual(trainer.best_val_auc, 0.70)

    def test_runs_all_epochs_when_patience_is_none(self):
        auc_sequence = [0.60, 0.61, 0.60, 0.59, 0.58]

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _make_full_trainer(tmp_dir, epochs=5, early_stopping_patience=None)
            with patch.object(
                trainer, "_val_epoch",
                side_effect=[(0.5, auc, {}) for auc in auc_sequence],
            ):
                trainer.train()

        self.assertEqual(len(trainer.history["val_auc_macro"]), 5)


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
