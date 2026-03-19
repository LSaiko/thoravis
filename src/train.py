"""
thoravis/src/train.py
─────────────────────────────────────────────────────────────────────────────
Training loop for ThoraVis with:
  - AdamW optimizer + cosine LR schedule with warmup
  - Per-epoch macro AUC-ROC tracking
  - Best-model checkpointing
  - Backbone warm-up strategy (freeze → unfreeze)

CLI usage
---------
    python src/train.py --subset 5000 --epochs 5 --batch_size 32 --lr 2e-5
"""

import argparse
import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from typing import Optional

from src.dataset import get_dataloaders, PATHOLOGY_LABELS, NUM_CLASSES
from src.model import ThoraVisClassifier, WeightedBCELoss


# ─── Trainer ─────────────────────────────────────────────────────────────────

class ThoraVisTrainer:
    """
    Full training / validation / evaluation pipeline for ThoraVisClassifier.

    Parameters
    ----------
    subset_size     : cap dataset size (None = full)
    epochs          : number of training epochs
    batch_size      : samples per batch
    lr              : peak learning rate
    warmup_epochs   : epochs to freeze backbone (head-only training)
    checkpoint_dir  : where to save model weights
    device          : 'cuda' / 'mps' / 'cpu'
    """

    def __init__(
        self,
        subset_size: Optional[int] = 5000,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 2e-5,
        warmup_epochs: int = 1,
        checkpoint_dir: str = "models",
        device: Optional[str] = None,
    ):
        self.subset_size    = subset_size
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.lr             = lr
        self.warmup_epochs  = warmup_epochs
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Auto-detect device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Device: {self.device}")

        # Data
        print("\n── Loading data ─────────────────────────────────────────────")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            subset_size=subset_size,
            batch_size=batch_size,
        )

        # Model (start with frozen backbone for warmup)
        print("\n── Building model ───────────────────────────────────────────")
        self.model = ThoraVisClassifier(
            num_classes=NUM_CLASSES,
            freeze_backbone=(warmup_epochs > 0),
        ).to(self.device)

        param_info = self.model.count_parameters()
        print(f"  Trainable params : {param_info['trainable']:,}")
        print(f"  Total params     : {param_info['total']:,}")

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=1e-2,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7
        )
        self.criterion = WeightedBCELoss()

        # History
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_auc_macro": [], "val_auc_per_class": [],
        }
        self.best_val_auc = 0.0

    # ── Epoch routines ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def _val_epoch(self, epoch: int):
        self.model.eval()
        total_loss   = 0.0
        all_probs    = []
        all_labels   = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [val]  ", leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss   = self.criterion(logits, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())

        all_probs  = np.vstack(all_probs)     # (N, 14)
        all_labels = np.vstack(all_labels)    # (N, 14)

        # Per-class AUC (skip classes with no positives)
        aucs = []
        auc_per_class = {}
        for i, label_name in enumerate(PATHOLOGY_LABELS):
            if all_labels[:, i].sum() > 0:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                aucs.append(auc)
                auc_per_class[label_name] = round(auc, 4)

        macro_auc = float(np.mean(aucs)) if aucs else 0.0
        val_loss  = total_loss / len(self.val_loader)

        return val_loss, macro_auc, auc_per_class

    # ── Main training loop ─────────────────────────────────────────────────────

    def train(self):
        print("\n── Training ─────────────────────────────────────────────────")
        start = time.time()

        for epoch in range(self.epochs):

            # Unfreeze backbone after warmup
            if epoch == self.warmup_epochs and self.warmup_epochs > 0:
                self.model.unfreeze_backbone()
                # Re-create optimizer now that all params are active
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=self.lr * 0.5, weight_decay=1e-2
                )

            train_loss             = self._train_epoch(epoch)
            val_loss, macro_auc, per_class = self._val_epoch(epoch)

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # History
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc_macro"].append(macro_auc)
            self.history["val_auc_per_class"].append(per_class)

            # Checkpoint
            if macro_auc > self.best_val_auc:
                self.best_val_auc = macro_auc
                ckpt_path = os.path.join(self.checkpoint_dir, "best_thoravis.pt")
                torch.save({
                    "epoch":       epoch + 1,
                    "state_dict":  self.model.state_dict(),
                    "val_auc":     macro_auc,
                    "history":     self.history,
                }, ckpt_path)
                ckpt_marker = " ✓ (saved)"
            else:
                ckpt_marker = ""

            elapsed = (time.time() - start) / 60
            print(
                f"Epoch {epoch+1:02d}/{self.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_AUC={macro_auc:.4f}"
                f"{ckpt_marker}  [{elapsed:.1f}m]"
            )

        print(f"\nTraining complete. Best Val AUC: {self.best_val_auc:.4f}")
        return self.history

    def print_per_class_auc(self):
        """Pretty-print the final epoch per-class AUC scores."""
        if not self.history["val_auc_per_class"]:
            print("No results yet — run train() first.")
            return
        last = self.history["val_auc_per_class"][-1]
        print("\n── Per-class AUC (final epoch) ──────────────────────────────")
        for label, auc in sorted(last.items(), key=lambda x: -x[1]):
            bar = "█" * int(auc * 20)
            print(f"  {label:<22s}  {auc:.4f}  {bar}")
        print(f"\n  Macro AUC: {self.history['val_auc_macro'][-1]:.4f}")


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ThoraVis classifier")
    parser.add_argument("--subset",     type=int,   default=5000)
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--warmup",     type=int,   default=1)
    parser.add_argument("--device",     type=str,   default=None)
    args = parser.parse_args()

    trainer = ThoraVisTrainer(
        subset_size=args.subset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup,
        device=args.device,
    )
    trainer.train()
    trainer.print_per_class_auc()


if __name__ == "__main__":
    main()
