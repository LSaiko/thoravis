"""
thoravis/src/dataset.py
─────────────────────────────────────────────────────────────────────────────
PyTorch Dataset wrapping the NIH ChestX-ray14 dataset loaded via HuggingFace.

Usage
-----
    from src.dataset import ChestXrayDataset, get_dataloaders, PATHOLOGY_LABELS

    train_loader, val_loader, test_loader = get_dataloaders(
        subset_size=5000,    # None = full 112k
        batch_size=32,
    )
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Optional, Tuple, List
from src.preprocessing import XRayPreprocessor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ─── Label definitions ────────────────────────────────────────────────────────

PATHOLOGY_LABELS: List[str] = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_CLASSES = len(PATHOLOGY_LABELS)


# ─── Dataset class ────────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    Multi-label chest X-ray dataset.

    Wraps `alkzar90/NIH-Chest-X-ray-dataset` from HuggingFace Hub.
    Each sample returns:
      - image  : torch.FloatTensor  (3, 224, 224)
      - labels : torch.FloatTensor  (15,) — binary multi-hot vector

    Parameters
    ----------
    hf_split    : HuggingFace dataset split ('train' / 'test')
    preprocessor: XRayPreprocessor instance
    indices     : explicit list of dataset indices to use (takes priority over subset_size)
    subset_size : cap on number of samples from index 0 (None = use all)
    """

    def __init__(
        self,
        hf_split: str = "train",
        preprocessor: Optional[XRayPreprocessor] = None,
        indices: Optional[List[int]] = None,
        subset_size: Optional[int] = None,
    ):
        print(f"Loading NIH ChestX-ray14 [{hf_split}] from HuggingFace Hub...")
        self.ds = load_dataset(
            "alkzar90/NIH-Chest-X-ray-dataset",
            "image-classification",
            split=hf_split,
            trust_remote_code=True,
        )

        if indices is not None:
            self.ds = self.ds.select(indices)
        elif subset_size is not None:
            self.ds = self.ds.select(range(min(subset_size, len(self.ds))))

        self.preprocessor = preprocessor or XRayPreprocessor()
        print(f"  → {len(self.ds):,} samples loaded.")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.ds[idx]
        pil_img = sample["image"]
        raw_labels: List[int] = sample["labels"]   # list of active class indices

        # OpenCV preprocessing pipeline
        image_tensor = self.preprocessor.preprocess_pil(pil_img)

        # Convert sparse label indices → dense multi-hot binary vector
        label_vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for lbl in raw_labels:
            if 0 <= lbl < NUM_CLASSES:
                label_vec[lbl] = 1.0

        return image_tensor, label_vec

    def class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights to handle label imbalance.
        Returns tensor of shape (NUM_CLASSES,).
        """
        print("Computing class weights (this may take a moment)...")
        counts = torch.zeros(NUM_CLASSES)
        for sample in self.ds:
            for lbl in sample["labels"]:
                if 0 <= lbl < NUM_CLASSES:
                    counts[lbl] += 1

        # Avoid division by zero; invert frequency
        total = len(self.ds)
        weights = total / (counts.clamp(min=1) * NUM_CLASSES)
        return weights


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    """Seed python/numpy/torch RNGs so a run is reproducible end to end."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    """Re-seed numpy/random per DataLoader worker (each forks with the same
    numpy RNG state otherwise, so RandomHorizontalFlip/ColorJitter etc. would
    repeat identically across workers)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


_patient_id_cache: Optional[np.ndarray] = None


def _load_patient_ids_for_train_split() -> np.ndarray:
    """
    Patient ID for every row of the HF 'train' split, in the exact order
    the dataset yields them — memoized per process (one network fetch,
    cached to disk afterward by huggingface_hub regardless).

    The image-classification config's own _generate_examples() never
    exposes patient ID or filename directly. But its row order matches
    NIH's own Data_Entry_2017 CSV, filtered to train_val_list.txt, in CSV
    order: verified empirically that for all 86,524 train_val rows, the
    dataset's label set at row i is found somewhere within row i's own
    Patient ID block in the CSV (some images get reordered *within* a
    patient's own follow-up sequence, which doesn't matter for grouping —
    none ever land in a different patient's block). That's what makes
    patient ID recoverable by position even though per-row label identity
    isn't guaranteed to align exactly.
    """
    global _patient_id_cache
    if _patient_id_cache is not None:
        return _patient_id_cache

    from huggingface_hub import hf_hub_download
    import pandas as pd

    csv_path = hf_hub_download(
        "alkzar90/NIH-Chest-X-ray-dataset", "data/Data_Entry_2017_v2020.csv", repo_type="dataset"
    )
    train_val_path = hf_hub_download(
        "alkzar90/NIH-Chest-X-ray-dataset", "data/train_val_list.txt", repo_type="dataset"
    )
    with open(train_val_path) as f:
        train_val_names = set(line.strip() for line in f if line.strip())

    df = pd.read_csv(csv_path)
    df = df[df["Image Index"].isin(train_val_names)].reset_index(drop=True)
    _patient_id_cache = df["Patient ID"].to_numpy()
    return _patient_id_cache


def _train_val_indices(
    total: int,
    val_split: float,
    test_reserve: int = 0,
    patient_ids: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Disjoint (train_indices, val_indices) covering a pool of `total` items,
    reserving `test_reserve` items off the top for a demo-mode test slice
    (0 when the real held-out test split is used instead, as in full
    dataset mode). Always returns non-overlapping ranges — regression guard
    for a past bug where a `None` total silently made train == val.

    If `patient_ids` is given, splits by GroupShuffleSplit keyed on patient
    ID instead of a raw index cut, so no patient's images can straddle the
    train/val boundary. Falls back to the plain index-range split
    otherwise (e.g. when patient IDs couldn't be fetched).
    """
    pool_end = total - test_reserve

    if patient_ids is not None:
        from sklearn.model_selection import GroupShuffleSplit

        pool_indices = np.arange(pool_end)
        pool_patient_ids = patient_ids[:pool_end]
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        train_idx, val_idx = next(splitter.split(pool_indices, groups=pool_patient_ids))
        return sorted(pool_indices[train_idx].tolist()), sorted(pool_indices[val_idx].tolist())

    n_val   = int(total * val_split)
    n_train = total - n_val - test_reserve
    train_indices = list(range(0, n_train))
    val_indices   = list(range(n_train, n_train + n_val))
    return train_indices, val_indices


# ─── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(
    subset_size: Optional[int] = 5000,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 2,
    seed: int = 42,
    use_patient_grouping: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from HuggingFace NIH ChestX-ray14.

    Args
    ----
    subset_size : total images to use (None = full dataset ~112k)
    batch_size  : samples per batch
    val_split   : fraction for validation
    test_split  : fraction for test
    num_workers : DataLoader workers
    seed        : reproducibility
    use_patient_grouping : split train/val by patient ID (GroupShuffleSplit)
                  instead of a raw index cut, so no patient's images can
                  appear in both. Requires one extra network fetch (NIH's
                  own metadata CSV); falls back to the index-range split
                  with a printed warning if that fetch fails.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    set_global_seed(seed)

    train_preprocessor = XRayPreprocessor(augment=True)
    val_preprocessor   = XRayPreprocessor(augment=False)

    patient_ids = None
    if use_patient_grouping:
        try:
            patient_ids = _load_patient_ids_for_train_split()
        except Exception as exc:
            print(f"Warning: couldn't fetch patient IDs ({exc}); "
                  f"falling back to index-range train/val split.")

    if subset_size is not None:
        # Demo mode: carve a small train/val/test triple out of subset_size,
        # sampling n_test from the *real* held-out test split (not from the
        # train pool) so it stays a genuine out-of-sample check.
        n_test = int(subset_size * test_split)
        train_indices, val_indices = _train_val_indices(
            subset_size, val_split, test_reserve=n_test, patient_ids=patient_ids, seed=seed,
        )
    else:
        # Full run: NIH ChestX-ray14 only ships train/test, no train/val —
        # carve val out of the full train split ourselves. Loading it here
        # just to read its length is cheap once cached (Arrow-backed, no
        # re-download); train_ds below reuses the same cache.
        full_train_len = len(load_dataset(
            "alkzar90/NIH-Chest-X-ray-dataset", "image-classification", split="train",
        ))
        n_test = None  # use the entire real test split, uncapped
        train_indices, val_indices = _train_val_indices(
            full_train_len, val_split, test_reserve=0, patient_ids=patient_ids, seed=seed,
        )

    train_ds = ChestXrayDataset(
        hf_split="train",
        preprocessor=train_preprocessor,
        indices=train_indices,
    )
    val_ds = ChestXrayDataset(
        hf_split="train",
        preprocessor=val_preprocessor,
        indices=val_indices,
    )
    test_ds = ChestXrayDataset(
        hf_split="test",
        preprocessor=val_preprocessor,
        subset_size=n_test,
    )

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, val_loader, test_loader
