# 🫁 ThoraVis — Thoracic Pathology Classifier

> **Multi-label chest X-ray classification using PyTorch + HuggingFace Transformers + OpenCV preprocessing**  
> Applied to the NIH ChestX-ray14 dataset (112,120 frontal-view images, 15 label classes)

**→ Read [MODEL_CARD.md](MODEL_CARD.md) before using this for anything beyond the portfolio/research use case it was built for.**

---

![Tests](https://github.com/LSaiko/thoravis/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

**ThoraVis** is a production-style medical imaging pipeline that:
- Loads and explores the **NIH ChestX-ray14** dataset via HuggingFace Datasets
- Applies clinical-grade **OpenCV preprocessing** (CLAHE enhancement, lung-field normalization, adaptive thresholding)
- Fine-tunes a **ViT-B/16** (Vision Transformer) backbone from HuggingFace for multi-label classification
- Implements a custom **PyTorch** training loop with AUC-ROC tracking per pathology
- Exports **Grad-CAM heatmaps** (OpenCV overlay) to visualize model attention on X-ray findings
- Achieves competitive **AUC ≥ 0.80** on high-prevalence pathologies (Effusion, Atelectasis, Cardiomegaly)

This project was built to concretely demonstrate:
| Skill | Demonstrated Via |
|---|---|
| **OpenCV** | CLAHE, bilateral filtering, Sobel edge maps, Grad-CAM overlays |
| **PyTorch** | Custom Dataset, DataLoader, training loop, loss functions |
| **HuggingFace** | `datasets` for data streaming, `transformers` ViT model backbone |
| **ML Engineering** | AUC-ROC metrics, class imbalance handling, checkpoint management |

---

## 🗂️ Repository Structure

```
thoravis/
├── README.md
├── requirements.txt
├── requirements-api.txt ← Serving-only deps for Docker (no jupyter/matplotlib)
├── MODEL_CARD.md        ← Read before using this beyond the portfolio use case
├── Dockerfile           ← CPU inference image (see "Run via Docker")
├── .dockerignore
├── .github/workflows/
│   └── tests.yml        ← CI: installs deps, runs tests/ on every push/PR
├── notebooks/
│   └── 01_thoravis_full_pipeline.ipynb   ← Main Jupyter showcase notebook
├── src/
│   ├── dataset.py       ← HuggingFace + PyTorch Dataset wrapper, global seeding
│   ├── preprocessing.py ← OpenCV clinical preprocessing pipeline
│   ├── model.py         ← ViT fine-tuning, MC-dropout uncertainty
│   ├── train.py         ← Training loop, AUC tracking, checkpointing
│   ├── predict.py       ← Single-image inference CLI
│   ├── api.py           ← FastAPI inference service (/predict, /predict/gradcam)
│   ├── export.py        ← TorchScript export CLI
│   ├── evaluate.py      ← Per-pathology AUC-ROC, calibration
│   ├── gradcam.py       ← Grad-CAM heatmap generation with OpenCV overlay
│   └── bbox_eval.py     ← Grad-CAM vs. radiologist bounding boxes
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_predict.py
│   ├── test_api.py
│   ├── test_export.py
│   ├── test_gradcam.py
│   ├── test_bbox_eval.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   └── test_dataset.py
├── models/
│   └── .gitkeep
├── results/
│   └── .gitkeep
└── assets/
    └── .gitkeep
```

---

## 🧬 Dataset: NIH ChestX-ray14

| Property | Value |
|---|---|
| Source | [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| HuggingFace Hub | [`alkzar90/NIH-Chest-X-ray-dataset`](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset) |
| Images | 112,120 frontal-view PNGs (1024×1024) |
| Patients | 30,805 unique |
| Labels | 15 label classes: 14 diseases + `No Finding` (multi-label) |
| License | No restrictions (NIH attribution required) |

**15 Label Classes (14 diseases + No Finding):**
`No Finding` · `Atelectasis` · `Cardiomegaly` · `Effusion` · `Infiltration` · `Mass` · `Nodule` · `Pneumonia` · `Pneumothorax` · `Consolidation` · `Edema` · `Emphysema` · `Fibrosis` · `Pleural Thickening` · `Hernia`

**Patient-grouped splitting.** The `image-classification` config used above never
exposes patient ID or filename — but its row order was verified (empirically,
against NIH's own `Data_Entry_2017_v2020.csv`) to match that CSV's row order
exactly for grouping purposes: some images get reordered *within* a patient's
own follow-up sequence, but none ever land in a different patient's block. So
`get_dataloaders()` recovers patient ID by row position and defaults to
`use_patient_grouping=True` — a `GroupShuffleSplit` on patient ID instead of a
raw index cut, guaranteeing no patient's images can straddle the train/val
boundary. Falls back to the old index-range split (with a printed warning) if
the metadata fetch fails. **The checkpoint reported below was trained under
this patient-grouped split** — see the Results section.

---

## 🔬 Pipeline Architecture

```
Raw X-ray PNG (1024×1024)
        │
        ▼
[OpenCV Preprocessing]
  ├─ Resize to 224×224
  ├─ CLAHE contrast enhancement (clipLimit=3.0, tileGrid=8×8)
  ├─ Bilateral noise filter (preserve edges)
  └─ Normalize to ImageNet stats
        │
        ▼
[HuggingFace ViT-B/16 Backbone]
  └─ google/vit-base-patch16-224-in21k (pretrained)
        │
        ▼
[Custom PyTorch Classification Head]
  └─ Linear(768 → 256) → GELU → Dropout(0.3) → Linear(256 → 15)
        │
        ▼
[Multi-label BCEWithLogitsLoss]
  └─ Weighted by inverse class frequency
        │
        ▼
[Output: 15-dim sigmoid probabilities]
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/LSaiko/thoravis.git
cd thoravis
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook notebooks/01_thoravis_full_pipeline.ipynb
```

This single notebook walks through the complete pipeline end-to-end, including:
- Dataset loading and exploration
- OpenCV preprocessing visualization
- Model training (configurable epochs / subset size)
- Evaluation with per-class AUC-ROC
- Grad-CAM heatmap generation

### 3. Train via Script

```bash
# Quick demo run on 5,000 images
python -m src.train --subset 5000 --epochs 5 --batch_size 32

# Full dataset run
python -m src.train --epochs 20 --batch_size 64 --lr 2e-5

# Disable mixed precision (on by default when a CUDA GPU is available)
python -m src.train --epochs 20 --batch_size 64 --no-amp

# Stop once val AUC hasn't improved for 3 epochs, instead of a fixed 20
python -m src.train --epochs 20 --batch_size 64 --early-stopping-patience 3
```

### 4. Run Inference on a Single Image

```bash
python -m src.predict --image path/to/xray.png --checkpoint models/best_thoravis.pt
```

Prints a sigmoid probability for each of the 15 pathology labels, marking the
ones at or above `--threshold` (default 0.5). Add `--uncertainty 20` to run
20-sample MC dropout instead and print `mean ± std` per label — see
[MODEL_CARD.md](MODEL_CARD.md) for what this uncertainty estimate does and
doesn't capture before treating a low std as reassuring.

### 5. Run the Inference API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Returns |
|---|---|---|
| `/health` | GET | device, checkpoint path, checkpoint epoch |
| `/predict` | POST (multipart `file`) | JSON: probability per pathology (add `?n_samples=20` for MC-dropout `uncertainty`) |
| `/predict/gradcam?label=Cardiomegaly` | POST (multipart `file`) | PNG Grad-CAM overlay for that label |

```bash
curl -F file=@xray.png "http://localhost:8000/predict?threshold=0.5"
curl -F file=@xray.png "http://localhost:8000/predict/gradcam?label=Effusion" -o gradcam.png
```

Set `THORAVIS_CHECKPOINT` to point at a different checkpoint (default `models/best_thoravis.pt`).

### 6. Run via Docker

```bash
docker build -t thoravis .

# Use an absolute host path to models/ — $(pwd) doesn't reliably translate
# through Docker Desktop for Windows' volume-mount path handling.
docker run -p 8000:8000 -v /absolute/path/to/thoravis/models:/app/models thoravis
```

Builds to ~3GB using `requirements-api.txt` (serving deps only — no
jupyter/notebook/training tooling) and CPU-only PyTorch pulled explicitly
from PyTorch's own CPU wheel index, since PyPI's default `torch` wheel now
bundles the full NVIDIA CUDA runtime as pip dependencies (~10GB) even with
no GPU involved. The checkpoint is mounted at runtime, not baked into the
image, since it's ~350MB and gitignored. For GPU inference, install a
CUDA-enabled torch build in the Dockerfile and run with `docker run --gpus all`.

Verified end to end against the real epoch-7 checkpoint: `docker build`,
`docker run` with the checkpoint mounted, and all three endpoints
(`/health`, `/predict`, `/predict/gradcam`) hit successfully from inside
the running container.

### 7. Export to TorchScript

```bash
python -m src.export --checkpoint models/best_thoravis.pt --out models/thoravis_traced.pt
```

Produces a single file that loads and runs with plain `torch` — no
`transformers`/HuggingFace Hub access needed at inference time. Verifies
eager vs. traced output match before saving.

### 8. Run Tests

```bash
python -m unittest discover -s tests -v
```

55 tests, all offline against a tiny local ViT checkpoint instead of the full
ViT-B/16 or the real dataset — preprocessing shapes, the model forward pass,
MC-dropout uncertainty, Grad-CAM (including a regression test for the
second-to-last-layer fix below), TorchScript export, the API's endpoints
(via FastAPI's TestClient), the bbox-eval pointing-game metric, and dataset
label/split logic. No GPU or dataset download required. (`src/bbox_eval.py`
itself needs real images and a real checkpoint — it's validated by an
actual run against the real data, documented in the Grad-CAM section below,
not a unit test.)

---

## 📊 Results (Full dataset: ~73.9k train / ~12.6k val, patient-grouped split, 20 epochs)

`python -m src.train --epochs 20 --batch_size 64 --lr 2e-5
--early-stopping-patience 3`, ~2.7h on a single RTX 5060 (early stopping cut
it well short of the full 20 epochs). Trained under the patient-grouped
split (`GroupShuffleSplit` on patient ID — no patient's images appear in
both train and val) described in the Dataset section below. Val AUC peaks
around epoch 9 and then degrades as the unfrozen ViT backbone overfits;
early stopping triggered after 3 epochs without improvement (epoch 12).

| Pathology | Val AUC (epoch 9, used for checkpoint selection) | Test AUC (held out, never touched) |
|---|---|---|
| Edema | 0.902 | 0.834 |
| Cardiomegaly | 0.868 | 0.846 |
| Effusion | 0.878 | 0.794 |
| Hernia | 0.864 | 0.846 |
| Pneumothorax | 0.823 | 0.807 |
| Emphysema | 0.808 | 0.789 |
| Consolidation | 0.803 | 0.717 |
| Mass | 0.801 | 0.755 |
| Pleural_Thickening | 0.779 | 0.725 |
| Fibrosis | 0.776 | 0.761 |
| Atelectasis | 0.776 | 0.719 |
| No Finding | 0.747 | 0.707 |
| Pneumonia | 0.714 | 0.669 |
| Nodule | 0.706 | 0.691 |
| Infiltration | 0.674 | 0.684 |
| **Macro Average** | **0.795** | **0.756** |

- Best checkpoint: `models/best_thoravis.pt`, epoch 9/20 (not committed — see `.gitignore`; regenerate with the command above). The prior index-split checkpoint (epoch 7, val 0.787 / test 0.753) is kept for comparison at `models/best_thoravis_oldsplit.pt`.
- **The test column matters more than the val column.** `get_dataloaders()` builds a `test_loader` that `ThoraVisTrainer.train()` never touches — every number anywhere else in this README (health check included) was on validation, which was also used to pick epoch 9 as "best," so it carries a small optimistic bias by construction. The 0.795 → 0.756 gap (val → true held-out test) is that bias, made visible instead of assumed away — a modest, expected-sized gap, not a red flag, and essentially the same size as the old index-split checkpoint's gap (0.787 → 0.753). **The patient-grouped split did not inflate the old numbers in any visible way** — test macro AUC actually moved slightly up (0.753 → 0.756), well within run-to-run noise for a single training run. That's a mildly reassuring result, not proof of anything: one run isn't enough to conclude the leak (when it existed) had zero effect, only that it wasn't large enough here to show up as an obvious drop.
- Early stopping (`--early-stopping-patience 3`) triggered after epoch 12, replacing the earlier hand-picked-after-the-fact epoch selection.
- `Infiltration`, `Nodule`, and `Pneumonia` are the weakest classes on test — consistent with them being genuinely hard, diffuse, low-prevalence findings in ChestX-ray14, not obviously a pipeline bug. Per-class ranking isn't perfectly stable between val and test — expected sampling noise at these per-class positive counts, not evidence of anything wrong.
- Test-set calibration: ECE 0.0144 (`results/test_set_evaluation_newsplit.txt`, regenerate, gitignored). Full per-epoch training log: `results/train_full_run.log`; the prior run's summary is kept at `results/run_summary_oldsplit.txt` for comparison.
- This checkpoint was trained and evaluated entirely under the patient-grouped split — see the Dataset section below for how patient ID is recovered and verified.

---

## 🎯 Calibration

A high AUC only says the model *ranks* positives above negatives — it says
nothing about whether `sigmoid(logits)` is a trustworthy probability. Before
treating any output as "70% chance of Effusion", check:

```python
from src.evaluate import collect_logits, fit_temperature, plot_reliability_diagram
import torch

logits, labels = collect_logits(model, val_loader, device)
plot_reliability_diagram(torch.sigmoid(torch.tensor(logits)).numpy(), labels,
                          save_path="results/reliability_diagram.png")

# Fit on a validation split, then apply sigmoid(logits / T) at inference time
temperature = fit_temperature(logits, labels)
```

`expected_calibration_error()` reports a single pooled ECE number if you just
need a scalar rather than the plot.

**On the best checkpoint above:** raw (uncalibrated) test-set ECE is
**0.0144** overall, but that's dominated by the large mass of easy,
correctly-low-confidence negatives across 15 labels. The 0.4–0.9 band (the
range that'd actually drive a decision) tells the real story: ECE **0.1684**
there — real, sizeable overconfidence. Temperature scaling (fit on val,
`T ≈ 1.05`) helps some (0.0112 overall, 0.1660 in-band) but doesn't erase
the in-band problem. Don't trust the single overall ECE number alone; look
at the band.

**Tried fixing that band specifically with isotonic regression**
(`fit_isotonic_calibration()`) since a single temperature can't in
principle correct miscalibration that varies across the confidence range.
Verified it works on synthetic data shaped like this exact problem — then,
on the real checkpoint, it made things *worse*: test-set ECE 0.0275 vs.
temperature scaling's 0.0112 overall, 0.2019 vs. 0.1660 in the 0.4–0.9 band
specifically (`results/calibration_comparison.txt`, regenerate,
gitignored). Same finding as with the prior (index-split) checkpoint, now
confirmed on a second, independently-trained model. Most likely cause:
pooling across 15 pathologies with different base rates gives isotonic
regression enough freedom to fit validation-set noise that doesn't
transfer to test. Temperature scaling remains the better real-world choice
here, despite being the theoretically cruder tool — worth knowing before
reaching for the fancier method by default.

---

## 🔥 Grad-CAM Visualization

Gradient-weighted Class Activation Maps highlight *which pixels drove each prediction*, overlaid on the original X-ray using OpenCV's `applyColorMap`.

```python
from src.gradcam import GradCAMViT
from src.preprocessing import overlay_heatmap

cam = GradCAMViT(model)
heatmap = cam.generate(image_tensor, target_class=2)  # Cardiomegaly
blended = overlay_heatmap(original_image_np, heatmap)
cam.remove_hooks()
```

**Which transformer layer to hook matters, and it's not the last one.** With
a CLS-token-only classification head, the final layer's own patch-token
outputs have *zero* gradient path to the logit — `ViTModel`'s last LayerNorm
operates per-token, so it can't mix patch information into the CLS
position. `GradCAMViT` hooks the second-to-last layer by default
(`layer_index=-2`); hooking the last layer produces an all-zero heatmap on
every input, silently — the class's own bounds ([0,1]) don't catch that a
constant heatmap satisfies them too. Verified with `tests/test_gradcam.py`.

### Validation: does it look where a radiologist would?

NIH released 984 hand-drawn bounding boxes across 8 pathologies
(`BBox_List_2017.csv`) — a real, checkable measure of localization, not
just a plausible-looking heatmap. `src/bbox_eval.py` runs Grad-CAM against
every annotated image and checks whether the heatmap's peak pixel falls
inside the radiologist's box ("pointing game" accuracy, Zhang et al. 2016),
alongside a trivial "always guess image center" baseline for comparison —
chest X-ray anatomy is roughly centered, so that baseline is a real bar to
clear, not a strawman.

```bash
python -m src.bbox_eval --checkpoint models/best_thoravis.pt --image-root <dir with extracted NIH images>
```

| Pathology | Model | Center baseline | n |
|---|---|---|---|
| Cardiomegaly | 0.336 | 0.973 | 146 |
| Pneumonia | 0.208 | 0.142 | 120 |
| Infiltration | 0.179 | 0.228 | 123 |
| Mass | 0.071 | 0.106 | 85 |
| Effusion | 0.065 | 0.026 | 153 |
| Pneumothorax | 0.041 | 0.010 | 98 |
| Atelectasis | 0.033 | 0.061 | 180 |
| Nodule | 0.013 | 0.000 | 79 |
| **Overall** | **0.125** | **0.215** | 984 |

The pooled "overall" row is honestly a bit misleading — don't read it on
its own. The model beats the naive center-guess on 4 of 8 pathologies
(Pneumonia, Effusion, Pneumothorax, Nodule), which is real evidence of
non-trivial localization for those findings. But Cardiomegaly (heart
enlargement) is anatomically central by nature, so the center baseline
scores 0.973 there by luck alone regardless of any actual detection —
that single pathology's baseline dominance drags the pooled average below
the model's own average. Read per pathology, not the aggregate. None of
these numbers are high in absolute terms (published weakly-supervised
CNN localization on this benchmark typically clears 40-60%+) — this is
evidence of *some* real signal, not a validated detector.

---

## 🛠️ Key Technical Choices

**Why ViT over CNN?**  
Vision Transformers capture global context across the entire X-ray in a single forward pass — crucial for diffuse pathologies like Cardiomegaly or Edema that span large anatomical regions.

**Why CLAHE preprocessing?**  
X-rays have extreme dynamic range. CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes local contrast without oversaturating bright bone structures — a standard step in clinical CAD systems.

**Handling class imbalance:**  
`No Finding` comprises ~53% of labels. We apply inverse-frequency weighting in `BCEWithLogitsLoss` and use macro AUC-ROC (not accuracy) as the primary metric.

---

## 📦 Requirements

```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.38.0
datasets>=2.18.0,<4.0.0
opencv-python>=4.9.0
numpy>=1.26.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
tqdm>=4.66.0
accelerate>=0.27.0
Pillow>=10.2.0
jupyter>=1.0.0
ipywidgets>=8.1.0
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
```

---

## 📚 Citation

```bibtex
@inproceedings{Wang_2017,
  doi       = {10.1109/cvpr.2017.369},
  year      = 2017,
  publisher = {IEEE},
  author    = {Xiaosong Wang and Yifan Peng and Le Lu and Zhiyong Lu
               and Mohammadhadi Bagheri and Ronald M. Summers},
  title     = {ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks},
  booktitle = {2017 IEEE Conference on Computer Vision and Pattern Recognition}
}
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE).  
Dataset attribution: NIH Clinical Center (required per dataset terms).

---

*Built to showcase applied medical imaging skills: OpenCV preprocessing · PyTorch training pipelines · HuggingFace model fine-tuning*
