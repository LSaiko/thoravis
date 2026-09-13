# 🫁 ThoraVis — Thoracic Pathology Classifier

> **Multi-label chest X-ray classification using PyTorch + HuggingFace Transformers + OpenCV preprocessing**  
> Applied to the NIH ChestX-ray14 dataset (112,120 frontal-view images, 15 label classes)

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
├── .github/workflows/
│   └── tests.yml        ← CI: installs deps, runs tests/ on every push/PR
├── notebooks/
│   └── 01_thoravis_full_pipeline.ipynb   ← Main Jupyter showcase notebook
├── src/
│   ├── dataset.py       ← HuggingFace + PyTorch Dataset wrapper, global seeding
│   ├── preprocessing.py ← OpenCV clinical preprocessing pipeline
│   ├── model.py         ← ViT fine-tuning with custom classification head
│   ├── train.py         ← Training loop, AUC tracking, checkpointing
│   ├── predict.py       ← Single-image inference CLI
│   ├── evaluate.py      ← Per-pathology AUC-ROC, confusion matrices
│   └── gradcam.py       ← Grad-CAM heatmap generation with OpenCV overlay
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_predict.py
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
```

### 4. Run Inference on a Single Image

```bash
python -m src.predict --image path/to/xray.png --checkpoint models/best_thoravis.pt
```

Prints a sigmoid probability for each of the 15 pathology labels, marking the
ones at or above `--threshold` (default 0.5).

### 5. Run Tests

```bash
python -m unittest discover -s tests -v
```

Covers preprocessing output shapes, the classifier's forward pass (against a tiny
ViT checkpoint, not the full ViT-B/16), and label-vector / class-weight construction
in `ChestXrayDataset` — no GPU or full dataset download required.

---

## 📊 Results (Full dataset: ~73.5k train / ~13k val, 20 epochs)

First full run of the pipeline end to end — `python -m src.train --epochs 20
--batch_size 64 --lr 2e-5`, ~4.7h on a single RTX 5060. Val AUC peaks early
and then degrades as the unfrozen ViT backbone overfits; the checkpointing
logic already keeps the best epoch rather than the last one.

| Pathology | AUC-ROC (epoch 7, best checkpoint) |
|---|---|
| Edema | 0.890 |
| Hernia | 0.888 |
| Effusion | 0.881 |
| Cardiomegaly | 0.869 |
| Pneumothorax | 0.839 |
| Mass | 0.823 |
| Emphysema | 0.821 |
| Atelectasis | 0.799 |
| Consolidation | 0.789 |
| Pleural_Thickening | 0.777 |
| No Finding | 0.730 |
| Pneumonia | 0.723 |
| Fibrosis | 0.696 |
| Nodule | 0.696 |
| Infiltration | 0.582 |
| **Macro Average** | **0.787** |

- Best checkpoint: `models/best_thoravis.pt`, epoch 7/20, val macro AUC **0.7867** (not committed — see `.gitignore`; regenerate with the command above).
- By epoch 20, train loss keeps falling (0.286 → 0.139) while val loss climbs back up (0.243 → 0.255) and macro AUC drifts down to 0.749 — the model is memorizing past epoch ~7. Worth an early-stopping callback rather than a fixed 20 epochs.
- `Infiltration` and `Nodule` are the weakest classes — consistent with them being genuinely hard, diffuse, low-prevalence findings in ChestX-ray14, not obviously a pipeline bug.
- Full per-epoch history and this table's provenance: `results/run_summary.txt` (regenerate, gitignored).

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

**On the best checkpoint above:** pooled ECE is a deceptively good **0.019**
— but that's dominated by the large mass of easy, correctly-low-confidence
negatives across 15 labels. The reliability diagram tells the real story:
predictions in the 0.4–0.9 range (the ones that'd actually drive a decision)
are consistently overconfident — e.g. ~0.83 predicted maps to only ~0.36
observed frequency. Fitted temperature is a mild `T ≈ 1.07`. Don't trust the
single ECE number alone; look at the curve.

---

## 🔥 Grad-CAM Visualization

Gradient-weighted Class Activation Maps highlight *which pixels drove each prediction*, overlaid on the original X-ray using OpenCV's `applyColorMap`.

```python
from src.gradcam import GradCAMVisualizer

viz = GradCAMVisualizer(model, target_layer="vit.encoder.layer[-1]")
heatmap = viz.generate(image_tensor, target_class=2)  # Cardiomegaly
viz.save_overlay(original_image, heatmap, "results/gradcam_cardiomegaly.png")
```

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
