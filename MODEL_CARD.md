# Model Card: ThoraVis

Format follows Mitchell et al., 2019 ("Model Cards for Model Reporting").

## Model Details

- **Architecture**: ViT-B/16 (`google/vit-base-patch16-224-in21k`, ImageNet-21k pretrained) + a 2-layer MLP head (`LayerNorm → Linear(768→256) → GELU → Dropout(0.3) → Linear(256→15)`), fine-tuned end to end. 86.6M parameters.
- **Task**: Multi-label classification — 15 sigmoid outputs (14 thoracic pathologies + "No Finding") per frontal chest X-ray.
- **Checkpoint evaluated in this card**: `models/best_thoravis.pt`, saved at epoch 9/20 of the run described in the "Results" section of [README.md](README.md) (not committed to git — regenerate via `python -m src.train --epochs 20 --batch_size 64 --lr 2e-5 --early-stopping-patience 3`, ~2.7h on an RTX 5060, early-stops before the full 20). Trained under the patient-grouped split (see Training Data below). The prior index-split checkpoint (epoch 7) is kept for comparison at `models/best_thoravis_oldsplit.pt`.
- **License**: MIT (code). The NIH ChestX-ray14 dataset itself requires attribution per NIH's terms — see README.
- **Developed by**: A single-author portfolio project (see README's own framing — this was built to demonstrate an OpenCV/PyTorch/HuggingFace pipeline, not as a funded clinical research effort).
- **Repository**: [github.com/LSaiko/thoravis](https://github.com/LSaiko/thoravis)

## Intended Use

**Primary intended uses**: ML engineering portfolio demonstration; a research baseline for benchmarking other architectures or preprocessing choices on ChestX-ray14.

**Plausible future uses, contingent on further validation** (not yet supported by evidence in this card): a triage-priority *aid* that reorders a radiologist's read queue — flagging studies with a suspected finding to be read sooner, never replacing or preceding a radiologist's read. A teaching aid pairing Grad-CAM overlays with NIH's ground-truth bounding boxes.

**Out of scope — do not use this for**:
- Any diagnostic decision, alone or as a "second opinion," for any patient.
- Any clinical deployment without prospective validation, regulatory clearance, and a much larger and more diverse evidence base than what's in this card.
- Populations, scanners, or acquisition protocols outside NIH ChestX-ray14's own (see Training Data below) — there is no evidence here that performance transfers.

## Training Data

- **Source**: NIH ChestX-ray14 (Wang et al., 2017), 112,120 frontal-view PNGs from 30,805 patients, via the `alkzar90/NIH-Chest-X-ray-dataset` HuggingFace wrapper.
- **Label provenance — read this before trusting any number in this card**: labels are **NLP-mined from radiology reports**, not adjudicated by a radiologist reading the image, per NIH's own dataset documentation. A model trained on these labels is learning to predict what an NLP pipeline extracted from a report, which correlates with but is not identical to ground truth. This ceiling applies to every metric below, not just this model's errors.
- **Split**: 73,916 train / 12,608 val / 25,596 test **for this checkpoint**. Val and test are NIH's own train_val/test partition; val is split out of the train_val pool by `GroupShuffleSplit` on patient ID (`get_dataloaders(use_patient_grouping=True)`, the default) — no patient's images can appear in both train and val. Patient ID is recovered by joining row position against NIH's own metadata CSV, since the classification config never exposes it directly; verified zero patient overlap on the real 86,524-row pool. A prior checkpoint trained under the old row-index split (73,546 train / 12,978 val, patient ID not accounted for) is kept for comparison at `models/best_thoravis_oldsplit.pt` — test macro AUC moved from 0.753 (old split) to 0.756 (patient-grouped split) on retraining, suggesting the leak, when present, wasn't large enough here to produce a visible inflation, though one run isn't proof of that in general.
- **Class balance**: "No Finding" is ~53% of the data; inverse-frequency weighting is applied in the training loss (`WeightedBCELoss`). Per-class positive counts vary by roughly two orders of magnitude across the 14 pathologies — see `results/run_summary.txt` for exact counts.

## Evaluation Data & Metrics

Two held-out evaluations exist, and they measure different things — see the README for the full table:

| | Macro AUC-ROC | What it actually measures |
|---|---|---|
| Validation (12,608 images) | 0.795 | Used to pick epoch 9 as "best" — carries a small optimistic bias by construction |
| **Test (25,596 images, never touched)** | **0.756** | The honest number |

**Calibration** (test set): pooled Expected Calibration Error 0.0144 — but read the README's Calibration section before trusting that alone. The pooled number is dominated by the large mass of easy, correctly-low-confidence negatives across 15 labels; the 0.4–0.9 predicted-probability band (the range that would actually drive a decision) shows real, systematic overconfidence: ECE 0.1684 there, vs. 0.0144 pooled. A fitted temperature of ≈1.05 helps a little (0.0112 pooled, 0.1660 in-band) but doesn't fix the in-band problem.

**Localization** (`src/bbox_eval.py`, 984 NIH-annotated boxes across 8 pathologies): Grad-CAM's attention peak beats a trivial "always guess image center" baseline on 4 of 8 pathologies (Pneumonia, Effusion, Pneumothorax, Nodule) — real evidence of *some* non-trivial localization. It does not beat the baseline on Cardiomegaly, Infiltration, Mass, or Atelectasis; Cardiomegaly's baseline (0.973) is inflated by the finding being anatomically central regardless of any real detection, and shouldn't be read as the model failing there specifically. In absolute terms (12.5% pooled pointing-game accuracy), this is far below published weakly-supervised CNN localization benchmarks (typically 40–60%+) — evidence of some signal, not a validated detector. See README for the full per-pathology table.

**Uncertainty**: `ThoraVisClassifier.predict_with_uncertainty()` provides MC-dropout std per class. One real limitation: the pretrained ViT-B/16 backbone itself ships with `hidden_dropout_prob=0.0` and `attention_probs_dropout_prob=0.0` — **all** MC-dropout variance comes from the classification head's single `Dropout(0.3)` layer, not the 86M-parameter backbone. It produces real, non-degenerate variance (std ≈ 0.08–0.18 observed on real inputs) but is a much weaker uncertainty signal than MC dropout over a fully-stochastic network — treat it as a coarse "is the head confident" check, not a rigorous epistemic uncertainty estimate.

## Ethical Considerations

- **Not a diagnostic device.** Nothing in this card constitutes clinical validation. See "Out of scope" above.
- **No demographic or fairness breakdown exists.** NIH ChestX-ray14's classification config (what this model trains on) does not expose age, sex, or other demographic fields, so no subgroup performance analysis has been done or is currently possible from this data source alone. Absence of a fairness problem has not been established — it hasn't been measured.
- **Single-institution, single-population data.** All images come from one clinical center's patient population and scanner set. No cross-institution or cross-scanner generalization evidence exists (see README's roadmap — cross-dataset validation against CheXpert/PadChest is an open, unattempted item).
- **Label noise is systematic, not random.** Because labels come from an NLP pipeline over reports rather than direct image review, errors likely correlate with report-writing style and reporting conventions at the source institution, not just random noise — this can bias what the model learns in ways a simple accuracy number won't reveal.

## Caveats and Recommendations

Before this model — or a successor trained the same way — is used for anything beyond what's in "Intended Use" above:

1. ~~**Close the patient-level split gap.**~~ Done — `get_dataloaders(use_patient_grouping=True)` (default), verified zero patient overlap on the real dataset, **and retrained under it**: this checkpoint's numbers above are patient-clean. Test macro AUC held steady (0.753 → 0.756 vs. the old index-split checkpoint), a mildly reassuring one-run result, not proof the leak never mattered.
2. **Validate cross-dataset.** An AUC on CheXpert or PadChest is the single most convincing thing this project could add — everything here is validated only against ChestX-ray14's own quirks.
3. **Recalibrate for the actionable range** — attempted with isotonic regression (`fit_isotonic_calibration()`), which can in principle correct the non-uniform miscalibration a single temperature can't. In practice it made things worse on this (retrained) checkpoint's real test set too (ECE 0.0275 vs. temperature scaling's 0.0112 overall; 0.2019 vs. 0.1660 in the 0.4–0.9 band) — same result as the prior checkpoint, now confirmed on a second independently-trained model. Likely overfitting validation-set noise across 15 pooled pathologies. Temperature scaling remains the better choice in practice; a per-class isotonic fit (rather than pooled) is the next thing worth trying, not yet attempted.
4. ~~**Add early stopping.**~~ Done — `train.py --early-stopping-patience N`. This checkpoint was itself selected via early stopping (patience=3, stopped at epoch 12, best was epoch 9), not hand-picked after the fact.
5. **Re-run `src/bbox_eval.py` on any future checkpoint before claiming localization has improved** — 984 images is a small, specific benchmark; don't extrapolate from it to "the model looks in the right place" in general.
