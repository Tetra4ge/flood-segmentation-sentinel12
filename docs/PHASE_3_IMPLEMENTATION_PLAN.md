# PHASE 3 — Implementation Plan

## Repository structure (generated for THIS project, not generic)

```
iew-reproduction/
├── configs/
│   ├── data.yaml                 # paths, band lists, patch size, split rules
│   ├── model_unet_baseline.yaml
│   ├── model_rupnet.yaml
│   ├── kd_student.yaml
│   ├── radar_model.yaml
│   └── experiment_registry.yaml  # maps experiment IDs (Phase 4) -> config combos
├── scripts/
│   ├── 00_inspect_dataset.py
│   ├── 01_visualize_samples.py
│   ├── 02_class_statistics.py
│   ├── 03_extract_patches.py
│   ├── 04_train.py
│   ├── 05_evaluate.py
│   ├── 06_distill_student.py
│   ├── 07_splitclass_postprocess.py
│   ├── 08_temporal_gapfill.py
│   └── 09_field_data_eval.py     # only runnable if field-data file exists
├── src/
│   ├── data/
│   │   ├── raster_io.py          # rasterio-based tif readers, band selection
│   │   ├── patch_dataset.py      # PyTorch Dataset over extracted patches
│   │   └── splits.py             # temporal split logic (BNPI/KNPI specific)
│   ├── models/
│   │   ├── unet.py
│   │   ├── rupnet.py             # [VERIFY architecture before writing]
│   │   └── registry.py
│   ├── losses/
│   │   ├── generalized_dice.py
│   │   ├── weighted_focal.py
│   │   └── kd_kl.py
│   ├── postprocess/
│   │   ├── splitclass.py         # Algorithm 1
│   │   └── temporal_gapfill.py   # Algorithm 2
│   └── eval/
│       └── metrics.py            # Pr/Rc/F1 per class, pixel-weighted aggregation
├── outputs/
│   ├── patches/                  # cached extracted patches (gitignored)
│   ├── checkpoints/
│   ├── logs/
│   ├── viz/
│   └── reports/
└── tests/
    └── test_algorithms.py        # unit tests for SplitClass + gap-filling logic
```

Rationale: patch caching is separated from raw tifs because with only 23 scenes but many patches, you'll re-extract with different strides during ablations — caching avoids recomputation and keeps raw data read-only.

---

## Milestone 1: Dataset Pipeline

**Tasks:**
- Load and validate S1, S2, GT, GT_0_8 tifs for all 23 scenes.
- Confirm band counts (S2=11, S1=3), GT class-ID sets, per-scene shape, nodata handling.
- Determine what `GT_0_8` actually represents (inspect values; do NOT assume it equals τ=0.98 SplitClass output — flag this explicitly in your report if unclear).
- Build the temporal-split logic from filenames.

**Commands:**
```bash
python scripts/00_inspect_dataset.py --root <DATASET_ROOT>/tifs
python scripts/02_class_statistics.py --root <DATASET_ROOT>/tifs --out outputs/reports/class_stats.csv
```

**Expected output:**
```
Dataset loaded successfully
Images: 14 train scenes, 9 test scenes
Masks: GT + GT_0_8 present for all scenes
Channels: S2=11, S1=3
Classes found in GT: {2,3,4,5,6,7,10,11}   <- must match; if not, STOP
Classes found in GT_0_8: {...}             <- report as-is, unknown meaning
Shape mismatches between S1/S2/GT for same scene: 0 (must be 0)
```

**Debugging strategy:** if S1/S2/GT shapes don't align per scene, check CRS/resampling — Sentinel-1 and Sentinel-2 have different native resolutions/grids and the paper notes acquisition dates differ by up to ±2 days, meaning the provider must have already co-registered them onto a common grid. If misaligned, do NOT proceed to patch extraction; re-derive a common grid first (e.g., resample S1 to S2's grid via rasterio `reproject`).

---

## Milestone 2: Baseline Model

**Before implementing the paper's method, build:**
- U-Net baseline (S2-only input, 11 channels → 6-class softmax over {2,3,4,5,6,7}).
- Standard training loop with checkpointing, logging (loss curves), and a simple config system.
- Evaluation pipeline computing per-class Pr/Rc/F1 and pixel-weighted overall F1 — build this once, reuse for every later milestone/experiment (this is your most reused piece of code, get it right first).

**Commands:**
```bash
python scripts/03_extract_patches.py --config configs/data.yaml --split train
python scripts/03_extract_patches.py --config configs/data.yaml --split test
python scripts/04_train.py --config configs/model_unet_baseline.yaml
python scripts/05_evaluate.py --checkpoint outputs/checkpoints/unet_baseline_best.pt --split test
```

**Expected output:**
```
Epoch 100/100 - train_loss: X.XXX - val_loss: X.XXX
Best checkpoint saved: outputs/checkpoints/unet_baseline_best.pt

Evaluation on test set:
Class 2 (water):   Pr=0.XX Rc=0.XX F1=0.XX
Class 3 (soil-3):  Pr=0.XX Rc=0.XX F1=0.XX
...
Overall (pixel-weighted): Pr=0.XX Rc=0.XX F1=0.XX
```

**Sanity target:** paper's U-Net row in Table 1 is Pr=78.58, Rc=80.20, F1=79.38 (classes 2–6 only, not including 7). Your number won't necessarily match exactly (small dataset, unstated augmentation/hyperparameters, unstated random seed) — treat >±5 F1 points as "plausibly consistent," not as pass/fail. Document your actual numbers, don't force-fit them to the paper's.

**Debugging strategy:** if loss doesn't decrease, check (a) ignore-index masking for classes 10/11 is actually wired into your loss function (easy to silently forget), (b) label remapping from raw class IDs {2..7} to contiguous {0..5} for softmax indexing, (c) class-weight computation isn't producing NaN/inf for any zero-count class in a given patch batch.

---

## Milestone 3: Paper Reproduction

**Implement:**
- RUPNet (teacher) — **[VERIFY architecture from the actual RUPNet paper before coding]**. Do not proceed on assumption.
- Train RUPNet as teacher on S2 (GDL+WFL, hard labels) → reproduce Table 1 row for RUPNet.
- Train RUPNet-architecture student via self-distillation (GDL+WFL+KL vs frozen teacher) → reproduce Table 2 "Student Top1" columns.
- Implement SplitClass (`src/postprocess/splitclass.py`, Algorithm 1) as inference-time post-processing on student softmax output, θ=0.98 → reproduce Table 2 "SplitClass τ=0.98" columns, using the "counts as correct if either class matches GT" evaluation rule.

**Files to create:** `src/models/rupnet.py`, `configs/model_rupnet.yaml`, `configs/kd_student.yaml`, `scripts/06_distill_student.py`, `src/losses/kd_kl.py`, `src/postprocess/splitclass.py`.

**Commands:**
```bash
python scripts/04_train.py --config configs/model_rupnet.yaml            # teacher
python scripts/06_distill_student.py --teacher outputs/checkpoints/rupnet_teacher_best.pt --config configs/kd_student.yaml
python scripts/07_splitclass_postprocess.py --checkpoint outputs/checkpoints/student_best.pt --theta 0.98 --split test
python scripts/05_evaluate.py --checkpoint outputs/checkpoints/student_best.pt --splitclass-theta 0.98 --split test
```

**Expected output:**
```
Teacher (RUPNet) Top1:  Pr=0.XX Rc=0.XX F1=0.XX  (compare to paper Table 1: Pr=76.35 Rc=86.70 F1=81.19)
Student Top1:           Pr=0.XX Rc=0.XX F1=0.XX  (compare to paper Table 2 "Student Top1", All row: Pr=0.98 Rc=0.97 F1=0.98)
Student SplitClass(0.98): Pr=0.XX Rc=0.XX F1=0.XX (compare to paper Table 2 "SplitClass", All row: Pr=0.99 Rc=0.99 F1=0.99)
SplitClass pixel coverage: XX.X% single-class, XX.X% dual-class (paper reports 6.33% dual-class overall)
```

**Debugging strategy:** if your SplitClass dual-class percentage is wildly different from 6.33%, your softmax confidence calibration differs from theirs (likely due to different training length/regularization/loss weights, all unstated in the excerpt) — this is expected and should be reported as a deviation, with a threshold-sensitivity plot (τ swept 0.5→0.99) rather than silently forcing τ to hit their exact percentage.

---

## Milestone 4: Temporal Gap Filling

**Implement:**
- Train the binary Radar_Model (S1-only, 3 channels, water/non-water) — separate small training run, binary GDL+WFL.
- Implement Algorithm 2 (`src/postprocess/temporal_gapfill.py`) combining S2_Model predictions across time + Radar_Model.
- If (and only if) you have a field-data file (GPS points + expert class + date): implement `scripts/09_field_data_eval.py` to reproduce Table 3-style single-day vs. gap-filled accuracy. If you don't have this file, **do not fabricate numbers** — report this experiment as blocked and explain why.

**Commands:**
```bash
python scripts/04_train.py --config configs/radar_model.yaml
python scripts/08_temporal_gapfill.py --s2-model outputs/checkpoints/student_best.pt --radar-model outputs/checkpoints/radar_best.pt --split test
python scripts/09_field_data_eval.py --field-data <PATH_IF_AVAILABLE> --predictions outputs/reports/gapfilled_predictions/
```

**Expected output:**
```
Radar Model (binary): Pr=0.XX Rc=0.XX F1=0.XX
Temporal gap-filling coverage: XX% of previously-missing S2 pixels resolved
[if field data available]
Single-day accuracy:      BNPI=XX.X%  KNPI=XX.X%   (paper: 58.25% / 59.45%)
Gap-filled accuracy:      BNPI=XX.X%  KNPI=XX.X%   (paper: 72.27% / 69.24%)
```
