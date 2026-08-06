# PHASE 2 — Technical Requirement Document (TRD)

> As with the PRD: architectural details for RUPNet specifically are **not fully specified in the paper excerpt you gave me**, and I am not certain of the exact layer structure from memory. Everywhere below marked **[VERIFY]** means: don't trust my recollection, check the cited source or run a controlled test yourself.

---

## 1. System Architecture (full pipeline)

```
Raw GeoTIFF (S1, S2, GT, GT_0_8)
        │
        ▼
Preprocessing
  - S2: keep 11 bands (drop B9, B10), reflectance scaling [VERIFY exact scale factor, likely /10000 for L2A BOA]
  - S1: Sigma-0 VV, VH (dB or linear? [VERIFY] paper says "Sigma nought (Sigma-0) backscatter coefficients"),
        + derived VV/VH ratio channel (compute yourself: ratio = VV/VH, handle VH=0 safely)
  - GT: keep raw class IDs {2,3,4,5,6,7,10,11}; classes 10/11 = ignore-index for loss
        │
        ▼
Patch Generation (64×64, matching paper §4.1 training image size)
  - Random/sliding-window patch extraction from full scenes
  - Only 14 train scenes total → patch extraction is your primary source of training volume;
    document exact stride/overlap used, since it directly affects reported metrics reproducibility
        │
        ▼
Data Loader (PyTorch Dataset/DataLoader)
  - Returns: s2_patch [11,64,64], s1_patch [3,64,64], gt_patch [64,64] (class-id map), ignore_mask
        │
        ▼
Segmentation Network (per-model — U-Net, RUPNet, etc., see §2)
        │
        ▼
Knowledge Distillation (teacher RUPNet frozen → student RUPNet, same arch)
        │
        ▼
SplitClass Post-processing (Algorithm 1, applied to student's softmax output at inference)
        │
        ▼
Temporal Gap-Filling (Algorithm 2, combines S2_Model output across time + Radar_Model)
        │
        ▼
Evaluation (Pr/Rc/F1 per class, weighted by pixel count; + field-data accuracy if available)
```

---

## 2. Model Design

### 2.1 Teacher Model
- **Architecture:** RUPNet [54] — selected because it had the best Table-1 F1 (81.19%) among 10 backbones tested (U-Net, U-Net++, FPN, PSPNet, DeepLabv3+, MANet, AttentionUNet, ResidualAttentionUNet, TransUNet, RUPNet).
- **[VERIFY]** RUPNet's actual structure — go to the SPIE 2023 paper "RUPNet: Residual upsampling network for real-time polyp segmentation" (Tomar, Bagci, Jha) before implementing; do not let me invent layer counts.
- **Input channels:** 11 (S2 only, per §3.2 "Both models share an identical architecture, differing only in the number of input channels").
- **Output classes:** the paper trains top-1 on "5 key classes" (2–6) per Table 1's caption ("Classes 2–6"), but Table 2 evaluates IDs {2,3,4,5,6,7} — i.e. 6 classes including "not affected" (7). **[AMBIGUOUS]** Determine your own consistent class set and document it; I'd recommend training on all non-ignored classes {2,3,4,5,6,7} (6-class softmax) since Table 2 clearly evaluates 7 as well.
- **Training objective:** hard cross-entropy-style supervision via GDL + WFL on ground-truth labels (§4.2), with cloud/shadow pixels (10, 11) excluded from the loss via masking (§3.2).

### 2.2 Student Model
- **Same architecture as teacher** (self-distillation, confirmed explicitly: "We employ a self-distillation approach in which both the teacher and student models share the same architecture" — §3.2).
- **Knowledge transfer mechanism:** student trained on combination of GDL loss + WFL loss (same hard-label losses as teacher) **plus** KL-divergence loss between student's output distribution and the *pre-trained, frozen* teacher's soft output distribution (§4.3).
- **Loss calculation:** total student loss = GDL + WFL + KL(student ‖ teacher). The paper does not give explicit weighting coefficients between these three terms in the excerpt — **[VERIFY/ABLATE]**: start with equal weighting (1:1:1) and treat the weighting as a hyperparameter you sweep, documenting that the paper's exact weights are not disclosed here.
- **Post-hoc SplitClass step:** applied only after student training completes, as inference-time post-processing (not part of the training loss).

### 2.3 Radar Model
- **Input:** Sentinel-1 channels only (VV, VH, VV/VH ratio — 3 channels).
- **Task:** simplified to **binary** water/non-water classification. All water-affected classes (2–6) are collapsed into one "water" label; class 7 ("not affected") is the negative class. Rationale given: radar backscatter is ambiguous between waterlogged soil/vegetation classes, so fine-grained radar classification is unreliable — binary is safer (§3.2).
- **Cloud handling:** N/A for radar itself (radar isn't blocked by clouds); however training labels near clouds (10/11) still get excluded from loss the same way, since those GT values themselves derive from optical S2 scene-classification maps and are unreliable there too.
- **Architecture:** same shared architecture as teacher/student, with input channel count changed to 3.

---

## 3. Loss Functions

### 3.1 Generalized Dice Loss (GDL) — Sudre et al. 2017 [51]
- Class-weighted Dice loss; weights set inversely proportional to class frequency in the training set to counter imbalance (explicit in §4.2).
- **[VERIFY]** exact weighting formula (`w_k = 1 / (sum of pixels of class k)^2` is the standard GDL formulation from the cited paper — confirm against the original before hardcoding, since GDL has more than one weighting variant in the literature).
- **Used for:** teacher training, student training (hard-label component).

### 3.2 Weighted Focal Loss (WFL) — Lin et al. 2017 [28]
- Focal loss (γ hyperparameter down-weights easy/well-classified pixels) with per-class weights, again addressing imbalance.
- **[VERIFY]** the paper doesn't state γ or the exact class-weight scheme in the excerpt — treat as an open hyperparameter, don't assume γ=2 (the original Focal Loss default) is what was used here.
- **Used for:** teacher training, student training (hard-label component).

### 3.3 KL Divergence Loss
- Standard `KL(softmax(student_logits/T) ‖ softmax(teacher_logits/T))`, classic Hinton-style distillation [18].
- **[VERIFY]** distillation temperature T is not stated in the excerpt — must be swept/ablated; start with T=1 (no temperature scaling) as the simplest baseline, or T=2–4 as common literature defaults, and report which you used.
- **Used for:** student training only, alongside GDL+WFL.

### Loss usage summary table

| Loss | Teacher | Student | Radar Model |
|---|---|---|---|
| GDL | ✓ | ✓ | ✓ (binary variant) |
| WFL | ✓ | ✓ | ✓ (binary variant) |
| KL (vs. frozen teacher) | ✗ | ✓ | ✗ |

---

## 4. Data Pipeline

- **Preprocessing:** band selection (S2 11/13 bands), Sigma-0 for S1, VV/VH ratio derivation, reflectance/backscatter normalization ([VERIFY] exact normalization stats — compute your own per-channel mean/std from the training set rather than assuming ImageNet-style stats, since these are non-RGB physical units).
- **Normalization:** per-channel z-score using **training-set-only** statistics (standard practice; not explicitly stated in excerpt but necessary to avoid train/test leakage).
- **Augmentation:** not mentioned in the excerpt at all. Given only 14 training scenes → tens of thousands of 64×64 patches from sliding windows, augmentation (flips/rotations at minimum) is strongly recommended as an *addition beyond the paper*, and should be logged as such (secondary objective, not reproduction).
- **Patch extraction:** 64×64, per §4.1. Recommend documenting stride explicitly (paper doesn't state it) — e.g., non-overlapping for a clean reproducibility baseline, then overlapping/sliding as a data-augmentation ablation.
- **Train/validation split:** the paper's "training set" (14 scenes) vs "test set" (9 scenes) is the only official split described. No internal train/val split is mentioned — you will need to carve your own validation subset out of the 14 training scenes (e.g., leave-one-scene-out or a small held-out fraction of patches), and this is a **design decision you're making, not something reproduced from the paper**.
- **Temporal split handling:** KNPI — train on pre-2023 dates, test on post-2023 dates (temporal generalization). BNPI — test set uses *earlier* dates than some training dates (reverse temporal split, explicitly to test generalization "across different time periods," §3.1.1). Implement this as metadata-driven splitting keyed off the date in each filename (`Location_YYYYMMDD.tif`), not a random split.

---

## 5. Algorithms (verbatim from paper, safe to implement directly)

### Algorithm 1 — SplitClass Segmentation
```
for each pixel x_i:
    C_top1 = argmax_k P(x_i, C_k)
    if P(x_i, C_top1) >= threshold (θ = 0.98 in paper's reported results):
        x_i = C_top1
    else:
        C_top2 = argmax_{k != C_top1} P(x_i, C_k)
        x_i = (C_top1, C_top2)   # composite dual-class label
```
Evaluation rule for SplitClass predictions (§4.3): a prediction counts as a true positive if **either** of the two assigned classes matches ground truth.

### Algorithm 2 — Temporal Gap-Filling
```
Inputs: s2_prev, s2_next, s2_day (may be "missing" where clouded), radar_mask (today's radar water/non-water)
mask = (s2_day == missing)
s2_day[mask & (radar_mask == water)] = water        # class 2, Open Surface Water only
if s2_prev == s2_next:
    s2_day = s2_prev
elif s2_prev is missing or s2_next is missing:
    s2_day = whichever is available
else:
    s2_day = (s2_prev, s2_next)   # composite dual-class, SplitClass-style
return s2_day
```
Note: this only resolves "water" via radar; non-water waterlogging severity classes under clouds are resolved purely via the prev/next S2 temporal logic, **not** via radar (radar model is binary water/non-water only, per §2.3).
