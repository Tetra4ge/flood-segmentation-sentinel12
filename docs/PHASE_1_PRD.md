# PHASE 1 — Product Requirement Document (PRD)
## Reproducing "Inland Excess Water (IEW) Monitoring Using Sentinel-1/2: A SplitClass Segmentation and Temporal Gap-Filling Approach" (ICCVW 2025, Ibrahim et al.)

> **Accuracy note (read first):** This PRD is built strictly from the text of the paper you uploaded and the dataset description you provided. I did **not** run a live search of GitHub/paper repositories for this document — I have no web-search tool enabled in this session. Any claim about "no official code exists" should be treated as "I could not verify code exists," not "code does not exist." Verify at https://waterdetection.github.io/water_detection/ (the site named in the paper) yourself before assuming reproduction-from-scratch is necessary.

---

## 0. Step 1 Result — Existing Implementation Analysis (honest version)

| Item | Status | Notes |
|---|---|---|
| Official GitHub repo | **Unverified** | Paper links a project website (waterdetection.github.io) but does not state a code repo is public. I cannot confirm current contents without browsing. |
| Dataset repo | **Unverified** | Same site is named as the dataset's home; you say you've already downloaded the data, which is good — but I can't confirm your copy matches the "official" release or if a newer version exists. |
| RUPNet implementation | RUPNet is a real prior architecture (Tomar, Bagci & Jha, 2023, *Medical Imaging 2023*, "RUPNet: Residual upsampling network for real-time polyp segmentation") — **for polyp segmentation**, not remote sensing. I recall this is a U-Net-style encoder-decoder with residual upsampling blocks, but I am **not fully certain of exact layer-by-layer details** from memory, and the IEW paper does not restate RUPNet's architecture in the excerpt you gave me. **You must locate the original RUPNet paper/code and verify architecture details before reimplementing** — do not let me guess layer widths/kernel sizes.
| SplitClass segmentation | Not a prior named method — introduced by this paper. Algorithm 1 is fully specified in the paper (see TRD). Reimplementable directly from the paper's pseudocode.
| Knowledge distillation (self-distillation, same architecture teacher/student) | Standard technique (Hinton et al. 2015, KD; self-distillation per Zhang et al. 2019 [67]). Generic implementations exist widely (e.g., in `torchdistill`, or hand-rolled). No paper-specific repo needed — implement from loss definitions in TRD.
| Temporal gap-filling (Algorithm 2) | Not a prior named method — introduced by this paper, fully specified as pseudocode. Reimplementable directly.

**Recommendation:** Reproduce from scratch. The paper gives sufficiently detailed algorithms (Algorithm 1, Algorithm 2), losses (GDL + WFL + KL), training hyperparameters (image size, batch size, epochs, LR), and evaluation protocol (Tables 1–3) to build an independent implementation. The one high-risk unknown is the **exact RUPNet architecture** — budget time to pin this down from the cited SPIE 2023 paper before assuming any structure.

---

## 1. Project Overview

**Problem:** Standard remote-sensing water segmentation is binary (water / non-water) and coarse. It fails to (a) represent gradations of waterlogging severity in soil/vegetation, (b) represent pixel-level classification ambiguity in transitional zones, and (c) handle Sentinel-2 optical data gaps caused by cloud cover.

**Why IEW monitoring matters:** Waterlogging and water-retention capacity are both climate-risk indicators — over 70% of Hungary studied in the paper has faced droughts in the past decade while up to 15% of arable land floods periodically, and these overlap. Fine-grained, temporally consistent IEW maps support land management, agriculture policy, and climate adaptation (paper §1).

**Why S1+S2 fusion is required:** Sentinel-2 (optical, 11 usable bands after excluding B9/B10) gives strong spectral discrimination of soil moisture/vegetation stress but is blocked by clouds. Sentinel-1 (SAR, VV/VH/VV-VH-ratio) penetrates clouds but cannot finely discriminate soil-moisture severity classes — the paper handles this by making the radar model a coarse binary water/non-water detector used only to gap-fill, not to replace S2 classification.

---

## 2. Research Objective

**Primary objective:** Reproduce the paper's core numerical results as closely as feasible on your copy of the IEW-Seg dataset:
- Table 1: top-1 DNN backbone comparison (Pr/Rc/F1) on 5 water-related classes (2–6), S2-only input, GDL+WFL loss.
- Table 2: Teacher vs Student (top-1) vs SplitClass (τ=0.98) per-class Pr/Rc/F1.
- Table 3: Single-day vs Temporal-Gap-Filling accuracy against field data for BNPI/KNPI (58.25%→72.27% and 59.45%→69.24%).

**Secondary objectives (your extensions):**
- Improve segmentation performance (e.g., try different KD temperature, alternate teacher backbones beyond RUPNet, try full soft-label distillation vs SplitClass).
- Improve generalization (test on your own out-of-distribution scene, analogous to their Doñana, Spain cross-site test).
- Additional experiments: ablation on SplitClass threshold τ; ablation on loss component weights; radar-only model performance ceiling; error analysis on the classes the paper itself flags as confusable ("vegetation affected by water" vs "not affected", 2.33% overlap; water vs non-affected, 0.13% overlap).

> **Known limitation to reproduce honestly, not just match numbers:** Field-data validation (Table 3) used 422 BNPI + 694 KNPI raw samples, filtered down to 321 / 439 after excluding "water under vegetation" and areas <100 m². Your reproduction's Table-3-equivalent numbers will only be comparable if you have the **field CSV/GeoPackage** with the same filtering fields (date, GPS, assigned class, wet area, water depth). Check whether your download includes this — the structure you gave me only lists GT/GT_0_8/S1/S2 tif folders, **no field-data file is mentioned**. If it's missing, Table 3 reproduction is not possible and should be marked as an "unreproducible without additional data" experiment rather than skipped silently.

---

## 3. Dataset Analysis

### 3.1 What you have vs. what the paper describes

| Paper (§3.1.1, §4.1) | Your download |
|---|---|
| 2 sites: BNPI (Bükk NP, "Heves Steppes"), KNPI (Kiskunság NP) | Not explicitly separated in your folder listing — filenames like `BNPI_20240129.tif` suggest site is encoded in filename prefix. **Verify KNPI-prefixed files also exist.** |
| Train: 7 BNPI + 7 KNPI = 14 scenes | Matches: 14 train scenes ✓ |
| Test: 4 BNPI + 5 KNPI = 9 scenes | Matches: 9 test scenes ✓ |
| BNPI resolution 5400×3600 px; KNPI 4800×2400 px | Not yet confirmed — verify per-file via script below (full scenes, not the 64×64 training patches). |
| S2: 11 bands (13 minus B9, B10) | Matches "11 spectral channels" you noted. |
| S1: Sigma-0 VV, VH, VV/VH ratio (3 channels) | Matches. |
| Labels: multi-class thematic map, IDs 2,3,4,5,6,7,10,11 (see class table below) | Your `GT` folder — verify pixel value set matches these exact IDs, not 0–7 remapped. |
| `GT_0_8`: **not described in the paper at all** | This is presumably something added by whoever prepared your dataset copy (maybe a pre-thresholded binary/8-bit mask, or corresponds to a specific τ). **I cannot tell you what it is — inspect it and, ideally, ask whoever supplied the dataset.** Do not assume it equals the paper's τ=0.98 SplitClass output. |

### 3.2 Full class table (paper Fig. 1 / class IDs)

| ID | Class | Notes |
|---|---|---|
| 2 | Open surface water | |
| 3 | Soil seriously affected by waterlogging | |
| 4 | Soil moderately affected by waterlogging | |
| 5 | Soil slightly affected by waterlogging | |
| 6 | Vegetation affected by waterlogging | |
| 7 | Not affected areas | |
| 10 | Cloud / missing data | excluded from loss (masked) |
| 11 | Shadows | excluded from loss (masked) |

Note IDs 0, 1, 8, 9 are apparently unused/reserved — confirm this empirically rather than assuming.

Base "water-related" classes used for SplitClass's C(n,2) composite-class count = 6 (paper states "6 base classes" → C(6,2) = 15 composite classes, confirmed arithmetic ✓). It is not 100% explicit in the excerpt whether the "6 base classes" are exactly {2,3,4,5,6,7} or exclude class 7 — re-read the Supplementary Material if available, since this materially changes which composite classes you need to enumerate. I'm flagging this as ambiguous rather than assuming.

### 3.3 Required inspection scripts

**Script 1 — `scripts/00_inspect_dataset.py`:** enumerate every `.tif`, print shape, dtype, band count, per-band min/max/mean, nodata value, CRS, and pixel value histogram for GT files.

**Script 2 — `scripts/01_visualize_samples.py`:** for N random scenes, render S2 RGB composite, S1 VV/VH/ratio false-color, and GT colorized per Fig. 1's color scheme, side by side.

**Script 3 — `scripts/02_class_statistics.py`:** aggregate pixel counts per class across Train and Test splits separately (to check for class imbalance and train/test class-distribution shift — critical since GDL/WFL are chosen specifically to counter imbalance).

```bash
# Expected commands once scripts exist (Phase 3 builds these)
python scripts/00_inspect_dataset.py --root /path/to/IEW_seg_DATASET_FINAL/tifs
python scripts/01_visualize_samples.py --root /path/to/IEW_seg_DATASET_FINAL/tifs --n 4 --out outputs/viz
python scripts/02_class_statistics.py --root /path/to/IEW_seg_DATASET_FINAL/tifs --out outputs/class_stats.csv
```

### 3.4 Expected outputs after dataset analysis

```
Dataset inspection report:
  Train scenes: 14 (BNPI: 7, KNPI: 7)
  Test scenes: 9 (BNPI: 4, KNPI: 5)
  S2 shape per scene: (11, H, W)   # H,W vary per scene, NOT fixed
  S1 shape per scene: (3, H, W)
  GT unique values: {2,3,4,5,6,7,10,11}  <- must verify, don't assume
  GT_0_8 unique values: ??? <- unknown, must inspect
  Class pixel-count table (Train): class 2: X px, class 3: Y px, ...
  Class pixel-count table (Test):  ...
  Missing-data files or shape mismatches: [] (must be empty or investigate)
```

If GT unique values or shapes don't match expectations, **stop and resolve before writing any model code** — this is the #1 place people silently reproduce wrong results.

---

## 4. Non-goals / explicit exclusions for this PRD
- Not attempting to reproduce the field-data collection process itself (QField app, GPS survey) — only consuming a field-data file if you already have one.
- Not claiming any numeric reproduction target is guaranteed reachable with only 23 total scenes (14 train / 9 test) — this is a very small dataset for deep segmentation networks; expect high variance run-to-run, and report mean ± std over multiple seeds rather than a single run, which the paper itself does not appear to do (another limitation worth noting in your writeup).
