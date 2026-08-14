# Implementation vs. Paper — Alignment Report

> Cross-check of this repository's actual, current state against `docs/PAPER_WORKFLOW.md` (the paper's method) and against the repo's own `README.md` (which describes a `src/`/`scripts/` layout that **no longer exists on disk** — see §5). Generated 2026-08-14.

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Is the project on the right conceptual track? | **Yes.** Data pipeline, class encoding, losses, and the backbone-comparison stage all match the paper's design. |
| Is it "in the correct flow" — i.e. doing steps in the paper's order? | **Yes, so far.** You are still inside the paper's **Stage C (Backbone Selection, §4.2)** — this is legitimately step 1 of the pipeline before distillation/SplitClass/gap-filling can happen. Nothing has been skipped ahead or done out of order. |
| Are the *results* tracking the paper? | **Partially, and one important divergence:** RUPNet — the paper's winning backbone — is currently the **worst** performer of your three trained models, the opposite of Table 1's ranking. This needs investigation before moving on (see §4). |
| Is the repo's documentation trustworthy? | **No, it's stale.** `README.md` describes `src/`, `scripts/`, and two differently-named notebooks that do not exist; the actual repo is three self-contained notebooks. Fix before it misleads you or a reviewer. |
| What's actually left to build? | Knowledge distillation (teacher/student), SplitClass Segmentation (Algorithm 1), the binary Radar_Model, Temporal Gap-Filling (Algorithm 2), and the two evaluation experiments that depend on them (§4.5 field validation, §4.6 cross-site generalization). None of these exist yet in code. |

---

## 2. What the paper's pipeline actually is (recap)

From `docs/PAPER_WORKFLOW.md`, the full pipeline has 7 stages:

```
A. Dataset construction (§3.1)        — GT map generation, masking, cloud/shadow mapping, field survey
B. Dual-model training (§3.2)         — S2_Model (6-class) + Radar_Model (binary S1)
C. Backbone selection (§4.2)          — 10 architectures compared, RUPNet wins (F1=81.19%)
D. Knowledge distillation (§3.2/§4.3) — RUPNet teacher → RUPNet student (self-distillation)
E. SplitClass Segmentation (§3.2.1)   — inference-time top-1/top-2 confidence splitting (θ=0.98)
F. Temporal Gap-Filling (§3.2.2)      — fills cloud gaps using S1 radar + flanking S2 predictions
G. Evaluation (§4)                    — Table 1 (backbones), Table 2 (KD+SplitClass), Table 3 (field data), §4.6 (cross-site)
```

Stages **A and C are the only ones this repo currently implements** (partially, for C). D–F are 0% implemented. This is the correct *order* to build in — you cannot distill a teacher/student pair, run SplitClass on its softmax output, or gap-fill S2 predictions, until you have a trained S2 backbone to do it with. So the current focus is legitimate and not out of sequence.

---

## 3. What actually exists in this repo right now

The repo is **four Jupyter notebooks + a config file + cached outputs** — no `src/` package, no `scripts/` folder, despite what `README.md` claims (see §5).

| File | Role | Paper stage |
|---|---|---|
| `sen12flood-eda.ipynb` | EDA on a *different*, earlier dataset (SEN12FLOOD) — bands, false-RGB visualization, global coverage, flood-event exploration. Not the IEW-Seg dataset. | Precursor/exploration only, not part of the paper's pipeline |
| `unet.ipynb` | Self-contained: data pipeline (GeoTIFF I/O, windowed S1/S2/GT reads, standardization, augmentation), GDL+WFL losses, U-Net (residual conv blocks, ~32.4M params), training (40 epochs, cosine LR), full-test-set evaluation, prediction visualization | §3.1 (data) + §4.2 (one of the 10 backbones) |
| `unetpp.ipynb` | Same structure as above, U-Net++ (nested/dense skip pathways, `base_channels=32`, ~9.16M params) | §4.2 (second backbone) |
| `rupnet.ipynb` | Same structure, RUPNet (paper's winning backbone, ~2.05M params — matches paper's 2.05M almost exactly) | §4.2 (third backbone, meant to become the KD teacher) |
| `configs/data.yaml` | dataset root, patch size (64), stride (64), channel counts (S2=11, S1=3), class list `[2,3,4,5,6,7]`, ignore `[10,11]` | §3.4 class encoding |
| `outputs/checkpoints/*.pt` | Best-val-loss checkpoints for each trained model | — |
| `outputs/class_stats.csv`, `outputs/patches/` | Cached class distribution + patch cache | §3.1 |

Each of the three model notebooks is structured identically (Imports → Data Pipeline → Visualization → Losses → Model → Dataloaders → Training → Loss Curves → Full-Test Evaluation → Prediction Visualization), which is good practice — they're directly comparable "apples to apples" experiments, which is exactly what Table 1 in the paper requires.

### 3.1 Dataset construction (§3.1) — alignment: **correct**
- Class encoding matches the paper exactly: classes `{2,3,4,5,6,7}` as the 6 base IEW classes, `{10,11}` (cloud/shadow) excluded from the loss.
- Channel counts match: 11-band S2, 3-band S1 (VV, VH, ratio).
- Patch-based windowed reading from GeoTIFFs, matching the paper's patch/tile-based training approach.
- Train/test scene split (14 train / 9 test, BNPI+KNPI) is documented in `README.md` and matches the paper's described temporal-generalization split.

### 3.2 Losses (§3.2) — alignment: **correct**
- Generalized Dice Loss (GDL) + Weighted Focal Loss (WFL), combined, exactly as specified in the paper. Training logs show both loss components tracked separately per epoch, which is a nice touch for debugging convergence.

### 3.3 Backbone comparison (§4.2, Table 1) — alignment: **in progress, 3 of 10 backbones done**
Paper compares 10 architectures. This repo has trained **3**: U-Net, U-Net++, RUPNet. Missing: FPN, PSPNet, DeepLabv3+, MANet, AttentionUNet, ResidualAttentionUNet, TransUNet — 7 backbones not yet implemented. (Note: an `AttentionUNet`/`ResConvBlock` implementation was referenced in an earlier commit — `007b379` — but is not present as a runnable notebook in the current tree; worth checking if that work was lost or just not yet turned into a notebook.)

---

## 4. Results comparison — this is the part that needs attention

All three models were trained for 40 epochs (GDL+WFL, cosine LR) and evaluated on the full, non-overlapping test set (32,691 patches / 9 scenes). Overall pixel-weighted results:

| Model | Params (this repo) | Params (paper) | F1 — this repo | F1 — paper (Table 1) |
|---|---|---|---|---|
| U-Net | 32.44M | 31.04M | **96.90%** | 79.38% |
| U-Net++ | 9.16M | — (not reported) | **96.79%** | 79.10% |
| RUPNet | 2.05M | 2.05M | **94.35%** | **81.19%** (winner) |

Per-class F1 (this repo):

| Class | U-Net | U-Net++ | RUPNet |
|---|---|---|---|
| Open water (2) | 97.65 | 96.80 | 94.01 |
| Soil seriously affected (3) | 57.22 | 46.55 | 29.38 |
| Soil moderately affected (4) | 51.92 | 41.24 | 26.68 |
| Soil slightly affected (5) | 51.70 | 50.89 | 37.88 |
| Vegetation affected (6) | 68.63 | 64.87 | 45.64 |
| Not affected (7) | 98.37 | 98.43 | 96.55 |

**Two things stand out:**

1. **Overall F1 numbers (94–97%) are far higher than the paper's (~79–81%).** This is almost certainly *not* a genuine improvement — it's a sign the two numbers aren't measuring the same thing. The paper's Table 1 F1 is computed **only over the 5 water-related classes (2–6)**, explicitly excluding the dominant "not affected" class (7) — which is >90% of pixels in most scenes. If this repo's evaluation script includes class 7 in the pixel-weighted average (as the per-class table's inclusion of class 7 suggests), the huge, easy, low-frequency-error class 7 is inflating the "ALL" row. **Action:** recompute an "ALL" metric restricted to classes 2–6 only, pixel-weighted, to get an apples-to-apples number against Table 1.

2. **RUPNet — the paper's best backbone — is the worst of the three here**, and by a wide margin on the ambiguous soil-severity classes (3–5: 26–38% F1 vs. U-Net's 52–57%). In the paper RUPNet *wins precisely because* of strong recall on exactly these hard classes. This inversion is the most important open question in the project right now. Plausible causes, roughly in order of likelihood:
   - **Undertrained relative to capacity mismatch** — RUPNet is 15× smaller than U-Net; if hyperparameters (LR schedule, epoch count, augmentation) were tuned/copied across notebooks unchanged, the same schedule may not suit a much smaller, differently-conditioned architecture.
   - **Training patch stride** — `configs/data.yaml` and `README.md` disagree: README's Table describes train stride 256 (3,388 patches) while `configs/data.yaml` sets `stride: 64` project-wide. Confirm which stride each notebook actually used — RUPNet may have trained on a different (smaller/larger) effective dataset than the others if the config drifted between runs.
   - **Loss curves**: RUPNet's val loss stops improving/plateaus around epoch 5–6 (`0.8433`→`0.8462`→`0.8556`→`0.8502`) while train loss keeps dropping — a classic mild-overfitting or LR-too-high-for-small-model signature, worth checking against U-Net's smoother monotonic descent.
   - Possibly a bug in the RUPNet architecture port itself (worth a unit-level sanity check against the original Tomar/Bagci/Jha spec), though the parameter count matching the paper's 2.05M almost exactly is a good sign the architecture is at least structurally correct.

   **This should be resolved before RUPNet is anointed the "chosen teacher" for Stage D (distillation)** — distilling from an underperforming teacher will just propagate the problem forward into SplitClass and Gap-Filling evaluation later.

---

## 5. Documentation drift — `README.md` is out of date

`README.md`'s "Directory Structure" section (§2) describes a repo layout that does not match reality:

| README claims | Reality |
|---|---|
| `scripts/00_inspect_dataset.py`, `02_class_statistics.py`, `03_extract_patches.py`, `04_train.py`, `05_evaluate.py`, `compute_stats.py` | No `scripts/` directory exists |
| `src/data/patch_dataset.py`, `raster_io.py`, `splits.py` | No `src/` directory exists |
| `src/losses/generalized_dice.py`, `weighted_focal.py` | Losses are defined inline in each notebook instead |
| `src/models/rupnet.py`, `unet.py` | Models are defined inline in each notebook instead |
| `tests/test_dataset.py` | No `tests/` directory exists |
| `unet_baseline.ipynb`, `rupnet_training.ipynb` | Actual files are `unet.ipynb`, `rupnet.ipynb` (plus `unetpp.ipynb`, not mentioned at all) |

It looks like the project pivoted from a planned modular `src/`+`scripts/` architecture to self-contained notebooks (each notebook's own markdown even says "no external `src/` package dependency"), but `README.md` was never updated to reflect that pivot. `docs/PAPER_WORKFLOW.md` §7 ("How this maps to the repository") has the same stale references. **This is worth fixing soon** — anyone (including future-you) reading the README will look for files that don't exist.

---

## 6. Flow check — are you doing things in the right order?

**Yes.** Mapping current work to the paper's dependency graph:

```
✅ A. Dataset construction         — done (patches, class encoding, splits)
🔶 B. Dual-model training          — only S2 side started; Radar_Model (S1, binary) not started
🔶 C. Backbone selection           — 3/10 backbones trained (U-Net, U-Net++, RUPNet)
⬜ D. Knowledge distillation       — not started (needs a confirmed-best backbone from C first — see §4 caveat)
⬜ E. SplitClass Segmentation      — not started (needs D's student model's softmax output)
⬜ F. Temporal Gap-Filling         — not started (needs E + a trained Radar_Model)
⬜ G. Evaluation (Tables 2/3, §4.6)— not started (needs D, E, F respectively)
```

No stage has been attempted before its prerequisite stage — the sequencing is sound. The only risk to flow, not order, is §4's finding: continuing forward (picking a teacher, starting KD) on top of an unresolved RUPNet regression would compound the issue rather than just delay it.

---

## 7. Recommended next steps, in order

1. **Fix the Table-1 comparability gap** — restrict the "ALL" evaluation metric to classes 2–6 (matching the paper), so U-Net/U-Net++/RUPNet numbers are directly comparable to the paper's 79.38/79.10/81.19.
2. **Diagnose the RUPNet regression** — check training stride consistency across the three notebooks (README says 256, config says 64), inspect RUPNet's val-loss plateau, and confirm architecture fidelity against the original paper spec before trusting it as "the winner."
3. **Decide on backbone-comparison scope** — either finish the remaining 7 backbones (FPN, PSPNet, DeepLabv3+, MANet, AttentionUNet, ResAttnUNet, TransUNet) for a full Table 1 replication, or explicitly scope down to U-Net/U-Net++/RUPNet only and document that decision.
4. **Only after 1–3**: pick the confirmed-best backbone and start Stage D (teacher/student self-distillation with GDL+WFL+KL).
5. **Update `README.md`** (§2 directory structure, §4 results table) and `docs/PAPER_WORKFLOW.md` §7 to reflect the real notebook-based structure — cheap to do now, expensive to leave stale.
6. Implement Radar_Model (binary S1) in parallel with distillation, since Gap-Filling (Stage F) needs it and it has no dependency on D/E.

---

## 8. One-line verdict

**On track, right order, one result anomaly to resolve before moving forward.** The project correctly follows the paper's dependency chain (dataset → backbone selection → [distillation → SplitClass → gap-filling]) and hasn't skipped ahead into any stage prematurely. The main issue isn't sequencing — it's that the backbone-selection results don't yet reproduce the paper's ranking (RUPNet should win, but currently loses badly on the soil-severity classes), which should be understood before RUPNet is committed to as the knowledge-distillation teacher.
