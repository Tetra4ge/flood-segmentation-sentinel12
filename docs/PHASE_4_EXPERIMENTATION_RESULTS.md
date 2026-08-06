# PHASE 4 — Experimentation & Results Documentation

This file is a **template** — fill in the "Result" rows once you actually run each experiment. Do not pre-fill numbers; that would be fabrication. Every "Expected result" below is either the paper's reported number (labeled as such) or a hypothesis to test (labeled as such), never presented as ground truth for your own run.

---

## Experiment Matrix

### Experiment 1 — Baseline (U-Net)
- **Objective:** Establish a simple, fast, reproducible lower/reference bound before attempting the paper's more complex pipeline.
- **Configuration:** `configs/model_unet_baseline.yaml` — S2 only (11 ch), classes {2..7}, GDL+WFL, 64×64 patches, batch 32, lr 1e-4, 100 epochs (matching paper's stated hyperparameters in §4.1).
- **Expected result (paper reference, Table 1, classes 2–6 only):** Pr=78.58, Rc=80.20, F1=79.38.
- **Evaluation metrics:** per-class + pixel-weighted overall Pr/Rc/F1.
- **Your result:** _(fill in after running)_

### Experiment 2 — Paper architecture (RUPNet, top-1)
- **Objective:** Reproduce the best-performing backbone identified by the paper's own model comparison.
- **Configuration:** `configs/model_rupnet.yaml`, same training regime as Exp 1, architecture per **[VERIFY]**-flagged RUPNet spec from Phase 2.
- **Expected result (paper reference, Table 1):** Pr=76.35, Rc=86.70, F1=81.19.
- **Evaluation metrics:** same as Exp 1.
- **Your result:** _(fill in after running)_

### Experiment 3 — Teacher-Student Knowledge Distillation
- **Objective:** Confirm whether self-distillation improves the student over the teacher, as the paper claims (citing Zhang et al. 2019 self-distillation effect [67]).
- **Configuration:** `configs/kd_student.yaml` — teacher = Exp 2's frozen RUPNet; student = fresh RUPNet trained on GDL+WFL+KL(vs teacher).
- **Expected result (paper reference, Table 2, "All" row, top-1 evaluation only):** Teacher Pr/Rc/F1 = 0.98/0.96/0.97; Student Pr/Rc/F1 = 0.98/0.97/0.98 (student marginally better).
- **Evaluation metrics:** per-class {2,3,4,5,6,7} + weighted "All" row, matching Table 2's layout.
- **Your result:** _(fill in after running)_
- **Note:** if your student does *not* outperform your teacher, this is a legitimate and reportable finding, not a bug to force-fix — self-distillation gains are known to be inconsistent across datasets/architectures in the broader literature.

### Experiment 4 — SplitClass Segmentation
- **Objective:** Reproduce the claim that SplitClass post-processing (top-2 labeling under low confidence) improves F1 specifically for underrepresented classes 3–6.
- **Configuration:** apply Algorithm 1 to Experiment 3's student output at θ=0.98; also sweep θ ∈ {0.5, 0.7, 0.9, 0.95, 0.98, 0.99} as an ablation beyond the paper.
- **Expected result (paper reference, Table 2, "SplitClass τ=0.98" All row):** Pr=0.99, Rc=0.99, F1=0.99; dual-class pixel coverage 6.33% overall; largest F1 gains on classes 3–6.
- **Evaluation metrics:** per-class Pr/Rc/F1 using the "either predicted class matches GT" rule; % of pixels assigned dual-class labels; confusion analysis for the specific class pairs the paper flags (water↔non-affected: 0.13% of pixels; vegetation-affected↔non-affected: 2.33% of pixels).
- **Your result:** _(fill in after running, include your θ-sweep as a table/plot)_

### Experiment 5 — Temporal Gap-Filling
- **Objective:** Reproduce the accuracy gain from combining S2_Model + Radar_Model + temporal consistency (Algorithm 2) vs. single-day prediction, validated against field data.
- **Configuration:** Radar_Model (binary, S1-only) trained per Phase 3 Milestone 4; Algorithm 2 applied on test-set cloud-gap scenarios.
- **Expected result (paper reference, Table 3):** Single-Day accuracy BNPI=58.25%, KNPI=59.45%; With Temporal Gap-Filling BNPI=72.27%, KNPI=69.24%.
- **Evaluation metrics:** accuracy vs. field-collected ground truth points (excluding "water under vegetation" and areas <100 m², per paper's filtering — replicate this filter if your field data has the equivalent fields).
- **Your result:** _(fill in after running — or mark BLOCKED if you don't have a field-data file, see Phase 1 §2 note)_

---

## Suggested Experiments Beyond the Paper (secondary objectives)

| # | Idea | Why |
|---|---|---|
| 6 | Full soft-label distillation (student trained to match full teacher softmax, no SplitClass thresholding) vs. SplitClass | Paper positions SplitClass as a "scalable and interpretable compromise" vs full soft labels — but never actually benchmarks against literal full soft-label output quantitatively in the excerpt provided. Worth directly comparing. |
| 7 | Alternate teacher backbone for KD (e.g., your Exp-1 U-Net as teacher instead of RUPNet) | Tests whether the self-distillation/SplitClass gains are RUPNet-specific or general. |
| 8 | Loss-term ablation on student (GDL only / WFL only / KL only / all combinations) | The paper's excerpt gives no ablation of loss term contributions — a genuine research gap you can fill. |
| 9 | KD temperature sweep | Temperature T is unstated in the paper (flagged in Phase 2 §3.3) — sweep and report sensitivity. |
| 10 | Cross-site generalization test (your own out-of-distribution scene, if you can source one) | Mirrors the paper's Doñana, Spain test (§4.6) — requires an external Sentinel-1/2 scene with some reference water layer (e.g., a public WIW-style product) for comparison, which is outside your current dataset. |

---

## Reporting Format for Each Completed Experiment
For every experiment, record:
1. Exact config file + git commit hash used.
2. Random seed(s) — run ≥3 seeds where feasible given the small dataset, report mean ± std, not a single number (the paper does not report seed variance, which is a limitation you can explicitly improve on).
3. Hardware + wall-clock training time.
4. Full per-class metrics table + confusion matrix.
5. Qualitative visualizations (predicted map vs GT vs input RGB) for at least 2 test scenes per experiment.
6. A short "deviation from paper" paragraph — what differed in your setup (unstated hyperparameters you had to choose) and how that plausibly explains any metric gap.
