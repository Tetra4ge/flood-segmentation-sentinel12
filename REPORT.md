# IEW Segmentation: Model Report

Inland Excess Water (IEW) mapping from Sentinel-1/2, six classes (open water, three soil-severity grades, vegetation affected, not affected). This report covers every model notebook in the repository except **notebook 14** (ViT + terrain UPerNet), which has not been run and has no results.

It describes each model, compares them on the same test set, and ranks them from best to worst. Every number comes from a saved results file or a saved notebook output. Where a notebook has no saved result, this report says so instead of filling in a number.

---

## 1. Summary table (ranked best to worst)

The ranking uses the paper's metric: **pixel-weighted F1 over classes 2–6 on the 9 unseen test scenes** (the same metric as Table 1 of the IEW paper). Where a notebook reports a calibrated result, the calibrated number is used. The raw number is shown in brackets.

| Rank | Model (notebook) | Status | Params | Paper F1, classes 2–6 | Macro-F1, 6 classes | Water F1 | Soil serious F1 | Soil moderate F1 | Soil slight F1 | Vegetation F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **PT-UPerNet** (12) | run, results saved | 59.66 M | **84.31** (raw 84.12) | **78.69** (raw 78.62) | 98.59 | 70.56 | 60.77 | 67.68 | 75.43 |
| 2 | **Soil-Severity Specialist** on NB12 (15) | run, results saved | NB12 (59.66 M) + specialist | **84.14** | **78.65** | 98.50 | 70.08 | 61.98 | 66.81 | 75.56 |
| 3 | **Prithvi-EO-2.0 + UPerNet** (13) | run, results saved | 313.5 M | **83.53** (raw 83.33) | **77.84** (raw 77.86) | 98.45 | 68.61 | 61.00 | 63.24 | 76.69 |
| 4 | **Spectral ResUNet** (10) | run, results saved | 5.23 M | **81.53** (raw 80.50) | **76.53** (raw 75.57) | 98.62 | 71.04 | 60.20 | 63.54 | 67.22 |
| 5 | **Temporal-Radar Former** (11) | run, results saved | 11.31 M | **79.00** (raw 80.30) | **73.77** (raw 75.59) | 98.40 | 67.43 | 57.77 | 56.44 | 64.18 |
| 6 | **SegFormer-B0** (09) | run, results saved | 3.73 M | **68.47** (raw only) | 56.44 (from NB10 table) | 92.75 | 24.52 | 27.65 | 40.42 | 55.52 |
| 7 | **RUPNet** (03) | run, results saved | 2.05 M | **63.97** (raw only) | not reported | 94.00 | 30.98 | 15.96 | 23.35 | 50.24 |
| – | U-Net (01) | not evaluated in saved output | 32.44 M | – | – | – | – | – | – | – |
| – | U-Net++ (02) | trained, test evaluation not saved | 9.16 M | – | – | – | – | – | – | – |
| – | Knowledge distillation (05) | run, different metric | 2.05 M × 2 | not reported (teacher 96.78 / student 96.70 on classes 2–7) | – | 95.62 (student) | 35.03 | 36.80 | 51.02 | 63.64 |
| – | TransUNet (08) | **invalid**: outputs are copied from NB03 | not printed | do not use | do not use | – | – | – | – | – |
| – | Radar model (04) | not run | 2–3 M (not printed) | – | – | – | – | – | – | – |
| – | SplitClass (06) | not run | – | – | – | – | – | – | – | – |
| – | Temporal gap-filling (07) | not run | – | – | – | – | – | – | – | – |
| excluded | ViT + terrain UPerNet (14) | **not run** | – | – | – | – | – | – | – | – |

Units are F1 in percent on the test set. "Water F1" is open surface water. The three soil columns are the three soil-severity classes. For the unranked rows, the numbers are the ones saved in their own notebook, on a different metric or without the paper-comparable split, so they cannot be placed in the ranking.

**Headline:** the top three models (12, 15, 13) are within about 1 F1 point of each other on the paper metric, and that difference is smaller than the spread between scenes (section 3.3). Notebook 10 is about 2 to 3 points behind that group. Notebooks 09 and 03 are about 15 to 20 points behind.

---

## 2. Comparison

### 2.1 What each model is trying to do

| Model | Core idea | Key difference from the others |
|---|---|---|
| 01 U-Net | Residual U-Net baseline | Largest model (32 M); the reference point for the paper pipeline |
| 02 U-Net++ | Dense nested skip connections | Smaller than U-Net, but more connections in the decoder |
| 03 RUPNet | Upsample directly from deep features with large jumps | The paper's chosen backbone (2 M parameters) |
| 04 Radar model | Binary water detection from Sentinel-1 only | Used only as a fallback for cloudy days (for notebook 07) |
| 05 Knowledge distillation | Frozen RUPNet teacher trains a fresh student | Tests whether self-distillation helps |
| 06 SplitClass | Output two labels when confidence is below 0.98 | Post-processing on top of the student |
| 07 Temporal gap-filling | Fill cloud gaps from radar and neighbouring clear dates | Post-processing across dates |
| 08 TransUNet | CNN encoder with a transformer bottleneck | Meant to add global context; see section 4 |
| 09 SegFormer-B0 | Hierarchical transformer encoder | Lightweight transformer; predicts at 1/4 resolution |
| 10 Spectral ResUNet | Residual U-Net plus a per-pixel spectral branch, rare-class sampling, logit calibration | First model built to fix the rare soil classes |
| 11 Temporal-Radar Former | Per-pixel transformer over up to 3 earlier dates and radar tokens; self-supervised pre-training | Adds time, not just a bigger network |
| 12 PT-UPerNet | Two SSL4EO-S12 pre-trained ResNet-50 encoders (optical, radar), plus terrain features from the Copernicus DEM | Adds pre-trained encoders and terrain |
| 13 Prithvi-EO-2.0 + UPerNet | NASA/IBM Prithvi-EO-2.0 ViT (300 M) as the encoder, plus radar and terrain | Adds a large geospatial foundation model |
| 15 Soil-Severity Specialist | A second network that only grades soil pixels, on top of frozen NB12 | Splits the job: "is it soil?" (NB12) and "how bad?" (NB15) |

### 2.2 Test-set results on the common metric

Per-class F1 (%) on the 9 test scenes, calibrated where available:

| Class | NB12 PT-UPerNet | NB15 Specialist | NB13 Prithvi | NB10 Spectral ResUNet | NB11 TRF | NB09 SegFormer | NB03 RUPNet |
|---|---|---|---|---|---|---|---|
| Open surface water | 98.59 | 98.50 | 98.45 | 98.62 | 98.40 | 92.75 | 94.00 |
| Soil, seriously affected | 70.56 | 70.08 | 68.61 | 71.04 | 67.43 | 24.52 | 30.98 |
| Soil, moderately affected | 60.77 | 61.98 | 61.00 | 60.20 | 57.77 | 27.65 | 15.96 |
| Soil, slightly affected | 67.68 | 66.81 | 63.24 | 63.54 | 56.44 | 40.42 | 23.35 |
| Vegetation affected | 75.43 | 75.56 | 76.69 | 67.22 | 64.18 | 55.52 | 50.24 |
| Not affected | 99.09 | 98.99 | 99.04 | 98.56 | 98.41 | 97.81 | 95.78 |
| **Macro-F1 (6 classes)** | **78.69** | **78.65** | **77.84** | **76.53** | **73.77** | 56.44 | not reported |
| **Paper F1 (classes 2–6)** | **84.31** | **84.14** | **83.53** | **81.53** | **79.00** | 68.47 | 63.97 |

Raw (uncalibrated) macro-F1 for the same models: NB12 78.62, NB15 (Stage A, NB12) 78.62, NB13 77.86, NB10 75.57, NB11 75.59.

Notes on the table:
- NB09 and NB03 are reported on the raw output only (no calibration step).
- "Not affected" for NB09 and NB03 is the same class as the others; their per-class values are taken from each notebook's printed table.
- NB09's "ALL" figure (95.65) includes the not-affected class and is not on the paper metric, so it is not used for ranking.

### 2.3 Model size and training budget

| Model | Params | Epochs | Samples per epoch | Patch | Pre-training | Test-time augmentation |
|---|---|---|---|---|---|---|
| 03 RUPNet | 2.05 M | not printed in summary | – | 64 (`configs/data.yaml`) | none | none |
| 09 SegFormer-B0 | 3.73 M | – | – | 128 | none | – |
| 10 Spectral ResUNet | 5.23 M | 30 | – | 96 | none | – |
| 11 TRF | 11.31 M | 40 | 3000 | 192 | self-supervised on 14 training scenes | 8-way (d4) |
| 12 PT-UPerNet | 59.66 M (47.04 M pre-trained) | 20 (first 2 frozen) | 1000 | 256 | SSL4EO-S12 ResNet-50 (MoCo) | flip |
| 13 Prithvi + UPerNet | 313.5 M | 25 (first 3 frozen) | 2000 | 224 | Prithvi-EO-2.0 (NASA/IBM) | flip |
| 15 Specialist | NB12 + specialist head | 30 (stage A: 20 epochs of NB12) | 1500 | 256 | none beyond NB12 | flip |

### 2.4 What the comparison shows

1. **Pre-trained encoders and terrain help most.** NB12 gains about 3 paper-F1 points over NB10 (84.31 vs 81.53 calibrated). The gain is largest on vegetation (75.43 vs 67.22) and the soil classes. In NB12's validation ablation, zeroing the terrain features drops vegetation F1 from 76.4 to 72.1 (about 4 points).
2. **Bigger is not better on its own.** NB13 has about 60 times the parameters of NB10 and about 5 times those of NB12 (313.5 M vs 5.23 M and 59.66 M), but it is slightly below NB12 on the paper metric. Its own ablation shows the Sentinel-1 radar input matters most, not the size of the encoder.
3. **Time helps less than expected.** NB11 (time-based TRF) is below NB10 on the paper metric, both raw and calibrated. Its validation macro-F1 is close to the other top models. In its own ablation, removing the earlier dates changes validation macro-F1 by less than 0.1 point. Radar is what matters (removing it drops validation macro-F1 by about 8 points).
4. **Calibration is not a reliable gain.** Calibration raised NB10 (+1.0 paper F1) and lowered NB11 (−1.3 paper F1 after calibration). For NB12 it changed almost nothing (α = 1.0, +0.2). Calibration fitted on two held-out training scenes does not transfer consistently.
5. **The specialist (NB15) adds almost nothing on test.** The Stage A output of NB12 already scores 84.12 raw on the paper metric. The specialist scores 84.14. The soil-grade macro-F1 on validation rose from 78.33 to 79.37, but that gain did not carry over to the test scenes.
6. **The soil-severity classes are the bottleneck.** Across all top models, soil moderate is still around 60 F1 and soil slight around 56 to 68. Water and "not affected" are above 98 for every model that ran. Vegetation is 64 to 77.
7. **Scene-level spread is large.** See section 3.3. The worst scene for every model is BNPI_20231125 or one of the KNPI 2024 scenes. The model ranking changes between scenes, so a single-scene result should not be read as a ranking.

---

## 3. Per-model details

Each section gives the design, what the notebook did, its results, and any problem found. Sections are in notebook order.

### 3.1 Results per model (detailed)

#### 01 U-Net (`01_unet_backbone.ipynb`)
- **Design:** residual U-Net baseline, 11 Sentinel-2 bands, six output classes. Loss is generalised Dice plus weighted focal.
- **Parameters:** 32,441,798.
- **Run state:** only the setup and architecture cells are executed (6 of 14 code cells). Training and test evaluation cells have no saved output.
- **Result:** none saved. `README.md` lists 96.90 (all six classes) without a source in this repository. Treat it as unverified.

#### 02 U-Net++ (`02_unetpp_backbone.ipynb`)
- **Design:** U-Net++ with dense nested skips, same loss and inputs as 01.
- **Parameters:** 9,162,150.
- **Run state:** training ran (the class-weight and data-loader cells are executed, the device is CUDA). The test-evaluation cell has no saved output.
- **Result:** none saved. `README.md` lists 96.79 (all six classes) without a source. Treat it as unverified.

#### 03 RUPNet (`03_rupnet_backbone.ipynb`)
- **Design:** RUPNet, a light residual upsampling network that jumps directly from deep features to the output resolution (paper's chosen backbone and the teacher for notebook 05).
- **Parameters:** 2,048,934.
- **Patch and data:** 64 × 64 patches, 11 S2 channels plus 3 S1 channels, cloud and shadow ignored in the loss.
- **Test result (saved output):** all classes 2–7 pixel-weighted F1 **93.46**. Paper-comparable (classes 2–6) F1 **63.97** (precision 58.58, recall 77.60). Per-class: water 94.00, soil serious 30.98, moderate 15.96, slight 23.35, vegetation 50.24, not affected 95.78.
- **Comparison with the paper:** the paper reports 81.19 for RUPNet on this metric. The 63.97 here is about 17 points lower. Section 4 explains why this number needs checking.
- **Conflict:** `README.md` lists RUPNet as 94.35 (all six). The saved output here says 93.46. The 94.35 figure has no source in this repository.

#### 04 Radar model (`04_radar_model.ipynb`)
- **Design:** binary water detection from Sentinel-1 only (VV, VH, ratio). Intended as the cloud-penetrating fallback for notebook 07.
- **Run state:** not run. The summary cell reads "fill in after running." A checkpoint `radar_model_best.pt` exists in `outputs/checkpoints/`, but it has no notebook output or results file in this repository, so its quality is unknown.

#### 05 Knowledge distillation (`05_knowledge_distillation.ipynb`)
- **Design:** the frozen RUPNet teacher (`rupnet_best.pt`, epoch 48) supervises a fresh RUPNet student with generalised Dice, weighted focal, and KL divergence.
- **Parameters:** 2,048,934 for each network.
- **Test result (saved output):** the metric is pixel-weighted over classes 2–7, so it is not the paper metric.
  - Teacher: precision 97.16, recall 96.51, F1 **96.78**.
  - Student: precision 97.25, recall 96.34, F1 **96.70**.
  - Per-class F1 (teacher / student): water 95.57 / 95.62; soil serious 32.08 / 35.03; soil moderate 35.65 / 36.80; soil slight 51.80 / 51.02; vegetation 62.09 / 63.64; not affected 98.54 / 98.44.
- **Finding:** the student did not beat the teacher overall (−0.07 F1). The notebook itself says this is a legitimate result, not a bug. Small gains on soil serious and vegetation are offset by losses on soil slight and not affected.
- **Conflict:** the teacher is `rupnet_best.pt`, the same checkpoint notebook 03 evaluates. Notebook 03 reports 93.46 on this metric, while notebook 05 reports 96.78 for the same checkpoint. The two evaluations disagree by 3.3 points and this is not explained in the repository. It needs to be resolved before either number is cited.

#### 06 SplitClass (`06_splitclass_segmentation.ipynb`)
- **Design:** post-processing from the paper's Algorithm 1. If the top class probability is at least θ = 0.98, keep one label; otherwise output a dual label (top-1 and top-2). A prediction counts as correct if either label matches.
- **Run state:** not run. The summary cell is still "fill in after running."

#### 07 Temporal gap-filling (`07_temporal_gapfilling.ipynb`)
- **Design:** paper's Algorithm 2. Fills cloud gaps in a day's prediction using the radar fallback and the nearest clear dates before and after the gap, with dual labels where those dates disagree.
- **Run state:** not run. The summary cell is still "fill in after running."

#### 08 TransUNet (`08_transunet_backbone.ipynb`)
- **Design (as written):** CNN encoder, four transformer blocks at the bottleneck, U-Net-style decoder with skip connections. Built by `build_transunet.py`.
- **Result: not valid.** The saved outputs in this notebook are identical to those in notebook 03. The model print line says `RUPNet params=2,048,934`, not TransUNet. The notebook was generated from notebook 03 and inherited its outputs. The 93.46 and 63.97 figures that `README.md` lists for TransUNet are notebook 03's RUPNet numbers. **No TransUNet result exists in this repository.**

#### 09 SegFormer-B0 (`09_segformer_b0_backbone.ipynb`)
- **Design:** SegFormer-B0 (Xie et al., 2021), 3.73 M parameters, 128 px patches (changed from 64 because the encoder downsamples by 32 in total).
- **Test result (saved output):** water-related classes (raw 2–6) pixel-weighted F1 **68.47** (precision 65.40, recall 75.83). Per-class F1: water 92.75, soil serious 24.52, moderate 27.65, slight 40.42, vegetation 55.52, not affected 97.81. ALL (with not-affected) 95.65.
- **Notes:** the soil classes are much weaker than in NB10 (24–40 vs 60–71 F1). Notebook 10 says SegFormer predicts at a quarter of the input resolution and upsamples, which blurs the thin soil-severity boundaries.

#### 10 Spectral ResUNet (`10_spectral_resunet.ipynb`)
- **Design:** residual U-Net with a per-pixel spectral branch, rare-class sampling, class weights (inverse square root), and a per-class logit bias fitted on the two held-out training scenes.
- **Parameters:** 5.23 M. Patch 96 px. 30 epochs (training shows checkpointing at epoch 24, best validation macro-F1 79.87).
- **Test result:** raw macro-F1 75.57, calibrated 76.53. Paper F1 (classes 2–6): raw 80.50, calibrated **81.53**. Per-class F1 calibrated: water 98.62, soil serious 71.04, moderate 60.20, slight 63.54, vegetation 67.22, not affected 98.56.
- **Important context from the notebook:** labels in this dataset come from per-pixel thresholds on SWIR1, NDVI, NDMI and an SWIR/RED index. A per-pixel classifier with no spatial context reaches roughly 70 / 52 / 70 F1 on the three soil classes on the same test scenes. That means the labels are an almost deterministic function of a few bands, which explains why spectral-branch models do well and why the soil grades are still hard.

#### 11 Temporal-Radar Former (`11_temporal_radar_former.ipynb`)
- **Design:** per-pixel temporal transformer over up to three earlier acquisitions of the same site (imagery only, no earlier labels), with Sentinel-1 radar tokens and a time embedding. A ConvNeXt encoder and FPN decoder follow. Four heads: flat six-class, coarse four-group, ordinal soil severity, and a reconstruction head used only in stage 1.
- **Training:** two stages. Stage 1 is self-supervised masked reconstruction on the training imagery. Stage 2 is supervised fine-tuning with EMA weights.
- **Parameters:** 11.31 M. Patch 192 px. 40 epochs, 3000 samples per epoch.
- **Test result:** raw macro-F1 75.59, calibrated 73.77. Paper F1 (classes 2–6): raw **80.30**, calibrated 79.00.
- **Ablation (validation):** removing the earlier dates changes validation macro-F1 from 79.94 to 79.86, a negligible drop. Removing Sentinel-1 radar drops it to 71.39, so radar carries most of the signal in this model.
- **Notes:** the calibrated result is worse than the raw result on the test scenes, even though calibration improved the held-out validation scenes. The README does not mention this drop.

#### 12 PT-UPerNet (`12_pretrained_terrain_upernet.ipynb`)
- **Design:** two ResNet-50 encoders with SSL4EO-S12 (MoCo) weights, one for Sentinel-2 and one for Sentinel-1, plus a small CNN for indices and terrain features. Terrain comes from the Copernicus 30 m DEM (relative elevation, slope, and topographic position at three scales). Encoders are fused at four scales, then a UPerNet decoder and three heads.
- **Training:** first two epochs with encoders frozen, then all layers trainable with a lower encoder learning rate. BatchNorm statistics of the pre-trained encoders are frozen.
- **Parameters:** 59.66 M, of which 47.04 M are pre-trained encoders.
- **Test result:** raw macro-F1 **78.62**, calibrated **78.69** (best overall). Paper F1 (classes 2–6): raw 84.12, calibrated **84.31** (best overall). Per-class F1 calibrated: water 98.59, soil serious 70.56, moderate 60.77, slight 67.68, vegetation 75.43, not affected 99.09.
- **Ablation (validation):** zeroing terrain lowers validation macro-F1 from 80.00 to 79.09. Zeroing Sentinel-1 lowers it to 74.70, so radar matters more than terrain.
- **Notes:** calibration is almost a no-op here (α = 1.0, per-class bias between −0.5 and +0.75).

#### 13 Prithvi-EO-2.0 + UPerNet (`13_prithvi_eo_upernet.ipynb`)
- **Design:** the Prithvi-EO-2.0 ViT (NASA/IBM, pre-trained as a masked autoencoder on Harmonized Landsat-Sentinel imagery) used as a multi-temporal encoder over four frames, with a UPerNet decoder. Sentinel-1 radar is an input. The saved configuration has no terrain features.
- **Parameters:** 313.5 M. Patch 224 px (fixed by the pre-trained model). Backbone frozen for 3 epochs, then trainable with a low learning rate. 25 epochs, 2000 samples per epoch.
- **Test result:** raw macro-F1 **77.86**, calibrated 77.84. Paper F1 (classes 2–6): raw 83.33, calibrated **83.53**. Blended-head result (α = 0.5) macro-F1 77.98, paper F1 83.24 (not used for ranking). Per-class F1 calibrated: water 98.45, soil serious 68.61, moderate 61.00, slight 63.24, vegetation 76.69, not affected 99.04.
- **Notes:** a ViT with 313 M parameters is not better than the 59.66 M ResNet model on this data. It also needs a separate environment (TerraTorch pins many dependencies).

#### 15 Soil-Severity Specialist (`15_soil_severity_specialist.ipynb`)
- **Design:** stage A is the frozen NB12 model (`ptuper_best.pt`, epoch 20, validation macro-F1 80.00). Stage B is a small dilated-convolution network trained only on soil pixels. It takes NB12's 64-channel feature map plus terrain, indices, radar, and validity flags, and outputs an ordinal head and a three-way grade head. The final soil grades mix the stage-B distribution with stage A's own grade choice at weight β.
- **Parameters:** stage A is 59.66 M (frozen). Stage B is not printed in the saved output.
- **Test result:** final macro-F1 **78.65** (stage A alone 78.62). Paper F1 (classes 2–6): final **84.14** (stage A 84.12). β = 1.0 was chosen on validation.
- **Validation:** grade macro-F1 rose from 78.33 (stage A) to 79.37 in training. That gain does not carry over to the test scenes.
- **Notes:** the specialist is effectively a no-op on test. Its own ablation shows the stage-B grade choice alone scores about the same as stage A on the soil grades (grade macro-F1 82.75 vs 82.74).

### 3.2 Models that are not run or not valid

| Notebook | Status | What is missing |
|---|---|---|
| 01 U-Net | Training and test cells not executed | Any saved test result. README number 96.90 is unsourced |
| 02 U-Net++ | Training ran, test evaluation not saved | Any saved test result. README number 96.79 is unsourced |
| 04 Radar model | Not run | Results. The checkpoint exists but is unverified |
| 05 KD | Run, but on a different metric | Paper-comparable figure. Teacher result conflicts with notebook 03 |
| 06 SplitClass | Not run | All results |
| 07 Gap-filling | Not run | All results |
| 08 TransUNet | Invalid | Outputs are copied from notebook 03. No TransUNet result exists |
| 14 ViT + terrain | Not run (excluded from this report) | All results |

### 3.3 Scene-level macro-F1 (test)

| Scene | NB11 TRF | NB12 PT-UPerNet | NB13 Prithvi | NB15 Specialist |
|---|---|---|---|---|
| BNPI_20220208 | 80.17 | 79.44 | 80.38 | 80.20 |
| BNPI_20220325 | 81.53 | 80.43 | 77.92 | 83.23 |
| BNPI_20231125 | 65.23 | 70.24 | 67.79 | 69.16 |
| BNPI_20240129 | 80.67 | 82.34 | 82.03 | 81.56 |
| KNPI_20230213 | 77.02 | 84.17 | 79.56 | 83.77 |
| KNPI_20230603 | 64.14 | 61.95 | 67.40 | 66.06 |
| KNPI_20240104 | 56.77 | 70.05 | 73.55 | 72.68 |
| KNPI_20240129 | 62.20 | 65.93 | 66.56 | 67.93 |
| KNPI_20240203 | 78.39 | 80.95 | 77.53 | 80.62 |

The top four models each win on some scenes and lose on others. KNPI_20240104 is the scene where Prithvi and the specialist clearly beat TRF. KNPI_20230213 is where PT-UPerNet and the specialist are strongest. The ranking is stable on average, but the per-scene spread is about 10 to 20 points.

---

## 4. Problems found

These affect how the numbers can be used.

1. **TransUNet (08) has no result.** Its saved outputs are copied from RUPNet (03). The "TransUNet" row in `README.md` is RUPNet's result. This needs to be re-run before any TransUNet claim is made.
2. **RUPNet has two conflicting numbers.** Notebook 03 gives 93.46 (all classes) and 63.97 (paper metric). `README.md` gives 94.35 with no source. Notebook 05 gives 96.78 for what should be the same checkpoint. The paper reports 81.19. The gap between the notebook and the paper is large enough to suggest a label or evaluation mismatch. This is the most important number to verify.
3. **Calibration is fitted on two scenes.** Its effect is small or negative on test (TRF −1.3 paper F1, NB12 +0.2, NB10 +1.0). Report both raw and calibrated numbers.
4. **The specialist (15) does not improve the test result.** Its validation gain did not carry over.
5. **Unsafe checkpoint loading.** Notebooks 10 through 15 load checkpoints with `torch.load(..., weights_only=False)`. Only load checkpoints from a trusted source.
6. **The README is out of date.** Section 7 of `README.md` still lists TRF and PT-UPerNet as "to be run," does not mention notebooks 13 and 15, and lists numbers for 01, 02, 03 and 08 that do not match the saved outputs.
7. **Labels are a function of spectral indices.** The ground truth is a threshold on SWIR1, NDVI, NDMI and an index, so models can reach high scores by learning those thresholds. High test scores should be read with this in mind.

---

## 5. Where each number comes from

| Model | Source |
|---|---|
| 03 RUPNet | `03_rupnet_backbone.ipynb`, final evaluation cell (saved output) |
| 05 KD | `05_knowledge_distillation.ipynb`, final evaluation cell (saved output) |
| 08 TransUNet | not valid; output copied from 03 |
| 09 SegFormer | `09_segformer_b0_backbone.ipynb`, final evaluation cell (saved output) |
| 10 Spectral ResUNet | `10_spectral_resunet.ipynb`, final evaluation cell (saved output) |
| 11 TRF | `outputs/trf_results.json` |
| 12 PT-UPerNet | `outputs/ptuper_results.json` |
| 13 Prithvi | `outputs/prithvi_results.json` |
| 15 Specialist | `outputs/soil_results.json` |

The paper-comparable value is the `weighted_water` triple (precision, recall, F1) in the JSON files, which is pixel-weighted over classes 2–6. This is the same definition notebook 13 uses in its comparison table.

---

## 6. Reproducing the table

Test-set numbers for 11, 12, 13 and 15 can be read straight from `outputs/*_results.json`. The per-class values in section 2.2 are in each file's `test_raw` and `test_calibrated` (or `test_final` for 15) under `f1`. Macro-F1 is `macro_f1_all`, and the paper metric is `weighted_water[2]`.

Models 03, 05, 09 and 10 print their results in their final notebook cell; the values in section 2.2 are copied from those printed tables.
