# Inland Excess Water (IEW) Monitoring using Sentinel-1/2

Replication codebase for **"Inland Excess Water (IEW) Monitoring Using Sentinel-1/2: A SplitClass Segmentation and Temporal Gap-Filling Approach"** (ICCVW 2025, Sustainability with Earth Observation and AI Workshop).

- **Paper:** [IEEE/CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025W/Sustainability/html/Ibrahim_Inland_Excess_Water_IEW_Monitoring_Using_Sentinel-12_A_SplitClass_Segmentation_ICCVW_2025_paper.html)
- **Dataset & official page:** [waterdetection.github.io/water_detection](https://waterdetection.github.io/water_detection/)
- **Full method walkthrough:** [`docs/PAPER_WORKFLOW.md`](docs/PAPER_WORKFLOW.md)
- **Implementation-vs-paper alignment report:** [`implementation.md`](implementation.md)

---

## Overview

Binary "water vs. non-water" segmentation misses inland waterlogging. This pipeline instead maps **6 fine-grained classes** (open water + 4 soil/vegetation severity levels + "not affected") by fusing two satellite sources per scene:

| Source | Bands | Role |
|---|---|---|
| **Sentinel-2** (optical) | 11 bands | Fine-grained 6-class segmentation |
| **Sentinel-1** (SAR) | 3 bands — VV, VH, VV/VH | Cloud-penetrating, binary water/non-water gap-filling |

## System Architecture

```mermaid
flowchart LR
    subgraph IN["Input"]
        S2["Sentinel-2 Optical\n11 bands"]
        S1["Sentinel-1 SAR\n3 bands"]
    end

    subgraph PRE["Preprocessing"]
        PP1["Per-channel standardization\n(mean / std)"]
        PP2["Sliding-window patching\n64x64, stride 64"]
        PP3["Flip / rotation augmentation"]
    end

    subgraph CORE["Segmentation Core"]
        BB["Backbone CNN\nU-Net / U-Net++ / RUPNet"]
        KD["Self-Distillation\nTeacher -> Student"]
        RM["Radar Model\nbinary S1-only"]
    end

    subgraph POST["Post-Processing"]
        SC["SplitClass Segmentation\nAlgorithm 1"]
        GF["Temporal Gap-Filling\nAlgorithm 2"]
    end

    OUT["6-Class IEW Map"]

    S2 --> PP1 --> PP2 --> PP3 --> BB
    S1 --> PP1
    BB --> KD --> SC
    PP3 --> RM --> GF
    KD --> GF
    SC --> OUT
    GF --> OUT

    classDef input fill:#2b6cb0,stroke:#1a365d,color:#fff
    classDef pre fill:#805ad5,stroke:#44337a,color:#fff
    classDef core fill:#2f855a,stroke:#22543d,color:#fff
    classDef post fill:#b7791f,stroke:#7b341e,color:#fff
    classDef output fill:#553c9a,stroke:#322659,color:#fff
    class S2,S1 input
    class PP1,PP2,PP3 pre
    class BB,KD,RM core
    class SC,GF post
    class OUT output
```

| Stage | What it does |
|---|---|
| Standardization | Per-channel mean/std computed from the training set, applied to both S1 and S2 rasters. |
| Patching | Scenes are cut into 64x64 patches on a stride-64 grid (`configs/data.yaml`), cached to `outputs/patches/`. |
| Augmentation | Random flips/rotations applied during training only. |
| Backbone | Learns the 6-class map from Sentinel-2. |
| Self-distillation | Compresses the trained teacher into a lighter student. |
| Radar model | Independently predicts binary water from Sentinel-1 (works through cloud cover). |
| SplitClass Segmentation | Refines confident pixels into dual (class, confidence) labels. |
| Temporal Gap-Filling | Uses the radar signal and neighboring dates to reconstruct cloud-obscured pixels in the optical map. |

## Pipeline: Notebooks & Data Flow

Each stage is one self-contained Jupyter notebook — no shared `src/` package, so every notebook can be read and run top-to-bottom on its own. Every model notebook's "Model Architecture" section walks through the convolution math, the encoder/decoder shape flow (with an ASCII diagram tracing exact tensor shapes end to end), and how the final layer's logits become a predicted class map — written to be read directly, not just skimmed as code.

```mermaid
flowchart TD
    A["01_unet_backbone.ipynb"]
    B["02_unetpp_backbone.ipynb"]
    C["03_rupnet_backbone.ipynb\n(paper's winning backbone)"]
    D["05_knowledge_distillation.ipynb\nTeacher → Student"]
    E["06_splitclass_segmentation.ipynb\nAlgorithm 1"]
    F["04_radar_model.ipynb\nS1-only binary model"]
    G["07_temporal_gapfilling.ipynb\nAlgorithm 2"]

    A --> D
    B -.-> D
    C --> D
    D --> E
    D --> G
    F --> G

    classDef backbone fill:#2b6cb0,stroke:#1a365d,color:#fff
    classDef stage fill:#2f855a,stroke:#22543d,color:#fff
    class A,B,C backbone
    class D,E,F,G stage
```

| Notebook | Produces |
|---|---|
| `01_unet_backbone.ipynb` | `outputs/checkpoints/unet_best.pt` |
| `02_unetpp_backbone.ipynb` | `outputs/checkpoints/unetpp_best.pt` |
| `03_rupnet_backbone.ipynb` | `outputs/checkpoints/rupnet_best.pt` (teacher) |
| `04_radar_model.ipynb` | `outputs/checkpoints/radar_model_best.pt` |
| `05_knowledge_distillation.ipynb` | `outputs/checkpoints/rupnet_student_best.pt` |
| `06_splitclass_segmentation.ipynb` | dual confidence-labeled maps (post-processing, no training) |
| `07_temporal_gapfilling.ipynb` | gap-filled temporal maps (post-processing, no training) |

**Reproduce end-to-end:** `03_rupnet_backbone.ipynb` → `04_radar_model.ipynb` (independent) → `05_knowledge_distillation.ipynb` → `06_splitclass_segmentation.ipynb` and `07_temporal_gapfilling.ipynb` (either order). `01_unet_backbone.ipynb` / `02_unetpp_backbone.ipynb` are backbone-comparison baselines only.

> `eda_sen12flood_precursor.ipynb` is precursor EDA on a different, earlier dataset — not part of this pipeline.

## Dataset

**IEW-Seg** — co-registered Sentinel-1/2 scenes over two Hungarian sites, all channels resampled to a common **10 m** grid.

| Site | Scene size | Ground area |
|---|---|---|
| BNPI | 3600 × 5400 px | 36 km × 54 km (1,944 km²) |
| KNPI | 2400 × 4800 px | 24 km × 48 km (1,152 km²) |

**Split** (temporal, to test generalization to unseen dates):

| Split | Scenes | Sites |
|---|---|---|
| Train | 14 | 7 BNPI + 7 KNPI |
| Test | 9 | 4 BNPI + 5 KNPI |

**Ground truth classes** (cloud/shadow classes 10 & 11 are masked out of loss and evaluation):

| Class | Meaning |
|---|---|
| 2 | Open surface water |
| 3 | Soil — seriously affected |
| 4 | Soil — moderately affected |
| 5 | Soil — slightly affected |
| 6 | Vegetation affected |
| 7 | Not affected |

**On-disk layout:**

```text
IEW_seg_DATASET_FINAL/tifs/
├── Train/{GT, GT_0_8, S1, S2}/
└── Test/{GT, GT_0_8, S1, S2}/
```

## Current Results (backbone comparison, Stage C)

40 epochs, GDL+WFL loss, evaluated on the full test set (32,691 patches / 9 scenes):

| Model | Params | F1 (all 6 classes) | Paper F1 (Table 1) |
|---|---|---|---|
| U-Net | 32.44M | 96.90% | 79.38% |
| U-Net++ | 9.16M | 96.79% | 79.10% |
| RUPNet | 2.05M | 94.35% | **81.19%** (paper's winner) |

RUPNet — the paper's best backbone — trails here despite matching the paper's parameter count almost exactly; the training log shows it was still improving when the 40-epoch run stopped (undertrained relative to its smaller capacity). See `implementation.md` for the fix and full diagnosis.

Stages D–F (`04`–`07`) run correctly but haven't been executed end-to-end in this repo, since the dev environment is CPU-only.

*(Stage lettering follows the paper: D = knowledge distillation (`05`), E = SplitClass segmentation (`06`), F = temporal gap-filling (`07`).)*

## Repository Layout

```text
.
├── configs/data.yaml                  # dataset root, patch size/stride, channels, classes
├── docs/                              # PRD, TRD, implementation plan, results tracker, paper walkthrough
├── outputs/                           # checkpoints, cached patches, class stats (generated, gitignored)
├── papers/                            # reference PDFs
├── *.ipynb                            # one notebook per pipeline stage (see table above)
└── implementation.md                  # implementation-vs-paper alignment report
```
