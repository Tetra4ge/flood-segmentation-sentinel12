<div align="center">

# Inland Excess Water (IEW) Monitoring with Sentinel-1/2

### Fine-grained waterlogging segmentation from satellite optical + radar imagery, from a paper replication to two new architectures

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-GPU%20training-76B900?style=for-the-badge&logo=nvidia&logoColor=white">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
</p>
<p>
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white">
  <img alt="Rasterio" src="https://img.shields.io/badge/Rasterio-GeoTIFF%20I%2FO-4B8BBE?style=flat-square">
  <img alt="torchgeo" src="https://img.shields.io/badge/torchgeo-SSL4EO--S12-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="timm" src="https://img.shields.io/badge/timm-ResNet--50-1F2937?style=flat-square">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=plotly&logoColor=white">
  <img alt="YAML" src="https://img.shields.io/badge/YAML-config-CB171E?style=flat-square&logo=yaml&logoColor=white">
  <img alt="Mermaid" src="https://img.shields.io/badge/Mermaid-diagrams-FF3670?style=flat-square&logo=mermaid&logoColor=white">
  <img alt="Git" src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white">
  <img alt="GitHub" src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
</p>
<p>
  <img alt="Sentinel-2" src="https://img.shields.io/badge/Sentinel--2-optical%2011%20bands-0B3D91?style=flat-square">
  <img alt="Sentinel-1" src="https://img.shields.io/badge/Sentinel--1-SAR%20VV%2FVH-0B3D91?style=flat-square">
  <img alt="Copernicus DEM" src="https://img.shields.io/badge/Copernicus-GLO--30%20DEM-0B3D91?style=flat-square">
  <img alt="Paper" src="https://img.shields.io/badge/ICCVW%202025-paper%20replication-8A2BE2?style=flat-square">
</p>

</div>

---

## Table of contents

1. [The problem](#1-the-problem)
2. [What this repository contains](#2-what-this-repository-contains)
3. [Dataset](#3-dataset)
4. [System overview](#4-system-overview)
5. [The paper pipeline (notebooks 01–10)](#5-the-paper-pipeline-notebooks-0110)
6. [New architectures (notebooks 11 and 12)](#6-new-architectures-notebooks-11-and-12)
7. [Results so far](#7-results-so-far)
8. [Training and inference protocol](#8-training-and-inference-protocol)
9. [Tech stack](#9-tech-stack)
10. [Quick start](#10-quick-start)
11. [Repository layout](#11-repository-layout)
12. [Documentation map](#12-documentation-map)
13. [Known limitations and housekeeping](#13-known-limitations-and-housekeeping)
14. [References and license](#14-references-and-license)

---

## 1. The problem

Most satellite water-mapping tools answer a yes/no question: *is this pixel open water?* **Inland excess water (IEW)**, waterlogged soil and water-stressed vegetation that never becomes a visible lake, is invisible to that kind of map, yet it damages crops and infrastructure across flat lowland regions such as the Hungarian plain.

This project builds a **6-class, pixel-level IEW map** from two complementary satellites:

| Satellite | Sees | Strength | Weakness |
|---|---|---|---|
| **Sentinel-2** (optical, 11 bands) | Colour, vegetation, soil moisture in the shortwave infrared | Fine-grained severity | Blocked by clouds |
| **Sentinel-1** (radar, VV/VH) | Surface roughness and moisture | Sees through clouds, day or night | Only rough water / non-water |

```mermaid
flowchart LR
    A["Cloudy day:<br/>optical blocked"] --> B["Radar still sees<br/>water"]
    C["Clear day:<br/>optical sees detail"] --> D["6-class severity map"]
    B --> E["Temporal gap-filling:<br/>combine both"]
    D --> E
    E --> F["Continuous IEW<br/>monitoring"]

    classDef bad fill:#fee2e2,stroke:#b91c1c,color:#111
    classDef good fill:#dcfce7,stroke:#15803d,color:#111
    classDef out fill:#dbeafe,stroke:#1d4ed8,color:#111
    class A bad
    class B,C,D good
    class E,F out
```

### The 6 target classes

| Index | Raw label | Class | Share of training pixels |
|---|---|---|---|
| 0 | 2 | Open surface water | 3.66 % |
| 1 | 3 | Soil, seriously affected | 0.32 % |
| 2 | 4 | Soil, moderately affected | 0.41 % |
| 3 | 5 | Soil, slightly affected | 2.22 % |
| 4 | 6 | Vegetation affected | 1.65 % |
| 5 | 7 | Not affected | 91.73 % |
| – | 10, 11 | Cloud / cloud shadow | masked out of loss and evaluation |

```mermaid
pie showData title Class share of training pixels (strong imbalance)
    "Not affected" : 91.73
    "Open water" : 3.66
    "Soil slightly affected" : 2.22
    "Vegetation affected" : 1.65
    "Soil moderately affected" : 0.41
    "Soil seriously affected" : 0.32
```

The three soil-severity classes make up under 3 % of pixels and are neighbouring grades of the same thing, which is why they are the hard part of the problem.

---

## 2. What this repository contains

The project has three layers:

```mermaid
flowchart TB
    subgraph L1["Layer 1: Paper replication (notebooks 01-07)"]
        A1["Backbones: U-Net, U-Net++, RUPNet"]
        A2["Radar model, knowledge distillation"]
        A3["SplitClass, temporal gap-filling"]
    end
    subgraph L2["Layer 2: Extra backbones and improvements (08-10)"]
        B1["TransUNet"]
        B2["SegFormer-B0"]
        B3["Spectral ResUNet<br/>rare-class focused"]
    end
    subgraph L3["Layer 3: New architectures (11-12)"]
        C1["Temporal-Radar Former<br/>time-series + self-supervised"]
        C2["PT-UPerNet<br/>pre-trained encoders + terrain"]
    end
    L1 --> L2 --> L3

    classDef l1 fill:#dbeafe,stroke:#1d4ed8,color:#111
    classDef l2 fill:#fef9c3,stroke:#a16207,color:#111
    classDef l3 fill:#dcfce7,stroke:#15803d,color:#111
    class A1,A2,A3 l1
    class B1,B2,B3 l2
    class C1,C2 l3
```

Every stage is a **self-contained Jupyter notebook** (no shared `src/` package), so each can be read and run top to bottom on its own.

---

## 3. Dataset

**IEW-Seg**: co-registered Sentinel-1/2 scenes over two Hungarian national-park sites, all channels resampled to a common **10 m** grid.

| Site | Scene size | Ground area | Train scenes | Test scenes |
|---|---|---|---|---|
| **BNPI** | 3600 × 5400 px | 36 × 54 km (1,944 km²) | 7 | 4 |
| **KNPI** | 2400 × 4800 px | 24 × 48 km (1,152 km²) | 7 | 5 |

The split is **temporal**: test scenes are dates the model never saw, so the score measures generalisation to new weather and seasons.

```mermaid
gantt
    title Acquisition dates per site (scene counts, train vs test)
    dateFormat YYYY-MM-DD
    axisFormat %Y
    section BNPI train
    7 scenes 2021-2024 :done, 2021-02-23, 2024-02-03
    section BNPI test
    4 scenes 2022-2024 :active, 2022-02-08, 2024-01-29
    section KNPI train
    7 scenes 2021-2022 :done, 2021-02-18, 2022-07-23
    section KNPI test
    5 scenes 2023-2024 :active, 2023-02-13, 2024-02-03
```

**Per-scene inputs**

| Input | Channels | Notes |
|---|---|---|
| Sentinel-2 | 11 bands (B01–B8A, B11, B12) | Reflectance, DN / 10000 |
| Sentinel-1 | 3: VV, VH, VV/VH ratio (dB) | Standardised |
| Ground truth | 1 | Classes 2–7, with 10 and 11 for cloud and shadow |

**On-disk layout expected by the notebooks**

```text
IEW_seg_DATASET_FINAL/tifs/
├── Train/{GT, GT_0_8, S1, S2}/<SITE>_<YYYYMMDD>.tif
└── Test/{GT, GT_0_8, S1, S2}/<SITE>_<YYYYMMDD>.tif
```

> The dataset is **not** included in this repository. Set `dataset_root` in [configs/data.yaml](configs/data.yaml) to wherever it lives on your machine.

Official dataset page: [waterdetection.github.io/water_detection](https://waterdetection.github.io/water_detection/)

---

## 4. System overview

The full target system combines all the pieces below. Notebooks 01–10 implement the baseline paper design; notebooks 11 and 12 swap in new cores.

```mermaid
flowchart LR
    subgraph IN["Inputs"]
        S2["Sentinel-2<br/>11 bands"]
        S1["Sentinel-1<br/>VV, VH, ratio"]
        DEM["Copernicus DEM<br/>terrain (NB12)"]
        HIST["Earlier dates<br/>of the same site (NB11)"]
    end

    subgraph PRE["Preprocessing"]
        P1["Per-channel standardisation"]
        P2["Spectral indices<br/>NDVI, NDMI, NDWI, MNDWI"]
        P3["Class-aware patch sampling"]
        P4["Flip / rotate / gain-noise augmentation"]
    end

    subgraph CORE["Segmentation core"]
        M1["Backbone CNNs<br/>NB01-03, 08-10"]
        M2["Temporal-Radar Former<br/>NB11"]
        M3["PT-UPerNet<br/>NB12"]
    end

    subgraph POST["Post-processing"]
        KD["Self-distillation<br/>NB05"]
        SC["SplitClass<br/>NB06"]
        GF["Temporal gap-filling<br/>NB07"]
        CAL["Head blending +<br/>logit-bias calibration"]
    end

    OUT["6-class IEW map"]

    S2 --> P1
    S1 --> P1
    DEM --> P1
    HIST --> P1
    P1 --> P2 --> P3 --> P4
    P4 --> M1
    P4 --> M2
    P4 --> M3
    M1 --> KD --> SC --> OUT
    M1 --> GF --> OUT
    M2 --> CAL
    M3 --> CAL
    CAL --> OUT

    classDef input fill:#dbeafe,stroke:#1d4ed8,color:#111
    classDef pre fill:#ede9fe,stroke:#6d28d9,color:#111
    classDef core fill:#dcfce7,stroke:#15803d,color:#111
    classDef post fill:#fef9c3,stroke:#a16207,color:#111
    classDef out fill:#fce7f3,stroke:#be185d,color:#111
    class S2,S1,DEM,HIST input
    class P1,P2,P3,P4 pre
    class M1,M2,M3 core
    class KD,SC,GF,CAL post
    class OUT out
```

### Data flow of one training step

```mermaid
sequenceDiagram
    autonumber
    participant S as Sampler
    participant D as Dataset (rasterio)
    participant N as Network
    participant L as Loss
    participant O as Optimizer + EMA
    S->>D: patch locations (rare classes oversampled)
    D->>D: read window, build features, augment
    D->>N: batch of tensors
    N->>L: main, coarse, ordinal, auxiliary outputs
    L->>O: weighted total loss
    O->>N: AdamW step, one-cycle LR
    O->>O: update EMA copy of weights
    Note over N,O: EMA weights are the ones validated and saved
```

---

## 5. The paper pipeline (notebooks 01–10)

Replication of **"Inland Excess Water (IEW) Monitoring Using Sentinel-1/2: A SplitClass Segmentation and Temporal Gap-Filling Approach"** (ICCV Workshops 2025). The full method walk-through is in [docs/PAPER_WORKFLOW.md](docs/PAPER_WORKFLOW.md).

```mermaid
flowchart TD
    A["01 U-Net"]
    B["02 U-Net++"]
    C["03 RUPNet<br/>(paper's winner)"]
    F["04 Radar model<br/>S1-only, binary"]
    D["05 Knowledge distillation<br/>teacher to student"]
    E["06 SplitClass<br/>Algorithm 1"]
    G["07 Temporal gap-filling<br/>Algorithm 2"]
    H["08 TransUNet"]
    I["09 SegFormer-B0"]
    J["10 Spectral ResUNet"]

    A -.baseline.-> D
    B -.baseline.-> D
    C --> D
    D --> E
    D --> G
    F --> G

    classDef bb fill:#dbeafe,stroke:#1d4ed8,color:#111
    classDef st fill:#dcfce7,stroke:#15803d,color:#111
    classDef ex fill:#fef9c3,stroke:#a16207,color:#111
    class A,B,C bb
    class D,E,F,G st
    class H,I,J ex
```

| # | Notebook | What it does | Produces |
|---|---|---|---|
| 01 | [01_unet_backbone](01_unet_backbone.ipynb) | Residual U-Net (32.4 M params), baseline | `unet_best.pt`* |
| 02 | [02_unetpp_backbone](02_unetpp_backbone.ipynb) | U-Net++ with dense skips (9.2 M) | `unetpp_best.pt` |
| 03 | [03_rupnet_backbone](03_rupnet_backbone.ipynb) | RUPNet (2.05 M), the paper's chosen teacher | `rupnet_best.pt` |
| 04 | [04_radar_model](04_radar_model.ipynb) | Binary water model from Sentinel-1 only | `radar_model_best.pt` |
| 05 | [05_knowledge_distillation](05_knowledge_distillation.ipynb) | Self-distillation: frozen RUPNet teacher trains a fresh student with GDL + WFL + KL | `rupnet_student_best.pt` |
| 06 | [06_splitclass_segmentation](06_splitclass_segmentation.ipynb) | Algorithm 1: keep a class where confidence ≥ θ (0.98), else output the top-2 pair | dual-label maps |
| 07 | [07_temporal_gapfilling](07_temporal_gapfilling.ipynb) | Algorithm 2: fill cloud gaps from radar plus flanking clear dates | gap-filled maps |
| 08 | [08_transunet_backbone](08_transunet_backbone.ipynb) | CNN encoder + transformer bottleneck | `transunet_best.pt` |
| 09 | [09_segformer_b0_backbone](09_segformer_b0_backbone.ipynb) | Hierarchical transformer encoder | `segformer_b0_best.pt` |
| 10 | [10_spectral_resunet](10_spectral_resunet.ipynb) | Residual U-Net + per-pixel spectral branch, rare-class sampling, logit calibration | `spectral_resunet_best.pt` |
| – | [eda_sen12flood_precursor](eda_sen12flood_precursor.ipynb) | Exploration of an earlier dataset, not part of the pipeline | – |

\* The `unet_best.pt` checkpoint is not in the repository's `outputs/` folder.

### SplitClass (Algorithm 1)

```mermaid
flowchart TD
    P["Softmax probabilities per pixel"] --> T1["Find top-1 class"]
    T1 --> Q{"top-1 probability<br/>≥ θ = 0.98 ?"}
    Q -- yes --> S["Single class label"]
    Q -- no --> T2["Find top-2 class"]
    T2 --> D["Dual label (top-1, top-2)<br/>counted correct if either matches"]
    classDef y fill:#dcfce7,stroke:#15803d,color:#111
    classDef n fill:#fef9c3,stroke:#a16207,color:#111
    class S y
    class D n
```

### Temporal gap-filling (Algorithm 2)

```mermaid
flowchart TD
    A["Today's optical prediction<br/>has cloud gaps"] --> B{"Radar says water<br/>in the gap?"}
    B -- yes --> W["Label: water"]
    B -- no --> C{"Previous and next<br/>clear dates agree?"}
    C -- yes --> E["Copy that class"]
    C -- "one is missing" --> F["Use the available one"]
    C -- "they disagree" --> G["Dual label from both"]
    classDef a fill:#dbeafe,stroke:#1d4ed8,color:#111
    classDef b fill:#dcfce7,stroke:#15803d,color:#111
    class A a
    class W,E,F,G b
```

### Losses

| Loss | Role |
|---|---|
| Generalised Dice (GDL) | Overlap-based, robust to class imbalance |
| Weighted Focal (WFL) | Down-weights easy pixels, class-frequency weights |
| KL divergence | Pulls the student toward the teacher (notebook 05 only) |

---

## 6. New architectures (notebooks 11 and 12)

Notebook 10 reached about 98 F1 on water and not-affected, but only 60–71 F1 on the soil-severity classes. The reason is what the model *sees*: one date, one small window, optical-dominated. The two new notebooks change the information, not just the network.

```mermaid
flowchart LR
    ROOT["Why soil severity<br/>is hard"] --> R1["Depends on<br/>soil moisture"]
    ROOT --> R2["Depends on<br/>history"]
    ROOT --> R3["Depends on<br/>terrain"]
    ROOT --> R4["Neighbouring grades<br/>are fuzzy"]
    R1 --> S1["Radar as its own<br/>token stream (NB11)"]
    R2 --> S2["Earlier dates as input (NB11)"]
    R3 --> S3["Elevation + depression<br/>features (NB12)"]
    R1 --> S4["Pre-trained S1/S2 encoders (NB12)"]
    R4 --> S5["Ordinal severity head<br/>(NB11 and NB12)"]

    classDef p fill:#fee2e2,stroke:#b91c1c,color:#111
    classDef s fill:#dcfce7,stroke:#15803d,color:#111
    class ROOT,R1,R2,R3,R4 p
    class S1,S2,S3,S4,S5 s
```

### 6.1 Notebook 11: Temporal-Radar Former (TRF)

[11_temporal_radar_former.ipynb](11_temporal_radar_former.ipynb) · trained on a GPU · ≈ 10–13 M parameters

**Idea:** classify each pixel from a short *history* instead of one snapshot. Each sample is the target date plus up to 3 earlier acquisitions of the same site (imagery only, never their labels).

```mermaid
flowchart TD
    subgraph DATES["Per date t (target + up to 3 earlier)"]
        S2T["Sentinel-2<br/>19 ch + cloud score + valid"]
        S1T["Sentinel-1<br/>4 ch + valid"]
    end
    S2T --> ST2["Optical date stem<br/>(shared over dates)"]
    S1T --> ST1["Radar date stem<br/>(shared over dates)"]
    ST2 --> TF
    ST1 --> TF
    TIME["Time embedding:<br/>days before target, season"] --> TF
    subgraph TF["Per-pixel temporal fusion"]
        TOK["2T tokens per location:<br/>S2 and S1 of every date"]
        ATT["2-layer transformer<br/>cloudy / missing tokens masked"]
        POOL["Attention pooling"]
        TOK --> ATT --> POOL
    end
    TF --> ENC["ConvNeXt encoder 1/4 to 1/32<br/>+ global self-attention"]
    ENC --> DEC["FPN decoder"]
    DEC --> UP["Upsample to full resolution<br/>+ spectral/local branch"]
    UP --> H1["Main head: 6 classes"]
    UP --> H2["Coarse head: 4 groups"]
    UP --> H3["Ordinal head:<br/>soil severity"]
    UP --> H4["Reconstruction head<br/>(stage 1 only)"]

    classDef a fill:#dbeafe,stroke:#1d4ed8,color:#111
    classDef b fill:#ede9fe,stroke:#6d28d9,color:#111
    classDef c fill:#dcfce7,stroke:#15803d,color:#111
    class S2T,S1T,TIME a
    class ST2,ST1,TOK,ATT,POOL b
    class ENC,DEC,UP,H1,H2,H3,H4 c
```

**Two-stage training**

```mermaid
flowchart LR
    A["Stage 1: self-supervised<br/>hide 50% of 16x16 blocks,<br/>25% of samples hide the whole<br/>target optical image"] --> B["Rebuild hidden bands from<br/>radar + history + neighbours<br/>(no labels used)"]
    B --> C["Pre-trained weights"]
    C --> D["Stage 2: supervised fine-tuning<br/>class-aware sampling, EMA,<br/>encoder LR x0.3"]
    D --> E["Calibrate on held-out<br/>training scenes"]
    E --> F["Test on 9 unseen scenes"]
    classDef s1 fill:#ede9fe,stroke:#6d28d9,color:#111
    classDef s2 fill:#dcfce7,stroke:#15803d,color:#111
    class A,B,C s1
    class D,E,F s2
```

### 6.2 Notebook 12: PT-UPerNet

[12_pretrained_terrain_upernet.ipynb](12_pretrained_terrain_upernet.ipynb) · trained on a GPU · ≈ 60 M parameters, ≈ 47 M pre-trained

**Idea:** start from encoders that already know what Sentinel imagery looks like, and add the terrain information that none of the earlier models had.

```mermaid
flowchart TD
    S2["Sentinel-2<br/>11 bands"] --> E2["ResNet-50<br/>SSL4EO-S12 S2 weights<br/>(first conv adapted 13 to 11 bands)"]
    S1["Sentinel-1<br/>VV, VH"] --> E1["ResNet-50<br/>SSL4EO-S12 S1 weights"]
    AUX["Aux channels: indices, cloud score,<br/>validity flags, ratio<br/>+ 5 terrain features"] --> EA["Light CNN encoder"]
    DEM["Copernicus 30 m DEM"] --> TERR["Relative elevation, slope,<br/>TPI 150 m / 500 m / 1.5 km"]
    TERR --> AUX
    E2 --> FUSE
    E1 --> FUSE
    EA --> FUSE
    subgraph FUSE["Fusion at 4 scales"]
        F1["concat, 1x1 conv, 3x3 conv"]
        F2["channel gate (squeeze-excitation)"]
        F1 --> F2
    end
    FUSE --> UP["UPerNet: pyramid pooling + FPN"]
    UP --> REF["Refine to full resolution<br/>with stem + spectral branch"]
    REF --> H1["Main: 6 classes"]
    REF --> H2["Coarse: 4 groups"]
    REF --> H3["Ordinal: soil severity"]
    UP --> AH["Auxiliary deep-supervision heads"]

    classDef a fill:#dbeafe,stroke:#1d4ed8,color:#111
    classDef b fill:#fef9c3,stroke:#a16207,color:#111
    classDef c fill:#dcfce7,stroke:#15803d,color:#111
    class S2,S1,DEM a
    class E2,E1,EA,TERR,AUX b
    class FUSE,UP,REF,H1,H2,H3,AH c
```

**Why terrain matters:** excess water collects in shallow depressions. The topographic position index (TPI) is a pixel's height minus the average height around it, so negative values mark depressions.

```mermaid
flowchart LR
    DEMTILE["Download Copernicus tiles<br/>(AWS, no account)"] --> ALIGN["Reproject to the scene's<br/>exact 10 m grid"]
    ALIGN --> FEAT["5 features:<br/>height, slope, TPI x3"]
    FEAT --> CACHE["Cache per site as .npy"]
    CACHE --> NET["Fed into the network<br/>as extra channels"]
    classDef n fill:#dbeafe,stroke:#1d4ed8,color:#111
    class DEMTILE,ALIGN,FEAT,CACHE,NET n
```

**Fine-tuning schedule**

```mermaid
flowchart LR
    A["Epochs 1-3<br/>encoders frozen,<br/>only new layers learn"] --> B["Epoch 4 onward<br/>everything trains<br/>encoders LR 1e-4, new layers 1e-3"]
    B --> C["BatchNorm statistics of the<br/>pre-trained encoders stay frozen"]
    C --> D["EMA weights saved<br/>as best checkpoint"]
    classDef n fill:#dcfce7,stroke:#15803d,color:#111
    class A,B,C,D n
```

### 6.3 Shared output design (notebooks 11 and 12)

Both models predict the classes in three ways and blend them:

```mermaid
flowchart TD
    FEAT["Per-pixel features"] --> MAIN["Flat head<br/>6-class softmax"]
    FEAT --> COARSE["Coarse head<br/>water / soil / vegetation / not affected"]
    FEAT --> ORD["Ordinal head<br/>P(at least moderate)<br/>P(serious | at least moderate)"]
    COARSE --> HIER["Hierarchical 6-class probabilities<br/>soil share x severity split"]
    ORD --> HIER
    MAIN --> BLEND["Blend: (1 - α)·flat + α·hierarchical<br/>α chosen on held-out training scenes"]
    HIER --> BLEND
    BLEND --> BIAS["Per-class logit bias<br/>fitted on held-out training scenes"]
    BIAS --> MAP["Final class map"]
    classDef n fill:#ede9fe,stroke:#6d28d9,color:#111
    class MAIN,COARSE,ORD,HIER,BLEND,BIAS n
```

```mermaid
stateDiagram-v2
    direction LR
    [*] --> NotSoil: coarse says not soil
    [*] --> Soil: coarse says soil
    Soil --> Slight: P(≥ moderate) low
    Soil --> Moderate: P(≥ moderate) high, P(serious) low
    Soil --> Serious: P(≥ moderate) high, P(serious) high
    Slight --> [*]
    Moderate --> [*]
    Serious --> [*]
    NotSoil --> [*]
```

This encodes slight < moderate < serious, so confusing serious with slight costs more than confusing serious with moderate. Training also smooths labels toward neighbouring grades (serious → 90 % serious + 10 % moderate).

### 6.4 How the two models differ

| | Notebook 11 TRF | Notebook 12 PT-UPerNet |
|---|---|---|
| Main bet | **Time**: earlier acquisitions | **Prior knowledge + terrain** |
| Starting weights | Random, then own self-supervised stage | SSL4EO-S12 pre-trained ResNet-50s |
| Extra data needed | None | Copernicus DEM (free) + internet once |
| Dates per sample | Target + up to 3 earlier | Target only |
| Core | Per-pixel temporal transformer → ConvNeXt | Two ResNet-50 + aux CNN → UPerNet |
| Patch / context | 256 px train, 640 px inference windows | 256 px train, 640 px inference windows |
| Handles clouds | Masks cloudy tokens, falls back to radar and history | Validity flags, radar encoder |
| Built-in ablation | Remove context dates, remove radar | Zero the terrain, zero radar |

Because they bet on different information, their errors should differ. **Averaging their calibrated probabilities** is the planned final step.

---

## 7. Results so far

> **Status:** notebooks 01–10 have been run. Notebooks 11 and 12 are **written but not yet run**; their result columns are intentionally empty and are filled by the notebooks themselves into `outputs/trf_results.json` and `outputs/ptuper_results.json`.

All numbers below are on the **9 unseen test scenes** (about 135 million labelled pixels).

### 7.1 Per-class F1 (%), latest run (notebook 10, calibrated)

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Open surface water | **98.62** | 98.19 | 99.04 |
| Soil, seriously affected | 71.04 | 71.25 | 70.83 |
| Soil, moderately affected | 60.20 | 61.22 | 59.21 |
| Soil, slightly affected | 63.54 | 52.50 | 80.46 |
| Vegetation affected | 67.22 | 53.67 | 89.93 |
| Not affected | **98.56** | 99.64 | 97.51 |
| **Macro-F1 (6 classes)** | **76.53** | | |
| **Pixel-weighted F1, classes 2–6 (paper Table 1 metric)** | **81.53** | 75.88 | 90.56 |

```mermaid
xychart-beta
    title "Notebook 10: test F1 per class (%)"
    x-axis ["Water", "Soil serious", "Soil moderate", "Soil slight", "Vegetation", "Not affected"]
    y-axis "F1 (%)" 0 --> 100
    bar [98.62, 71.04, 60.20, 63.54, 67.22, 98.56]
```

### 7.2 Model comparison

The paper's Table 1 reports F1 over the **five water-related classes only** (2–6), excluding the huge not-affected class. Rows that include class 7 look much higher and are **not comparable** to the paper.

| Model | Params | F1, all 6 classes | F1, classes 2–6 (paper-comparable) |
|---|---|---|---|
| U-Net (NB01) | 32.4 M | 96.90 | not recomputed |
| U-Net++ (NB02) | 9.2 M | 96.79 | not recomputed |
| RUPNet (NB03) | 2.05 M | 94.35 | not recomputed |
| TransUNet (NB08) | – | 93.46 | 63.97 |
| SegFormer-B0 (NB09) | – | 95.65 | 68.47 |
| **Spectral ResUNet (NB10), raw** | 5.2 M | 75.57 macro* | 80.50 |
| **Spectral ResUNet (NB10), calibrated** | 5.2 M | 76.53 macro* | **81.53** |
| *Paper's best (RUPNet, Table 1)* | 2.05 M | – | *81.19* |
| TRF (NB11) | ≈ 10–13 M | to be run | to be run |
| PT-UPerNet (NB12) | ≈ 60 M | to be run | to be run |

\* Notebook 10 reports macro-F1 over the 6 classes, a different aggregation from the pixel-weighted figure used for NB01–03, 08 and 09, so those cells are not directly comparable. The classes 2–6 column is the like-for-like one.

```mermaid
xychart-beta
    title "Classes 2-6 pixel-weighted F1 (%), test set"
    x-axis ["TransUNet", "SegFormer-B0", "Spectral ResUNet raw", "Spectral ResUNet cal", "Paper best"]
    y-axis "F1 (%)" 50 --> 90
    bar [63.97, 68.47, 80.50, 81.53, 81.19]
```

### 7.3 Where the headroom is

| Class | Current F1 | Gap to 90 |
|---|---|---|
| Water | 98.62 | already above |
| Not affected | 98.56 | already above |
| Soil seriously affected | 71.04 | 19 |
| Vegetation affected | 67.22 | 23 |
| Soil slightly affected | 63.54 | 26 |
| Soil moderately affected | 60.20 | 30 |

Reaching 90+ on every class is **not guaranteed**: the boundaries between serious, moderate and slight in the ground truth are themselves fuzzy, and the paper's own best model reached about 81 F1 averaged over classes 2–6. Notebooks 11 and 12 are the attempt to close this gap by adding information, not just capacity.

---

## 8. Training and inference protocol

All of notebooks 10, 11 and 12 use the same protocol, so their numbers can be compared directly.

```mermaid
flowchart TD
    A["14 training scenes"] --> B["12 for training"]
    A --> C["2 held out:<br/>BNPI_20230208, KNPI_20220310"]
    B --> D["Train with class-aware sampling"]
    D --> E["Validate each epoch on fixed<br/>patches of the 2 held-out scenes"]
    E --> F["Keep best checkpoint (macro-F1)"]
    F --> G["Full-scene inference on the<br/>2 held-out scenes"]
    C --> E
    C --> G
    G --> H["Fit head-blend weight α<br/>and per-class logit bias"]
    H --> I["Apply to the 9 test scenes<br/>(never used for any fitting)"]
    classDef n fill:#dcfce7,stroke:#15803d,color:#111
    classDef t fill:#fee2e2,stroke:#b91c1c,color:#111
    class A,B,C,D,E,F,G,H n
    class I t
```

| Component | Setting |
|---|---|
| Optimiser | AdamW with a one-cycle learning-rate schedule |
| Precision | Mixed (bf16 / fp16 on CUDA) |
| Patch size | 96 px (NB10), 256 px (NB11, NB12) |
| Class balance | Per-epoch shares drawn from patches rich in each rare class, plus inverse-square-root class weights |
| Weight averaging | EMA of weights (NB11, NB12) |
| Inference | 640 px windows, centre 512 px kept, no seams; 8-way flip/rotation averaging |
| Calibration | Fitted only on held-out training scenes, never on test |
| Metrics | Per-class precision / recall / F1, macro-F1, and pixel-weighted F1 over classes 2–6 |

---

## 9. Tech stack

| Area | Tools |
|---|---|
| Language and runtime | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white) |
| Deep learning | ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![CUDA](https://img.shields.io/badge/-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white) ![torchgeo](https://img.shields.io/badge/-torchgeo-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![timm](https://img.shields.io/badge/-timm-1F2937?style=flat-square) |
| Geospatial | ![Rasterio](https://img.shields.io/badge/-Rasterio-4B8BBE?style=flat-square) ![GeoTIFF](https://img.shields.io/badge/-GeoTIFF-2E7D32?style=flat-square) ![Copernicus](https://img.shields.io/badge/-Copernicus%20DEM-0B3D91?style=flat-square) |
| Satellite data | ![Sentinel-1](https://img.shields.io/badge/-Sentinel--1-0B3D91?style=flat-square) ![Sentinel-2](https://img.shields.io/badge/-Sentinel--2-0B3D91?style=flat-square) |
| Numerics and plots | ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square&logo=plotly&logoColor=white) |
| Config and docs | ![YAML](https://img.shields.io/badge/-YAML-CB171E?style=flat-square&logo=yaml&logoColor=white) ![Markdown](https://img.shields.io/badge/-Markdown-000000?style=flat-square&logo=markdown&logoColor=white) ![Mermaid](https://img.shields.io/badge/-Mermaid-FF3670?style=flat-square&logo=mermaid&logoColor=white) |
| Version control | ![Git](https://img.shields.io/badge/-Git-F05032?style=flat-square&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white) |

**Architectures used across the notebooks:** residual U-Net, U-Net++, RUPNet, TransUNet, SegFormer, Spectral ResUNet, ConvNeXt, transformer encoders, FPN, UPerNet, squeeze-and-excitation, ResNet-50 (self-supervised pre-trained).

---

## 10. Quick start

### Requirements

- Python 3.10+ and a CUDA GPU (notebooks 11 and 12 are not practical on CPU; notebooks 01–10 were developed on CPU, but slowly)
- For notebook 12: internet access once (pre-trained weights and elevation tiles)

```bash
pip install torch torchvision          # use the CUDA build for your machine
pip install rasterio numpy pyyaml matplotlib
pip install torchgeo                   # notebook 12 only (also installs timm)
```

### Steps

```mermaid
flowchart LR
    A["1. Get the IEW-Seg<br/>dataset"] --> B["2. Edit dataset_root in<br/>configs/data.yaml"]
    B --> C["3. Open a notebook<br/>in Jupyter"]
    C --> D["4. Dry run:<br/>QUICK_TEST = True"]
    D --> E{"Runs without<br/>errors?"}
    E -- yes --> F["5. QUICK_TEST = False,<br/>Restart and Run All"]
    E -- no --> G["Fix path / memory<br/>settings"]
    G --> D
    F --> H["6. Read outputs/<br/>*_results.json"]
    classDef n fill:#dbeafe,stroke:#1d4ed8,color:#111
    classDef ok fill:#dcfce7,stroke:#15803d,color:#111
    class A,B,C,D,F,G n
    class H ok
```

### Recommended run order

| Goal | Run |
|---|---|
| Reproduce the paper pipeline | 03 → 04 → 05 → 06 and 07 |
| Best current single model | 10 |
| Try the new architectures | 11 and 12 (independent of each other) |
| Final combined result | run 11 and 12, then average their calibrated probabilities |

### GPU memory

The defaults (`PATCH = 256`, `BATCH_SIZE = 8`, mixed precision) need roughly 8–12 GB. If you hit *CUDA out of memory*, set `BATCH_SIZE = 4`, then `PATCH = 192` (must stay divisible by 32).

### Resuming

Notebooks 10, 11 and 12 save a checkpoint every epoch and resume automatically.

---

## 11. Repository layout

```text
.
├── 01_unet_backbone.ipynb               paper baseline backbone
├── 02_unetpp_backbone.ipynb             paper baseline backbone
├── 03_rupnet_backbone.ipynb             paper's winning backbone / KD teacher
├── 04_radar_model.ipynb                 Sentinel-1-only binary model
├── 05_knowledge_distillation.ipynb      teacher to student
├── 06_splitclass_segmentation.ipynb     Algorithm 1
├── 07_temporal_gapfilling.ipynb         Algorithm 2
├── 08_transunet_backbone.ipynb          extra backbone
├── 09_segformer_b0_backbone.ipynb       extra backbone
├── 10_spectral_resunet.ipynb            rare-class-focused model
├── 11_temporal_radar_former.ipynb       NEW: multi-date transformer
├── 12_pretrained_terrain_upernet.ipynb  NEW: pre-trained encoders + terrain
├── eda_sen12flood_precursor.ipynb       exploration of an earlier dataset
├── build_transunet.py                   script that generated notebook 08
├── configs/data.yaml                    dataset path, patch size, classes
├── docs/                                PRD, TRD, plan, results template, paper walk-through
├── papers/                              reference PDFs
├── outputs/                             checkpoints, patch cache, class stats (generated)
├── implementation.md                    implementation-vs-paper report (snapshot, 2026-08-14)
├── LICENSE
└── README.md
```

```mermaid
flowchart TD
    ROOT["flood-segmentation-sentinel12"] --> NB["12 pipeline notebooks<br/>+ 1 EDA notebook"]
    ROOT --> CFG["configs/data.yaml"]
    ROOT --> DOC["docs/ and implementation.md"]
    ROOT --> PAP["papers/ (8 PDFs)"]
    ROOT --> OUT["outputs/<br/>checkpoints, patches, dem, results json"]
    classDef n fill:#dbeafe,stroke:#1d4ed8,color:#111
    class ROOT,NB,CFG,DOC,PAP,OUT n
```

### Generated outputs

| Path | Written by |
|---|---|
| `outputs/checkpoints/*.pt` | each training notebook |
| `outputs/checkpoints/trf_*`, `ptuper_*` | notebooks 11 and 12 (checkpoints, calibration JSON) |
| `outputs/dem/*.npy` | notebook 12 (terrain features per site) |
| `outputs/trf_results.json`, `outputs/ptuper_results.json` | notebooks 11 and 12 (all test metrics) |
| `outputs/class_stats.csv` | class pixel counts per scene |

---

## 12. Documentation map

| Document | Contents |
|---|---|
| [docs/PAPER_WORKFLOW.md](docs/PAPER_WORKFLOW.md) | The paper end to end: problem, dataset, method, experiments |
| [docs/PHASE_1_PRD.md](docs/PHASE_1_PRD.md) | Product requirements |
| [docs/PHASE_2_TRD.md](docs/PHASE_2_TRD.md) | Technical requirements |
| [docs/PHASE_3_IMPLEMENTATION_PLAN.md](docs/PHASE_3_IMPLEMENTATION_PLAN.md) | Milestones |
| [docs/PHASE_4_EXPERIMENTATION_RESULTS.md](docs/PHASE_4_EXPERIMENTATION_RESULTS.md) | Experiment matrix and results template |
| [implementation.md](implementation.md) | Implementation vs. paper alignment, written when only 3 backbones existed |

```mermaid
flowchart LR
    P1["Phase 1<br/>PRD"] --> P2["Phase 2<br/>TRD"] --> P3["Phase 3<br/>Plan"] --> P4["Phase 4<br/>Experiments"]
    P4 --> NB["Notebooks 01-12"]
    classDef n fill:#ede9fe,stroke:#6d28d9,color:#111
    class P1,P2,P3,P4,NB n
```

---

## 13. Known limitations and housekeeping

| Item | Detail |
|---|---|
| Notebooks 11 and 12 | Written and syntax-checked but **not yet run**; expect a first trial run (`QUICK_TEST`) to surface small fixes |
| Notebooks 04–07 | Run correctly in design but not executed end to end in this repository's results |
| 90+ F1 on every class | Not guaranteed; soil-severity labels are inherently fuzzy |
| Metric comparability | Only the classes 2–6 pixel-weighted F1 is comparable to the paper's Table 1 |
| `.gitignore` | The rule for `outputs/checkpoints/*.pt` is commented out, so checkpoints (about 250 MB) are tracked unless it is enabled |
| `configs/data.yaml` | `dataset_root` is a Windows path; edit it for your machine |
| `implementation.md` | A snapshot from before notebooks 08–12 existed; some of its status statements are outdated |
| Transductive option | Notebook 11 can also pre-train on test-split *imagery* (`SSL_USE_TEST_IMAGERY`); off by default and must be disclosed if enabled |

---

## 14. References and license

**Main paper:** Ibrahim, Belényesi, Liu, Richter-Cserey, Simon, Szirányi, Benedek. *Inland Excess Water (IEW) Monitoring Using Sentinel-1/2: A SplitClass Segmentation and Temporal Gap-Filling Approach.* ICCV Workshops 2025 (Sustainability with Earth Observation and AI).
[Paper](https://openaccess.thecvf.com/content/ICCV2025W/Sustainability/html/Ibrahim_Inland_Excess_Water_IEW_Monitoring_Using_Sentinel-12_A_SplitClass_Segmentation_ICCVW_2025_paper.html) · [Dataset page](https://waterdetection.github.io/water_detection/)

**Other references in [papers/](papers/):** Sen1Floods11, H2O-Net, SkySense, EarthDial, SatMAE++, and a multi-sensor geospatial foundation-model paper (CVPR 2024).

**Pre-trained weights and data used by notebook 12:** SSL4EO-S12 (via torchgeo) and the Copernicus GLO-30 digital elevation model.

Released under the [MIT License](LICENSE).
