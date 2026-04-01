# SpatialLM: Training Large Language Models for Structured Indoor Modeling

**Paper:** [arXiv:2506.07491](https://arxiv.org/abs/2506.07491)
**Repository:** [manycore-research/SpatialLM](https://github.com/manycore-research/SpatialLM)
**Venue:** NeurIPS 2025
**Authors:** Yongsen Mao, Junhao Zhong, Chuan Fang, Jia Zheng, Rui Tang, Hao Zhu, Ping Tan, Zihan Zhou

---

## 1. Overview

SpatialLM is a large language model designed to process **3D point cloud data** and generate **structured 3D scene understanding outputs**. These outputs include architectural elements (walls, doors, windows) and oriented 3D object bounding boxes with semantic categories. Unlike prior methods that rely on task-specific network designs, SpatialLM adheres to a standard multimodal LLM architecture and is fine-tuned directly from open-source LLMs.

**Core idea:** Encode a 3D point cloud into feature tokens, inject them into an LLM, and auto-regressively generate structured text that describes the scene.

---

## 2. Architecture

SpatialLM follows a three-stage multimodal LLM pipeline:

```
3D Point Cloud (PLY)
    → Point Cloud Encoder (Sonata / SceneScript)
    → Feature Projection (Linear / MLP)
    → LLM Backbone (Llama 3.2 / Qwen 2.5)
    → Structured Text Output (Wall, Door, Window, Bbox)
```

### 2.1 Point Cloud Encoder

| Version | Encoder | Description |
|---|---|---|
| v1.0 | SceneScript | Sparse tensor representation with configurable conv layers and bins |
| v1.1 | Sonata | Hierarchical encoding with doubled point cloud resolution |

### 2.2 Feature Projection

Point cloud features are projected into the LLM's embedding space via a linear layer or 2-layer MLP with GELU activation. Features are injected into the token sequence between special `point_start_token` and `point_end_token` markers.

### 2.3 LLM Backbone

Standard open-source LLMs fine-tuned with language modeling loss. Point cloud token positions are masked (`IGNORE_INDEX = -100`) so only text generation contributes to the loss.

### 2.4 Point Cloud Serialization

Uses **Hilbert curve ordering** to linearize 3D point clouds into 1D sequences while preserving spatial locality — nearby 3D points map to nearby positions in the sequence, which is critical for transformer processing.

---

## 3. Models

| Model | Base LLM | Encoder | Params | License |
|---|---|---|---|---|
| SpatialLM1.1-Llama-1B | Llama-3.2-1B-Instruct | Sonata | ~1B | Llama 3.2 |
| SpatialLM1.1-Qwen-0.5B | Qwen2.5-0.5B-Instruct | Sonata | ~0.5B | CC-BY-NC-4.0 |
| SpatialLM1.0-Llama-1B | Llama-3.2-1B-Instruct | SceneScript | ~1B | Llama 3.2 |
| SpatialLM1.0-Qwen-0.5B | Qwen2.5-0.5B-Instruct | SceneScript | ~0.5B | Apache 2.0 |

**v1.0 → v1.1 improvements:** Doubled point cloud resolution, replaced SceneScript encoder with the more powerful Sonata encoder, added user-specified category detection.

---

## 4. Input / Output

### 4.1 Input

**Format:** Axis-aligned point cloud in PLY format (z-axis up, metric scale: 1 = 1 meter)

**Point cloud sources:**
- Monocular RGB video → 3D reconstruction via MASt3R-SLAM or SLAM3R
- RGBD images
- LiDAR scans

**Preprocessing:** Positive shift normalization, color normalization, grid sampling to create discretized tensor features.

### 4.2 Output

Structured text using Python dataclass syntax:

```python
Wall(a_x, a_y, a_z, b_x, b_y, b_z, height, thickness)
# Two 3D endpoints defining wall center line + dimensions

Door(wall_id, position_x, position_y, position_z, width, height)
# Wall-referenced opening

Window(wall_id, position_x, position_y, position_z, width, height)
# Wall-referenced opening

Bbox(class_name, position_x, position_y, position_z, angle_z, scale_x, scale_y, scale_z)
# Oriented 3D bounding box with semantic label
```

**Coordinate discretization bins:**
- World coordinates: 0–32.0
- Height/Width: 0–25.6
- Scale: 0–20.0
- Angle (radians): -6.28 to 6.28

### 4.3 Three Task Modes

| Mode | `--detect_type` | Output |
|---|---|---|
| Structured Reconstruction | `all` | Walls + Doors + Windows + Bounding Boxes |
| Layout Estimation | `arch` | Walls + Doors + Windows |
| 3D Object Detection | `object` | Bounding Boxes only |

**Conditional detection (v1.1):** Specify which of the 59 supported categories to detect via `--category`.

---

## 5. Training Data: SpatialLM-Dataset

| Spec | Value |
|---|---|
| Total scenes | 12,328 |
| Total rooms | 54,778 |
| Room type categories | 70 |
| Source | Synthetic (professional 3D designers) |
| Train / Val / Test scenes | 11,328 / 500 / 500 |
| Train / Val / Test point clouds | 199,286 / 500 / 500 |
| Format | PLY + TXT layout + ShareGPT-style JSON |
| License | CC-BY-NC-4.0 |
| HuggingFace | `manycore-research/SpatialLM-Dataset` |

**4 point cloud sampling configurations:**
- Config 0: Most complete (8 panoramic views)
- Config 1: Most sparse (8 perspective views)
- Config 2: Less complete (16 perspective views)
- Config 3: Less sparse (24 perspective views)

**Real-world test set:** 107 point clouds reconstructed from RGB videos via MASt3R-SLAM (`manycore-research/SpatialLM-Testset`).

---

## 6. Benchmark Results

### 6.1 Layout Estimation (Structured3D, finetuned)

| Method | F1 @.25 IoU | F1 @.5 IoU |
|---|---|---|
| RoomFormer | 83.4 | 81.4 |
| SceneScript (finetuned) | 90.4 | 89.2 |
| **SpatialLM1.1-Qwen-0.5B** | **94.3** | **93.5** |

### 6.2 3D Object Detection (ScanNet, 18 categories, finetuned)

| Method | F1 @.25 IoU | F1 @.5 IoU |
|---|---|---|
| V-DETR | 65.1 | 56.8 |
| SceneScript (finetuned) | 49.1 | 36.8 |
| **SpatialLM1.1-Qwen-0.5B** | **65.6** | **52.6** |

### 6.3 Zero-Shot Detection on Videos (SpatialLM-Testset)

| Category | Llama-1B (F1 @.25) | Qwen-0.5B (F1 @.25) |
|---|---|---|
| **Layout** | | |
| wall | 68.9 | 68.2 |
| door | 49.1 | 47.4 |
| window | 47.0 | 51.4 |
| **Objects** | | |
| bed | 96.8 | 95.2 |
| sofa | 66.9 | 69.1 |
| nightstand | 62.8 | 67.0 |
| coffee table | 56.4 | 64.9 |
| chandelier | 53.5 | 36.8 |
| dresser | 46.7 | 46.7 |
| dining table | 40.7 | 24.2 |
| carpet | 40.3 | 24.1 |
| curtain | 34.9 | 37.0 |
| painting | 34.9 | 38.2 |
| wardrobe | 29.4 | 39.6 |
| plants | 29.5 | 26.3 |
| stool | 17.6 | 30.8 |
| chair | 20.8 | 32.3 |
| tv cabinet | 34.4 | 27.3 |
| air conditioner | 16.7 | 24.0 |
| tv | 16.0 | 18.0 |
| cabinet | 15.2 | 11.2 |
| side table | 14.6 | 9.7 |
| refrigerator | 0.0 | 16.7 |

---

## 7. Supported Object Categories (59)

sofa, chair, dining_chair, bar_chair, stool, bed, pillow, wardrobe, nightstand, tv_cabinet, wine_cabinet, bathroom_cabinet, shoe_cabinet, entrance_cabinet, decorative_cabinet, washing_cabinet, wall_cabinet, sideboard, cupboard, coffee_table, dining_table, side_table, dressing_table, desk, integrated_stove, gas_stove, range_hood, micro-wave_oven, sink, stove, refrigerator, hand_sink, shower, shower_room, toilet, tub, illumination, chandelier, floor-standing_lamp, wall_decoration, painting, curtain, carpet, plants, potted_bonsai, tv, computer, air_conditioner, washing_machine, clothes_rack, mirror, bookcase, cushion, bar, screen, combination_sofa, dining_table_combination, leisure_table_and_chair_combination, multifunctional_combination_bed

---

## 8. Installation

**Requirements:** Python 3.11, PyTorch 2.4.1, CUDA 12.4

```bash
git clone https://github.com/manycore-research/SpatialLM.git
cd SpatialLM

conda create -n spatiallm python=3.11
conda activate spatiallm
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit conda-forge::sparsehash

pip install poetry && poetry config virtualenvs.create false --local
poetry install

# For SpatialLM1.0 (SceneScript encoder):
poe install-torchsparse

# For SpatialLM1.1 (Sonata encoder):
poe install-sonata          # Building flash-attn wheel takes a while

# For finetuning:
poe install-training
```

---

## 9. Usage

### 9.1 Inference

```bash
# Download example point cloud
huggingface-cli download manycore-research/SpatialLM-Testset \
    pcd/scene0000_00.ply --repo-type dataset --local-dir .

# Full structured reconstruction
python inference.py \
    --point_cloud pcd/scene0000_00.ply \
    --output scene0000_00.txt \
    --model_path manycore-research/SpatialLM1.1-Qwen-0.5B

# Object detection with specific categories
python inference.py \
    --point_cloud pcd/scene0000_00.ply \
    --output scene0000_00.txt \
    --model_path manycore-research/SpatialLM1.1-Qwen-0.5B \
    --detect_type object \
    --category bed nightstand

# Batch inference on folder
python inference.py \
    --point_cloud SpatialLM-Testset/pcd \
    --output SpatialLM-Testset/pred \
    --model_path manycore-research/SpatialLM1.1-Qwen-0.5B
```

**Inference arguments:**

| Argument | Default | Description |
|---|---|---|
| `--point_cloud` | (required) | Path to PLY file or folder |
| `--output` | (required) | Path to output TXT or folder |
| `--model_path` | `SpatialLM-Llama-1B` | HuggingFace model name or local path |
| `--detect_type` | `all` | `all` / `arch` / `object` |
| `--category` | `[]` | Subset of 59 categories to detect |
| `--top_k` | 10 | Top-k filtering |
| `--top_p` | 0.95 | Nucleus sampling |
| `--temperature` | 0.6 | Sampling temperature |
| `--num_beams` | 1 | Beam search width |
| `--inference_dtype` | `bfloat16` | `bfloat16` or `float32` |
| `--seed` | -1 | Random seed (-1 = none) |

### 9.2 Visualization

```bash
python visualize.py \
    --point_cloud pcd/scene0000_00.ply \
    --layout scene0000_00.txt \
    --save scene0000_00.rrd

rerun scene0000_00.rrd
```

### 9.3 Evaluation

```bash
python eval.py \
    --metadata SpatialLM-Testset/test.csv \
    --gt_dir SpatialLM-Testset/layout \
    --pred_dir SpatialLM-Testset/pred \
    --label_mapping SpatialLM-Testset/benchmark_categories.tsv
```

---

## 10. Custom Video Pipeline

To use SpatialLM on your own RGB video, four steps are required:

### Step 1: Reconstruct Point Cloud

Use [SLAM3R](https://github.com/PKU-VCL-3DV/SLAM3R) or [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) to reconstruct a 3D point cloud from RGB video.

### Step 2: Axis Alignment

Align the point cloud so the z-axis points up (ScanNet convention). Methods:
- VanishingPoint estimation + Manhattan Frame estimation
- [U-ARE-ME](https://github.com/naver/uareme)
- [Perspective Fields](https://github.com/jinlinyi/PerspectiveFields)
- Manual alignment in Blender

### Step 3: Metric Scaling

Scale the point cloud to real-world metric (1 unit = 1 meter). Heuristic: resize so wall height is approximately 2.5m. Alternatively, use [Depth Pro](https://github.com/apple/ml-depth-pro) for metric depth reference from keyframes.

### Step 4: Run Inference

```bash
python inference.py --point_cloud your_scene.ply --output your_scene.txt
```

---

## 11. Fine-Tuning

**Config:** `configs/spatiallm_sft.yaml`

| Parameter | Value |
|---|---|
| Learning rate | 1.0e-5 (aggressive: 5.0e-5 or 1.0e-4) |
| Epochs | 10 |
| Batch size per device | 1 |
| LR scheduler | Cosine with 0.03 warmup ratio |
| Cutoff length | 8192 |
| Max new tokens | 4096 |
| VRAM required | ~60GB (full fine-tuning) |
| LoRA / Quantized | Not yet supported |

```bash
# Single-node fine-tuning
python train.py configs/spatiallm_sft.yaml

# Inference with fine-tuned model
python inference.py -d object \
    -p arkitscenes-spatiallm/pcd/42446137.ply \
    -o 42446137.txt \
    --model_path ysmao/SpatialLM1.1-Qwen-0.5B-ARKitScenes-SFT
```

**Example fine-tuning dataset:** ARKitScenes-SpatialLM at `ysmao/arkitscenes-spatiallm`

**Distributed training:** Set `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NODE_RANK`, `NPROC_PER_NODE` environment variables.

---

## 12. Source Layout

```
SpatialLM/
├── inference.py              # Main inference entry point
├── train.py                  # Fine-tuning entry point
├── eval.py                   # Evaluation script
├── visualize.py              # Rerun-based 3D visualization
├── code_template.txt         # Output schema (Wall/Door/Window/Bbox dataclasses)
├── pyproject.toml            # Poetry config (version 0.1.1)
├── configs/
│   └── spatiallm_sft.yaml   # SFT fine-tuning config
├── spatiallm/
│   ├── layout/               # Layout parsing and processing
│   ├── model/                # Model definitions (spatiallm_qwen, spatiallm_llama)
│   ├── pcd/                  # Point cloud loading, preprocessing
│   └── tuner/                # Dataset creation, training utilities
├── figures/                  # Documentation images
├── EXAMPLE.md                # Custom video pipeline guide
├── FINETUNE.md               # Fine-tuning instructions
└── LICENSE.txt               # Llama 3.2 Community License
```

---

## 13. License

| Component | License |
|---|---|
| SpatialLM-Llama variants | Llama 3.2 Community License |
| SpatialLM1.0-Qwen-0.5B | Apache 2.0 |
| SpatialLM1.1-Qwen-0.5B | CC-BY-NC-4.0 |
| SceneScript encoder (v1.0) | CC-BY-NC-4.0 |
| Sonata encoder weights (v1.1) | CC-BY-NC-4.0 |
| Sonata code (Pointcept) | Apache 2.0 |
| SpatialLM-Dataset | CC-BY-NC-4.0 |
| Paper | CC BY 4.0 |

---

## 14. Citation

```bibtex
@inproceedings{SpatialLM,
  title     = {SpatialLM: Training Large Language Models for Structured Indoor Modeling},
  author    = {Mao, Yongsen and Zhong, Junhao and Fang, Chuan and Zheng, Jia
               and Tang, Rui and Zhu, Hao and Tan, Ping and Zhou, Zihan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

---

*Generated: 2026-04-01*
