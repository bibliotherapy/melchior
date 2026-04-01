# Server Execution Guide

**Purpose:** Tracks all commands and steps that must be run on the V100 server (not local MacBook) as each phase is implemented locally.

**Server Specs:** 2x NVIDIA Tesla V100-DGXS-32GB, CUDA 12.4, Driver 550.144.03

---

## Phase 0: Project Setup & Environment

### 0.2 — Install dependencies

```bash
# PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# MMPose ecosystem (multi-person pose estimation)
pip install mmcv mmpose mmdet

# Core dependencies
pip install opencv-python-headless numpy scipy
pip install pyyaml tqdm matplotlib seaborn

# Feature importance analysis
pip install shap
```

### Verify installation

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# Expected: True 2

python -c "from mmpose.apis import MMPoseInferencer; print('MMPose OK')"
```

### Update config paths

Edit `configs/default.yaml` — change `data_root` from `./data` to the server's absolute path:

```yaml
data_root: /path/to/CP_dataset   # <-- set to actual server path
```

---

*Subsequent phases will be appended below as they are implemented.*
