# Widar Wi-CBR Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class Wi-CBR reproduction baseline for Widar3.0-G6D before continuing CSI-CARAT model design.

**Architecture:** Build Wi-CBR-style tensors directly from grouped raw `.dat` files, keeping all three antennas and six receivers per sample. The cache stores CSI-ratio phase images and Doppler velocity spectrum images, then a two-branch saliency-fusion classifier trains with CE plus proxy contrastive loss.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, optional `csiread` for raw Widar files, optional `torchvision` for the ResNet18 reproduction backbone.

## Global Constraints

- Do not commit raw data, caches, checkpoints, or server-only artifacts.
- Keep synthetic tests independent of server data and optional dependencies.
- Reuse `WidarFeatureDataset` and ERM evaluation where the schema permits it.
- Preserve existing cleaned-cache baselines unchanged.
- Keep Wi-CBR outputs under `wicbr_cache/` and `results/widar3_wicbr/`.

---

### Task 1: Wi-CBR Feature Cache

**Files:**
- Create: `tests/test_wicbr_features.py`
- Create: `src/csi_carat/data/wicbr_features.py`
- Create: `scripts/extract_widar3_wicbr_features.py`

**Interfaces:**
- Consumes: raw Widar `.dat` groups with receiver IDs `r1` through `r6`.
- Produces: cache keys `wicbr_phase_image`, `wicbr_dfs_image`, plus activity/environment/user/domain metadata compatible with `WidarFeatureDataset`.

- [x] **Step 1: Write failing CSI-ratio phase, DFS, resize, and cache-grouping tests**
- [x] **Step 2: Run targeted tests and observe RED**
- [x] **Step 3: Implement raw antenna reader, six-receiver grouping, phase image, DFS image, and extraction CLI**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 2: Wi-CBR Model And Loss

**Files:**
- Create: `tests/test_wicbr_model.py`
- Create: `src/csi_carat/models/wicbr.py`
- Create: `src/csi_carat/engine/wicbr.py`

**Interfaces:**
- Consumes: tensors `wicbr_phase_image: [B, 3, H, W]` and `wicbr_dfs_image: [B, 3, H, W]`.
- Produces: logits `[B, 6]` and embeddings for proxy contrastive loss.

- [x] **Step 1: Write failing spatial-gate, fusion, proxy-loss, and classifier tests**
- [x] **Step 2: Run targeted tests and observe RED**
- [x] **Step 3: Implement Wi-CBR saliency fusion, small CNN smoke backbone, optional ResNet18 backbone, and training step**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 3: Server Runner And Documentation

**Files:**
- Create: `scripts/train_widar3_wicbr.py`
- Modify: `tests/test_cli_config.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: Wi-CBR train/test feature caches.
- Produces: `results/widar3_wicbr/wicbr_metrics.json`, `results/widar3_wicbr/wicbr_metrics.md`, and an ignored checkpoint.

- [x] **Step 1: Write failing CLI/config import test**
- [x] **Step 2: Run targeted tests and observe RED**
- [x] **Step 3: Implement runner, config, docs, and optional dependency metadata**
- [x] **Step 4: Run full verification, commit, and push**
