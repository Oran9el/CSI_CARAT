# Widar Transformer Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stronger three-branch Transformer baseline for Widar3.0 before implementing the full CSI-CARAT disentanglement/gate/TTA stack.

**Architecture:** Add reusable temporal Transformer branch encoders to `src/csi_carat/models/baselines.py`, keep existing ERM train/eval helpers, and expose a server runner through `scripts/train_widar3_transformer_multibranch.py`. This baseline uses amplitude, phase-difference, and Doppler/spectrogram branches, but remains plain supervised ERM.

**Tech Stack:** Python 3.10+, PyTorch, pytest.

## Global Constraints

- Do not commit generated checkpoints or caches.
- Preserve existing amplitude, multibranch CNN, and risk-aware baseline behavior.
- Keep synthetic tests independent of server data.
- Output metrics under `results/widar3_erm/` with run name `transformer_multibranch`.

---

### Task 1: Transformer Multibranch Model

**Files:**
- Modify: `tests/test_erm_baseline.py`
- Modify: `src/csi_carat/models/baselines.py`

**Interfaces:**
- Consumes: tensors `amplitude: [B, S, T]`, `phase_difference: [B, S, T]`, and `doppler_spectrogram: [B, S, F, W]`.
- Produces: `MultiBranchTransformerClassifier(...) -> logits [B, num_classes]`.

- [x] **Step 1: Write failing forward-shape test**
- [x] **Step 2: Run `python -m pytest tests/test_erm_baseline.py -q` and observe RED**
- [x] **Step 3: Implement Transformer branch encoders and classifier**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 2: Transformer Server Runner

**Files:**
- Create: `scripts/train_widar3_transformer_multibranch.py`
- Modify: `tests/test_cli_config.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: Widar feature train/test caches.
- Produces: `results/widar3_erm/transformer_multibranch_metrics.json`, `results/widar3_erm/transformer_multibranch_metrics.md`, and an ignored checkpoint.

- [x] **Step 1: Write failing CLI/config test**
- [x] **Step 2: Run `python -m pytest tests/test_cli_config.py -q` and observe RED**
- [x] **Step 3: Implement runner, config, and docs**
- [x] **Step 4: Run full verification and commit**
