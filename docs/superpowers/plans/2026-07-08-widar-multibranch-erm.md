# Widar Multibranch ERM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-branch Widar3.0 ERM baseline using amplitude, phase-difference, and Doppler/spectrogram features before building CSI-CARAT proper.

**Architecture:** Keep branch fusion in `src/csi_carat/models/baselines.py`, make ERM helpers accept configurable feature keys, and expose a server runner through `scripts/train_widar3_multibranch_erm.py`. This remains a baseline/diagnostic, not the CSI-CARAT model.

**Tech Stack:** Python 3.10+, PyTorch, JSON, Markdown, pytest.

## Global Constraints

- Do not commit generated checkpoints or caches.
- Keep tests synthetic and independent of server data.
- Preserve amplitude-only behavior as the default path for existing scripts.
- Use the existing feature cache keys: `amplitude`, `phase_difference`, and `doppler_spectrogram`.

---

### Task 1: Multibranch Baseline Model

**Files:**
- Modify: `tests/test_erm_baseline.py`
- Modify: `src/csi_carat/models/baselines.py`

**Interfaces:**
- Consumes: tensors `amplitude: [B, S, T]`, `phase_difference: [B, S, T]`, and `doppler_spectrogram: [B, S, F, W]`.
- Produces: `MultiBranchCnnClassifier(num_subcarriers, window_size, doppler_bins, doppler_frames, num_classes)`.

- [x] **Step 1: Write failing forward-shape test**
- [x] **Step 2: Run `python -m pytest tests/test_erm_baseline.py -q` and observe RED**
- [x] **Step 3: Implement multibranch model**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 2: Feature-Key Aware ERM Helpers

**Files:**
- Modify: `tests/test_erm_baseline.py`
- Modify: `src/csi_carat/engine/erm.py`

**Interfaces:**
- Consumes: `feature_keys=("amplitude", "phase_difference", "doppler_spectrogram")`.
- Produces: `train_one_erm_step`, `run_erm_epoch`, and `evaluate_erm` that can call either single-tensor or keyword-branch models.

- [x] **Step 1: Write failing training-step test with multibranch inputs**
- [x] **Step 2: Run `python -m pytest tests/test_erm_baseline.py -q` and observe RED**
- [x] **Step 3: Implement feature-key aware forwarding**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 3: Server Runner And Docs

**Files:**
- Create: `scripts/train_widar3_multibranch_erm.py`
- Modify: `tests/test_cli_config.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: train/test feature cache paths inferred from `/home/ccl/data/csi-carat`.
- Produces: `results/widar3_erm/multibranch_metrics.json`, `results/widar3_erm/multibranch_metrics.md`, and an ignored checkpoint.

- [x] **Step 1: Write failing CLI import/config test**
- [x] **Step 2: Run `python -m pytest tests/test_cli_config.py -q` and observe RED**
- [x] **Step 3: Implement runner, config, and README command**
- [x] **Step 4: Run full verification and commit**
