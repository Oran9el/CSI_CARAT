# Wi-CBR Ablation And CARAT Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Wi-CBR ablation runners, source-val checkpoint selection, and a first Wi-CBR-backed CSI-CARAT training/TTA path.

**Architecture:** Keep the Wi-CBR feature cache as the strong input representation. Extend Wi-CBR models with branch/fusion switches for ablations, add deterministic source-val splits for checkpoint selection, and wrap Wi-CBR embeddings with CSI-CARAT factor heads, risk gates, domain adversarial loss, risk-aware source objective, and conservative TTA parameter collection.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, pytest.

---

### Task 1: Source-Val Selection And Ablation Runners

**Files:**
- Modify: `tests/test_wicbr_model.py`
- Create: `tests/test_wicbr_training.py`
- Modify: `src/csi_carat/models/wicbr.py`
- Create: `src/csi_carat/data/splits.py`
- Modify: `scripts/train_widar3_wicbr.py`
- Create: `scripts/train_widar3_wicbr_ablation.py`
- Modify: `tests/test_cli_config.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`

**Interfaces:**
- `WiCbrCnnClassifier(..., branch_mode="phase"|"dfs"|"both", use_fusion=False)` returns logits and embeddings.
- `stratified_source_val_indices(labels, domains, val_fraction, seed)` returns deterministic train/val indices.
- `scripts/train_widar3_wicbr.py --source-val-fraction 0.1 --selection-metric macro_f1` selects checkpoints from source-val metrics, not target-test metrics.
- `scripts/train_widar3_wicbr_ablation.py --runs phase_only,dfs_only,no_fusion,no_contrastive` launches repeated Wi-CBR runs.

- [x] **Step 1: Write failing branch/fusion tests**
- [x] **Step 2: Write failing source-val and ablation-spec tests**
- [x] **Step 3: Run targeted tests and observe RED**
- [x] **Step 4: Implement model switches, deterministic source-val split, checkpoint selection, ablation runner, config, and docs**
- [x] **Step 5: Run targeted tests and observe GREEN**

### Task 2: Wi-CBR-Backed CSI-CARAT Model

**Files:**
- Modify: `tests/test_wicbr_model.py`
- Create: `tests/test_wicbr_carat.py`
- Modify: `src/csi_carat/models/wicbr.py`
- Create: `src/csi_carat/engine/wicbr_carat.py`
- Create: `scripts/train_widar3_wicbr_carat.py`
- Modify: `tests/test_cli_config.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`

**Interfaces:**
- `WiCbrCaratClassifier(...)` consumes `wicbr_phase_image` and `wicbr_dfs_image`, returns logits by default and full CARAT outputs with `return_outputs=True`.
- `train_one_wicbr_carat_step(...)` returns CE, risk, domain, disentangle, and optional contrastive loss components.
- `adapt_wicbr_carat_tta_step(...)` updates only TTA-eligible parameters using target entropy minimization.
- `scripts/train_widar3_wicbr_carat.py` trains the first Wi-CBR-CARAT variant with source-val checkpoint selection.

- [x] **Step 1: Write failing Wi-CBR-CARAT forward/loss/TTA tests**
- [x] **Step 2: Run targeted tests and observe RED**
- [x] **Step 3: Implement CARAT wrapper, training helper, TTA helper, runner, config, and docs**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 3: Verification And Delivery

**Files:**
- All modified files above.

- [x] **Step 1: Run `python -m pytest -q`**
- [x] **Step 2: Run `python -m py_compile` on new scripts/modules**
- [x] **Step 3: Run `git diff --check`**
- [x] **Step 4: Commit and push branch**
