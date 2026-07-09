# Wi-CBR LODO Validation And Domain 8 Focus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace random source-val selection with leave-one-source-domain-out validation and add a domain 8 focused experiment launcher.

**Architecture:** Extend the split helpers with a full-domain holdout strategy, thread it through Wi-CBR, ablation, and Wi-CBR-CARAT training scripts, and add a thin domain8 sweep runner that launches fair baseline and robust candidates under the same LODO checkpoint-selection protocol.

**Tech Stack:** Python 3.10+, PyTorch, pytest.

---

### Task 1: Leave-One-Source-Domain-Out Validation

**Files:**
- Modify: `tests/test_wicbr_training.py`
- Modify: `src/csi_carat/data/splits.py`
- Modify: `scripts/train_widar3_wicbr.py`
- Modify: `scripts/train_widar3_wicbr_ablation.py`
- Modify: `scripts/train_widar3_wicbr_carat.py`

- [x] **Step 1: Write failing split and CLI plumbing tests**
- [x] **Step 2: Run targeted tests and observe RED**
- [x] **Step 3: Implement `leave_one_domain_source_val_indices`, `--source-val-strategy`, and `--source-val-domain`**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 2: Domain 8 Focused Sweep Runner

**Files:**
- Modify: `tests/test_cli_config.py`
- Modify: `tests/test_wicbr_training.py`
- Create: `scripts/sweep_widar3_domain8_focus.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`

- [x] **Step 1: Write failing domain8 sweep spec/import tests**
- [x] **Step 2: Run targeted tests and observe RED**
- [x] **Step 3: Implement sweep specs for full Wi-CBR, phase/no-fusion candidates, and Wi-CBR-CARAT**
- [x] **Step 4: Document server commands for fair baseline, LODO validation, and domain8-focused sweep**
- [x] **Step 5: Run targeted tests and observe GREEN**

### Task 3: Verification And Delivery

**Files:**
- All modified files above.

- [x] **Step 1: Run `python -m pytest -q`**
- [x] **Step 2: Run `python -m py_compile` on changed modules/scripts**
- [x] **Step 3: Run `git diff --check`**
- [x] **Step 4: Commit and push branch**
