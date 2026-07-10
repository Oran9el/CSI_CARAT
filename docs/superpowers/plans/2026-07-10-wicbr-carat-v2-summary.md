# Wi-CBR CARAT V2 And LODO Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable LODO result summary script and implement a branch-aware Wi-CBR-CARAT v2 model.

**Architecture:** Keep Wi-CBR phase/DFS image caches as the input representation. Summarize existing LODO result folders into one CSV/Markdown table, then add a v2 CARAT head that factorizes phase and DFS branches separately and learns branch-wise/channel-wise gates instead of factorizing a pre-fused embedding.

**Tech Stack:** Python 3.10+, PyTorch, pytest.

---

### Task 1: LODO Result Summary

**Files:**
- Create: `scripts/report_widar3_lodo_results.py`
- Modify: `tests/test_wicbr_training.py`
- Modify: `tests/test_cli_config.py`
- Modify: `README.md`

- [x] **Step 1: Write failing summary parser and report tests**
- [x] **Step 2: Run targeted tests and observe RED**
- [x] **Step 3: Implement JSON collection, aggregate mean/std, CSV/Markdown writers, and CLI**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 2: Wi-CBR-CARAT V2

**Files:**
- Modify: `src/csi_carat/models/wicbr.py`
- Modify: `scripts/train_widar3_wicbr_carat.py`
- Modify: `tests/test_wicbr_model.py`
- Modify: `tests/test_wicbr_training.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`

- [x] **Step 1: Write failing v2 forward and training tests**
- [x] **Step 2: Run targeted tests and observe RED**
- [x] **Step 3: Implement branch encoders, per-branch factor heads, branch gates, and `--carat-version v1|v2`**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 3: Verification And Delivery

**Files:**
- All modified files above.

- [x] **Step 1: Run `python -m pytest -q`**
- [x] **Step 2: Run `python -m py_compile` on changed modules/scripts**
- [x] **Step 3: Run `git diff --check`**
- [x] **Step 4: Commit and push branch**
