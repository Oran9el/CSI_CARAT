# Widar Learnability Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add learnability gates that distinguish source learnability failures from cross-domain generalization failures before building CSI-CARAT proper.

**Architecture:** Extend the ERM baseline report with optional source-train evaluation and add a separate tiny-subset overfit diagnostic script. Keep reusable balanced-subset selection in the script for now because it is a diagnostic tool, not core training infrastructure.

**Tech Stack:** Python 3.10+, PyTorch, JSON, Markdown, pytest.

## Global Constraints

- Do not commit generated checkpoints or caches.
- Keep synthetic tests independent of server data.
- Do not treat low target-domain ERM as CSI-CARAT failure until source learnability and tiny-overfit gates are checked.
- Keep all diagnostic outputs under `results/widar3_erm/`.

---

### Task 1: Source-Train Evaluation In ERM Baseline

**Files:**
- Modify: `tests/test_erm_baseline.py`
- Modify: `scripts/train_widar3_erm_baseline.py`

**Interfaces:**
- Consumes: baseline `payload` with optional `train_eval` metrics.
- Produces: Markdown section `## Source Train Evaluation` and JSON `history[*].train_eval`.

- [x] **Step 1: Write failing Markdown test**
- [x] **Step 2: Run `python -m pytest tests/test_erm_baseline.py -q` and observe RED**
- [x] **Step 3: Implement train evaluation and report section**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 2: Tiny-Subset Overfit Diagnostic

**Files:**
- Create: `scripts/overfit_widar3_erm_subset.py`
- Modify: `tests/test_erm_baseline.py`
- Modify: `tests/test_cli_config.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Widar feature train cache.
- Produces: `results/widar3_erm/overfit_subset_metrics.json` and `results/widar3_erm/overfit_subset_metrics.md`.

- [x] **Step 1: Write failing tests for balanced subset selection and CLI import**
- [x] **Step 2: Run `python -m pytest tests/test_erm_baseline.py tests/test_cli_config.py -q` and observe RED**
- [x] **Step 3: Implement overfit diagnostic script and docs**
- [x] **Step 4: Run full verification and commit**
