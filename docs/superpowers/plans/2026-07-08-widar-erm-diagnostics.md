# Widar ERM Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Widar3.0 ERM baseline outputs so each run records best epochs and per-domain/per-class diagnostics.

**Architecture:** Keep metric decomposition in `src/csi_carat/metrics/classification.py`, expose optional detailed evaluation from `src/csi_carat/engine/erm.py`, and keep report assembly in `scripts/train_widar3_erm_baseline.py`. Existing result files remain valid; new runs will include richer `best`, `per_domain`, and `per_class` sections.

**Tech Stack:** Python 3.10+, PyTorch, JSON, Markdown, pytest.

## Global Constraints

- Do not commit generated checkpoints or caches.
- Keep synthetic tests independent of server data.
- Preserve existing summary keys: `accuracy`, `macro_f1`, `worst_domain_accuracy`, `worst_domain_macro_f1`, `domain_std_accuracy`.
- Make JSON outputs deterministic and easy to review in GitHub.

---

### Task 1: Detailed Classification Breakdown

**Files:**
- Modify: `tests/test_losses_metrics_tta.py`
- Modify: `src/csi_carat/metrics/classification.py`
- Modify: `src/csi_carat/engine/erm.py`

**Interfaces:**
- Consumes: `y_true`, `y_pred`, `domains`, and `num_classes`.
- Produces: `classification_breakdown(...) -> dict[str, dict[str, dict[str, float | int]]]` and `evaluate_erm(..., include_breakdown=True)`.

- [x] **Step 1: Write failing tests for per-class and per-domain diagnostics**
- [x] **Step 2: Run `python -m pytest tests/test_losses_metrics_tta.py tests/test_erm_baseline.py -q` and observe RED**
- [x] **Step 3: Implement breakdown helpers and optional evaluation output**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 2: Best Epoch Selection In Baseline Report

**Files:**
- Modify: `tests/test_erm_baseline.py`
- Modify: `scripts/train_widar3_erm_baseline.py`

**Interfaces:**
- Consumes: baseline `history`.
- Produces: `select_best_epoch(history, metric)` and `summarize_best_epochs(history)` used in JSON and Markdown.

- [x] **Step 1: Write failing tests for best epoch selection**
- [x] **Step 2: Run `python -m pytest tests/test_erm_baseline.py -q` and observe RED**
- [x] **Step 3: Implement best epoch helpers and report sections**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 3: Verification And Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-08-widar-erm-diagnostics.md`

**Interfaces:**
- Consumes: updated baseline script.
- Produces: server rerun instructions that explain the richer metrics.

- [x] **Step 1: Update README command notes**
- [x] **Step 2: Run `git diff --check`**
- [x] **Step 3: Run full `python -m pytest -q`**
- [x] **Step 4: Commit and push**
