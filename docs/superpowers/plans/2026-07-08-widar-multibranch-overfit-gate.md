# Widar Multibranch Overfit Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the tiny-subset overfit diagnostic so it can test both amplitude-only and multibranch ERM learnability.

**Architecture:** Keep the diagnostic script self-contained, with small helpers for selecting branches, building a model from the first sample, and naming output files. The script will continue to use shared ERM train/eval helpers and emit Git-friendly Markdown/JSON results under `results/widar3_erm/`.

**Tech Stack:** Python 3.10+, PyTorch, pytest.

## Global Constraints

- Do not commit generated checkpoints or caches.
- Preserve existing amplitude-only overfit behavior.
- Use model-specific result names so amplitude and multibranch diagnostics can coexist.
- Keep synthetic tests independent of server data.

---

### Task 1: Model-Selectable Overfit Diagnostic

**Files:**
- Modify: `tests/test_erm_baseline.py`
- Modify: `scripts/overfit_widar3_erm_subset.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `--model amplitude|multibranch`.
- Produces: `model_feature_keys(model_name)`, model-specific JSON/Markdown file names, and a training path using the right feature keys.

- [x] **Step 1: Write failing tests for model feature keys and output names**
- [x] **Step 2: Run `python -m pytest tests/test_erm_baseline.py -q` and observe RED**
- [x] **Step 3: Implement model-selectable overfit logic**
- [x] **Step 4: Run targeted tests and full verification**
