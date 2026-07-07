# Widar ERM Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible amplitude-only source ERM baseline that trains on Widar3.0 train features, evaluates on target test features, and writes metrics for GitHub review.

**Architecture:** Keep reusable training and evaluation loops in `src/csi_carat/engine/erm.py`, keep Widar feature loading in `src/csi_carat/data/widar3_dataset.py`, and expose the full experiment through `scripts/train_widar3_erm_baseline.py`. The existing `scripts/train_widar3_erm.py` remains a short smoke test.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, JSON, pytest.

## Global Constraints

- Do not commit generated `.pkl`, `.pt`, `.pth`, `.npy`, or `.npz` files.
- Keep tests synthetic and independent of server data.
- Preserve raw metadata labels while allowing mapped domain labels for future domain objectives.
- Write experiment metrics under `results/` so server outputs can be pushed and reviewed locally.

---

### Task 1: Domain Mapping In Dataset

**Files:**
- Modify: `tests/test_widar3_dataset.py`
- Modify: `src/csi_carat/data/widar3_dataset.py`

**Interfaces:**
- Consumes: `WidarFeatureDataset(cache_path, branches=..., domain_map=None)`.
- Produces: `domain` as mapped label when `domain_map` is provided, and `domain_raw` as the original cache label.

- [x] **Step 1: Write failing test**
- [x] **Step 2: Run `python -m pytest tests/test_widar3_dataset.py -q` and observe RED**
- [x] **Step 3: Implement optional `domain_map` and `domain_raw`**
- [x] **Step 4: Run dataset tests and observe GREEN**

### Task 2: ERM Train/Eval Helpers

**Files:**
- Modify: `tests/test_erm_baseline.py`
- Modify: `src/csi_carat/engine/erm.py`

**Interfaces:**
- Consumes: DataLoader batches with `amplitude`, `activity`, and `domain`.
- Produces: `run_erm_epoch(...) -> dict[str, float | int]` and `evaluate_erm(...) -> dict[str, float]`.

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run `python -m pytest tests/test_erm_baseline.py -q` and observe RED**
- [x] **Step 3: Implement reusable train/eval helpers**
- [x] **Step 4: Run ERM tests and observe GREEN**

### Task 3: Full Baseline Script And Docs

**Files:**
- Create: `scripts/train_widar3_erm_baseline.py`
- Modify: `tests/test_cli_config.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: train/test feature cache paths inferred from `/home/ccl/data/csi-carat`.
- Produces: `results/widar3_erm/amplitude_only_metrics.json`, `results/widar3_erm/amplitude_only_metrics.md`, and an ignored checkpoint file.

- [x] **Step 1: Write failing CLI import test**
- [x] **Step 2: Run `python -m pytest tests/test_cli_config.py -q` and observe RED**
- [x] **Step 3: Implement baseline script, config, and README command**
- [x] **Step 4: Run full suite and observe GREEN**
