# Widar Feature Dataset And ERM Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sanity report, PyTorch Dataset, and amplitude-only ERM smoke path for Widar3.0 feature caches.

**Architecture:** Keep cache validation in `data/feature_report.py`, dataset loading in `data/widar3_dataset.py`, and baseline training utilities in `models/baselines.py` plus `engine/erm.py`. Scripts should be import-safe and only run work from `main()`.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, pickle, pytest.

## Global Constraints

- Do not commit generated `.pkl` caches.
- Keep synthetic tests independent of real server data.
- Preserve the feature cache keys `amplitude`, `phase_difference`, `doppler_spectrogram`, `activities`, `domains`, `environments`, `users`, `source_indices`, and `window_starts`.

---

### Task 1: Feature Cache Sanity Report

**Files:**
- Create: `tests/test_feature_report.py`
- Create: `src/csi_carat/data/feature_report.py`
- Create: `scripts/report_widar3_features.py`

**Interfaces:**
- Consumes: feature cache pickle with branch arrays and metadata arrays.
- Produces: `summarize_feature_cache(path: str | Path) -> dict[str, object]`; `write_feature_report(summary, output_path) -> None`.

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run `python -m pytest tests/test_feature_report.py -q` and observe RED**
- [x] **Step 3: Implement report module and script**
- [x] **Step 4: Run feature report tests and observe GREEN**

### Task 2: Widar Feature Dataset

**Files:**
- Create: `tests/test_widar3_dataset.py`
- Create: `src/csi_carat/data/widar3_dataset.py`

**Interfaces:**
- Consumes: feature cache pickle.
- Produces: `WidarFeatureDataset(cache_path, branches=("amplitude", ...))` returning tensor dictionaries.

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run `python -m pytest tests/test_widar3_dataset.py -q` and observe RED**
- [x] **Step 3: Implement dataset module**
- [x] **Step 4: Run dataset tests and observe GREEN**

### Task 3: Amplitude-Only ERM Smoke

**Files:**
- Create: `tests/test_erm_baseline.py`
- Create: `src/csi_carat/models/baselines.py`
- Create: `src/csi_carat/engine/erm.py`
- Create: `scripts/train_widar3_erm.py`
- Modify: `tests/test_cli_config.py`
- Modify: `README.md`
- Modify: `configs/widar3_g6d.yaml`

**Interfaces:**
- Consumes: dataset batches with `amplitude` and `activity`.
- Produces: `AmplitudeCnnClassifier` and `train_one_erm_step(model, batch, optimizer)`.

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run `python -m pytest tests/test_erm_baseline.py tests/test_cli_config.py -q` and observe RED**
- [x] **Step 3: Implement baseline model, train step, script, config, and README command**
- [x] **Step 4: Run full suite and observe GREEN**
