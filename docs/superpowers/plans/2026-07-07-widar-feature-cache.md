# Widar3 Feature Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the third Widar3.0 preprocessing layer that converts cleaned complex CSI windows into model-ready amplitude, phase-difference, and Doppler/spectrogram feature caches.

**Architecture:** Keep feature extraction separate from raw and clean cache generation. The feature layer reads `clean_cache/*.pkl` containing complex windows, computes three feature branches with NumPy-only operations, copies label/domain/window metadata, and writes `feature_cache/*.pkl` for training and evaluation loaders.

**Tech Stack:** Python 3.10+, NumPy, pickle, pytest.

---

### Task 1: Feature Branch Operations

**Files:**
- Create: `tests/test_widar3_features.py`
- Create: `src/csi_carat/data/widar3_features.py`

- [ ] **Step 1: Write failing tests for branch extraction**

Create tests that verify:

- `amplitude_branch(csi)` returns non-negative float32 magnitudes with the same `[F, T]` shape.
- `phase_difference_branch(csi)` unwraps phase across subcarriers and returns same `[F, T]` shape with a zero first row.
- `doppler_spectrogram_branch(csi, n_fft=8, hop_length=4)` returns non-negative float32 `[F, n_fft//2+1, frames]` features.

- [ ] **Step 2: Run tests and observe RED**

Run:

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_widar3_features.py -q
```

Expected: import failure because `csi_carat.data.widar3_features` does not exist.

- [ ] **Step 3: Implement branch functions**

Implement the three branch functions with NumPy only.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_widar3_features.py -q
```

Expected: feature branch tests pass.

### Task 2: Feature Cache Builder And CLI

**Files:**
- Modify: `tests/test_widar3_features.py`
- Create: `scripts/extract_widar3_features.py`
- Modify: `tests/test_cli_config.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests for cache builder and CLI import**

Create tests that verify:

- `build_feature_widar_cache` writes `amplitude`, `phase_difference`, and `doppler_spectrogram` arrays.
- Metadata arrays such as `activities`, `domains`, `source_indices`, and `window_starts` are preserved.
- `scripts.extract_widar3_features` exposes a callable `main`.

- [ ] **Step 2: Run tests and observe RED**

Run:

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_widar3_features.py tests/test_cli_config.py -q
```

Expected: missing builder or missing CLI import failure.

- [ ] **Step 3: Implement builder, CLI, config, and README command**

The CLI should accept:

```bash
python scripts/extract_widar3_features.py --data-root /home/ccl/data/csi-carat --split BOTH
```

- [ ] **Step 4: Verify full suite**

Run:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass.
