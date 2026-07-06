# Widar3 Clean Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the second Widar3.0 preprocessing layer that converts DATTA-compatible raw complex CSI caches into cleaned, resampled, windowed caches.

**Architecture:** Keep raw cache generation separate from clean cache generation. The clean layer reads pickle caches containing `csiComplex` and metadata, infers or reads valid packet lengths, applies Hampel outlier filtering, linear packet resampling, sliding window extraction, and per-window instance normalization, then writes a new pickle with `csi`, labels, domains, window metadata, and preprocessing config.

**Tech Stack:** Python 3.10+, NumPy, pickle, pytest.

---

### Task 1: Clean Signal Operations

**Files:**
- Create: `tests/test_widar3_clean.py`
- Create: `src/csi_carat/data/widar3_clean.py`

- [ ] **Step 1: Write failing tests for signal operations**

Cover:

- `infer_valid_length` ignores zero-padded trailing packets.
- `hampel_filter_complex` suppresses an amplitude outlier while preserving shape.
- `resample_packets` maps a short sequence to a fixed packet length.
- `instance_normalize_window` produces near-zero mean and unit variance amplitude.

- [ ] **Step 2: Run test and observe RED**

Run:

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_widar3_clean.py -q
```

Expected: import failure because `csi_carat.data.widar3_clean` does not exist.

- [ ] **Step 3: Implement signal operations**

Implement the above functions with NumPy only.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_widar3_clean.py -q
```

Expected: tests pass.

### Task 2: Clean Cache Builder And CLI

**Files:**
- Modify: `tests/test_widar3_clean.py`
- Create: `scripts/clean_widar3_g6d.py`
- Modify: `tests/test_cli_config.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests for cache builder and CLI import**

Cover:

- `build_clean_widar_cache` expands samples into multiple windows and repeats metadata.
- CLI module `scripts.clean_widar3_g6d` is importable.

- [ ] **Step 2: Run tests and observe RED**

Run:

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_widar3_clean.py tests/test_cli_config.py -q
```

Expected: missing builder or missing CLI import failure.

- [ ] **Step 3: Implement builder, CLI, config, and README command**

The CLI should accept:

```bash
python scripts/clean_widar3_g6d.py --data-root /home/ccl/data/csi-carat --split BOTH
```

- [ ] **Step 4: Verify full suite**

Run:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass.
