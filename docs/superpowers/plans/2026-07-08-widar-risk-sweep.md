# Widar Risk Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible risk-weight sweep for the risk-aware multibranch baseline.

**Architecture:** Keep each risk setting as a normal `train_widar3_risk_multibranch.py` run with a unique `run_name`, then summarize the generated metrics into one JSON/Markdown table. The sweep script is orchestration only; model and training logic stay in existing modules.

**Tech Stack:** Python 3.10+, subprocess, JSON, Markdown, pytest.

## Global Constraints

- Do not commit generated checkpoints or caches.
- Keep tests synthetic and independent of server data.
- Default sweep should be small enough to run on the server without blocking the project.
- Summary must include average macro-F1, worst-domain macro-F1, and domain 8 metrics.

---

### Task 1: Risk Sweep Helpers

**Files:**
- Modify: `tests/test_erm_baseline.py`
- Create: `scripts/sweep_widar3_risk_multibranch.py`

**Interfaces:**
- Consumes: comma-separated float lists such as `"0.25,0.5,1.0"`.
- Produces: `parse_float_list`, `risk_run_name`, and `summarize_completed_runs`.

- [x] **Step 1: Write failing helper tests**
- [x] **Step 2: Run `python -m pytest tests/test_erm_baseline.py -q` and observe RED**
- [x] **Step 3: Implement helper functions**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 2: Sweep CLI And Docs

**Files:**
- Modify: `tests/test_cli_config.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`
- Modify: `scripts/sweep_widar3_risk_multibranch.py`

**Interfaces:**
- Consumes: Widar feature caches and risk sweep settings.
- Produces: `results/widar3_erm/risk_sweep_summary.json` and `results/widar3_erm/risk_sweep_summary.md`.

- [x] **Step 1: Write failing CLI/config test**
- [x] **Step 2: Run `python -m pytest tests/test_cli_config.py -q` and observe RED**
- [x] **Step 3: Implement CLI orchestration and docs**
- [x] **Step 4: Run full verification and commit**
