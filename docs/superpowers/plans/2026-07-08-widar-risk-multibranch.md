# Widar Risk-Aware Multibranch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a risk-aware multibranch ERM baseline that optimizes average source CE plus smooth worst-source-domain risk.

**Architecture:** Put reusable domain-risk train helpers in `src/csi_carat/engine/erm.py`, reuse the existing `MultiBranchCnnClassifier`, and expose a server runner through `scripts/train_widar3_risk_multibranch.py`. This is a minimal CSI-CARAT risk component, not the full causal/TTA model.

**Tech Stack:** Python 3.10+, PyTorch, pytest.

## Global Constraints

- Do not commit generated checkpoints or caches.
- Preserve existing ERM behavior.
- Risk loss uses source-domain labels from batches and `logsumexp_risk`.
- Output metrics under `results/widar3_erm/` with run name `risk_multibranch`.

---

### Task 1: Risk-Aware ERM Helpers

**Files:**
- Modify: `tests/test_erm_baseline.py`
- Modify: `src/csi_carat/engine/erm.py`

**Interfaces:**
- Consumes: batches with `activity`, `domain`, and multibranch feature keys.
- Produces: `domain_ce_losses(...)`, `train_one_risk_aware_step(...)`, and `run_risk_aware_epoch(...)`.

- [x] **Step 1: Write failing tests for domain losses and one risk-aware step**
- [x] **Step 2: Run `python -m pytest tests/test_erm_baseline.py -q` and observe RED**
- [x] **Step 3: Implement risk-aware helpers**
- [x] **Step 4: Run targeted tests and observe GREEN**

### Task 2: Risk-Aware Server Runner

**Files:**
- Create: `scripts/train_widar3_risk_multibranch.py`
- Modify: `tests/test_cli_config.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: Widar feature train/test caches.
- Produces: `results/widar3_erm/risk_multibranch_metrics.json`, `results/widar3_erm/risk_multibranch_metrics.md`, and an ignored checkpoint.

- [x] **Step 1: Write failing CLI/config test**
- [x] **Step 2: Run `python -m pytest tests/test_cli_config.py -q` and observe RED**
- [x] **Step 3: Implement runner, config, and docs**
- [x] **Step 4: Run full verification and commit**
