# Widar Risk-Aware Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a risk-aware three-branch Transformer baseline for Widar3.0 so the current strongest temporal encoder is evaluated with the smooth worst-source-domain risk objective.

**Architecture:** Reuse `MultiBranchTransformerClassifier` from `src/csi_carat/models/baselines.py` and `run_risk_aware_epoch` from `src/csi_carat/engine/erm.py`. Add a server runner that consumes the existing Widar feature caches and writes metrics under `results/widar3_erm/` with run name `risk_transformer_multibranch`.

**Tech Stack:** Python 3.10+, PyTorch, pytest.

## Global Constraints

- Do not commit generated checkpoints or caches.
- Preserve existing CNN risk-aware and Transformer ERM behavior.
- Keep tests synthetic and import/config focused so they do not require server data.
- Use `risk_weight=0.25` and `risk_eta=2.0` as the first run because the previous risk sweep made 0.25 the best macro-F1 starting point.

---

### Task 1: Risk-Aware Transformer Runner

**Files:**
- Modify: `tests/test_cli_config.py`
- Create: `scripts/train_widar3_risk_transformer_multibranch.py`
- Modify: `configs/widar3_g6d.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: Widar feature train/test caches with `amplitude`, `phase_difference`, and `doppler_spectrogram`.
- Produces: `results/widar3_erm/risk_transformer_multibranch_metrics.json`, `results/widar3_erm/risk_transformer_multibranch_metrics.md`, and an ignored checkpoint.

- [x] **Step 1: Write failing CLI/config test**
- [x] **Step 2: Run `python -m pytest tests/test_cli_config.py -q` and observe RED**
- [x] **Step 3: Implement runner, config, and docs**
- [x] **Step 4: Run targeted tests and observe GREEN**
- [x] **Step 5: Run full verification and commit**
