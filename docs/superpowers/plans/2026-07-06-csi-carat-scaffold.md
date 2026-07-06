# CSI-CARAT Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the first runnable CSI-CARAT Python package scaffold with synthetic tests for data schemas, model forward passes, losses, metrics, and lightweight TTA parameter selection.

**Architecture:** The scaffold uses a `src/csi_carat` package with small modules for data, augmentations, models, losses, TTA, metrics, and training/evaluation entry points. The first model combines a DATTA-style CSI Transformer backbone with an MDTA-style factor head and a CARAT-style adapter/gate fusion path. Real dataset preprocessing is represented by path-aware utilities and sample schemas first; server-side Widar3.0 preprocessing will be added after the local package skeleton is testable.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, pytest, optional scikit-learn for future experiments.

---

### Task 1: Project Package Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.gitignore`
- Create: `src/csi_carat/__init__.py`
- Create: `src/csi_carat/data/__init__.py`
- Create: `src/csi_carat/augmentations/__init__.py`
- Create: `src/csi_carat/models/__init__.py`
- Create: `src/csi_carat/models/backbones/__init__.py`
- Create: `src/csi_carat/losses/__init__.py`
- Create: `src/csi_carat/tta/__init__.py`
- Create: `src/csi_carat/engine/__init__.py`
- Create: `src/csi_carat/metrics/__init__.py`

- [x] **Step 1: Create package metadata**

Write `pyproject.toml` with editable-install friendly metadata:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "csi-carat"
version = "0.1.0"
description = "CSI-CARAT research scaffold for cross-domain WiFi CSI sensing"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "torch>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [x] **Step 2: Create README and package init files**

Write `README.md` with the project goal, server data root `/home/ccl/data/csi-carat`, and first milestone commands:

```markdown
# CSI-CARAT

CSI-CARAT is a PyTorch research scaffold for causal adaptive risk-aware augmentation and test-time calibration for cross-domain WiFi CSI sensing.

Default server data root:

```text
/home/ccl/data/csi-carat
```

Run local synthetic checks:

```bash
python -m pytest
```
```

Each package `__init__.py` should be valid Python and may expose only `__all__ = []` until concrete APIs exist.

- [x] **Step 3: Verify package discovery**

Run:

```bash
python -m pytest
```

Expected: pytest starts; if there are no tests yet, it may report no tests collected.

### Task 2: Data Schema And Widar3 Path Planning

**Files:**
- Create: `tests/test_data_schema.py`
- Create: `src/csi_carat/data/sample.py`
- Create: `src/csi_carat/data/paths.py`

- [x] **Step 1: Write failing tests for data schema and paths**

Create `tests/test_data_schema.py`:

```python
from pathlib import Path

import torch

from csi_carat.data.sample import CsiSample, collate_csi_samples
from csi_carat.data.paths import WidarG6DPaths


def test_csi_sample_collate_preserves_domain_metadata():
    samples = [
        CsiSample(
            x=torch.ones(1, 30, 220),
            activity=0,
            domain=9,
            environment=2,
            user=1,
        ),
        CsiSample(
            x=torch.zeros(1, 30, 220),
            activity=1,
            domain=10,
            environment=2,
            user=2,
        ),
    ]

    batch = collate_csi_samples(samples)

    assert batch["x"].shape == (2, 1, 30, 220)
    assert batch["activity"].tolist() == [0, 1]
    assert batch["domain"].tolist() == [9, 10]
    assert batch["environment"].tolist() == [2, 2]
    assert batch["user"].tolist() == [1, 2]


def test_widar_paths_use_server_raw_and_cache_roots():
    paths = WidarG6DPaths.from_data_root(Path("/home/ccl/data/csi-carat"))

    assert paths.raw_root == Path("/home/ccl/data/csi-carat/widar3/widar3g6d/raw")
    assert paths.cache_root == Path("/home/ccl/data/csi-carat/widar3/widar3g6d/cache")
    assert paths.train_cache.name == "widar3-g6_csi_domain_train_cache.pkl"
    assert paths.test_cache.name == "widar3-g6_csi_domain_test_cache.pkl"
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest tests/test_data_schema.py -q
```

Expected: import failure because `csi_carat.data.sample` and `csi_carat.data.paths` do not exist yet.

- [x] **Step 3: Implement sample and path modules**

Create `src/csi_carat/data/sample.py` with a dataclass `CsiSample` and function `collate_csi_samples(samples) -> dict[str, torch.Tensor]`.

Create `src/csi_carat/data/paths.py` with frozen dataclass `WidarG6DPaths` containing `raw_root`, `cache_root`, `train_cache`, and `test_cache`, plus `from_data_root`.

- [x] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest tests/test_data_schema.py -q
```

Expected: both tests pass.

### Task 3: Model Forward Skeleton

**Files:**
- Create: `tests/test_models.py`
- Create: `src/csi_carat/models/backbones/wiflexformer.py`
- Create: `src/csi_carat/models/disentanglement.py`
- Create: `src/csi_carat/models/adapters.py`
- Create: `src/csi_carat/models/gate.py`
- Create: `src/csi_carat/models/carat.py`

- [x] **Step 1: Write failing model tests**

Create `tests/test_models.py`:

```python
import torch

from csi_carat.models.carat import CsiCaratModel


def test_csi_carat_model_forward_returns_expected_keys_and_shapes():
    model = CsiCaratModel(
        input_subcarriers=30,
        window_size=64,
        feature_dim=32,
        factor_dim=16,
        num_classes=6,
        num_domains=7,
    )
    x = torch.randn(4, 1, 30, 64)

    out = model(x)

    assert out["logits"].shape == (4, 6)
    assert out["domain_logits"].shape == (4, 7)
    assert out["gate"].shape == (4, 1)
    assert out["factors"]["action"].shape == (4, 16)
    assert out["factors"]["environment"].shape == (4, 16)
    assert out["factors"]["position"].shape == (4, 16)
    assert out["factors"]["orientation"].shape == (4, 16)
    assert out["factors"]["user"].shape == (4, 16)
    assert out["factors"]["residual"].shape == (4, 16)
    assert torch.all(out["gate"] >= 0)
    assert torch.all(out["gate"] <= 1)
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest tests/test_models.py -q
```

Expected: import failure because model modules do not exist yet.

- [x] **Step 3: Implement minimal model modules**

Implement:

- `WiFlexFormerBackbone`: accepts `[B, C, F, T]`, uses Conv1d stem plus Transformer encoder, returns `[B, feature_dim]`.
- `GradientReversal`: autograd function for DATTA-style adversarial training.
- `FactorHead`: maps encoder feature into six named factors.
- `Adapter`: maps concatenated spurious factors to `factor_dim`.
- `RiskGate`: sigmoid MLP returning `[B, 1]`.
- `CsiCaratModel`: combines backbone, factor head, adapter, gate, classifier, and domain discriminator.

- [x] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest tests/test_models.py -q
```

Expected: model forward test passes.

### Task 4: Losses, Metrics, And TTA Utilities

**Files:**
- Create: `tests/test_losses_metrics_tta.py`
- Create: `src/csi_carat/losses/adversarial.py`
- Create: `src/csi_carat/losses/disentangle.py`
- Create: `src/csi_carat/losses/risk.py`
- Create: `src/csi_carat/losses/consistency.py`
- Create: `src/csi_carat/metrics/classification.py`
- Create: `src/csi_carat/tta/calibrator.py`
- Create: `src/csi_carat/tta/prototype_memory.py`

- [x] **Step 1: Write failing tests**

Create `tests/test_losses_metrics_tta.py`:

```python
import torch

from csi_carat.losses.consistency import symmetric_kl
from csi_carat.losses.disentangle import covariance_penalty
from csi_carat.losses.risk import logsumexp_risk
from csi_carat.metrics.classification import classification_summary
from csi_carat.models.carat import CsiCaratModel
from csi_carat.tta.calibrator import collect_trainable_tta_parameters
from csi_carat.tta.prototype_memory import PrototypeMemory


def test_losses_return_finite_scalars():
    z1 = torch.randn(8, 16)
    z2 = torch.randn(8, 16)
    logits_a = torch.randn(8, 6)
    logits_b = torch.randn(8, 6)
    domain_losses = torch.tensor([0.8, 1.0, 0.4])

    values = [
        covariance_penalty(z1, z2),
        symmetric_kl(logits_a, logits_b),
        logsumexp_risk(domain_losses, eta=2.0),
    ]

    for value in values:
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_classification_summary_reports_worst_domain_macro_f1():
    y_true = torch.tensor([0, 1, 0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 1, 1, 0, 0])
    domains = torch.tensor([0, 0, 1, 1, 2, 2])

    summary = classification_summary(y_true, y_pred, domains, num_classes=2)

    assert set(summary) == {
        "accuracy",
        "macro_f1",
        "worst_domain_accuracy",
        "worst_domain_macro_f1",
        "domain_std_accuracy",
    }
    assert 0.0 <= summary["worst_domain_macro_f1"] <= 1.0


def test_tta_parameter_selection_is_lightweight():
    model = CsiCaratModel(
        input_subcarriers=30,
        window_size=64,
        feature_dim=32,
        factor_dim=16,
        num_classes=6,
        num_domains=7,
    )

    params = collect_trainable_tta_parameters(model)
    names = {name for name, _ in params}

    assert any(name.startswith("adapter.") for name in names)
    assert any(name.startswith("gate.") for name in names)
    assert all(not name.startswith("backbone.stem") for name in names)


def test_prototype_memory_updates_high_confidence_samples():
    memory = PrototypeMemory(num_classes=3, feature_dim=4, momentum=0.5)
    features = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    labels = torch.tensor([0, 1])
    confidence = torch.tensor([0.9, 0.2])

    memory.update(features, labels, confidence, threshold=0.5)

    assert memory.counts.tolist() == [1, 0, 0]
    assert torch.allclose(memory.prototypes[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest tests/test_losses_metrics_tta.py -q
```

Expected: import failure because loss, metric, and TTA modules do not exist yet.

- [x] **Step 3: Implement minimal loss, metric, and TTA modules**

Implement finite scalar losses, torch-only classification metrics, TTA parameter selection for LayerNorm/adapter/gate/temperature if present, and a momentum prototype memory.

- [x] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest tests/test_losses_metrics_tta.py -q
```

Expected: all tests pass.

### Task 5: Config And CLI Smoke Entrypoints

**Files:**
- Create: `configs/widar3_g6d.yaml`
- Create: `scripts/train.py`
- Create: `scripts/evaluate.py`
- Create: `tests/test_cli_config.py`

- [x] **Step 1: Write failing tests for config and CLI importability**

Create `tests/test_cli_config.py`:

```python
from pathlib import Path


def test_widar_config_exists_and_points_to_server_root():
    config = Path("configs/widar3_g6d.yaml")

    text = config.read_text(encoding="utf-8")

    assert "/home/ccl/data/csi-carat" in text
    assert "widar3g6d" in text


def test_scripts_are_importable():
    import scripts.train as train_script
    import scripts.evaluate as evaluate_script

    assert callable(train_script.main)
    assert callable(evaluate_script.main)
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest tests/test_cli_config.py -q
```

Expected: failure because config and scripts do not exist yet.

- [x] **Step 3: Implement config and scripts**

Create `configs/widar3_g6d.yaml` with the server data root, raw/cache paths, model shape defaults, and training defaults.

Create importable script modules with `main(argv: list[str] | None = None) -> int` and no expensive work at import time.

- [x] **Step 4: Verify GREEN and full suite**

Run:

```bash
python -m pytest -q
```

Expected: all synthetic scaffold tests pass.

## Self-Review

Spec coverage:

- DATTA-style package baseline is covered by model backbone, GRL, TTA parameter selection, and Widar path planning.
- MDTA-style multi-factor disentanglement is covered by `FactorHead`, named factors, reconstruction-ready factor outputs, and factor separation loss.
- CSI-CARAT gated fusion is covered by adapter, gate, and final model forward path.
- Dataset server layout is covered by `WidarG6DPaths` and `configs/widar3_g6d.yaml`.
- Initial tests are synthetic and do not require real data.

Placeholder scan:

- No placeholders are required for milestone 1. Phase-two items are explicitly scoped out of the first scaffold.

Type consistency:

- The model forward returns `logits`, `domain_logits`, `gate`, and `factors`.
- TTA parameter selection operates on named parameters from the same model.
- Metrics consume tensor class labels and domain ids.
