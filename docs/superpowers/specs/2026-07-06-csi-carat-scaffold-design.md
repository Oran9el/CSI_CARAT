# CSI-CARAT Scaffold Design

## Goal

Build a modular PyTorch research scaffold for CSI-CARAT: Causal Adaptive Risk-aware Augmentation and Test-time Calibration for cross-domain WiFi CSI sensing.

The first milestone is not a final paper model. It is a runnable project skeleton that can reproduce DATTA-style baselines, absorb MDTA-style multi-factor disentanglement, and leave clean interfaces for CSI-CARAT augmentations, risk-aware gating, and lightweight test-time calibration.

## Recommended Foundation

Use DATTA as the engineering baseline and MDTA as the method baseline for disentanglement.

- DATTA contributes the Widar3.0-G6D data path, WiFlexFormer-style CSI encoder, domain-adversarial training loop, source statistic collection, online test-time adaptation, and random reset against drift.
- MDTA contributes multi-factor feature disentanglement, factor-specific discriminators, reconstruction-based information preservation, factor separation losses, and memory/prototype-based test-time expansion.
- CSI-CARAT combines both: a DATTA-compatible train/test/TTA pipeline with an MDTA-inspired factor head and a CARAT-specific gate.

## Project Shape

```text
configs/
scripts/
src/csi_carat/
  data/
  augmentations/
  models/
    backbones/
    disentanglement.py
    adapters.py
    gate.py
    carat.py
  losses/
    adversarial.py
    disentangle.py
    risk.py
    consistency.py
  tta/
    stat_alignment.py
    prototype_memory.py
    calibrator.py
  engine/
    train.py
    evaluate.py
  metrics/
tests/
```

## Model Design

The first CSI-CARAT model should use a DATTA/WiFlexFormer-compatible encoder:

```text
x -> encoder -> h
```

Then replace the single DATTA feature with an MDTA-style factor head:

```text
h -> z_action, z_env, z_pos, z_ori, z_user, z_res
```

Map the factors into CSI-CARAT notation:

```text
z_c = z_action
z_s = concat(z_env, z_pos, z_ori, z_user, z_res)
logits = classifier(z_c + gate(x, h) * adapter(z_s))
```

This gives us the simple paper-facing story:

- `z_c` is the stable action-causal factor.
- `z_s` contains domain-specific or spurious factors.
- `gate` controls when domain-specific factors are helpful versus risky.
- `adapter` makes `z_s` usable without letting it dominate the classifier.

## Training Losses

The initial implementation should include these losses:

- Activity classification: cross entropy on final logits.
- Domain adversarial loss: DATTA-style gradient reversal on `z_action`.
- Factor discrimination: MDTA-style heads for action, environment/location/orientation/user/device when labels exist.
- Reconstruction loss: reconstruct encoder feature `h` from all factors.
- Factor separation: covariance penalty in milestone 1; HSIC or triplet separation in milestone 2 after the covariance baseline is validated.
- Risk loss: log-sum-exp over source-domain losses to emphasize worst-domain behavior.
- Consistency loss: KL or Jensen-Shannon consistency between original and augmented views.

Do not implement every loss as mandatory in the first run. Every loss should be config-gated so ablations are simple.

## Test-Time Calibration

The first TTA path should be conservative:

- Freeze the backbone.
- Update only LayerNorm affine parameters, adapters, gate, temperature, and optional class prototypes.
- Align target stream feature statistics to source statistics, following DATTA.
- Use random reset or EMA rollback to reduce continual TTA drift.
- Use prototype memory for high-confidence target samples.
- Add motion-preserving versus motion-destroying sample selection after the baseline TTA path works.

MDTA's graph KNN propagation is valuable but should be phase-two. It adds `torch_geometric` and extra runtime complexity, so the first scaffold should use plain prototype memory.

## Dataset Plan

All server datasets should live under:

```text
/home/ccl/data/csi-carat/
```

Recommended layout:

```text
/home/ccl/data/csi-carat/
  widar3/
    raw/
    widar3g6d/
      raw/
      cache/
  mmfi/
    raw/
  xrf55/
    raw/
  mdta_dfs/
    raw/
```

Priority:

1. Widar3.0 is required first. It is the primary dataset for reproducing DATTA-style training and for early CSI-CARAT experiments.
2. MM-Fi is required second. Use WiFi CSI first; RGB/pose/depth are optional for teacher or upper-bound experiments.
3. XRF55 is the large-scale stress test. Use WiFi first; mmWave/RFID/Kinect can remain optional.
4. MDTA author's DFS data is optional. If available through author permission, use it as an extra cross-location/cross-user sanity check, not as the core benchmark.

## Download Checklist For Server

### Widar3.0

Download from the official Widar3.0 page: https://tns.thss.tsinghua.edu.cn/widar3.0/

For the DATTA-compatible Widar3.0-G6D subset, download these CSI archives:

```text
CSI_20181130.zip
CSI_20181211.zip
CSI_20181209.zip
CSI_20181204.zip
```

Place them under:

```text
/home/ccl/data/csi-carat/widar3/widar3g6d/raw/
```

The Widar preprocessing command will generate cache files equivalent to:

```text
widar3-g6_csi_domain_train_cache.pkl
widar3-g6_csi_domain_test_cache.pkl
```

### MM-Fi

Use the official project/toolbox pages:

- Project page: https://ntu-aiot-lab.github.io/mm-fi
- Toolbox and download links: https://github.com/ybhbingo/MMFi_dataset

Download the dataset parts from the provided Google Drive or Baidu Netdisk links. For CSI-CARAT, WiFi CSI is required. Other modalities are optional.

Place the unzipped dataset root under:

```text
/home/ccl/data/csi-carat/mmfi/raw/
```

Expected upstream layout begins with environment folders such as:

```text
E01/
E02/
E03/
E04/
```

and each sample contains a `wifi-csi` modality directory.

### XRF55

Use the official project/repository:

- Project page: https://aiotgroup.github.io/XRF55/
- Code and download instructions: https://github.com/aiotgroup/XRF55-repo

Download `dataset.zip` from the Kaggle link referenced by the XRF55 repository. For CSI-CARAT, the required modality is WiFi.

Place the raw unzipped data under:

```text
/home/ccl/data/csi-carat/xrf55/raw/
```

Expected upstream raw layout includes:

```text
Raw_dataset/
  WiFi/
  mmWave/
  RFID/
```

Only `Raw_dataset/WiFi` is required for the main CSI-only experiments.

### MDTA DFS Data

The public MDTA code does not include a dataset downloader. If the author provides the original DFS/CSI data, place it under:

```text
/home/ccl/data/csi-carat/mdta_dfs/raw/
```

Keep the author's original directory structure intact. The adapter should map its labels into:

```text
activity, position, orientation, user, environment/domain
```

## Error Handling

Dataset loaders should fail with clear messages:

- Missing root path.
- Missing required split/cache files.
- Missing modality such as `wifi-csi` or `WiFi`.
- Unsupported protocol or domain factor.
- Shape mismatch after preprocessing.

No training script should silently skip a dataset because files are missing.

## Testing

Initial tests should not require real datasets. Use synthetic tensors to verify:

- Dataset sample schema.
- Backbone forward pass.
- Factor head output shapes.
- Loss functions return finite scalars.
- TTA calibrator only exposes allowed trainable parameters.
- Metrics compute average accuracy, macro-F1, worst-domain accuracy, and worst-domain macro-F1.

Real-data smoke tests can run on the server after Widar3.0-G6D caches are generated.

## Approval State

The user approved the combined DATTA + MDTA foundation on 2026-07-06.

This workspace was not a git repository when the design was written, so the design document could not be committed locally without initializing git.
