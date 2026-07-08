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

Build Widar3.0-G6D raw complex CSI caches on the server:

```bash
python -m pip install -e ".[preprocess]"
python scripts/preprocess_widar3_g6d.py \
  --data-root /home/ccl/data/csi-carat \
  --split BOTH
```

Expected raw data root:

```text
/home/ccl/data/csi-carat/widar3/widar3g6d/raw
```

Expected cache outputs:

```text
/home/ccl/data/csi-carat/widar3/widar3g6d/cache/widar3-g6_csi_domain_train_cache.pkl
/home/ccl/data/csi-carat/widar3/widar3g6d/cache/widar3-g6_csi_domain_test_cache.pkl
```

Build cleaned/windowed Widar3.0-G6D caches:

```bash
python scripts/clean_widar3_g6d.py \
  --data-root /home/ccl/data/csi-carat \
  --split BOTH \
  --target-packets 220 \
  --window-size 128 \
  --stride 64
```

Expected clean cache outputs:

```text
/home/ccl/data/csi-carat/widar3/widar3g6d/clean_cache/widar3-g6_clean_train_cache.pkl
/home/ccl/data/csi-carat/widar3/widar3g6d/clean_cache/widar3-g6_clean_test_cache.pkl
```

Extract model-ready Widar3.0-G6D feature caches:

```bash
python scripts/extract_widar3_features.py \
  --data-root /home/ccl/data/csi-carat \
  --split BOTH \
  --n-fft 32 \
  --hop-length 16
```

Expected feature cache outputs:

```text
/home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_train_cache.pkl
/home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_test_cache.pkl
```

Generate feature-cache sanity reports before training:

```bash
python scripts/report_widar3_features.py \
  --data-root /home/ccl/data/csi-carat \
  --split BOTH \
  --output-dir results/widar3_features
```

Expected report outputs:

```text
results/widar3_features/train_feature_report.md
results/widar3_features/test_feature_report.md
```

Run the first amplitude-only ERM smoke baseline:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/train_widar3_erm.py \
  --data-root /home/ccl/data/csi-carat \
  --batch-size 64 \
  --max-steps 20 \
  --device cuda
```

The sanity report checks feature shapes, finite values, label counts, and domain/user/environment counts. The ERM smoke is not the final CSI-CARAT baseline; it validates the feature cache, PyTorch Dataset, DataLoader, model forward pass, loss, and optimizer path before adding multi-branch fusion and CARAT/TTA objectives.

Run the reproducible amplitude-only source ERM baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_widar3_erm_baseline.py \
  --data-root /home/ccl/data/csi-carat \
  --batch-size 256 \
  --epochs 10 \
  --device cuda \
  --output-dir results/widar3_erm
```

Expected baseline outputs:

```text
results/widar3_erm/amplitude_only_metrics.json
results/widar3_erm/amplitude_only_metrics.md
```

The metrics include source-train evaluation, final-epoch target scores, best epochs by target accuracy / macro-F1 / worst-domain macro-F1, plus per-domain and per-class diagnostics at the best macro-F1 epoch. The checkpoint is written to `results/widar3_erm/amplitude_only_checkpoint.pt` and is ignored by Git.

Run a tiny-subset overfit diagnostic if target-domain ERM looks suspiciously low:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/overfit_widar3_erm_subset.py \
  --data-root /home/ccl/data/csi-carat \
  --model amplitude \
  --samples-per-class 16 \
  --epochs 100 \
  --device cuda \
  --output-dir results/widar3_erm
```

Expected diagnostic outputs:

```text
results/widar3_erm/overfit_subset_metrics.json
results/widar3_erm/overfit_subset_metrics.md
```

Run the same diagnostic for the three-branch model:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/overfit_widar3_erm_subset.py \
  --data-root /home/ccl/data/csi-carat \
  --model multibranch \
  --samples-per-class 16 \
  --epochs 100 \
  --device cuda \
  --output-dir results/widar3_erm
```

Expected multibranch diagnostic outputs:

```text
results/widar3_erm/overfit_subset_multibranch_metrics.json
results/widar3_erm/overfit_subset_multibranch_metrics.md
```

If the tiny subset cannot approach the target accuracy, debug labels/features/model capacity before building CSI-CARAT. If it overfits but target-domain ERM stays low, the failure is cross-domain generalization, which is exactly the setting CSI-CARAT is meant to address.

Run the three-branch ERM baseline before CSI-CARAT:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_widar3_multibranch_erm.py \
  --data-root /home/ccl/data/csi-carat \
  --batch-size 256 \
  --epochs 10 \
  --device cuda \
  --output-dir results/widar3_erm
```

Expected multibranch outputs:

```text
results/widar3_erm/multibranch_metrics.json
results/widar3_erm/multibranch_metrics.md
```

This baseline uses amplitude, phase-difference, and Doppler/spectrogram branches. It is the next source-learnability gate before adding CSI-CARAT losses, gates, and test-time calibration.

Run the risk-aware three-branch baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_widar3_risk_multibranch.py \
  --data-root /home/ccl/data/csi-carat \
  --batch-size 256 \
  --epochs 10 \
  --risk-weight 0.5 \
  --risk-eta 2.0 \
  --device cuda \
  --output-dir results/widar3_erm
```

Expected risk-aware outputs:

```text
results/widar3_erm/risk_multibranch_metrics.json
results/widar3_erm/risk_multibranch_metrics.md
```

This baseline is the smallest measurable version of CSI-CARAT's risk-aware objective: average CE plus smooth worst-source-domain risk.

Run a compact risk sweep:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sweep_widar3_risk_multibranch.py \
  --data-root /home/ccl/data/csi-carat \
  --risk-weights 0.25,0.5,1.0 \
  --risk-etas 2.0 \
  --batch-size 256 \
  --epochs 10 \
  --device cuda \
  --output-dir results/widar3_erm
```

Expected sweep summary outputs:

```text
results/widar3_erm/risk_sweep_summary.json
results/widar3_erm/risk_sweep_summary.md
```

Each sweep setting also writes its own `risk_multibranch_w*_eta*_metrics.json` and `.md` files for detailed inspection.

Run the three-branch Transformer ERM baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_widar3_transformer_multibranch.py \
  --data-root /home/ccl/data/csi-carat \
  --batch-size 128 \
  --epochs 10 \
  --learning-rate 0.0003 \
  --feature-dim 96 \
  --num-heads 4 \
  --num-layers 2 \
  --device cuda \
  --output-dir results/widar3_erm
```

Expected Transformer outputs:

```text
results/widar3_erm/transformer_multibranch_metrics.json
results/widar3_erm/transformer_multibranch_metrics.md
```

This baseline tests whether stronger temporal encoding lifts source learnability before adding CSI-CARAT factor disentanglement, gates, or TTA.

Run the risk-aware three-branch Transformer baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_widar3_risk_transformer_multibranch.py \
  --data-root /home/ccl/data/csi-carat \
  --batch-size 128 \
  --epochs 10 \
  --learning-rate 0.0003 \
  --feature-dim 96 \
  --num-heads 4 \
  --num-layers 2 \
  --risk-weight 0.25 \
  --risk-eta 2.0 \
  --device cuda \
  --output-dir results/widar3_erm
```

Expected risk-aware Transformer outputs:

```text
results/widar3_erm/risk_transformer_multibranch_metrics.json
results/widar3_erm/risk_transformer_multibranch_metrics.md
```

This run combines the strongest temporal encoder tested so far with the smooth worst-source-domain risk objective. Use it to decide whether CSI-CARAT should prioritize the risk path, the Transformer backbone, or both before adding causal factor heads and TTA.
