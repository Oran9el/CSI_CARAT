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

Extract Wi-CBR reproduction feature caches directly from raw `.dat` files:

```bash
python -m pip install -e ".[preprocess,wicbr]"
python scripts/extract_widar3_wicbr_features.py \
  --data-root /home/ccl/data/csi-carat \
  --split BOTH \
  --image-size 224 \
  --packet-downsample 1 \
  --n-fft 128 \
  --hop-length 32
```

Expected Wi-CBR cache outputs:

```text
/home/ccl/data/csi-carat/widar3/widar3g6d/wicbr_cache/widar3-g6_wicbr_train_cache.pkl
/home/ccl/data/csi-carat/widar3/widar3g6d/wicbr_cache/widar3-g6_wicbr_test_cache.pkl
```

This path keeps all three receive antennas and groups the six receiver `.dat` files for each Widar trial before computing CSI-ratio phase images and Doppler velocity spectrum images.

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

Run the Wi-CBR reproduction baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_widar3_wicbr.py \
  --data-root /home/ccl/data/csi-carat \
  --batch-size 10 \
  --epochs 30 \
  --learning-rate 0.0001 \
  --contrastive-weight 0.1 \
  --temperature 0.1 \
  --source-val-fraction 0.1 \
  --source-val-strategy leave_one_domain \
  --source-val-domain -1 \
  --selection-split source_val \
  --selection-metric macro_f1 \
  --backbone resnet18 \
  --device cuda \
  --output-dir results/widar3_wicbr
```

Expected Wi-CBR outputs:

```text
results/widar3_wicbr/wicbr_metrics.json
results/widar3_wicbr/wicbr_metrics.md
```

For a dependency-light smoke test, replace `--backbone resnet18` with `--backbone small --max-steps-per-epoch 20 --no-checkpoint`. The ResNet18 path is closer to the Wi-CBR code release; the small path only validates the CSI-CARAT training plumbing.

The checkpoint's `best_model_state_dict` is selected by source-val metrics when `--source-val-fraction` is positive. Use `--source-val-strategy leave_one_domain` to hold out one full source domain instead of a random stratified split. Target-test best epochs are still written as diagnostics, but should not be used for formal model selection.

Run Wi-CBR ablations:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_widar3_wicbr_ablation.py \
  --data-root /home/ccl/data/csi-carat \
  --runs phase_only,dfs_only,no_fusion,no_contrastive \
  --batch-size 10 \
  --epochs 30 \
  --learning-rate 0.0001 \
  --source-val-fraction 0.1 \
  --source-val-strategy leave_one_domain \
  --source-val-domain -1 \
  --selection-split source_val \
  --selection-metric macro_f1 \
  --backbone resnet18 \
  --device cuda \
  --output-dir results/widar3_wicbr_ablation
```

Expected ablation outputs include:

```text
results/widar3_wicbr_ablation/wicbr_phase_only_metrics.json
results/widar3_wicbr_ablation/wicbr_dfs_only_metrics.json
results/widar3_wicbr_ablation/wicbr_no_fusion_metrics.json
results/widar3_wicbr_ablation/wicbr_no_contrastive_metrics.json
```

Run the first Wi-CBR-backed CSI-CARAT baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_widar3_wicbr_carat.py \
  --data-root /home/ccl/data/csi-carat \
  --batch-size 10 \
  --epochs 30 \
  --learning-rate 0.0001 \
  --risk-weight 0.25 \
  --risk-eta 2.0 \
  --domain-weight 0.1 \
  --disentangle-weight 0.1 \
  --contrastive-weight 0.1 \
  --source-val-fraction 0.1 \
  --source-val-strategy leave_one_domain \
  --source-val-domain -1 \
  --selection-split source_val \
  --selection-metric macro_f1 \
  --backbone resnet18 \
  --device cuda \
  --output-dir results/widar3_wicbr_carat
```

Expected Wi-CBR-CARAT outputs:

```text
results/widar3_wicbr_carat/wicbr_carat_metrics.json
results/widar3_wicbr_carat/wicbr_carat_metrics.md
```

Run the domain-8-focused LODO sweep:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sweep_widar3_domain8_focus.py \
  --data-root /home/ccl/data/csi-carat \
  --candidates wicbr_full,phase_only,no_fusion,wicbr_carat \
  --batch-size 10 \
  --epochs 30 \
  --learning-rate 0.0001 \
  --source-val-strategy leave_one_domain \
  --source-val-domain -1 \
  --selection-split source_val \
  --selection-metric macro_f1 \
  --backbone resnet18 \
  --device cuda \
  --output-dir results/widar3_domain8_focus
```

`--source-val-domain -1` deterministically holds out the highest source domain id. Repeat with explicit values such as `--source-val-domain 9` through `15` when running a fuller leave-one-source-domain-out audit.
