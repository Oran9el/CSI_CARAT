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
