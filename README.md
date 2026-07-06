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
