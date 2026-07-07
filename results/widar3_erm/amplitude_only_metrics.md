# Widar3 ERM Baseline

run_name: amplitude_only
train_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_train_cache.pkl
test_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_test_cache.pkl
num_train: 48964
num_test: 68332

## Final Metrics

| split | loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.238266 |  |  |  |  |  |
| test | 1.579278 | 0.331645 | 0.341409 | 0.211554 | 0.098741 | 0.044998 |

## Epochs

| epoch | train_loss | test_accuracy | test_macro_f1 | worst_domain_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.489478 | 0.366812 | 0.363985 | 0.118010 |
| 2 | 1.380286 | 0.354695 | 0.327994 | 0.151110 |
| 3 | 1.345702 | 0.254010 | 0.222138 | 0.027549 |
| 4 | 1.327609 | 0.366197 | 0.388557 | 0.130962 |
| 5 | 1.309499 | 0.281464 | 0.200823 | 0.240954 |
| 6 | 1.297056 | 0.301162 | 0.274637 | 0.252575 |
| 7 | 1.280352 | 0.248507 | 0.214618 | 0.196766 |
| 8 | 1.264477 | 0.285445 | 0.254263 | 0.231086 |
| 9 | 1.255888 | 0.292250 | 0.270294 | 0.061266 |
| 10 | 1.238266 | 0.331645 | 0.341409 | 0.211554 |
