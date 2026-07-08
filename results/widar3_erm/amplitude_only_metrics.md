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

## Best Epochs

| metric | epoch | test_loss | accuracy | macro_f1 | worst_domain_macro_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 1 | 1.467194 | 0.366812 | 0.363985 | 0.079729 |
| macro_f1 | 4 | 1.560669 | 0.366197 | 0.388557 | 0.100301 |
| worst_domain_macro_f1 | 6 | 2.026828 | 0.301162 | 0.274637 | 0.129775 |

## Per-Domain Metrics At Best Macro-F1

| domain | support | accuracy | macro_f1 |
| ---: | ---: | ---: | ---: |
| 0 | 7930 | 0.411854 | 0.432827 |
| 1 | 7714 | 0.405367 | 0.436589 |
| 2 | 8668 | 0.359714 | 0.373567 |
| 3 | 7730 | 0.360543 | 0.373199 |
| 4 | 8046 | 0.364902 | 0.378134 |
| 5 | 7612 | 0.439175 | 0.473379 |
| 6 | 8156 | 0.334600 | 0.345093 |
| 7 | 7612 | 0.404624 | 0.424704 |
| 8 | 4864 | 0.130962 | 0.100301 |

## Per-Class Metrics At Best Macro-F1

| class | support | precision | recall | f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13386 | 0.285160 | 0.439265 | 0.345821 |
| 1 | 9734 | 0.536361 | 0.468256 | 0.500000 |
| 2 | 8074 | 0.929840 | 0.389027 | 0.548550 |
| 3 | 12882 | 0.380245 | 0.363686 | 0.371781 |
| 4 | 11986 | 0.278769 | 0.260721 | 0.269443 |
| 5 | 12270 | 0.295327 | 0.296170 | 0.295748 |

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
