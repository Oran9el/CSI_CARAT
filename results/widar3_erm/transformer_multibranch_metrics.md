# Widar3 ERM Baseline

run_name: transformer_multibranch
train_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_train_cache.pkl
test_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_test_cache.pkl
num_train: 48964
num_test: 68332

## Final Metrics

| split | loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.198182 |  |  |  |  |  |
| test | 1.481104 | 0.365334 | 0.392654 | 0.070724 | 0.055792 | 0.103566 |

## Source Train Evaluation

| loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.192365 | 0.475860 | 0.520307 | 0.209505 | 0.057738 | 0.101280 |

## Best Epochs

| metric | epoch | test_loss | accuracy | macro_f1 | worst_domain_macro_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 2 | 1.378932 | 0.433999 | 0.431389 | 0.083144 |
| macro_f1 | 4 | 1.414025 | 0.427077 | 0.447578 | 0.068383 |
| worst_domain_macro_f1 | 2 | 1.378932 | 0.433999 | 0.431389 | 0.083144 |

## Per-Domain Metrics At Best Macro-F1

| domain | support | accuracy | macro_f1 |
| ---: | ---: | ---: | ---: |
| 0 | 7930 | 0.434678 | 0.460602 |
| 1 | 7714 | 0.431294 | 0.468645 |
| 2 | 8668 | 0.421089 | 0.433167 |
| 3 | 7730 | 0.434541 | 0.457048 |
| 4 | 8046 | 0.505220 | 0.511699 |
| 5 | 7612 | 0.459406 | 0.484927 |
| 6 | 8156 | 0.449485 | 0.463986 |
| 7 | 7612 | 0.490804 | 0.532857 |
| 8 | 4864 | 0.089638 | 0.068383 |

## Per-Class Metrics At Best Macro-F1

| class | support | precision | recall | f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13386 | 0.352329 | 0.563947 | 0.433701 |
| 1 | 9734 | 0.564537 | 0.629957 | 0.595455 |
| 2 | 8074 | 0.893329 | 0.670052 | 0.765747 |
| 3 | 12882 | 0.376160 | 0.185685 | 0.248636 |
| 4 | 11986 | 0.256373 | 0.250876 | 0.253595 |
| 5 | 12270 | 0.394370 | 0.382478 | 0.388333 |

## Epochs

| epoch | train_loss | test_accuracy | test_macro_f1 | worst_domain_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.451715 | 0.400720 | 0.421068 | 0.066201 |
| 2 | 1.311730 | 0.433999 | 0.431389 | 0.091694 |
| 3 | 1.284986 | 0.424881 | 0.445434 | 0.068462 |
| 4 | 1.266638 | 0.427077 | 0.447578 | 0.089638 |
| 5 | 1.253841 | 0.433574 | 0.439563 | 0.070312 |
| 6 | 1.247166 | 0.424867 | 0.430869 | 0.081620 |
| 7 | 1.235761 | 0.416759 | 0.425543 | 0.065584 |
| 8 | 1.222570 | 0.419086 | 0.444279 | 0.112664 |
| 9 | 1.212338 | 0.367193 | 0.392401 | 0.068668 |
| 10 | 1.198182 | 0.365334 | 0.392654 | 0.070724 |
