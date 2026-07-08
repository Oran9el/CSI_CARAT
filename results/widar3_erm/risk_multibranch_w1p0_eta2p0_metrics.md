# Widar3 ERM Baseline

run_name: risk_multibranch_w1p0_eta2p0
train_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_train_cache.pkl
test_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_test_cache.pkl
num_train: 48964
num_test: 68332

## Final Metrics

| split | loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 3.427939 |  |  |  |  |  |
| test | 1.455187 | 0.407657 | 0.433118 | 0.105880 | 0.081333 | 0.106551 |

## Source Train Evaluation

| loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.210912 | 0.482722 | 0.511882 | 0.431058 | 0.100405 | 0.039903 |

## Best Epochs

| metric | epoch | test_loss | accuracy | macro_f1 | worst_domain_macro_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 10 | 1.455187 | 0.407657 | 0.433118 | 0.081333 |
| macro_f1 | 10 | 1.455187 | 0.407657 | 0.433118 | 0.081333 |
| worst_domain_macro_f1 | 1 | 1.454815 | 0.358163 | 0.341342 | 0.100782 |

## Per-Domain Metrics At Best Macro-F1

| domain | support | accuracy | macro_f1 |
| ---: | ---: | ---: | ---: |
| 0 | 7930 | 0.455612 | 0.470532 |
| 1 | 7714 | 0.434146 | 0.469574 |
| 2 | 8668 | 0.435971 | 0.448540 |
| 3 | 7730 | 0.410349 | 0.434170 |
| 4 | 8046 | 0.427417 | 0.454999 |
| 5 | 7612 | 0.415003 | 0.460458 |
| 6 | 8156 | 0.376900 | 0.407369 |
| 7 | 7612 | 0.493431 | 0.539552 |
| 8 | 4864 | 0.105880 | 0.081333 |

## Per-Class Metrics At Best Macro-F1

| class | support | precision | recall | f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13386 | 0.348453 | 0.326535 | 0.337138 |
| 1 | 9734 | 0.481565 | 0.701767 | 0.571178 |
| 2 | 8074 | 0.852345 | 0.697795 | 0.767366 |
| 3 | 12882 | 0.435276 | 0.291570 | 0.349217 |
| 4 | 11986 | 0.274350 | 0.314200 | 0.292926 |
| 5 | 12270 | 0.276806 | 0.285086 | 0.280885 |

## Epochs

| epoch | train_loss | test_accuracy | test_macro_f1 | worst_domain_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.203575 | 0.358163 | 0.341342 | 0.148438 |
| 2 | 3.665560 | 0.393213 | 0.401335 | 0.151316 |
| 3 | 3.599058 | 0.399242 | 0.400077 | 0.114926 |
| 4 | 3.574684 | 0.394398 | 0.416277 | 0.176398 |
| 5 | 3.549341 | 0.404262 | 0.412601 | 0.119449 |
| 6 | 3.515989 | 0.403647 | 0.421440 | 0.151933 |
| 7 | 3.512432 | 0.407174 | 0.428098 | 0.159745 |
| 8 | 3.472744 | 0.374451 | 0.400630 | 0.197574 |
| 9 | 3.448491 | 0.405754 | 0.430286 | 0.149054 |
| 10 | 3.427939 | 0.407657 | 0.433118 | 0.105880 |
