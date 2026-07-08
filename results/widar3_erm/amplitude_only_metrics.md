# Widar3 ERM Baseline

run_name: amplitude_only
train_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_train_cache.pkl
test_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_test_cache.pkl
num_train: 48964
num_test: 68332

## Final Metrics

| split | loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.240489 |  |  |  |  |  |
| test | 2.046346 | 0.292718 | 0.288596 | 0.074630 | 0.043467 | 0.075909 |

## Source Train Evaluation

| loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.573591 | 0.384241 | 0.413432 | 0.156627 | 0.045139 | 0.085828 |

## Best Epochs

| metric | epoch | test_loss | accuracy | macro_f1 | worst_domain_macro_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 1 | 1.467194 | 0.366812 | 0.363985 | 0.079729 |
| macro_f1 | 1 | 1.467194 | 0.366812 | 0.363985 | 0.079729 |
| worst_domain_macro_f1 | 4 | 1.843219 | 0.314772 | 0.305474 | 0.112203 |

## Per-Domain Metrics At Best Macro-F1

| domain | support | accuracy | macro_f1 |
| ---: | ---: | ---: | ---: |
| 0 | 7930 | 0.391425 | 0.395467 |
| 1 | 7714 | 0.402385 | 0.401750 |
| 2 | 8668 | 0.360060 | 0.352961 |
| 3 | 7730 | 0.362484 | 0.353081 |
| 4 | 8046 | 0.386155 | 0.374765 |
| 5 | 7612 | 0.447977 | 0.455507 |
| 6 | 8156 | 0.337052 | 0.330957 |
| 7 | 7612 | 0.406463 | 0.403727 |
| 8 | 4864 | 0.118010 | 0.079729 |

## Per-Class Metrics At Best Macro-F1

| class | support | precision | recall | f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13386 | 0.281619 | 0.436650 | 0.342404 |
| 1 | 9734 | 0.426790 | 0.615369 | 0.504018 |
| 2 | 8074 | 0.907188 | 0.483032 | 0.630405 |
| 3 | 12882 | 0.334290 | 0.370595 | 0.351508 |
| 4 | 11986 | 0.299959 | 0.366678 | 0.329980 |
| 5 | 12270 | 0.519355 | 0.013121 | 0.025596 |

## Epochs

| epoch | train_loss | test_accuracy | test_macro_f1 | worst_domain_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.489478 | 0.366812 | 0.363985 | 0.118010 |
| 2 | 1.380435 | 0.321753 | 0.313425 | 0.244860 |
| 3 | 1.349884 | 0.327767 | 0.285166 | 0.208265 |
| 4 | 1.328210 | 0.314772 | 0.305474 | 0.228618 |
| 5 | 1.311137 | 0.339182 | 0.329862 | 0.081826 |
| 6 | 1.296251 | 0.320728 | 0.312173 | 0.137541 |
| 7 | 1.282884 | 0.327782 | 0.298403 | 0.202714 |
| 8 | 1.267868 | 0.306591 | 0.302846 | 0.062706 |
| 9 | 1.256445 | 0.351768 | 0.322483 | 0.076275 |
| 10 | 1.240489 | 0.292718 | 0.288596 | 0.074630 |
