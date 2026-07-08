# Widar3 ERM Baseline

run_name: multibranch
train_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_train_cache.pkl
test_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_test_cache.pkl
num_train: 48964
num_test: 68332

## Final Metrics

| split | loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.234225 |  |  |  |  |  |
| test | 1.425229 | 0.421398 | 0.430914 | 0.072368 | 0.055495 | 0.118926 |

## Source Train Evaluation

| loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.237051 | 0.455845 | 0.472063 | 0.328648 | 0.082452 | 0.055315 |

## Best Epochs

| metric | epoch | test_loss | accuracy | macro_f1 | worst_domain_macro_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 9 | 1.392684 | 0.433677 | 0.443516 | 0.054933 |
| macro_f1 | 9 | 1.392684 | 0.433677 | 0.443516 | 0.054933 |
| worst_domain_macro_f1 | 1 | 1.458284 | 0.369139 | 0.365963 | 0.079587 |

## Per-Domain Metrics At Best Macro-F1

| domain | support | accuracy | macro_f1 |
| ---: | ---: | ---: | ---: |
| 0 | 7930 | 0.476040 | 0.475800 |
| 1 | 7714 | 0.454887 | 0.473757 |
| 2 | 8668 | 0.454315 | 0.442919 |
| 3 | 7730 | 0.410996 | 0.413299 |
| 4 | 8046 | 0.481357 | 0.494260 |
| 5 | 7612 | 0.461640 | 0.502567 |
| 6 | 8156 | 0.438818 | 0.454773 |
| 7 | 7612 | 0.524172 | 0.548056 |
| 8 | 4864 | 0.057360 | 0.054933 |

## Per-Class Metrics At Best Macro-F1

| class | support | precision | recall | f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13386 | 0.329618 | 0.165696 | 0.220532 |
| 1 | 9734 | 0.515585 | 0.710294 | 0.597477 |
| 2 | 8074 | 0.839670 | 0.705722 | 0.766891 |
| 3 | 12882 | 0.404428 | 0.523288 | 0.456244 |
| 4 | 11986 | 0.275821 | 0.175121 | 0.214227 |
| 5 | 12270 | 0.348181 | 0.486064 | 0.405728 |

## Epochs

| epoch | train_loss | test_accuracy | test_macro_f1 | worst_domain_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.602612 | 0.369139 | 0.365963 | 0.096012 |
| 2 | 1.334516 | 0.413232 | 0.410896 | 0.074424 |
| 3 | 1.302287 | 0.403808 | 0.415326 | 0.062089 |
| 4 | 1.295072 | 0.423330 | 0.438746 | 0.079770 |
| 5 | 1.275589 | 0.415354 | 0.422152 | 0.060650 |
| 6 | 1.268023 | 0.423213 | 0.429238 | 0.090049 |
| 7 | 1.259982 | 0.423638 | 0.439509 | 0.084293 |
| 8 | 1.250298 | 0.409808 | 0.434878 | 0.122944 |
| 9 | 1.240237 | 0.433677 | 0.443516 | 0.057360 |
| 10 | 1.234225 | 0.421398 | 0.430914 | 0.072368 |
