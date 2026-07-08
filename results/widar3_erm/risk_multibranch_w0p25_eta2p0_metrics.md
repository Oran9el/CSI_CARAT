# Widar3 ERM Baseline

run_name: risk_multibranch_w0p25_eta2p0
train_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_train_cache.pkl
test_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_test_cache.pkl
num_train: 48964
num_test: 68332

## Final Metrics

| split | loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.789960 |  |  |  |  |  |
| test | 1.431478 | 0.420213 | 0.435654 | 0.091694 | 0.071775 | 0.113652 |

## Source Train Evaluation

| loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.236919 | 0.468324 | 0.489916 | 0.429222 | 0.102457 | 0.028331 |

## Best Epochs

| metric | epoch | test_loss | accuracy | macro_f1 | worst_domain_macro_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 9 | 1.406451 | 0.423403 | 0.446908 | 0.082477 |
| macro_f1 | 9 | 1.406451 | 0.423403 | 0.446908 | 0.082477 |
| worst_domain_macro_f1 | 4 | 1.399439 | 0.409896 | 0.430024 | 0.090423 |

## Per-Domain Metrics At Best Macro-F1

| domain | support | accuracy | macro_f1 |
| ---: | ---: | ---: | ---: |
| 0 | 7930 | 0.452963 | 0.473804 |
| 1 | 7714 | 0.438164 | 0.474027 |
| 2 | 8668 | 0.418666 | 0.434377 |
| 3 | 7730 | 0.396248 | 0.414755 |
| 4 | 8046 | 0.479617 | 0.511326 |
| 5 | 7612 | 0.439963 | 0.473826 |
| 6 | 8156 | 0.406817 | 0.438770 |
| 7 | 7612 | 0.535076 | 0.577270 |
| 8 | 4864 | 0.137541 | 0.082477 |

## Per-Class Metrics At Best Macro-F1

| class | support | precision | recall | f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13386 | 0.291106 | 0.471911 | 0.360087 |
| 1 | 9734 | 0.536355 | 0.647935 | 0.586889 |
| 2 | 8074 | 0.870719 | 0.686525 | 0.767729 |
| 3 | 12882 | 0.422611 | 0.378978 | 0.399607 |
| 4 | 11986 | 0.294929 | 0.192641 | 0.233056 |
| 5 | 12270 | 0.391628 | 0.291280 | 0.334081 |

## Epochs

| epoch | train_loss | test_accuracy | test_macro_f1 | worst_domain_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.261036 | 0.362612 | 0.353262 | 0.115132 |
| 2 | 1.925803 | 0.397383 | 0.396354 | 0.137747 |
| 3 | 1.878323 | 0.406676 | 0.417244 | 0.071752 |
| 4 | 1.861759 | 0.409896 | 0.430024 | 0.153372 |
| 5 | 1.841920 | 0.415398 | 0.426481 | 0.071546 |
| 6 | 1.832686 | 0.422335 | 0.440424 | 0.112459 |
| 7 | 1.824780 | 0.405535 | 0.424781 | 0.168791 |
| 8 | 1.812336 | 0.401730 | 0.426088 | 0.157278 |
| 9 | 1.799441 | 0.423403 | 0.446908 | 0.137541 |
| 10 | 1.789960 | 0.420213 | 0.435654 | 0.091694 |
