# Widar3 ERM Baseline

run_name: risk_multibranch_w0p5_eta2p0
train_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_train_cache.pkl
test_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_test_cache.pkl
num_train: 48964
num_test: 68332

## Final Metrics

| split | loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2.323139 |  |  |  |  |  |
| test | 1.463487 | 0.413569 | 0.422329 | 0.099095 | 0.070932 | 0.108564 |

## Source Train Evaluation

| loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.218561 | 0.477902 | 0.494179 | 0.435857 | 0.105624 | 0.037847 |

## Best Epochs

| metric | epoch | test_loss | accuracy | macro_f1 | worst_domain_macro_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 9 | 1.412844 | 0.414154 | 0.434968 | 0.076281 |
| macro_f1 | 9 | 1.412844 | 0.414154 | 0.434968 | 0.076281 |
| worst_domain_macro_f1 | 1 | 1.501790 | 0.326465 | 0.330009 | 0.124673 |

## Per-Domain Metrics At Best Macro-F1

| domain | support | accuracy | macro_f1 |
| ---: | ---: | ---: | ---: |
| 0 | 7930 | 0.449180 | 0.451235 |
| 1 | 7714 | 0.432201 | 0.458538 |
| 2 | 8668 | 0.434933 | 0.434830 |
| 3 | 7730 | 0.381630 | 0.392096 |
| 4 | 8046 | 0.472160 | 0.493920 |
| 5 | 7612 | 0.422622 | 0.478776 |
| 6 | 8156 | 0.402158 | 0.440102 |
| 7 | 7612 | 0.512743 | 0.542846 |
| 8 | 4864 | 0.099712 | 0.076281 |

## Per-Class Metrics At Best Macro-F1

| class | support | precision | recall | f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13386 | 0.280018 | 0.184148 | 0.222182 |
| 1 | 9734 | 0.538684 | 0.554346 | 0.546403 |
| 2 | 8074 | 0.842780 | 0.699777 | 0.764650 |
| 3 | 12882 | 0.398776 | 0.541143 | 0.459177 |
| 4 | 11986 | 0.260282 | 0.224929 | 0.241318 |
| 5 | 12270 | 0.342174 | 0.417441 | 0.376078 |

## Epochs

| epoch | train_loss | test_accuracy | test_macro_f1 | worst_domain_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.907937 | 0.326465 | 0.330009 | 0.218544 |
| 2 | 2.505089 | 0.388017 | 0.387729 | 0.163446 |
| 3 | 2.452189 | 0.403266 | 0.411454 | 0.113487 |
| 4 | 2.434877 | 0.401101 | 0.420388 | 0.159128 |
| 5 | 2.415799 | 0.403047 | 0.404594 | 0.113076 |
| 6 | 2.389694 | 0.409398 | 0.427671 | 0.107319 |
| 7 | 2.382410 | 0.408769 | 0.425309 | 0.153577 |
| 8 | 2.358556 | 0.390842 | 0.422724 | 0.135691 |
| 9 | 2.337863 | 0.414154 | 0.434968 | 0.099712 |
| 10 | 2.323139 | 0.413569 | 0.422329 | 0.099095 |
