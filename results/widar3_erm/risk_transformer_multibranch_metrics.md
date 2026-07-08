# Widar3 ERM Baseline

run_name: risk_transformer_multibranch
train_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_train_cache.pkl
test_cache: /home/ccl/data/csi-carat/widar3/widar3g6d/feature_cache/widar3-g6_features_test_cache.pkl
num_train: 48964
num_test: 68332

## Final Metrics

| split | loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.758667 |  |  |  |  |  |
| test | 1.411562 | 0.383773 | 0.408371 | 0.127673 | 0.086809 | 0.091147 |

## Source Train Evaluation

| loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.190961 | 0.480026 | 0.520715 | 0.454072 | 0.120240 | 0.039784 |

## Best Epochs

| metric | epoch | test_loss | accuracy | macro_f1 | worst_domain_macro_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 7 | 1.394559 | 0.425555 | 0.434917 | 0.070093 |
| macro_f1 | 7 | 1.394559 | 0.425555 | 0.434917 | 0.070093 |
| worst_domain_macro_f1 | 4 | 1.440114 | 0.394369 | 0.400600 | 0.098002 |

## Per-Domain Metrics At Best Macro-F1

| domain | support | accuracy | macro_f1 |
| ---: | ---: | ---: | ---: |
| 0 | 7930 | 0.461665 | 0.459353 |
| 1 | 7714 | 0.450350 | 0.468425 |
| 2 | 8668 | 0.426511 | 0.422446 |
| 3 | 7730 | 0.406598 | 0.403517 |
| 4 | 8046 | 0.465946 | 0.479804 |
| 5 | 7612 | 0.463347 | 0.500198 |
| 6 | 8156 | 0.418710 | 0.445978 |
| 7 | 7612 | 0.509984 | 0.537529 |
| 8 | 4864 | 0.109169 | 0.070093 |

## Per-Class Metrics At Best Macro-F1

| class | support | precision | recall | f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13386 | 0.241566 | 0.209697 | 0.224506 |
| 1 | 9734 | 0.516555 | 0.735669 | 0.606942 |
| 2 | 8074 | 0.890735 | 0.644166 | 0.747646 |
| 3 | 12882 | 0.393917 | 0.611240 | 0.479085 |
| 4 | 11986 | 0.311893 | 0.128650 | 0.182162 |
| 5 | 12270 | 0.372112 | 0.366259 | 0.369163 |

## Epochs

| epoch | train_loss | test_accuracy | test_macro_f1 | worst_domain_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.075318 | 0.379968 | 0.384098 | 0.137952 |
| 2 | 1.905548 | 0.412193 | 0.417672 | 0.128289 |
| 3 | 1.877776 | 0.418925 | 0.433802 | 0.125411 |
| 4 | 1.851342 | 0.394369 | 0.400600 | 0.171464 |
| 5 | 1.837722 | 0.409047 | 0.413719 | 0.147821 |
| 6 | 1.823360 | 0.415589 | 0.416316 | 0.134046 |
| 7 | 1.804633 | 0.425555 | 0.434917 | 0.109169 |
| 8 | 1.787462 | 0.399886 | 0.415145 | 0.179688 |
| 9 | 1.778869 | 0.377466 | 0.397202 | 0.164885 |
| 10 | 1.758667 | 0.383773 | 0.408371 | 0.127673 |
