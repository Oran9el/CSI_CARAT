# Widar3 LODO Result Summary

## Aggregate

| run | n | target_macro_f1_mean | target_macro_f1_std | worst_domain_macro_f1_mean | worst_domain_macro_f1_std | domain8_macro_f1_mean | domain8_macro_f1_std | source_val_macro_f1_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wicbr_carat_lodo | 7 | 0.831935 | 0.015936 | 0.455029 | 0.059999 | 0.455029 | 0.059999 | 0.811055 |
| wicbr_carat_v2_lodo | 7 | 0.840234 | 0.028676 | 0.500269 | 0.070647 | 0.500269 | 0.070647 | 0.822208 |
| wicbr_lodo_domain10 | 1 | 0.699454 | 0.000000 | 0.418532 | 0.000000 | 0.418532 | 0.000000 | 0.322314 |
| wicbr_lodo_domain11 | 1 | 0.854712 | 0.000000 | 0.563628 | 0.000000 | 0.563628 | 0.000000 | 0.948518 |
| wicbr_lodo_domain12 | 1 | 0.847244 | 0.000000 | 0.429412 | 0.000000 | 0.429412 | 0.000000 | 0.957309 |
| wicbr_lodo_domain13 | 1 | 0.840528 | 0.000000 | 0.415909 | 0.000000 | 0.415909 | 0.000000 | 0.857602 |
| wicbr_lodo_domain14 | 1 | 0.812792 | 0.000000 | 0.433256 | 0.000000 | 0.433256 | 0.000000 | 0.853893 |
| wicbr_lodo_domain15 | 1 | 0.862070 | 0.000000 | 0.519085 | 0.000000 | 0.519085 | 0.000000 | 0.933115 |
| wicbr_lodo_domain9 | 1 | 0.828972 | 0.000000 | 0.495899 | 0.000000 | 0.495899 | 0.000000 | 0.918670 |
| wicbr_lodo_full | 8 | 0.825980 | 0.050356 | 0.474351 | 0.053240 | 0.474351 | 0.053240 | 0.840567 |
| wicbr_lodo_no_fusion | 7 | 0.818825 | 0.038905 | 0.488647 | 0.086990 | 0.488647 | 0.086990 | 0.807595 |
| wicbr_lodo_phase_only | 7 | 0.811451 | 0.037605 | 0.630152 | 0.044081 | 0.630152 | 0.044081 | 0.798584 |

## Selected Epoch Records

| run | source_val_domain | selected_epoch | source_val_macro_f1 | target_macro_f1 | worst_domain_macro_f1 | domain8_macro_f1 | metrics_path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| wicbr_carat_lodo | 9 | 30 | 0.893192 | 0.855332 | 0.542882 | 0.542882 | results/widar3_domain8_focus_lodo_d9/wicbr_carat_lodo_metrics.json |
| wicbr_carat_lodo | 10 | 6 | 0.279070 | 0.811425 | 0.359279 | 0.359279 | results/widar3_domain8_focus_lodo_d10/wicbr_carat_lodo_metrics.json |
| wicbr_carat_lodo | 11 | 22 | 0.900744 | 0.832919 | 0.460567 | 0.460567 | results/widar3_domain8_focus_lodo_d11/wicbr_carat_lodo_metrics.json |
| wicbr_carat_lodo | 12 | 27 | 0.966636 | 0.840417 | 0.476018 | 0.476018 | results/widar3_domain8_focus_lodo_d12/wicbr_carat_lodo_metrics.json |
| wicbr_carat_lodo | 13 | 18 | 0.836006 | 0.813606 | 0.377230 | 0.377230 | results/widar3_domain8_focus_lodo_d13/wicbr_carat_lodo_metrics.json |
| wicbr_carat_lodo | 14 | 26 | 0.885234 | 0.848759 | 0.480244 | 0.480244 | results/widar3_domain8_focus_lodo_d14/wicbr_carat_lodo_metrics.json |
| wicbr_carat_lodo | 15 | 26 | 0.916506 | 0.821090 | 0.488982 | 0.488982 | results/widar3_domain8_focus_lodo_d15/wicbr_carat_lodo_metrics.json |
| wicbr_carat_v2_lodo | 9 | 26 | 0.907715 | 0.813226 | 0.396138 | 0.396138 | results/widar3_domain8_focus_lodo_d9/wicbr_carat_v2_lodo_metrics.json |
| wicbr_carat_v2_lodo | 10 | 12 | 0.294643 | 0.844832 | 0.505121 | 0.505121 | results/widar3_domain8_focus_lodo_d10/wicbr_carat_v2_lodo_metrics.json |
| wicbr_carat_v2_lodo | 11 | 22 | 0.925478 | 0.813210 | 0.437702 | 0.437702 | results/widar3_domain8_focus_lodo_d11/wicbr_carat_v2_lodo_metrics.json |
| wicbr_carat_v2_lodo | 12 | 30 | 0.960082 | 0.879203 | 0.603474 | 0.603474 | results/widar3_domain8_focus_lodo_d12/wicbr_carat_v2_lodo_metrics.json |
| wicbr_carat_v2_lodo | 13 | 27 | 0.860483 | 0.841082 | 0.568618 | 0.568618 | results/widar3_domain8_focus_lodo_d13/wicbr_carat_v2_lodo_metrics.json |
| wicbr_carat_v2_lodo | 14 | 21 | 0.862322 | 0.808234 | 0.446498 | 0.446498 | results/widar3_domain8_focus_lodo_d14/wicbr_carat_v2_lodo_metrics.json |
| wicbr_carat_v2_lodo | 15 | 27 | 0.944736 | 0.881850 | 0.544333 | 0.544333 | results/widar3_domain8_focus_lodo_d15/wicbr_carat_v2_lodo_metrics.json |
| wicbr_lodo_domain10 | 10 | 25 | 0.322314 | 0.699454 | 0.418532 | 0.418532 | results/widar3_wicbr_lodo/wicbr_lodo_domain10_metrics.json |
| wicbr_lodo_domain11 | 11 | 30 | 0.948518 | 0.854712 | 0.563628 | 0.563628 | results/widar3_wicbr_lodo/wicbr_lodo_domain11_metrics.json |
| wicbr_lodo_domain12 | 12 | 25 | 0.957309 | 0.847244 | 0.429412 | 0.429412 | results/widar3_wicbr_lodo/wicbr_lodo_domain12_metrics.json |
| wicbr_lodo_domain13 | 13 | 25 | 0.857602 | 0.840528 | 0.415909 | 0.415909 | results/widar3_wicbr_lodo/wicbr_lodo_domain13_metrics.json |
| wicbr_lodo_domain14 | 14 | 27 | 0.853893 | 0.812792 | 0.433256 | 0.433256 | results/widar3_wicbr_lodo/wicbr_lodo_domain14_metrics.json |
| wicbr_lodo_domain15 | 15 | 28 | 0.933115 | 0.862070 | 0.519085 | 0.519085 | results/widar3_wicbr_lodo/wicbr_lodo_domain15_metrics.json |
| wicbr_lodo_domain9 | 9 | 26 | 0.918670 | 0.828972 | 0.495899 | 0.495899 | results/widar3_wicbr_lodo/wicbr_lodo_domain9_metrics.json |
| wicbr_lodo_full | 9 | 26 | 0.918670 | 0.828972 | 0.495899 | 0.495899 | results/widar3_domain8_focus_lodo_d9/wicbr_lodo_full_metrics.json |
| wicbr_lodo_full | 10 | 25 | 0.322314 | 0.699454 | 0.418532 | 0.418532 | results/widar3_domain8_focus_lodo_d10/wicbr_lodo_full_metrics.json |
| wicbr_lodo_full | 11 | 30 | 0.948518 | 0.854712 | 0.563628 | 0.563628 | results/widar3_domain8_focus_lodo_d11/wicbr_lodo_full_metrics.json |
| wicbr_lodo_full | 12 | 25 | 0.957309 | 0.847244 | 0.429412 | 0.429412 | results/widar3_domain8_focus_lodo_d12/wicbr_lodo_full_metrics.json |
| wicbr_lodo_full | 13 | 25 | 0.857602 | 0.840528 | 0.415909 | 0.415909 | results/widar3_domain8_focus_lodo_d13/wicbr_lodo_full_metrics.json |
| wicbr_lodo_full | 14 | 27 | 0.853893 | 0.812792 | 0.433256 | 0.433256 | results/widar3_domain8_focus_lodo_d14/wicbr_lodo_full_metrics.json |
| wicbr_lodo_full | 15 | 28 | 0.933115 | 0.862070 | 0.519085 | 0.519085 | results/widar3_domain8_focus_lodo_d15/wicbr_lodo_full_metrics.json |
| wicbr_lodo_full | 15 | 28 | 0.933115 | 0.862070 | 0.519085 | 0.519085 | results/widar3_wicbr_lodo/wicbr_lodo_full_metrics.json |
| wicbr_lodo_no_fusion | 9 | 21 | 0.915815 | 0.836476 | 0.504739 | 0.504739 | results/widar3_domain8_focus_lodo_d9/wicbr_lodo_no_fusion_metrics.json |
| wicbr_lodo_no_fusion | 10 | 25 | 0.275430 | 0.774431 | 0.396011 | 0.396011 | results/widar3_domain8_focus_lodo_d10/wicbr_lodo_no_fusion_metrics.json |
| wicbr_lodo_no_fusion | 11 | 30 | 0.876900 | 0.829200 | 0.570669 | 0.570669 | results/widar3_domain8_focus_lodo_d11/wicbr_lodo_no_fusion_metrics.json |
| wicbr_lodo_no_fusion | 12 | 28 | 0.950932 | 0.762544 | 0.366123 | 0.366123 | results/widar3_domain8_focus_lodo_d12/wicbr_lodo_no_fusion_metrics.json |
| wicbr_lodo_no_fusion | 13 | 22 | 0.845643 | 0.801607 | 0.439301 | 0.439301 | results/widar3_domain8_focus_lodo_d13/wicbr_lodo_no_fusion_metrics.json |
| wicbr_lodo_no_fusion | 14 | 28 | 0.864936 | 0.845012 | 0.516559 | 0.516559 | results/widar3_domain8_focus_lodo_d14/wicbr_lodo_no_fusion_metrics.json |
| wicbr_lodo_no_fusion | 15 | 29 | 0.923506 | 0.882508 | 0.627125 | 0.627125 | results/widar3_domain8_focus_lodo_d15/wicbr_lodo_no_fusion_metrics.json |
| wicbr_lodo_phase_only | 9 | 29 | 0.901282 | 0.814922 | 0.673316 | 0.673316 | results/widar3_domain8_focus_lodo_d9/wicbr_lodo_phase_only_metrics.json |
| wicbr_lodo_phase_only | 10 | 16 | 0.282642 | 0.762146 | 0.628029 | 0.628029 | results/widar3_domain8_focus_lodo_d10/wicbr_lodo_phase_only_metrics.json |
| wicbr_lodo_phase_only | 11 | 24 | 0.836582 | 0.828219 | 0.614878 | 0.614878 | results/widar3_domain8_focus_lodo_d11/wicbr_lodo_phase_only_metrics.json |
| wicbr_lodo_phase_only | 12 | 30 | 0.913591 | 0.861304 | 0.701325 | 0.701325 | results/widar3_domain8_focus_lodo_d12/wicbr_lodo_phase_only_metrics.json |
| wicbr_lodo_phase_only | 13 | 29 | 0.873365 | 0.753696 | 0.556427 | 0.556427 | results/widar3_domain8_focus_lodo_d13/wicbr_lodo_phase_only_metrics.json |
| wicbr_lodo_phase_only | 14 | 28 | 0.862546 | 0.811629 | 0.599430 | 0.599430 | results/widar3_domain8_focus_lodo_d14/wicbr_lodo_phase_only_metrics.json |
| wicbr_lodo_phase_only | 15 | 28 | 0.920081 | 0.848242 | 0.637656 | 0.637656 | results/widar3_domain8_focus_lodo_d15/wicbr_lodo_phase_only_metrics.json |
