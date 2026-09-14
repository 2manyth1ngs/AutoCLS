# WP16-TCLSC-L1 tables（126 实例）

读数码 **L1-FLAT**

| 臂 | 配对 acc | top-10 内 acc | Spearman | regret@1 | regret@3 | n |
|---|---:|---:|---:|---:|---:|---:|
| cls_only_binned | 0.758 | 0.528 | 0.402 | 0.066 | 0.032 | 126 |
| cls_only_pod | 0.741 | 0.434 | 0.375 | 0.081 | 0.033 | 126 |
| cls_only_scalar | 0.746 | 0.437 | 0.376 | 0.074 | 0.033 | 126 |
| main_pod | 0.735 | 0.385 | 0.351 | 0.074 | 0.038 | 126 |
| mlp_z_binned | 0.764 | 0.543 | 0.414 | 0.074 | 0.030 | 126 |
| mlp_z_scalar | 0.727 | 0.449 | 0.354 | 0.116 | 0.039 | 126 |
| random | — | — | — | 0.134 | 0.052 | — |

## 逐折配对 acc

| 折 | cls_only_binned | cls_only_pod | cls_only_scalar | main_pod | mlp_z_binned | mlp_z_scalar |
|---|---:|---:|---:|---:|---:|---:|
| Cricket | 0.820 | 0.859 | 0.855 | 0.856 | 0.830 | 0.798 |
| EthanolConcentration | 0.519 | 0.469 | 0.473 | 0.459 | 0.498 | 0.535 |
| HandMovementDirection | 0.758 | 0.692 | 0.723 | 0.784 | 0.710 | 0.657 |
| Handwriting | 0.783 | 0.729 | 0.716 | 0.720 | 0.778 | 0.722 |
| PhonemeSpectra | 0.784 | 0.739 | 0.760 | 0.648 | 0.806 | 0.773 |
| UWaveGestureLibrary | 0.786 | 0.865 | 0.855 | 0.856 | 0.863 | 0.829 |
| UniMiB-SHAR | 0.843 | 0.804 | 0.816 | 0.816 | 0.839 | 0.748 |

## 门

| 门 | 值 | 过 |
|---|---|---|
| G1 | {"delta": 0.012423429574058509, "n_folds_better": 5, "n_folds": 7, "min_delta": 0.02, "min_folds": 5} | False |
| G2 | {"delta": null, "n_folds_better": null, "note": "阶段 2 未跑（main_binned 缺席）"} | None |
| G3 | {"value": 0.7643950842580871, "min": 0.765} | False |
| G3p | {"delta": 0.037506748696069625, "min": 0.02} | True |

## 配对检验（Wilcoxon，H1：后者更高）

- cls_only_scalar → cls_only_binned: n=121 p=0.1246894418191622 胜/负/平 56/53/12 Δ̄=0.012423429574058429
- mlp_z_scalar → mlp_z_binned: n=121 p=2.8068243598530714e-05 胜/负/平 73/36/12 Δ̄=0.03750674869606957
- cls_only_binned → mlp_z_binned: n=121 p=0.07929918594987478 胜/负/平 60/45/16 Δ̄=0.006450705696865967
- cls_only_pod → cls_only_scalar: n=121 p=0.14465032142115442 胜/负/平 48/41/32 Δ̄=0.0049845158988759365

## 预测对账

- P1 C0 硬门过: True
- P2 G1 过（编码 +0.02 且 5/7）: False
- P3 G3 过（mlp_z_binned ≥ 0.765）: False
- P4 G3′ 过: True
- P5 mlp_z_binned ≥ hgb_binned 0.780: False
- P6 C0-local 在 0.02 带内: True（0.004984515898875941）
- P7 编码收益集中在 PS / Ethanol（两折 Δ 都高于 pooled Δ）: True（{'Cricket': -0.03581755308990986, 'EthanolConcentration': 0.04593401211183196, 'HandMovementDirection': 0.03550813662694774, 'Handwriting': 0.066807166011708, 'PhonemeSpectra': 0.02403138559848772, 'UWaveGestureLibrary': -0.06929589773326661, 'UniMiB-SHAR': 0.026788391423128166}）
