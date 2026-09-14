# WP16-TCLSC-L2 tables（126 实例，7 折 LODO cls_only）

读数码 **L2-FLAT**

| 臂 | 配对 acc | top-10 内 acc | Spearman | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|
| gin_full_mean | 0.756 | 0.495 | 0.411 | 0.072 | 0.030 |
| gin_none_mean | 0.769 | 0.487 | 0.416 | 0.073 | 0.032 |
| gin_pipe_hub | 0.761 | 0.499 | 0.420 | 0.080 | 0.029 |
| gin_pipe_hub_noscalar | 0.764 | 0.507 | 0.426 | 0.081 | 0.031 |
| gin_pipe_mean | 0.763 | 0.515 | 0.415 | 0.069 | 0.031 |
| mlp_binned | 0.758 | 0.528 | 0.402 | 0.066 | 0.032 |

## 逐折配对 acc

| 折 | gin_full_mean | gin_none_mean | gin_pipe_hub | gin_pipe_hub_noscalar | gin_pipe_mean | mlp_binned |
|---|---:|---:|---:|---:|---:|---:|
| Cricket | 0.821 | 0.842 | 0.809 | 0.806 | 0.827 | 0.820 |
| EthanolConcentration | 0.515 | 0.504 | 0.530 | 0.529 | 0.505 | 0.519 |
| HandMovementDirection | 0.738 | 0.761 | 0.733 | 0.747 | 0.790 | 0.758 |
| Handwriting | 0.746 | 0.796 | 0.771 | 0.776 | 0.782 | 0.783 |
| PhonemeSpectra | 0.782 | 0.795 | 0.784 | 0.782 | 0.789 | 0.784 |
| UWaveGestureLibrary | 0.883 | 0.821 | 0.825 | 0.837 | 0.811 | 0.786 |
| UniMiB-SHAR | 0.792 | 0.851 | 0.857 | 0.857 | 0.829 | 0.843 |

## 门

| 门 | 值 | 过 |
|---|---|---|
| G_L2a | {"delta": 0.0031149741730637226, "n_folds_better": 4, "min": 0.02, "min_folds": 5} | False |
| G_L2b | {"delta": 0.004711054617163368, "n_folds_better": 4, "min": 0.01, "min_folds": 5} | False |

C0: {"mlp_binned": 0.7579443785612212, "expected": 0.758, "gin_none_mean": 0.769370052452315, "pass": true}

## 配对检验（Wilcoxon，H1：后者更高）

- mlp_binned → gin_pipe_hub: n=121 p=0.20656782493729248 胜/负/平 58/47/16 Δ̄=0.0031149741730638987
- mlp_binned → gin_pipe_mean: n=121 p=0.17974366621679694 胜/负/平 55/47/19 Δ̄=0.005180129870413405
- mlp_binned → gin_full_mean: n=121 p=0.7515339273238955 胜/负/平 47/58/16 Δ̄=-0.00159608044409951
- mlp_binned → gin_none_mean: n=121 p=0.025523534823405844 胜/负/平 57/42/22 Δ̄=0.011425673891093948
- mlp_binned → gin_pipe_hub_noscalar: n=121 p=0.05643576667544198 胜/负/平 60/41/20 Δ̄=0.006498338409463982
- gin_full_mean → gin_pipe_hub: n=121 p=0.21102430246525783 胜/负/平 52/49/20 Δ̄=0.004711054617163407
- gin_pipe_mean → gin_pipe_hub: n=121 p=0.5045278609820947 胜/负/平 47/57/17 Δ̄=-0.0020651556973495045
- gin_pipe_hub_noscalar → gin_pipe_hub: n=121 p=0.949619389758023 胜/负/平 33/49/39 Δ̄=-0.0033833642364000834
- gin_none_mean → gin_pipe_hub: n=121 p=0.8468721865321609 胜/负/平 45/59/17 Δ̄=-0.00831069971803005

## 预测对账

- P1 C0 过: True
- P2 G-L2a 过: False
- P3 G-L2b 过（条件于 P2）: None
- P4 hub ≥ mean（pipe）: False（[0.7610593527342849, 0.7631245084316346]）
- P5 标量序数边际 ≥ +0.005: False（-0.003383364236400177）
- P6 任一 GIN 臂 top-10 acc ≥ 0.55: False（0.5154092338689862）
