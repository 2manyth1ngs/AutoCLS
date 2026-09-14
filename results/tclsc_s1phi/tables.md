# WP16-TCLSC 阶段 1 φ 臂 tables（126 实例，7 折 LODO）

读数码 **PHI-DEAD**

## normal

| 臂 | M1 好半区 acc | M2 | M0 δ 对 acc | Spearman | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|---:|
| blind_diff_hard | 0.499 | 0.731 | 0.756 | 0.406 | 0.066 | 0.029 |
| phi_concat_hard | 0.528 | 0.730 | 0.759 | 0.402 | 0.078 | 0.030 |
| phi_diff_hard | 0.499 | 0.748 | 0.780 | 0.431 | 0.077 | 0.031 |
| ref L1 cls_only_binned | 0.528 | — | 0.758 | — | 0.066 | 0.032 |

## 消融 Δ（normal − cross 3 种子均值 / normal − zero；Wilcoxon H1 normal 更高）

| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|---|
| phi_concat_hard | cross | acc_top10 | 0.528 | 0.518 | +0.010 | 0.12 | 10/8/33 |
| phi_concat_hard | cross | wacc | 0.730 | 0.728 | +0.002 | 0.211 | 55/57/14 |
| phi_concat_hard | cross | pair_acc | 0.759 | 0.758 | +0.002 | 0.0331 | 50/38/33 |
| phi_concat_hard | zero | acc_top10 | 0.528 | 0.525 | +0.003 | 0.4 | 4/3/44 |
| phi_concat_hard | zero | wacc | 0.730 | 0.731 | -0.001 | 0.843 | 47/51/28 |
| phi_concat_hard | zero | pair_acc | 0.759 | 0.761 | -0.002 | 0.368 | 22/22/77 |
| phi_diff_hard | cross | acc_top10 | 0.499 | 0.488 | +0.011 | 0.117 | 18/16/17 |
| phi_diff_hard | cross | wacc | 0.748 | 0.722 | +0.026 | 9.96e-12 | 99/27/0 |
| phi_diff_hard | cross | pair_acc | 0.780 | 0.745 | +0.035 | 3.47e-11 | 92/23/6 |
| phi_diff_hard | zero | acc_top10 | 0.499 | 0.499 | +0.000 | 0.422 | 6/6/39 |
| phi_diff_hard | zero | wacc | 0.748 | 0.743 | +0.005 | 0.072 | 69/54/3 |
| phi_diff_hard | zero | pair_acc | 0.780 | 0.774 | +0.007 | 0.00989 | 56/36/29 |

## 逐折 Δ_M1（cross）

| 折 | phi_concat_hard | phi_diff_hard |
|---|---:|---:|
| Cricket | +0.000 | +0.000 |
| EthanolConcentration | +0.000 | -0.020 |
| HandMovementDirection | +0.051 | +0.060 |
| Handwriting | +0.015 | +0.060 |
| PhonemeSpectra | +0.010 | -0.027 |
| UWaveGestureLibrary | +0.000 | -0.028 |
| UniMiB-SHAR | +0.000 | +0.069 |

## 价值门（regret@3 配对，H1 φ 臂更低）

- blind_diff_hard → phi_concat_hard: n=126 0.029427115528154295 → 0.029653457655289366 p=0.43509925431841734 胜/负/平 24/20/82
- blind_diff_hard → phi_diff_hard: n=126 0.029427115528154295 → 0.03108100375369745 p=0.624567798437852 胜/负/平 21/20/85

## 门

| 门 | 值 | 过 |
|---|---|---|
| GP1 | {"arm": "phi_diff_hard", "delta": {"arm": "phi_diff_hard", "metric": "acc_top10", "n": 51, "normal": 0.49889855839136477, "cross": 0.4876878494775964, "mean_delta": 0.011210708913768474, "p": 0.11733729681562538, "wins": 18, "losses": 16, "ties": 17}} | False |
| GP2 | {"metric": "regret3", "blind": 0.029427115528154295, "phi": 0.03108100375369745, "p": 0.624567798437852} | False |
| GP3 | {"arm": "phi_concat_hard", "delta": {"arm": "phi_concat_hard", "metric": "acc_top10", "n": 51, "normal": 0.5276534039828521, "cross": 0.5176922265294147, "mean_delta": 0.009961177453437517, "p": 0.1197697730382804, "wins": 10, "losses": 8, "ties": 33}} | True |

## 预测对账

- P1 C0 带内: True
- P2 G-P1 过: False
- P3 G-P2 过: False
- P4 G-P3 零（concat 仍零）: True
- P5 逐折 Δ_M1 > 0 的折 ≥ 5/7（若 G-P1 过）: None（3）
- P6 phi_diff_zdiff regret@3 < phi_diff_hard（若跑）: None（[None, 0.03108100375369745]）
