# WP16-TCLSC 阶段 1 真 E 臂 tables（126 实例，7 折 LODO）

读数码 **E-INERT**

| 臂 | M1 好半区 acc | M2 | M0 δ 对 acc | Spearman | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|---:|
| e_concat_hard | 0.501 | 0.730 | 0.758 | 0.399 | 0.074 | 0.033 |
| e_diff_hard | 0.492 | 0.742 | 0.768 | 0.427 | 0.079 | 0.032 |
| blind_diff_hard | 0.499 | 0.731 | 0.756 | 0.406 | 0.066 | 0.029 |

## 排名口径消融 Δ（normal − cross 3 种子均值 / normal − zero）

| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|---|
| e_concat_hard | cross | acc_top10 | 0.501 | 0.508 | -0.007 | 0.901 | 4/10/37 |
| e_concat_hard | cross | wacc | 0.730 | 0.730 | +0.001 | 0.414 | 43/47/36 |
| e_concat_hard | cross | pair_acc | 0.758 | 0.758 | +0.000 | 0.484 | 39/39/43 |
| e_concat_hard | zero | acc_top10 | 0.501 | 0.509 | -0.008 | 0.827 | 1/4/46 |
| e_concat_hard | zero | wacc | 0.730 | 0.730 | +0.000 | 0.44 | 58/56/12 |
| e_concat_hard | zero | pair_acc | 0.758 | 0.758 | +0.000 | 0.343 | 34/28/59 |
| e_diff_hard | cross | acc_top10 | 0.492 | 0.489 | +0.003 | 0.884 | 2/7/42 |
| e_diff_hard | cross | wacc | 0.742 | 0.741 | +0.001 | 0.07 | 27/23/76 |
| e_diff_hard | cross | pair_acc | 0.768 | 0.766 | +0.002 | 0.0319 | 23/16/82 |
| e_diff_hard | zero | acc_top10 | 0.492 | 0.484 | +0.008 | 0.345 | 8/7/36 |
| e_diff_hard | zero | wacc | 0.742 | 0.720 | +0.022 | 1.33e-10 | 90/36/0 |
| e_diff_hard | zero | pair_acc | 0.768 | 0.740 | +0.028 | 6.77e-11 | 71/21/29 |

## 逐折 Δ_M1（cross）

- e_concat_hard: Cricke +0.000, Ethano +0.000, HandMo -0.022, Handwr -0.014, Phonem +0.011, UWaveG -0.083, UniMiB +0.000
- e_diff_hard: Cricke +0.000, Ethano -0.004, HandMo +0.000, Handwr -0.000, Phonem -0.000, UWaveG +0.042, UniMiB +0.000

## E 诊断（seed 0，逐折 / 中位）

- e_diff_hard 中位：cos cross 0.5766147244254121 / within 0.9999140758613331 / norm 162.68719071647126
  - Handwriting: cross 0.5766 within 1.0000 norm 519.371
  - UniMiB-SHAR: cross 0.5571 within 0.9986 norm 162.687
  - PhonemeSpectra: cross 0.6023 within 0.9997 norm 279.215
  - UWaveGestureLibrary: cross 1.0000 within 1.0000 norm 98.242
  - EthanolConcentration: cross 0.4324 within 0.9990 norm 80.171
  - Cricket: cross 0.5595 within 0.9999 norm 181.240
  - HandMovementDirection: cross 0.5835 within 1.0000 norm 110.481
- e_concat_hard 逐折：Handwr 0.2155, UniMiB 0.4004, Phonem 0.1745, UWaveG 0.2194, Ethano 0.4400, Cricke 0.2613, HandMo 0.5100

## 33 号 run json（δ 对口径）cross 档

- e_concat_hard: Handwr -0.0011 ambiguous/detectable/zero, UniMiB +0.0050 ambiguous/zero, Phonem -0.0044 zero, UWaveG -0.0248 ambiguous/zero, Ethano +0.0005 ambiguous/zero, Cricke +0.0097 ambiguous/detectable/pass, HandMo -0.0032 zero
- e_diff_hard: Handwr +0.0149 ambiguous/detectable, UniMiB +0.0112 detectable/zero, Phonem +0.0096 detectable, UWaveG +0.0029 ambiguous/zero, Ethano -0.0013 zero, Cricke +0.0056 detectable/zero, HandMo +0.0164 ambiguous

## 价值（regret@3 配对，H1 E 臂更低）

- blind_diff_hard → e_concat_hard: n=126 0.029427115528154295 → 0.03320023866811791 p=0.8842695993187712 胜/负/平 21/24/81
- blind_diff_hard → e_diff_hard: n=126 0.029427115528154295 → 0.03196375721990427 p=0.8366311844477087 胜/负/平 18/19/89

## 门

| 门 | 值 | 过 |
|---|---|---|
| GE0 | {"cos_cross": 0.5766147244254121, "cos_within": 0.9999140758613331} | True |
| GE1 | {"arm": "e_diff_hard", "delta": {"arm": "e_diff_hard", "metric": "acc_top10", "n": 51, "normal": 0.4921258596317784, "cross": 0.48940254808493744, "mean_delta": 0.0027233115468409527, "p": 0.8842097012512444, "wins": 2, "losses": 7, "ties": 42}} | False |
| GE2 | {"metric": "regret3", "blind": 0.029427115528154295, "e": 0.03196375721990427, "p": 0.8366311844477087} | False |
| GE3 | {"arm": "e_concat_hard", "delta": {"arm": "e_concat_hard", "metric": "acc_top10", "n": 51, "normal": 0.5008676896971379, "cross": 0.5080949080698671, "mean_delta": -0.007227218372729212, "p": 0.9011596764748095, "wins": 4, "losses": 10, "ties": 37}} | True |

C0: {"concat_M0": 0.7583223919425555, "concat_cross_tiers": [["ambiguous", "detectable", "zero"], ["ambiguous", "zero"], ["zero"], ["ambiguous", "zero"], ["ambiguous", "zero"], ["ambiguous", "detectable", "pass"], ["zero"]], "pass": false}

## 预测对账

- P1 C0 过: False
- P2 G-E0 不塌缩: True
- P3 G-E1 过: False
- P4 G-E2 过: False
- P5 e_diff_hard M0 cross Δ 显著（p<0.01，排名口径或 δ 对档位 pass）: False（[0.03189110549503491, [['ambiguous', 'detectable'], ['detectable', 'zero'], ['detectable'], ['ambiguous', 'zero'], ['zero'], ['detectable', 'zero'], ['ambiguous']]]）
- P6 e_diff_hard M0 ≥ e_concat_hard + 0.01: False（0.009662836675638942）
