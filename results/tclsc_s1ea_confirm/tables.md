# WP16-TCLSC 阶段 1 A′ 确认版 tables

读数码 **EA-NOT-CONFIRMED+UCR-REP**

## UEA 8 集 7 折 LODO（主）（126 实例）

| 臂 | M1 | M2 | M0 | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|
| blind_diff_hard | 0.499 | 0.731 | 0.756 | 0.0657 | 0.0294 |
| phi_diff_hard | 0.499 | 0.748 | 0.780 | 0.0771 | 0.0311 |
| eaphi_diff_hard | 0.521 | 0.756 | 0.793 | 0.0791 | 0.0354 |
| ea_diff_hard | 0.504 | 0.752 | 0.780 | 0.0807 | 0.0314 |

### 价值（配对 Wilcoxon，H1 b < a）

- regret@3 phi_diff_hard → eaphi_diff_hard: n=126 0.0311 → 0.0354 p=0.983 胜/负/平 9/22/95
- regret@3 blind_diff_hard → eaphi_diff_hard: n=126 0.0294 → 0.0354 p=0.977 胜/负/平 20/31/75
- regret@3 phi_diff_hard → ea_diff_hard: n=126 0.0311 → 0.0314 p=0.812 胜/负/平 14/25/87
- regret@3 blind_diff_hard → ea_diff_hard: n=126 0.0294 → 0.0314 p=0.745 胜/负/平 26/28/72
- regret@1 phi_diff_hard → eaphi_diff_hard: n=126 0.0771 → 0.0791 p=0.619 胜/负/平 16/13/97
- regret@1 blind_diff_hard → eaphi_diff_hard: n=126 0.0657 → 0.0791 p=0.999 胜/负/平 16/27/83
- regret@1 phi_diff_hard → ea_diff_hard: n=126 0.0771 → 0.0807 p=0.839 胜/负/平 16/18/92
- regret@1 blind_diff_hard → ea_diff_hard: n=126 0.0657 → 0.0807 p=0.999 胜/负/平 17/35/74

### 消融下的 regret（H1 消融更差）

- eaphi_diff_hard regret@1 cross: 0.0791 → 0.0751 p=0.876 更差/更好/平 37/43/46
- eaphi_diff_hard regret@1 zero: 0.0791 → 0.0718 p=0.984 更差/更好/平 2/7/117
- eaphi_diff_hard regret@3 cross: 0.0354 → 0.0351 p=0.428 更差/更好/平 40/33/53
- eaphi_diff_hard regret@3 zero: 0.0354 → 0.0332 p=0.975 更差/更好/平 4/13/109
- ea_diff_hard regret@1 cross: 0.0807 → 0.0816 p=0.69 更差/更好/平 34/42/50
- ea_diff_hard regret@1 zero: 0.0807 → 0.0727 p=0.952 更差/更好/平 11/20/95
- ea_diff_hard regret@3 cross: 0.0314 → 0.0353 p=0.132 更差/更好/平 49/39/38
- ea_diff_hard regret@3 zero: 0.0314 → 0.0295 p=0.823 更差/更好/平 16/23/87

### 消融 Δ（配对精度）

| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|---|
| eaphi_diff_hard | cross | acc_top10 | 0.521 | 0.497 | +0.025 | 0.0835 | 17/14/20 |
| eaphi_diff_hard | cross | wacc | 0.756 | 0.728 | +0.029 | 2.25e-13 | 99/27/0 |
| eaphi_diff_hard | cross | pair_acc | 0.793 | 0.753 | +0.040 | 4.71e-14 | 95/19/7 |
| ea_diff_hard | cross | acc_top10 | 0.504 | 0.488 | +0.017 | 0.18 | 18/16/17 |
| ea_diff_hard | cross | wacc | 0.752 | 0.723 | +0.029 | 1.22e-11 | 95/31/0 |
| ea_diff_hard | cross | pair_acc | 0.780 | 0.744 | +0.036 | 2.16e-09 | 80/28/13 |
| eaphi_diff_hard | zero | acc_top10 | 0.521 | 0.526 | -0.005 | 0.232 | 4/2/45 |
| eaphi_diff_hard | zero | wacc | 0.756 | 0.749 | +0.008 | 4.77e-05 | 75/46/5 |
| eaphi_diff_hard | zero | pair_acc | 0.793 | 0.785 | +0.008 | 0.00254 | 44/24/53 |
| ea_diff_hard | zero | acc_top10 | 0.504 | 0.504 | +0.000 | 0.378 | 7/4/40 |
| ea_diff_hard | zero | wacc | 0.752 | 0.741 | +0.011 | 0.00016 | 74/51/1 |
| ea_diff_hard | zero | pair_acc | 0.780 | 0.773 | +0.008 | 0.000306 | 58/27/36 |

### 逐折 regret@3

- blind_diff_hard: Cricket 0.0147, EthanolConcentration 0.0434, HandMovementDirection 0.0302, Handwriting 0.0516, PhonemeSpectra 0.0302, UWaveGestureLibrary 0.0181, UniMiB-SHAR 0.0177
- phi_diff_hard: Cricket 0.0072, EthanolConcentration 0.0370, HandMovementDirection 0.0281, Handwriting 0.0730, PhonemeSpectra 0.0424, UWaveGestureLibrary 0.0126, UniMiB-SHAR 0.0172
- eaphi_diff_hard: Cricket 0.0134, EthanolConcentration 0.0463, HandMovementDirection 0.0348, Handwriting 0.0762, PhonemeSpectra 0.0441, UWaveGestureLibrary 0.0117, UniMiB-SHAR 0.0215
- ea_diff_hard: Cricket 0.0125, EthanolConcentration 0.0406, HandMovementDirection 0.0440, Handwriting 0.0457, PhonemeSpectra 0.0424, UWaveGestureLibrary 0.0130, UniMiB-SHAR 0.0215

## UCR 84 种子 3/4/5（副）（252 实例）

| 臂 | M1 | M2 | M0 | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|
| u_blind_diff_hard | 0.615 | 0.753 | 0.774 | 0.0670 | 0.0197 |
| u_phi_diff_hard | 0.627 | 0.749 | 0.771 | 0.0544 | 0.0223 |
| u_eaphi_diff_hard | 0.667 | 0.767 | 0.791 | 0.0426 | 0.0188 |

### 价值（配对 Wilcoxon，H1 b < a）

- regret@3 u_phi_diff_hard → u_eaphi_diff_hard: n=252 0.0223 → 0.0188 p=0.00115 胜/负/平 42/18/192
- regret@3 u_blind_diff_hard → u_eaphi_diff_hard: n=252 0.0197 → 0.0188 p=0.00266 胜/负/平 45/20/187
- regret@3 u_blind_diff_hard → u_phi_diff_hard: n=252 0.0197 → 0.0223 p=0.685 胜/负/平 30/29/193
- regret@1 u_phi_diff_hard → u_eaphi_diff_hard: n=252 0.0544 → 0.0426 p=0.0597 胜/负/平 37/32/183
- regret@1 u_blind_diff_hard → u_eaphi_diff_hard: n=252 0.0670 → 0.0426 p=0.00774 胜/负/平 39/32/181
- regret@1 u_blind_diff_hard → u_phi_diff_hard: n=252 0.0670 → 0.0544 p=0.0951 胜/负/平 29/24/199

### 消融下的 regret（H1 消融更差）

- u_eaphi_diff_hard regret@1 cross: 0.0426 → 0.0493 p=0.0141 更差/更好/平 79/62/111
- u_eaphi_diff_hard regret@1 zero: 0.0426 → 0.0776 p=9.49e-05 更差/更好/平 66/43/143
- u_eaphi_diff_hard regret@3 cross: 0.0188 → 0.0221 p=0.000136 更差/更好/平 93/49/110
- u_eaphi_diff_hard regret@3 zero: 0.0188 → 0.0184 p=0.528 更差/更好/平 44/39/169

### 消融 Δ（配对精度）

| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|---|
| u_eaphi_diff_hard | cross | acc_top10 | 0.667 | 0.606 | +0.061 | 0.00233 | 50/28/8 |
| u_eaphi_diff_hard | cross | wacc | 0.767 | 0.743 | +0.024 | 1.02e-12 | 170/74/2 |
| u_eaphi_diff_hard | cross | pair_acc | 0.791 | 0.764 | +0.027 | 4.11e-12 | 150/70/18 |
| u_eaphi_diff_hard | zero | acc_top10 | 0.667 | 0.642 | +0.024 | 0.159 | 29/23/34 |
| u_eaphi_diff_hard | zero | wacc | 0.767 | 0.744 | +0.022 | 6.11e-11 | 167/76/3 |
| u_eaphi_diff_hard | zero | pair_acc | 0.791 | 0.766 | +0.025 | 1.42e-11 | 139/56/43 |

### 逐折 regret@3

- u_blind_diff_hard: fold0 0.0078, fold1 0.0219, fold2 0.0159, fold3 0.0161, fold4 0.0177, fold5 0.0435, fold6 0.0153
- u_phi_diff_hard: fold0 0.0232, fold1 0.0241, fold2 0.0126, fold3 0.0180, fold4 0.0192, fold5 0.0446, fold6 0.0147
- u_eaphi_diff_hard: fold0 0.0233, fold1 0.0164, fold2 0.0129, fold3 0.0103, fold4 0.0194, fold5 0.0358, fold6 0.0135

## 门

| 门 | 值 | 过 |
|---|---|---|
| GC1 | {"metric": "regret3", "phi": 0.03108100375369745, "eaphi": 0.03542239641158655, "p": 0.9828565304402509} | False |
| GC2 | {"metric": "regret3", "normal": 0.03542239641158655, "cross": 0.03505731678026622, "p": 0.4280181405226207} | False |
| GC3 | {"metric": "regret3", "phi": 0.022338387764348586, "eaphi": 0.018806890470017084, "p": 0.001152590112039555} | True |
| C0 blind_diff_hard | ref 0.7557259601443286 got 0.7557259601443286 | True |
| C0 phi_diff_hard | ref 0.7804929708961402 got 0.7804929708961402 | True |

## 预测对账

- P1 C0 逐位: True
- P2 G-C1 过: False
- P3 G-C2 过: False
- P4 G-C3 过: True
- P5 UEA eaphi M0 ≥ φ: True（[0.7928683759126642, 0.7804929708961402]）
- P6 UEA eaphi regret@1 < φ（p<0.05）: False（[0.07711909006788457, 0.07906606033074709, 0.6189670673641543]）
- P7 UEA eaphi regret@3 < blind（p<0.05）: False（[0.029427115528154295, 0.03542239641158655, 0.9773241681303476]）
