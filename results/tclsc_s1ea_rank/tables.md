# WP16-TCLSC 阶段 1 R 检验 tables（field 头）

读数码 **E-OR-N**

## UEA 7 折 LODO（n=7，126 实例）

| 臂 | M1 | M2 | M0 | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|
| blind_diff_hard | 0.499 | 0.731 | 0.756 | 0.0657 | 0.0294 |
| phi_diff_hard | 0.499 | 0.748 | 0.780 | 0.0771 | 0.0311 |
| eaphi_diff_hard | 0.521 | 0.756 | 0.793 | 0.0791 | 0.0354 |
| fw4_eaphi_hard | 0.485 | 0.736 | 0.766 | 0.0764 | 0.0325 |
| fw_eaphi_hard | 0.492 | 0.754 | 0.788 | 0.0758 | 0.0269 |

### 价值（配对 Wilcoxon，H1 b < a）

- regret@3 blind_diff_hard → fw4_eaphi_hard: n=126 0.0294 → 0.0325 p=0.87 胜/负/平 21/23/82
- regret@3 phi_diff_hard → fw4_eaphi_hard: n=126 0.0311 → 0.0325 p=0.872 胜/负/平 19/27/80
- regret@3 eaphi_diff_hard → fw4_eaphi_hard: n=126 0.0354 → 0.0325 p=0.203 胜/负/平 34/25/67
- regret@3 blind_diff_hard → fw_eaphi_hard: n=126 0.0294 → 0.0269 p=0.035 胜/负/平 36/21/69
- regret@3 phi_diff_hard → fw_eaphi_hard: n=126 0.0311 → 0.0269 p=0.0166 胜/负/平 27/17/82
- regret@3 eaphi_diff_hard → fw_eaphi_hard: n=126 0.0354 → 0.0269 p=6.8e-05 胜/负/平 37/14/75
- regret@3 fw_eaphi_hard → fw4_eaphi_hard: n=126 0.0269 → 0.0325 p=0.988 胜/负/平 18/30/78
- regret@1 blind_diff_hard → fw4_eaphi_hard: n=126 0.0657 → 0.0764 p=0.995 胜/负/平 15/27/84
- regret@1 phi_diff_hard → fw4_eaphi_hard: n=126 0.0771 → 0.0764 p=0.425 胜/负/平 19/17/90
- regret@1 eaphi_diff_hard → fw4_eaphi_hard: n=126 0.0791 → 0.0764 p=0.255 胜/负/平 22/16/88
- regret@1 blind_diff_hard → fw_eaphi_hard: n=126 0.0657 → 0.0758 p=0.955 胜/负/平 31/39/56
- regret@1 phi_diff_hard → fw_eaphi_hard: n=126 0.0771 → 0.0758 p=0.356 胜/负/平 34/30/62
- regret@1 eaphi_diff_hard → fw_eaphi_hard: n=126 0.0791 → 0.0758 p=0.212 胜/负/平 35/30/61
- regret@1 fw_eaphi_hard → fw4_eaphi_hard: n=126 0.0758 → 0.0764 p=0.56 胜/负/平 26/28/72

### 消融下的 regret（H1 消融更差）

- fw4_eaphi_hard regret@1 cross: 0.0764 → 0.0727 p=0.746 更差/更好/平 46/40/40
- fw4_eaphi_hard regret@1 zero: 0.0764 → 0.0727 p=0.895 更差/更好/平 8/12/106
- fw4_eaphi_hard regret@3 cross: 0.0325 → 0.0319 p=0.489 更差/更好/平 43/33/50
- fw4_eaphi_hard regret@3 zero: 0.0325 → 0.0324 p=0.644 更差/更好/平 12/14/100
- fw_eaphi_hard regret@1 cross: 0.0758 → 0.0715 p=0.77 更差/更好/平 61/59/6
- fw_eaphi_hard regret@1 zero: 0.0758 → 0.0655 p=1 更差/更好/平 7/24/95
- fw_eaphi_hard regret@3 cross: 0.0269 → 0.0321 p=0.000292 更差/更好/平 68/33/25
- fw_eaphi_hard regret@3 zero: 0.0269 → 0.0333 p=0.000461 更差/更好/平 20/8/98

### 消融 Δ（配对精度）

| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|---|
| fw4_eaphi_hard | cross | acc_top10 | 0.485 | 0.509 | -0.024 | 0.618 | 16/18/17 |
| fw4_eaphi_hard | cross | wacc | 0.736 | 0.723 | +0.013 | 2.72e-05 | 83/43/0 |
| fw4_eaphi_hard | cross | pair_acc | 0.766 | 0.750 | +0.016 | 0.000194 | 77/40/4 |
| fw_eaphi_hard | cross | acc_top10 | 0.492 | 0.503 | -0.011 | 0.569 | 21/20/10 |
| fw_eaphi_hard | cross | wacc | 0.754 | 0.706 | +0.049 | 3.43e-14 | 95/31/0 |
| fw_eaphi_hard | cross | pair_acc | 0.788 | 0.728 | +0.061 | 1.66e-13 | 93/24/4 |
| fw4_eaphi_hard | zero | acc_top10 | 0.485 | 0.495 | -0.010 | 0.352 | 9/8/34 |
| fw4_eaphi_hard | zero | wacc | 0.736 | 0.735 | +0.000 | 0.334 | 62/61/3 |
| fw4_eaphi_hard | zero | pair_acc | 0.766 | 0.765 | +0.001 | 0.391 | 43/38/40 |
| fw_eaphi_hard | zero | acc_top10 | 0.492 | 0.501 | -0.008 | 0.707 | 7/11/33 |
| fw_eaphi_hard | zero | wacc | 0.754 | 0.751 | +0.003 | 0.255 | 61/64/1 |
| fw_eaphi_hard | zero | pair_acc | 0.788 | 0.784 | +0.005 | 0.484 | 48/50/23 |

### 逐折 regret@3

- blind_diff_hard: Cricket 0.0147, EthanolConcentration 0.0434, HandMovementDirection 0.0302, Handwriting 0.0516, PhonemeSpectra 0.0302, UWaveGestureLibrary 0.0181, UniMiB-SHAR 0.0177
- phi_diff_hard: Cricket 0.0072, EthanolConcentration 0.0370, HandMovementDirection 0.0281, Handwriting 0.0730, PhonemeSpectra 0.0424, UWaveGestureLibrary 0.0126, UniMiB-SHAR 0.0172
- eaphi_diff_hard: Cricket 0.0134, EthanolConcentration 0.0463, HandMovementDirection 0.0348, Handwriting 0.0762, PhonemeSpectra 0.0441, UWaveGestureLibrary 0.0117, UniMiB-SHAR 0.0215
- fw4_eaphi_hard: Cricket 0.0100, EthanolConcentration 0.0460, HandMovementDirection 0.0323, Handwriting 0.0608, PhonemeSpectra 0.0464, UWaveGestureLibrary 0.0166, UniMiB-SHAR 0.0156
- fw_eaphi_hard: Cricket 0.0080, EthanolConcentration 0.0321, HandMovementDirection 0.0370, Handwriting 0.0604, PhonemeSpectra 0.0231, UWaveGestureLibrary 0.0099, UniMiB-SHAR 0.0181

## UCR 84 分组 7 折（n=72，252 实例）

| 臂 | M1 | M2 | M0 | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|
| u_blind_diff_hard | 0.602 | 0.752 | 0.772 | 0.0611 | 0.0220 |
| u_phi_diff_hard | 0.611 | 0.760 | 0.782 | 0.0609 | 0.0209 |
| u_eaphi_diff_hard | 0.607 | 0.771 | 0.794 | 0.0458 | 0.0202 |
| u_fw4_eaphi_hard | 0.676 | 0.776 | 0.801 | 0.0417 | 0.0189 |
| u_fw_eaphi_hard | 0.646 | 0.771 | 0.794 | 0.0385 | 0.0183 |

### 价值（配对 Wilcoxon，H1 b < a）

- regret@3 u_blind_diff_hard → u_fw4_eaphi_hard: n=252 0.0220 → 0.0189 p=0.000386 胜/负/平 61/30/161
- regret@3 u_phi_diff_hard → u_fw4_eaphi_hard: n=252 0.0209 → 0.0189 p=0.0501 胜/负/平 52/35/165
- regret@3 u_eaphi_diff_hard → u_fw4_eaphi_hard: n=252 0.0202 → 0.0189 p=0.22 胜/负/平 32/27/193
- regret@3 u_blind_diff_hard → u_fw_eaphi_hard: n=252 0.0220 → 0.0183 p=5.72e-05 胜/负/平 85/38/129
- regret@3 u_phi_diff_hard → u_fw_eaphi_hard: n=252 0.0209 → 0.0183 p=0.00614 胜/负/平 73/40/139
- regret@3 u_eaphi_diff_hard → u_fw_eaphi_hard: n=252 0.0202 → 0.0183 p=0.0251 胜/负/平 57/37/158
- regret@3 u_fw_eaphi_hard → u_fw4_eaphi_hard: n=252 0.0183 → 0.0189 p=0.869 胜/负/平 29/45/178
- regret@1 u_blind_diff_hard → u_fw4_eaphi_hard: n=252 0.0611 → 0.0417 p=0.00174 胜/负/平 62/38/152
- regret@1 u_phi_diff_hard → u_fw4_eaphi_hard: n=252 0.0609 → 0.0417 p=0.0275 胜/负/平 48/37/167
- regret@1 u_eaphi_diff_hard → u_fw4_eaphi_hard: n=252 0.0458 → 0.0417 p=0.0854 胜/负/平 46/32/174
- regret@1 u_blind_diff_hard → u_fw_eaphi_hard: n=252 0.0611 → 0.0385 p=0.000129 胜/负/平 99/65/88
- regret@1 u_phi_diff_hard → u_fw_eaphi_hard: n=252 0.0609 → 0.0385 p=0.00323 胜/负/平 79/59/114
- regret@1 u_eaphi_diff_hard → u_fw_eaphi_hard: n=252 0.0458 → 0.0385 p=0.0173 胜/负/平 74/57/121
- regret@1 u_fw_eaphi_hard → u_fw4_eaphi_hard: n=252 0.0385 → 0.0417 p=0.888 胜/负/平 63/71/118

### 消融下的 regret（H1 消融更差）

- u_fw4_eaphi_hard regret@1 cross: 0.0417 → 0.0522 p=1.01e-05 更差/更好/平 112/56/84
- u_fw4_eaphi_hard regret@1 zero: 0.0417 → 0.0437 p=0.753 更差/更好/平 28/40/184
- u_fw4_eaphi_hard regret@3 cross: 0.0189 → 0.0216 p=0.00323 更差/更好/平 91/53/108
- u_fw4_eaphi_hard regret@3 zero: 0.0189 → 0.0217 p=0.00581 更差/更好/平 46/27/179
- u_fw_eaphi_hard regret@1 cross: 0.0385 → 0.0508 p=1.8e-06 更差/更好/平 136/73/43
- u_fw_eaphi_hard regret@1 zero: 0.0385 → 0.0407 p=0.108 更差/更好/平 76/64/112
- u_fw_eaphi_hard regret@3 cross: 0.0183 → 0.0210 p=1.35e-06 更差/更好/平 118/50/84
- u_fw_eaphi_hard regret@3 zero: 0.0183 → 0.0199 p=0.0618 更差/更好/平 51/35/166

### 消融 Δ（配对精度）

| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|---|
| u_fw4_eaphi_hard | cross | acc_top10 | 0.676 | 0.596 | +0.080 | 0.000222 | 51/22/13 |
| u_fw4_eaphi_hard | cross | wacc | 0.776 | 0.746 | +0.031 | 2.5e-19 | 179/65/2 |
| u_fw4_eaphi_hard | cross | pair_acc | 0.801 | 0.761 | +0.039 | 1.63e-21 | 175/56/7 |
| u_fw_eaphi_hard | cross | acc_top10 | 0.646 | 0.588 | +0.059 | 0.0101 | 54/27/5 |
| u_fw_eaphi_hard | cross | wacc | 0.771 | 0.735 | +0.037 | 3.49e-15 | 168/75/3 |
| u_fw_eaphi_hard | cross | pair_acc | 0.794 | 0.752 | +0.042 | 1.72e-15 | 164/66/8 |
| u_fw4_eaphi_hard | zero | acc_top10 | 0.676 | 0.636 | +0.040 | 0.0685 | 32/18/36 |
| u_fw4_eaphi_hard | zero | wacc | 0.776 | 0.759 | +0.017 | 1.16e-06 | 148/95/3 |
| u_fw4_eaphi_hard | zero | pair_acc | 0.801 | 0.778 | +0.022 | 1.22e-07 | 122/64/52 |
| u_fw_eaphi_hard | zero | acc_top10 | 0.646 | 0.637 | +0.009 | 0.216 | 38/24/24 |
| u_fw_eaphi_hard | zero | wacc | 0.771 | 0.764 | +0.007 | 0.0439 | 134/109/3 |
| u_fw_eaphi_hard | zero | pair_acc | 0.794 | 0.783 | +0.010 | 0.0214 | 108/87/43 |

### 逐折 regret@3

- u_blind_diff_hard: fold0 0.0317, fold1 0.0151, fold2 0.0163, fold3 0.0147, fold4 0.0163, fold5 0.0430, fold6 0.0171
- u_phi_diff_hard: fold0 0.0234, fold1 0.0224, fold2 0.0127, fold3 0.0141, fold4 0.0173, fold5 0.0395, fold6 0.0171
- u_eaphi_diff_hard: fold0 0.0277, fold1 0.0201, fold2 0.0116, fold3 0.0123, fold4 0.0169, fold5 0.0379, fold6 0.0148
- u_fw4_eaphi_hard: fold0 0.0272, fold1 0.0114, fold2 0.0105, fold3 0.0123, fold4 0.0188, fold5 0.0388, fold6 0.0136
- u_fw_eaphi_hard: fold0 0.0275, fold1 0.0186, fold2 0.0092, fold3 0.0090, fold4 0.0186, fold5 0.0324, fold6 0.0126

## 门

| 门 | 值 | 过 |
|---|---|---|
| GR1 | {"metric": "regret3", "blind": 0.029427115528154295, "fw4": 0.0325338691882323, "p": 0.8699722302545703} | False |
| GR2 | {"metric": "regret3", "normal": 0.0325338691882323, "cross": 0.03187738469582182, "p": 0.48864115483223663} | False |
| GR3 | {"metric": "regret3", "u_blind": 0.02202857906502077, "u_fw4": 0.01894982037060755, "p": 0.00038643394558623983} | True |
| C0 uea blind_diff_hard | ref 0.029427115528154295 got 0.029427115528154295 | True |
| C0 uea phi_diff_hard | ref 0.03108100375369745 got 0.03108100375369745 | True |
| C0 uea eaphi_diff_hard | ref 0.03542239641158655 got 0.03542239641158655 | True |
| C0 ucr u_blind_diff_hard | ref 0.02202857906502077 got 0.02202857906502077 | True |
| C0 ucr u_phi_diff_hard | ref 0.020926898926290304 got 0.020926898926290304 | True |
| C0 ucr u_eaphi_diff_hard | ref 0.02017956430652682 got 0.02017956430652682 | True |

## 预测对账

- P1 C0: True
- P2 G-R1 过: False
- P3 G-R2 过: False
- P4 G-R3 过: True
- P5 UEA fw4 regret@3 < eaphi_diff（p<0.05）: False（[0.03542239641158655, 0.0325338691882323, 0.20318681556605694]）
- P6 UEA fw（不限秩）regret@3 > fw4: False（[0.026948913578002193, 0.0325338691882323]）
- P7 UCR fw4 M1 ≥ 0.60: True（0.6759364162930759）
