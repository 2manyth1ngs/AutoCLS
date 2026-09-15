# WP16-TCLSC 阶段 1 A″ 跨语料部署检验 tables（UEA 7 靶 126 实例）

读数码 **XFER-DEAD-CONTENT**

| 臂 | 训练侧 | M1 | M2 | M0 | regret@1 | regret@3 |
|---|---|---:|---:|---:|---:|---:|
| blind_diff_hard | UEA 7 折 | 0.499 | 0.731 | 0.756 | 0.0657 | 0.0294 |
| phi_diff_hard | UEA 7 折 | 0.499 | 0.748 | 0.780 | 0.0771 | 0.0311 |
| eaphi_diff_hard | UEA 7 折 | 0.521 | 0.756 | 0.793 | 0.0791 | 0.0354 |
| u_blind_diff_hard | UCR 84 | 0.486 | 0.733 | 0.749 | 0.1069 | 0.0374 |
| u_phi_diff_hard | UCR 84 | 0.508 | 0.729 | 0.744 | 0.0909 | 0.0368 |
| u_eaphi_diff_hard | UCR 84 | 0.518 | 0.741 | 0.757 | 0.0851 | 0.0352 |

## 价值（配对 Wilcoxon，H1 b < a）

- regret@3 blind_diff_hard → u_eaphi_diff_hard: n=126 0.0294 → 0.0352 p=0.755 胜/负/平 45/53/28
- regret@3 u_blind_diff_hard → u_eaphi_diff_hard: n=126 0.0374 → 0.0352 p=0.093 胜/负/平 31/19/76
- regret@3 blind_diff_hard → u_blind_diff_hard: n=126 0.0294 → 0.0374 p=0.971 胜/负/平 35/56/35
- regret@3 phi_diff_hard → u_phi_diff_hard: n=126 0.0311 → 0.0368 p=0.855 胜/负/平 45/56/25
- regret@3 blind_diff_hard → u_phi_diff_hard: n=126 0.0294 → 0.0368 p=0.942 胜/负/平 46/58/22
- regret@3 eaphi_diff_hard → u_eaphi_diff_hard: n=126 0.0354 → 0.0352 p=0.285 胜/负/平 54/46/26
- regret@3 u_phi_diff_hard → u_eaphi_diff_hard: n=126 0.0368 → 0.0352 p=0.135 胜/负/平 30/18/78
- regret@3 u_blind_diff_hard → u_phi_diff_hard: n=126 0.0374 → 0.0368 p=0.512 胜/负/平 25/28/73
- regret@1 blind_diff_hard → u_eaphi_diff_hard: n=126 0.0657 → 0.0851 p=0.989 胜/负/平 45/70/11
- regret@1 u_blind_diff_hard → u_eaphi_diff_hard: n=126 0.1069 → 0.0851 p=0.00167 胜/负/平 32/18/76
- regret@1 blind_diff_hard → u_blind_diff_hard: n=126 0.0657 → 0.1069 p=1 胜/负/平 33/78/15
- regret@1 phi_diff_hard → u_phi_diff_hard: n=126 0.0771 → 0.0909 p=0.919 胜/负/平 48/63/15
- regret@1 blind_diff_hard → u_phi_diff_hard: n=126 0.0657 → 0.0909 p=0.999 胜/负/平 41/70/15
- regret@1 eaphi_diff_hard → u_eaphi_diff_hard: n=126 0.0791 → 0.0851 p=0.655 胜/负/平 54/63/9
- regret@1 u_phi_diff_hard → u_eaphi_diff_hard: n=126 0.0909 → 0.0851 p=0.102 胜/负/平 26/23/77
- regret@1 u_blind_diff_hard → u_phi_diff_hard: n=126 0.1069 → 0.0909 p=0.0189 胜/负/平 27/16/83

## 消融下的 regret（H1 消融更差）

- u_phi_diff_hard regret@1 cross: 0.0909 → 0.1100 p=0.00233 更差/更好/平 42/20/64
- u_phi_diff_hard regret@1 zero: 0.0909 → 0.1136 p=0.000378 更差/更好/平 27/6/93
- u_phi_diff_hard regret@3 cross: 0.0368 → 0.0359 p=0.577 更差/更好/平 40/40/46
- u_phi_diff_hard regret@3 zero: 0.0368 → 0.0342 p=0.708 更差/更好/平 26/28/72
- u_eaphi_diff_hard regret@1 cross: 0.0851 → 0.0863 p=0.348 更差/更好/平 35/36/55
- u_eaphi_diff_hard regret@1 zero: 0.0851 → 0.0894 p=0.209 更差/更好/平 34/30/62
- u_eaphi_diff_hard regret@3 cross: 0.0352 → 0.0363 p=0.415 更差/更好/平 42/39/45
- u_eaphi_diff_hard regret@3 zero: 0.0352 → 0.0329 p=0.703 更差/更好/平 28/31/67

## 消融 Δ（配对精度）

| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|---|
| u_phi_diff_hard | cross | acc_top10 | 0.508 | 0.445 | +0.064 | 0.0926 | 18/16/17 |
| u_phi_diff_hard | cross | wacc | 0.729 | 0.721 | +0.008 | 0.00153 | 77/49/0 |
| u_phi_diff_hard | cross | pair_acc | 0.744 | 0.740 | +0.004 | 0.0073 | 71/44/6 |
| u_eaphi_diff_hard | cross | acc_top10 | 0.518 | 0.510 | +0.008 | 0.187 | 20/16/15 |
| u_eaphi_diff_hard | cross | wacc | 0.741 | 0.732 | +0.009 | 0.000255 | 81/45/0 |
| u_eaphi_diff_hard | cross | pair_acc | 0.757 | 0.748 | +0.009 | 0.000235 | 80/35/6 |
| u_phi_diff_hard | zero | acc_top10 | 0.508 | 0.445 | +0.063 | 0.0821 | 12/9/30 |
| u_phi_diff_hard | zero | wacc | 0.729 | 0.723 | +0.006 | 0.0104 | 75/51/0 |
| u_phi_diff_hard | zero | pair_acc | 0.744 | 0.744 | -0.000 | 0.0343 | 57/40/24 |
| u_eaphi_diff_hard | zero | acc_top10 | 0.518 | 0.472 | +0.046 | 0.0238 | 17/8/26 |
| u_eaphi_diff_hard | zero | wacc | 0.741 | 0.731 | +0.010 | 0.00107 | 78/46/2 |
| u_eaphi_diff_hard | zero | pair_acc | 0.757 | 0.757 | +0.000 | 0.015 | 61/35/25 |

## 逐靶 regret@3

- blind_diff_hard: Cricket 0.0147, EthanolConcentration 0.0434, HandMovementDirection 0.0302, Handwriting 0.0516, PhonemeSpectra 0.0302, UWaveGestureLibrary 0.0181, UniMiB-SHAR 0.0177
- phi_diff_hard: Cricket 0.0072, EthanolConcentration 0.0370, HandMovementDirection 0.0281, Handwriting 0.0730, PhonemeSpectra 0.0424, UWaveGestureLibrary 0.0126, UniMiB-SHAR 0.0172
- eaphi_diff_hard: Cricket 0.0134, EthanolConcentration 0.0463, HandMovementDirection 0.0348, Handwriting 0.0762, PhonemeSpectra 0.0441, UWaveGestureLibrary 0.0117, UniMiB-SHAR 0.0215
- u_blind_diff_hard: Cricket 0.0113, EthanolConcentration 0.0458, HandMovementDirection 0.0803, Handwriting 0.0350, PhonemeSpectra 0.0463, UWaveGestureLibrary 0.0177, UniMiB-SHAR 0.0253
- u_phi_diff_hard: Cricket 0.0124, EthanolConcentration 0.0569, HandMovementDirection 0.0858, Handwriting 0.0363, PhonemeSpectra 0.0179, UWaveGestureLibrary 0.0173, UniMiB-SHAR 0.0309
- u_eaphi_diff_hard: Cricket 0.0094, EthanolConcentration 0.0321, HandMovementDirection 0.1018, Handwriting 0.0338, PhonemeSpectra 0.0275, UWaveGestureLibrary 0.0151, UniMiB-SHAR 0.0268

## 附录：['SelfRegulationSCP2']（9 实例，无参照臂）regret@3

- u_blind_diff_hard: 0.0341
- u_phi_diff_hard: 0.0302
- u_eaphi_diff_hard: 0.0244

## 门

| 门 | 值 | 过 |
|---|---|---|
| GX1 | {"metric": "regret3", "blind_uea": 0.029427115528154295, "u_eaphi": 0.035212943503065454, "p": 0.7546682337345476} | False |
| GX2 | {"metric": "regret3", "u_blind_ucr": 0.037401212933068494, "u_eaphi": 0.035212943503065454, "p": 0.09298689350222605} | False |
| GX3 | {"metric": "regret3", "normal": 0.035212943503065454, "cross": 0.036343089581990694, "p": 0.41518382993192204} | False |
| C0 blind_diff_hard | ref 0.029427115528154295 got 0.029427115528154295 | True |
| C0 phi_diff_hard | ref 0.03108100375369745 got 0.03108100375369745 | True |
| C0 eaphi_diff_hard | ref 0.03542239641158655 got 0.03542239641158656 | True |

## 预测对账

- P1 C0: True
- P2 G-X1 过: False
- P3 G-X2 过: False
- P4 G-X3 过: False
- P5 u_blind(UCR) regret@3 ≥ blind(UEA): True（[0.029427115528154295, 0.037401212933068494]）
- P6 u_eaphi M0 ≥ 0.75: True（0.7569181944280476）
- P7 u_eaphi regret@1 < blind(UEA)（p<0.05）: False（[0.06571105634778879, 0.08508530309084135, 0.9892253420389966]）
