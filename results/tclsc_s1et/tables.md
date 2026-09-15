# WP16 T 检验 tables（转导型 E_T，field 头）

读数码 **ET-EQUIV+CONTENT**

## UEA n=7（126 实例）

| 臂 | M1 | M2 | M0 | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|
| blind_diff_hard | 0.499 | 0.731 | 0.756 | 0.0657 | 0.0294 |
| fw_eaphi_hard | 0.492 | 0.754 | 0.788 | 0.0758 | 0.0269 |
| fwb_hard | 0.516 | 0.726 | 0.754 | 0.0669 | 0.0304 |
| ft60_hard | 0.562 | 0.753 | 0.788 | 0.0679 | 0.0207 |
| ft290_hard | 0.500 | 0.745 | 0.780 | 0.0743 | 0.0221 |
| ftea_hard | 0.546 | 0.752 | 0.788 | 0.0760 | 0.0296 |

### 价值（配对 Wilcoxon，H1 b < a）

- regret@3 blind_diff_hard → ft60_hard: n=126 0.0294 → 0.0207 p=0.00108 胜/负/平 51/24/51
- regret@3 fw_eaphi_hard → ft60_hard: n=126 0.0269 → 0.0207 p=0.0536 胜/负/平 29/15/82
- regret@3 fwb_hard → ft60_hard: n=126 0.0304 → 0.0207 p=0.000885 胜/负/平 52/23/51
- regret@3 blind_diff_hard → ft290_hard: n=126 0.0294 → 0.0221 p=0.00155 胜/负/平 47/24/55
- regret@3 fw_eaphi_hard → ft290_hard: n=126 0.0269 → 0.0221 p=0.181 胜/负/平 26/20/80
- regret@3 fwb_hard → ft290_hard: n=126 0.0304 → 0.0221 p=0.00101 胜/负/平 48/23/55
- regret@3 blind_diff_hard → ftea_hard: n=126 0.0294 → 0.0296 p=0.318 胜/负/平 41/30/55
- regret@3 fw_eaphi_hard → ftea_hard: n=126 0.0269 → 0.0296 p=0.928 胜/负/平 18/22/86
- regret@3 fwb_hard → ftea_hard: n=126 0.0304 → 0.0296 p=0.245 胜/负/平 42/31/53
- regret@3 ft60_hard → ft290_hard: n=126 0.0207 → 0.0221 p=0.702 胜/负/平 16/22/88
- regret@3 ft60_hard → ftea_hard: n=126 0.0207 → 0.0296 p=0.995 胜/负/平 13/29/84
- regret@1 blind_diff_hard → ft60_hard: n=126 0.0657 → 0.0679 p=0.557 胜/负/平 39/33/54
- regret@1 fw_eaphi_hard → ft60_hard: n=126 0.0758 → 0.0679 p=0.0229 胜/负/平 29/16/81
- regret@1 fwb_hard → ft60_hard: n=126 0.0669 → 0.0679 p=0.537 胜/负/平 41/37/48
- regret@1 blind_diff_hard → ft290_hard: n=126 0.0657 → 0.0743 p=0.893 胜/负/平 39/36/51
- regret@1 fw_eaphi_hard → ft290_hard: n=126 0.0758 → 0.0743 p=0.181 胜/负/平 33/22/71
- regret@1 fwb_hard → ft290_hard: n=126 0.0669 → 0.0743 p=0.836 胜/负/平 40/39/47
- regret@1 blind_diff_hard → ftea_hard: n=126 0.0657 → 0.0760 p=0.938 胜/负/平 35/45/46
- regret@1 fw_eaphi_hard → ftea_hard: n=126 0.0758 → 0.0760 p=0.423 胜/负/平 22/26/78
- regret@1 fwb_hard → ftea_hard: n=126 0.0669 → 0.0760 p=0.889 胜/负/平 36/44/46
- regret@1 ft60_hard → ft290_hard: n=126 0.0679 → 0.0743 p=0.884 胜/负/平 23/26/77
- regret@1 ft60_hard → ftea_hard: n=126 0.0679 → 0.0760 p=0.973 胜/负/平 13/27/86

### 消融下的 regret（H1 消融更差）

- ft60_hard regret@1 cross: 0.0679 → 0.0752 p=0.0173 更差/更好/平 74/45/7
- ft60_hard regret@1 zero: 0.0679 → 0.0715 p=0.149 更差/更好/平 18/13/95
- ft60_hard regret@3 cross: 0.0207 → 0.0338 p=1.67e-07 更差/更好/平 72/24/30
- ft60_hard regret@3 zero: 0.0207 → 0.0321 p=0.000357 更差/更好/平 38/13/75
- ft290_hard regret@1 cross: 0.0743 → 0.0764 p=0.364 更差/更好/平 62/54/10
- ft290_hard regret@1 zero: 0.0743 → 0.0688 p=0.869 更差/更好/平 20/23/83
- ft290_hard regret@3 cross: 0.0221 → 0.0320 p=9.65e-06 更差/更好/平 74/28/24
- ft290_hard regret@3 zero: 0.0221 → 0.0310 p=0.00315 更差/更好/平 31/15/80
- ftea_hard regret@1 cross: 0.0760 → 0.0712 p=0.66 更差/更好/平 60/57/9
- ftea_hard regret@1 zero: 0.0760 → 0.0730 p=0.733 更差/更好/平 17/22/87
- ftea_hard regret@3 cross: 0.0296 → 0.0356 p=0.000881 更差/更好/平 73/36/17
- ftea_hard regret@3 zero: 0.0296 → 0.0329 p=0.0666 更差/更好/平 23/14/89

### 逐折 regret@3

- blind_diff_hard: Cricket 0.0147, EthanolConcentration 0.0434, HandMovementDirection 0.0302, Handwriting 0.0516, PhonemeSpectra 0.0302, UWaveGestureLibrary 0.0181, UniMiB-SHAR 0.0177
- fw_eaphi_hard: Cricket 0.0080, EthanolConcentration 0.0321, HandMovementDirection 0.0370, Handwriting 0.0604, PhonemeSpectra 0.0231, UWaveGestureLibrary 0.0099, UniMiB-SHAR 0.0181
- fwb_hard: Cricket 0.0183, EthanolConcentration 0.0459, HandMovementDirection 0.0300, Handwriting 0.0500, PhonemeSpectra 0.0340, UWaveGestureLibrary 0.0145, UniMiB-SHAR 0.0199
- ft60_hard: Cricket 0.0107, EthanolConcentration 0.0240, HandMovementDirection 0.0463, Handwriting 0.0091, PhonemeSpectra 0.0298, UWaveGestureLibrary 0.0086, UniMiB-SHAR 0.0163
- ft290_hard: Cricket 0.0068, EthanolConcentration 0.0307, HandMovementDirection 0.0360, Handwriting 0.0240, PhonemeSpectra 0.0284, UWaveGestureLibrary 0.0116, UniMiB-SHAR 0.0176
- ftea_hard: Cricket 0.0093, EthanolConcentration 0.0271, HandMovementDirection 0.0355, Handwriting 0.0716, PhonemeSpectra 0.0349, UWaveGestureLibrary 0.0081, UniMiB-SHAR 0.0210

## UCR n=72（252 实例）

| 臂 | M1 | M2 | M0 | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|
| u_blind_diff_hard | 0.602 | 0.752 | 0.772 | 0.0611 | 0.0220 |
| u_fw_eaphi_hard | 0.646 | 0.771 | 0.794 | 0.0385 | 0.0183 |
| u_fwb_hard | 0.617 | 0.757 | 0.775 | 0.0508 | 0.0196 |
| u_ft60_hard | 0.600 | 0.776 | 0.799 | 0.0399 | 0.0197 |
| u_ft290_hard | 0.583 | 0.768 | 0.789 | 0.0438 | 0.0210 |
| u_ftea_hard | 0.635 | 0.778 | 0.798 | 0.0358 | 0.0186 |

### 价值（配对 Wilcoxon，H1 b < a）

- regret@3 u_blind_diff_hard → u_ft60_hard: n=252 0.0220 → 0.0197 p=0.000538 胜/负/平 80/38/134
- regret@3 u_fw_eaphi_hard → u_ft60_hard: n=252 0.0183 → 0.0197 p=0.943 胜/负/平 24/38/190
- regret@3 u_fwb_hard → u_ft60_hard: n=252 0.0196 → 0.0197 p=0.03 胜/负/平 68/39/145
- regret@3 u_blind_diff_hard → u_ft290_hard: n=252 0.0220 → 0.0210 p=0.00312 胜/负/平 73/40/139
- regret@3 u_fw_eaphi_hard → u_ft290_hard: n=252 0.0183 → 0.0210 p=0.993 胜/负/平 24/43/185
- regret@3 u_fwb_hard → u_ft290_hard: n=252 0.0196 → 0.0210 p=0.183 胜/负/平 60/45/147
- regret@3 u_blind_diff_hard → u_ftea_hard: n=252 0.0220 → 0.0186 p=0.000314 胜/负/平 79/34/139
- regret@3 u_fw_eaphi_hard → u_ftea_hard: n=252 0.0183 → 0.0186 p=0.77 胜/负/平 23/27/202
- regret@3 u_fwb_hard → u_ftea_hard: n=252 0.0196 → 0.0186 p=0.0174 胜/负/平 70/36/146
- regret@3 u_ft60_hard → u_ft290_hard: n=252 0.0197 → 0.0210 p=0.855 胜/负/平 28/32/192
- regret@3 u_ft60_hard → u_ftea_hard: n=252 0.0197 → 0.0186 p=0.173 胜/负/平 32/23/197
- regret@1 u_blind_diff_hard → u_ft60_hard: n=252 0.0611 → 0.0399 p=0.00102 胜/负/平 98/68/86
- regret@1 u_fw_eaphi_hard → u_ft60_hard: n=252 0.0385 → 0.0399 p=0.615 胜/负/平 46/48/158
- regret@1 u_fwb_hard → u_ft60_hard: n=252 0.0508 → 0.0399 p=0.0216 胜/负/平 89/66/97
- regret@1 u_blind_diff_hard → u_ft290_hard: n=252 0.0611 → 0.0438 p=0.00326 胜/负/平 93/68/91
- regret@1 u_fw_eaphi_hard → u_ft290_hard: n=252 0.0385 → 0.0438 p=0.68 胜/负/平 60/58/134
- regret@1 u_fwb_hard → u_ft290_hard: n=252 0.0508 → 0.0438 p=0.0703 胜/负/平 88/73/91
- regret@1 u_blind_diff_hard → u_ftea_hard: n=252 0.0611 → 0.0358 p=0.00187 胜/负/平 91/70/91
- regret@1 u_fw_eaphi_hard → u_ftea_hard: n=252 0.0385 → 0.0358 p=0.533 胜/负/平 39/38/175
- regret@1 u_fwb_hard → u_ftea_hard: n=252 0.0508 → 0.0358 p=0.00492 胜/负/平 93/68/91
- regret@1 u_ft60_hard → u_ft290_hard: n=252 0.0399 → 0.0438 p=0.595 胜/负/平 47/40/165
- regret@1 u_ft60_hard → u_ftea_hard: n=252 0.0399 → 0.0358 p=0.26 胜/负/平 39/31/182

### 消融下的 regret（H1 消融更差）

- u_ft60_hard regret@1 cross: 0.0399 → 0.0447 p=0.0158 更差/更好/平 125/82/45
- u_ft60_hard regret@1 zero: 0.0399 → 0.0373 p=0.846 更差/更好/平 61/79/112
- u_ft60_hard regret@3 cross: 0.0197 → 0.0224 p=0.000137 更差/更好/平 110/54/88
- u_ft60_hard regret@3 zero: 0.0197 → 0.0197 p=0.354 更差/更好/平 50/45/157
- u_ft290_hard regret@1 cross: 0.0438 → 0.0480 p=0.00446 更差/更好/平 122/86/44
- u_ft290_hard regret@1 zero: 0.0438 → 0.0422 p=0.717 更差/更好/平 65/77/110
- u_ft290_hard regret@3 cross: 0.0210 → 0.0215 p=0.00221 更差/更好/平 99/61/92
- u_ft290_hard regret@3 zero: 0.0210 → 0.0186 p=0.872 更差/更好/平 47/53/152
- u_ftea_hard regret@1 cross: 0.0358 → 0.0474 p=7.42e-05 更差/更好/平 130/71/51
- u_ftea_hard regret@1 zero: 0.0358 → 0.0392 p=0.351 更差/更好/平 64/67/121
- u_ftea_hard regret@3 cross: 0.0186 → 0.0221 p=8.2e-06 更差/更好/平 118/57/77
- u_ftea_hard regret@3 zero: 0.0186 → 0.0187 p=0.395 更差/更好/平 49/40/163

### 逐折 regret@3

- u_blind_diff_hard: fold0 0.0317, fold1 0.0151, fold2 0.0163, fold3 0.0147, fold4 0.0163, fold5 0.0430, fold6 0.0171
- u_fw_eaphi_hard: fold0 0.0275, fold1 0.0186, fold2 0.0092, fold3 0.0090, fold4 0.0186, fold5 0.0324, fold6 0.0126
- u_fwb_hard: fold0 0.0218, fold1 0.0165, fold2 0.0155, fold3 0.0165, fold4 0.0153, fold5 0.0372, fold6 0.0144
- u_ft60_hard: fold0 0.0262, fold1 0.0138, fold2 0.0085, fold3 0.0162, fold4 0.0188, fold5 0.0371, fold6 0.0172
- u_ft290_hard: fold0 0.0244, fold1 0.0214, fold2 0.0124, fold3 0.0115, fold4 0.0155, fold5 0.0411, fold6 0.0210
- u_ftea_hard: fold0 0.0276, fold1 0.0198, fold2 0.0113, fold3 0.0125, fold4 0.0171, fold5 0.0319, fold6 0.0100

## 特征诊断（UEA 45 切片）：@60 vs @290 的 Spearman、@60 vs 原始空间同名列

| 列 | ρ(60,290) | ρ(60,raw) |
|---|---:|---:|
| rep_jitter | +0.44 | +0.20 |
| rep_rescale | +0.25 | +0.14 |
| rep_point_mask | +0.70 | +0.31 |
| rep_freq_mask | +0.94 | +0.78 |
| rep_polarity | +0.87 | +0.82 |
| rep_time_reverse | +0.77 | +0.58 |
| rep_inst_norm | +0.37 | +0.26 |
| rep_detrend | +0.10 | +0.24 |
| rep_highpass05 | +0.70 | +0.62 |
| rep_lowpass25 | -0.18 | +0.05 |
| rep_diff1 | +0.49 | +0.10 |
| rcos_jitter | +0.90 | +0.31 |
| rcos_rescale | +0.93 | +0.44 |
| rcos_point_mask | +0.90 | +0.24 |
| rcos_freq_mask | +0.90 | +0.37 |
| rcos_time_reverse | +0.47 | -0.01 |
| rcos_inst_norm | +0.96 | +0.79 |
| rcos_detrend | +0.94 | +0.87 |
| rcos_highpass05 | +0.93 | +0.76 |
| rcos_lowpass25 | +0.92 | +0.87 |
| rcos_diff1 | +0.89 | +0.70 |
| rinfo | +0.94 | — |

## 门

| 门 | 值 | 过 |
|---|---|---|
| GT1 | {"metric": "regret3", "fw": 0.026948913578002193, "ft60": 0.020688755454388367, "p": 0.053639856214967266} | False |
| GT2 | {"metric": "regret3", "blind": 0.029427115528154295, "ft60": 0.020688755454388367, "p": 0.0010771812892111043} | True |
| GT3 | {"metric": "regret3", "normal": 0.020688755454388367, "cross": 0.03383965611772463, "p": 1.667963239714292e-07} | True |
| C0 blind_diff_hard | ref 0.0294 got 0.029427115528154295 | True |
| C0 fw_eaphi_hard | ref 0.0269 got 0.026948913578002193 | True |

## 预测对账

- P1 C0: True
- P2 G-T1 过: False
- P3 G-T2 过: True
- P4 G-T3 过: True
- P5 UEA ft290 < ft60: False（[0.020688755454388367, 0.02213814424451346]）
- P6 UEA ftea < min(ft60, fw): False（[0.029635731689735427, 0.020688755454388367, 0.026948913578002193]）
- P7 UCR u_ft60 < u_blind（p<0.05）: True（[0.02202857906502077, 0.019693229220338934, 0.0005384938195568516]）
