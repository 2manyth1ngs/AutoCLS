# LC2 检验 tables：未见配置学习曲线

读数码 **LC2-FLAT**（门 {'G-L1': False, 'G-L2': False, 'G-L3': False}）

标签：{'pool_rows': 7830, 'truth_rows': 7830, 'random_configs': 870, 'splus_random_configs': 540, 'splus_share': 0.6206896551724138, 'points_rows': {'R0': 2610, 'R3': 3393, 'R6': 4065, 'RA': 4230, 'RN': 6840}, 'points_nonshared_per_slice': {'R0': 0.0, 'R3': 3.0, 'R6': 5.574712643678161, 'RA': 6.206896551724138, 'RN': 16.20689655172414}}

| 点 | 臂 | n | unseen pair acc | unseen good pair acc | unseen regret@3 | over | above_trust pred / true |
|---|---|---:|---:|---:|---:|---:|---|
| R0 | ft60_hard | 252 | 0.649 | 0.682 | 0.0204 | -0.312 | 0.11 / 0.91 |
| R0 | blind_diff_hard | 252 | 0.646 | 0.664 | 0.0254 | -0.434 | 0.01 / 0.91 |
| R3 | ft60_hard | 252 | 0.601 | 0.636 | 0.0328 | -0.070 | 1.03 / 0.91 |
| R3 | blind_diff_hard | 252 | 0.525 | 0.553 | 0.0485 | 0.224 | 3.02 / 0.91 |
| R6 | ft60_hard | 252 | 0.620 | 0.637 | 0.0374 | -0.077 | 0.95 / 0.91 |
| R6 | blind_diff_hard | 252 | 0.562 | 0.583 | 0.0564 | 0.045 | 1.49 / 0.91 |
| RA | ft60_hard | 252 | 0.619 | 0.641 | 0.0363 | -0.066 | 0.91 / 0.91 |
| RA | blind_diff_hard | 252 | 0.540 | 0.569 | 0.0524 | 0.116 | 2.04 / 0.91 |
| RN | ft60_hard | 252 | 0.696 | 0.676 | 0.0289 | -0.001 | 1.67 / 0.91 |
| RN | blind_diff_hard | 252 | 0.646 | 0.646 | 0.0325 | 0.271 | 4.55 / 0.91 |
| R0k1 | ft60_hard | 252 | 0.639 | 0.672 | 0.0199 | -0.300 | 0.27 / 0.91 |
| R0k1 | blind_diff_hard | 252 | 0.642 | 0.662 | 0.0242 | -0.434 | 0.00 / 0.91 |

## 台阶（unseen good pair acc，同实例配对，H1 = 后点更高）

| 台阶 | 臂 | n | 前 | 后 | Δ | p | 胜:负 |
|---|---|---:|---:|---:|---:|---:|---|
| R0_R3 | ft60_hard | 222 | 0.682 | 0.636 | -0.0467 | 0.998 | 71:107 |
| R0_R3 | blind_diff_hard | 222 | 0.664 | 0.553 | -0.1108 | 1 | 63:129 |
| R3_R6 | ft60_hard | 222 | 0.636 | 0.637 | +0.0009 | 0.357 | 94:80 |
| R3_R6 | blind_diff_hard | 222 | 0.553 | 0.583 | +0.0294 | 0.00626 | 103:74 |
| R6_RA | ft60_hard | 222 | 0.637 | 0.641 | +0.0044 | 0.0396 | 83:65 |
| R6_RA | blind_diff_hard | 222 | 0.583 | 0.569 | -0.0133 | 0.862 | 62:80 |
| R3_RA | ft60_hard | 222 | 0.636 | 0.641 | +0.0054 | 0.0803 | 100:76 |
| R3_RA | blind_diff_hard | 222 | 0.553 | 0.569 | +0.0160 | 0.0123 | 108:72 |
| RA_RN | ft60_hard | 222 | 0.641 | 0.676 | +0.0350 | 0.00232 | 109:71 |
| RA_RN | blind_diff_hard | 222 | 0.569 | 0.646 | +0.0763 | 1.98e-05 | 130:68 |
| R3_RN | ft60_hard | 222 | 0.636 | 0.676 | +0.0403 | 0.00113 | 117:73 |
| R3_RN | blind_diff_hard | 222 | 0.553 | 0.646 | +0.0923 | 6.33e-09 | 142:52 |
| R0k1_R0 | ft60_hard | 222 | 0.672 | 0.682 | +0.0104 | 0.168 | 91:68 |
| R0k1_R0 | blind_diff_hard | 222 | 0.662 | 0.664 | +0.0024 | 0.549 | 50:55 |
| R0k1_R3 | ft60_hard | 222 | 0.672 | 0.636 | -0.0363 | 0.997 | 77:105 |
| R0k1_R3 | blind_diff_hard | 222 | 0.662 | 0.553 | -0.1084 | 1 | 61:125 |

E_T − 盲（unseen good pair acc）：{'R0': 0.0182, 'R3': 0.0823, 'R6': 0.0538, 'RA': 0.0716, 'RN': 0.0303, 'R0k1': 0.0102}；E vs 盲配对：{'R0': (0.0182, 0.058), 'R3': (0.0823, 0.0), 'R6': (0.0538, 0.0), 'RA': (0.0716, 0.0), 'RN': (0.0303, 0.001), 'R0k1': (0.0102, 0.104)}
E_T over：{'R0': -0.312, 'R3': -0.07, 'R6': -0.077, 'RA': -0.066, 'RN': -0.001, 'R0k1': -0.3}

逐折 unseen good pair acc：

- R0: ft60_hard {Cricke 0.738, Ethano 0.612, HandMo 0.782, Handwr 0.773, Phonem 0.578, UWaveG 0.715, UniMiB 0.624}; blind_diff_hard {Cricke 0.766, Ethano 0.600, HandMo 0.834, Handwr 0.685, Phonem 0.520, UWaveG 0.701, UniMiB 0.630}
- R3: ft60_hard {Cricke 0.717, Ethano 0.531, HandMo 0.633, Handwr 0.746, Phonem 0.537, UWaveG 0.637, UniMiB 0.653}; blind_diff_hard {Cricke 0.567, Ethano 0.483, HandMo 0.687, Handwr 0.556, Phonem 0.454, UWaveG 0.608, UniMiB 0.571}
- R6: ft60_hard {Cricke 0.727, Ethano 0.490, HandMo 0.589, Handwr 0.802, Phonem 0.518, UWaveG 0.642, UniMiB 0.670}; blind_diff_hard {Cricke 0.577, Ethano 0.476, HandMo 0.693, Handwr 0.669, Phonem 0.468, UWaveG 0.606, UniMiB 0.627}
- RA: ft60_hard {Cricke 0.683, Ethano 0.504, HandMo 0.572, Handwr 0.816, Phonem 0.530, UWaveG 0.633, UniMiB 0.714}; blind_diff_hard {Cricke 0.513, Ethano 0.499, HandMo 0.649, Handwr 0.650, Phonem 0.461, UWaveG 0.603, UniMiB 0.630}
- RN: ft60_hard {Cricke 0.690, Ethano 0.493, HandMo 0.576, Handwr 0.781, Phonem 0.647, UWaveG 0.748, UniMiB 0.735}; blind_diff_hard {Cricke 0.745, Ethano 0.497, HandMo 0.669, Handwr 0.676, Phonem 0.551, UWaveG 0.699, UniMiB 0.694}
- R0k1: ft60_hard {Cricke 0.699, Ethano 0.626, HandMo 0.789, Handwr 0.743, Phonem 0.553, UWaveG 0.693, UniMiB 0.652}; blind_diff_hard {Cricke 0.761, Ethano 0.577, HandMo 0.810, Handwr 0.701, Phonem 0.511, UWaveG 0.708, UniMiB 0.637}

预测：{'P1': {'desc': 'G-L1', 'hit': False}, 'P2': {'desc': 'RN 点 E_T > 盲（p<0.05）', 'value': {'n': 222, 'mean_a': 0.6456372529951887, 'mean_b': 0.6759379710296252, 'delta': 0.03030071803443644, 'p': 0.0007286613187169211, 'wins': 109, 'losses': 74}, 'hit': True}, 'P3': {'desc': 'G-L3', 'hit': False}, 'P4': {'desc': 'RN 是 E_T 最好点', 'value': 'R0', 'hit': False}, 'P5': {'desc': 'R0 的 unseen_good_pair_acc ≤ 0.55', 'value': 0.6822958240029572, 'hit': False}}

