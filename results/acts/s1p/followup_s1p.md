# S1′ 零 GPU 后续核算（用户八条判断）

## §7 全局冠军 regret（任务感知余量的上界；标签 = shared 两种子均值套地板）

全局冠军（24 任务平均排名最好）= 2a667428c99d triplet/euclidean/layer_norm τ=0.01 aug_max=0.9；前 3 = ['2a667428', 'f64cea8e', '79db9191']

| 数据集 | 口径 | regret@1 vs shared oracle 均值 / 中位 | regret@1 vs 40 条 oracle 均值 | regret@3（全局前 3 取最好）vs shared oracle | 任务 regret@1 > 2σ_mean 的比例 | σ_mean 中位 |
|---|---|---|---|---|---|---|
| PhonemeSpectra | in-sample | 0.004 / 0.003 | 0.007 | 0.004 | 0.12 | 0.005 |
| PhonemeSpectra | LODO | 0.030 / 0.029 | 0.032 | 0.004 | 0.88 | 0.005 |
| Cricket | in-sample | 0.027 / 0.029 | 0.048 | 0.014 | 0.50 | 0.008 |
| Cricket | LODO | 0.027 / 0.029 | 0.048 | 0.006 | 0.50 | 0.008 |
| UWaveGestureLibrary | in-sample | 0.019 / 0.001 | 0.026 | 0.014 | 0.25 | 0.013 |
| UWaveGestureLibrary | LODO | 0.019 / 0.001 | 0.026 | 0.014 | 0.25 | 0.013 |
| 合并 | in-sample | 0.017 / 0.005 | 0.027 | 0.010 | 0.29 | 0.009 |

逐任务 regret@1（in-sample 全局冠军；acc 单位）：

| 任务 | oracle shared | 全局冠军 | regret@1 | 全局冠军排名 | σ_mean | 冠军是否 failed |
|---|---:|---:|---:|---:|---:|---|
| PhonemeSpectra_a00 | 0.138 | 0.131 | 0.007 | 2 | 0.011 |  |
| PhonemeSpectra_a01 | 0.189 | 0.189 | 0.000 | 1 | 0.005 |  |
| PhonemeSpectra_a02 | 0.162 | 0.161 | 0.001 | 2 | 0.006 |  |
| PhonemeSpectra_a03 | 0.183 | 0.183 | 0.000 | 1 | 0.005 |  |
| PhonemeSpectra_a04 | 0.159 | 0.159 | 0.000 | 1 | 0.004 |  |
| PhonemeSpectra_a05 | 0.128 | 0.123 | 0.005 | 4 | 0.006 |  |
| PhonemeSpectra_a06 | 0.130 | 0.116 | 0.014 | 2 | 0.009 |  |
| PhonemeSpectra_a07 | 0.127 | 0.118 | 0.008 | 3 | 0.004 |  |
| UWaveGestureLibrary_a00 | 0.500 | 0.400 | 0.100 | 15 | 0.014 |  |
| UWaveGestureLibrary_a01 | 0.836 | 0.831 | 0.005 | 2 | 0.012 |  |
| UWaveGestureLibrary_a02 | 0.819 | 0.819 | 0.000 | 1 | 0.016 |  |
| UWaveGestureLibrary_a03 | 0.924 | 0.924 | 0.000 | 1 | 0.009 |  |
| UWaveGestureLibrary_a04 | 0.618 | 0.618 | 0.000 | 1 | 0.013 |  |
| UWaveGestureLibrary_a05 | 0.916 | 0.913 | 0.003 | 2 | 0.012 |  |
| UWaveGestureLibrary_a06 | 0.862 | 0.814 | 0.048 | 3 | 0.019 |  |
| UWaveGestureLibrary_a07 | 0.923 | 0.923 | 0.000 | 1 | 0.004 |  |
| Cricket_a00 | 0.573 | 0.501 | 0.072 | 4 | 0.011 |  |
| Cricket_a01 | 0.957 | 0.948 | 0.010 | 3 | 0.008 |  |
| Cricket_a02 | 0.978 | 0.948 | 0.030 | 12 | 0.006 | failed |
| Cricket_a03 | 0.840 | 0.813 | 0.027 | 12 | 0.016 | failed |
| Cricket_a04 | 0.638 | 0.595 | 0.043 | 4 | 0.021 |  |
| Cricket_a05 | 0.996 | 0.996 | 0.000 | 1 | 0.007 |  |
| Cricket_a06 | 0.995 | 0.963 | 0.032 | 14 | 0.003 | failed |
| Cricket_a07 | 0.867 | 0.861 | 0.006 | 11 | nan | failed |

## §5 shared vs random：构成与失败率

- shared-20：τ {'1.0': 4, '0.1': 7, '0.01': 6, '10.0': 3}，loss {'triplet': 12, 'info_nce': 8}，triplet/euclidean/l2 2 条，aug_max 均值 0.92
- random（试点 480 条）：τ {'0.1': 136, '1.0': 139, '10.0': 66, '0.01': 139}，loss {'triplet': 291, 'info_nce': 189}，triplet/euclidean/l2 33 条（0.069），aug_max 均值 0.87
- random（全 72 任务 1440 条）：τ {'0.1': 419, '0.01': 411, '1.0': 409, '10.0': 201}，triplet/euclidean/l2 0.056

| 数据集 | shared failed（两种子均值） | shared failed（只 seed 0） | random failed（seed 0） | shared 逐配置失败率 sd/√20 |
|---|---:|---:|---:|---:|
| PhonemeSpectra | 0.59 | 0.59 | 0.61 | 0.087 |
| Cricket | 0.65 | 0.62 | 0.51 | 0.063 |
| UWaveGestureLibrary | 0.40 | 0.39 | 0.29 | 0.083 |

failed 按 loss × τ（shared+random 合计，括号 n）：

| loss \ τ | 0.01 | 0.1 | 1.0 | 10.0 |
|---|---:|---:|---:|---:|
| info_nce | 0.40 (130) | 0.48 (161) | 0.58 (90) | — |
| triplet | 0.29 (153) | 0.43 (143) | 0.64 (145) | 0.78 (138) |

triplet/euclidean/l2 按 τ：τ=0.01: 0.57 (7)；τ=0.1: 0.50 (8)；τ=1.0: 1.00 (11)；τ=10.0: 1.00 (55)

## §6 Cricket 逐任务存活（ADM 用 a01 / a07）

| 任务 | 档 | n | 地板 | 探针 3 种子 | shared 两种子都存活 | seed0 存活 | seed1 存活 | 两种子均值最高 | 存活 ρ(seed0,seed1) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| Cricket_a00 | c0w0n0 | 98 | 0.466 | [0.481, 0.501, 0.418] | 7 | 9 | 11 | 0.573 | 0.86 |
| Cricket_a01 | c1w2n3 | 153 | 0.897 | [0.888, 0.895, 0.908] | 9 | 11 | 10 | 0.957 | 0.63 |
| Cricket_a02 | c2w3n1 | 116 | 0.948 | [0.957, 0.93, 0.957] | 4 | 5 | 4 | 0.978 | 0.40 |
| Cricket_a03 | c3w1n2 | 137 | 0.813 | [0.796, 0.818, 0.825] | 2 | 3 | 3 | 0.840 | nan |
| Cricket_a04 | c0w1n1 | 105 | 0.492 | [0.495, 0.495, 0.486] | 11 | 14 | 12 | 0.638 | 0.55 |
| Cricket_a05 | c1w3n2 | 132 | 0.938 | [0.918, 0.94, 0.955] | 7 | 9 | 11 | 0.996 | 0.86 |
| Cricket_a06 | c2w2n0 | 99 | 0.963 | [0.95, 0.97, 0.97] | 5 | 7 | 6 | 0.995 | 0.89 |
| Cricket_a07 | c3w0n3 | 180 | 0.861 | [0.861, 0.872, 0.85] | 0 | 3 | 3 | 0.867 | nan |
| **Cricket 汇总** | | | | | 中位 6.0，≥ 8 的任务 2/8 | | | | |
| PhonemeSpectra_a00 | c0w0n0 | 374 | 0.107 | [0.11, 0.107, 0.104] | 5 | 10 | 5 | 0.138 | -0.21 |
| PhonemeSpectra_a01 | c1w2n3 | 984 | 0.145 | [0.135, 0.152, 0.147] | 8 | 8 | 8 | 0.189 | 0.64 |
| PhonemeSpectra_a02 | c2w3n1 | 501 | 0.132 | [0.152, 0.124, 0.122] | 8 | 9 | 8 | 0.162 | 0.14 |
| PhonemeSpectra_a03 | c3w1n2 | 734 | 0.133 | [0.153, 0.124, 0.123] | 11 | 11 | 13 | 0.183 | 0.87 |
| PhonemeSpectra_a04 | c0w1n1 | 474 | 0.135 | [0.154, 0.118, 0.133] | 6 | 7 | 7 | 0.159 | 0.58 |
| PhonemeSpectra_a05 | c1w3n2 | 721 | 0.118 | [0.118, 0.114, 0.122] | 2 | 7 | 4 | 0.128 | nan |
| PhonemeSpectra_a06 | c2w2n0 | 280 | 0.102 | [0.107, 0.089, 0.111] | 4 | 6 | 8 | 0.130 | 0.63 |
| PhonemeSpectra_a07 | c3w0n3 | 955 | 0.101 | [0.102, 0.092, 0.108] | 7 | 8 | 8 | 0.127 | 0.83 |
| **PhonemeSpectra 汇总** | | | | | 中位 6.5，≥ 8 的任务 3/8 | | | | |
| UWaveGestureLibrary_a00 | c0w0n0 | 95 | 0.379 | [0.358, 0.4, 0.379] | 12 | 14 | 15 | 0.500 | 0.40 |
| UWaveGestureLibrary_a01 | c1w2n3 | 393 | 0.770 | [0.779, 0.764, 0.766] | 10 | 12 | 10 | 0.836 | 0.38 |
| UWaveGestureLibrary_a02 | c2w3n1 | 160 | 0.669 | [0.688, 0.681, 0.637] | 13 | 13 | 14 | 0.819 | 0.74 |
| UWaveGestureLibrary_a03 | c3w1n2 | 263 | 0.855 | [0.867, 0.859, 0.84] | 8 | 11 | 9 | 0.924 | 0.83 |
| UWaveGestureLibrary_a04 | c0w1n1 | 148 | 0.559 | [0.527, 0.588, 0.561] | 8 | 11 | 8 | 0.618 | -0.01 |
| UWaveGestureLibrary_a05 | c1w3n2 | 196 | 0.796 | [0.791, 0.801, 0.796] | 12 | 13 | 12 | 0.916 | 0.50 |
| UWaveGestureLibrary_a06 | c2w2n0 | 105 | 0.737 | [0.743, 0.705, 0.762] | 11 | 13 | 11 | 0.862 | 0.38 |
| UWaveGestureLibrary_a07 | c3w0n3 | 388 | 0.850 | [0.861, 0.851, 0.838] | 11 | 11 | 13 | 0.923 | 0.90 |
| **UWaveGestureLibrary 汇总** | | | | | 中位 11.0，≥ 8 的任务 8/8 | | | | |

Cricket_a07 逐配置（地板 0.861；seed0 / seed1 终点 acc，* = 超地板）：

fe1eaebb info_nce/euclidean/none τ0.01: 0.872*/0.856; 55558e02 triplet/euclidean/layer_norm τ0.01: 0.867*/0.850; f59dbb24 info_nce/cosine/l2 τ0.1: 0.867*/0.850; cb08a823 triplet/cosine/layer_norm τ0.1: 0.861/0.872*; dae137f2 info_nce/dot/layer_norm τ0.01: 0.856/0.844; 72e67053 triplet/cosine/layer_norm τ0.1: 0.856/0.833; 79db9191 triplet/dot/none τ1.0: 0.844/0.872*; 1ec63b4f triplet/dot/l2 τ0.01: 0.844/0.872*; 9d4ef59d triplet/cosine/none τ0.1: 0.844/0.856; f64cea8e info_nce/dot/layer_norm τ0.1: 0.839/0.844; e3d73768 info_nce/dot/layer_norm τ0.01: 0.833/0.850; 2a667428 triplet/euclidean/layer_norm τ0.01: 0.828/0.844; 00f48d40 triplet/euclidean/none τ10.0: 0.822/0.844; 00cc0074 info_nce/euclidean/layer_norm τ0.1: 0.767/0.789; 9bf55c58 triplet/cosine/l2 τ1.0: 0.733/0.683; 482e382e triplet/cosine/l2 τ1.0: 0.678/0.683; 63a3ded8 triplet/euclidean/l2 τ10.0: 0.594/0.700; fc7fd37f info_nce/cosine/none τ1.0: 0.467/0.406; 8c482976 info_nce/dot/layer_norm τ0.1: 0.456/0.450; 57531c09 triplet/euclidean/l2 τ10.0: 0.394/0.461

## §4 配对规则产量（存活–存活对里 |gap| ≥ 2·sd_gap 的比例；sd_gap 由该任务两种子噪声 σ 推）

| 数据集 | 存活数/任务 中位 | 存活–存活对 | 其中 ≥ 2σ 的对 | 比例 | failed–存活对 | σ 中位 |
|---|---:|---:|---:|---:|---:|---:|
| PhonemeSpectra | 15.5 | 977 | 211 | 0.22 | 2999 | 0.008 |
| Cricket | 21.0 | 1169 | 415 | 0.36 | 2537 | 0.011 |
| UWaveGestureLibrary | 27.5 | 2702 | 852 | 0.32 | 2786 | 0.018 |

## (b) 地板之下（两种子都 failed 的 shared）的原始 acc 种子一致性 Spearman

- PhonemeSpectra：ρ 均值 0.68（8 任务；[-0.11, 0.85, 0.83, 0.93, 0.84, 0.87, 0.54, 0.71]）
- Cricket：ρ 均值 0.62（8 任务；[0.39, 0.17, 0.87, 0.96, 0.7, 0.14, 0.85, 0.85]）
- UWaveGestureLibrary：ρ 均值 0.77（7 任务；[0.69, 0.71, 0.81, 0.83, 0.71, 0.68, 0.96]）

