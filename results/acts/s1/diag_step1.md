# S1 零 GPU 诊断（用户建议第 1 条，2026-09-19；探索读数，为 S1′ 协议定参数）

### failed 份额按 温度

| 组 | failed 份额 | n |
|---|---:|---:|
| 10.0 | 0.68 | 138 |
| 1.0 | 0.59 | 237 |
| 0.1 | 0.52 | 308 |
| 0.01 | 0.46 | 277 |

### failed 份额按 损失类型

| 组 | failed 份额 | n |
|---|---:|---:|
| triplet | 0.56 | 571 |
| info_nce | 0.53 | 389 |

### failed 份额按 损失 × 温度

| 组 | failed 份额 | n |
|---|---:|---:|
| triplet/τ=10.0 | 0.68 | 138 |
| info_nce/τ=1.0 | 0.60 | 90 |
| triplet/τ=1.0 | 0.58 | 147 |
| info_nce/τ=0.1 | 0.54 | 165 |
| triplet/τ=0.1 | 0.50 | 143 |
| info_nce/τ=0.01 | 0.46 | 134 |
| triplet/τ=0.01 | 0.46 | 143 |

### failed 份额按 归一化 × 相似度

| 组 | failed 份额 | n |
|---|---:|---:|
| l2+euclidean | 0.66 | 118 |
| l2+cosine | 0.58 | 120 |
| l2+dot | 0.58 | 76 |
| none+cosine | 0.57 | 106 |
| none+dot | 0.53 | 75 |
| layer_norm+cosine | 0.53 | 93 |
| layer_norm+euclidean | 0.51 | 129 |
| none+euclidean | 0.49 | 93 |
| layer_norm+dot | 0.46 | 150 |

### failed 份额按 对比对构造

| 组 | failed 份额 | n |
|---|---:|---:|
| T=1 CS=1 k=3 | 0.64 | 74 |
| T=0 CS=1 k=3 | 0.60 | 143 |
| T=1 CS=0 k=5 | 0.60 | 35 |
| T=0 CS=1 k=5 | 0.59 | 56 |
| T=1 CS=1 k=2 | 0.58 | 24 |
| T=1 CS=0 k=3 | 0.55 | 83 |
| T=1 CS=0 k=0 | 0.54 | 102 |
| T=1 CS=0 k=2 | 0.53 | 77 |
| T=1 CS=1 k=5 | 0.50 | 32 |
| T=0 CS=0 k=None | 0.50 | 308 |
| T=0 CS=1 k=2 | 0.38 | 26 |

### failed 份额按 增强强度

| 组 | failed 份额 | n |
|---|---:|---:|
| aug>0.8 | 0.54 | 771 |
| aug(0.5,0.8] | 0.54 | 172 |
| aug≤0.5 | 0.53 | 17 |

### failed 份额 数据集 × 温度

| 数据集 | τ=0.01 | τ=0.1 | τ=1 | τ=10 |
|---|---:|---:|---:|---:|
| Cricket | 0.40 (88) | 0.51 (108) | 0.62 (74) | 0.78 (50) |
| HandMovementDirection | 0.63 (92) | 0.61 (104) | 0.57 (82) | 0.57 (42) |
| PhonemeSpectra | 0.36 (97) | 0.45 (96) | 0.57 (81) | 0.67 (46) |

### 逐任务可靠性（锚 = 2 条 × seed 0/1；spread = 存活配置 best-of-3 的 sd）

| 任务 | n | n_eval | k | n_eval/k | failed | 锚 mean\|Δacc\| | 存活 sd | SNR = sd/\|Δ\| | 锚状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PhonemeSpectra_a00 | 145 | 47 | 39 | 1.2 | 0.88 | 0.018 | 0.009 | 0.5 | FF/SF |
| PhonemeSpectra_a01 | 956 | 279 | 39 | 7.2 | 0.30 | 0.004 | 0.021 | 5.9 | FF/SS |
| PhonemeSpectra_a02 | 257 | 85 | 39 | 2.2 | 0.35 | 0.000 | 0.016 | inf | FF/SS |
| PhonemeSpectra_a03 | 536 | 165 | 39 | 4.2 | 0.53 | 0.021 | 0.023 | 1.1 | FF/SS |
| PhonemeSpectra_a04 | 234 | 78 | 39 | 2.0 | 0.33 | 0.013 | 0.014 | 1.1 | SS/SS |
| PhonemeSpectra_a05 | 519 | 160 | 39 | 4.1 | 0.33 | 0.009 | 0.018 | 2.0 | FF/SS |
| PhonemeSpectra_a06 | 82 | 40 | 39 | 1.0 | 0.53 | 0.000 | 0.011 | inf | FF/FF |
| PhonemeSpectra_a07 | 907 | 276 | 39 | 7.1 | 0.65 | 0.002 | 0.012 | 6.6 | FF/SS |
| HandMovementDirection_a00 | 40 | 13 | 4 | 3.2 | 0.88 | 0.000 | 0.000 | inf | FF/FF |
| HandMovementDirection_a01 | 207 | 61 | 4 | 15.2 | 0.53 | 0.025 | 0.025 | 1.0 | SF/SS |
| HandMovementDirection_a02 | 67 | 19 | 4 | 4.8 | 0.70 | 0.053 | 0.055 | 1.0 | SF/FF |
| HandMovementDirection_a03 | 124 | 38 | 4 | 9.5 | 0.35 | 0.013 | 0.054 | 4.1 | SF/SS |
| HandMovementDirection_a04 | 73 | 21 | 4 | 5.2 | 0.75 | 0.071 | 0.022 | 0.3 | FS/SF |
| HandMovementDirection_a05 | 103 | 31 | 4 | 7.8 | 0.50 | 0.016 | 0.064 | 3.9 | SS/FF |
| HandMovementDirection_a06 | 41 | 13 | 4 | 3.2 | 0.28 | 0.077 | 0.064 | 0.8 | SS/FS |
| HandMovementDirection_a07 | 223 | 67 | 4 | 16.8 | 0.82 | 0.022 | 0.007 | 0.3 | FS/SF |
| Cricket_a00 | 46 | 15 | 12 | 1.2 | 0.82 | 0.000 | 0.000 | inf | FF/SS |
| Cricket_a01 | 126 | 37 | 12 | 3.1 | 0.25 | 0.000 | 0.040 | inf | FF/SS |
| Cricket_a02 | 67 | 26 | 12 | 2.2 | 0.72 | 0.000 | 0.019 | inf | FF/FF |
| Cricket_a03 | 98 | 25 | 12 | 2.1 | 0.53 | 0.000 | 0.034 | inf | FF/SS |
| Cricket_a04 | 54 | 14 | 12 | 1.2 | 0.55 | 0.000 | 0.049 | inf | SS/FF |
| Cricket_a05 | 91 | 26 | 12 | 2.2 | 0.25 | 0.038 | 0.041 | 1.1 | FF/SS |
| Cricket_a06 | 47 | 15 | 12 | 1.2 | 0.75 | 0.000 | 0.020 | inf | FF/FF |
| Cricket_a07 | 180 | 48 | 12 | 4.0 | 0.50 | 0.000 | 0.024 | inf | FF/SS |

Spearman（24 任务）：failed~log n -0.36；failed~n_eval/k -0.12；|Δ|~log n +0.11；|Δ|~n_eval/k +0.52；SNR~log n +0.53；SNR~n_eval/k +0.13

| n_eval 档 | 任务数 | failed 均值 | 锚\|Δacc\| 均值 | 两边存活对占比 |
|---|---:|---:|---:|---:|
| [0, 20) | 6 | 0.66 | 0.022 | 0.25 |
| [20, 40) | 7 | 0.48 | 0.020 | 0.36 |
| [40, 80) | 6 | 0.60 | 0.013 | 0.33 |
| [80, 400) | 5 | 0.43 | 0.007 | 0.50 |

### 保真度两个切面（25% → 100% 截点 Spearman，逐任务）

| 任务 | 全部 40 | 只算存活 | 存活数 |
|---|---:|---:|---:|
| PhonemeSpectra_a00 | 0.56 | -0.19 | 5 |
| PhonemeSpectra_a01 | 0.75 | 0.42 | 28 |
| PhonemeSpectra_a02 | 0.72 | 0.36 | 26 |
| PhonemeSpectra_a03 | 0.79 | 0.39 | 19 |
| PhonemeSpectra_a04 | 0.09 | -0.25 | 27 |
| PhonemeSpectra_a05 | 0.66 | 0.20 | 27 |
| PhonemeSpectra_a06 | 0.05 | -0.20 | 19 |
| PhonemeSpectra_a07 | 0.80 | 0.31 | 14 |
| HandMovementDirection_a00 | 0.06 | -0.89 | 5 |
| HandMovementDirection_a01 | 0.17 | -0.42 | 19 |
| HandMovementDirection_a02 | 0.31 | -0.24 | 12 |
| HandMovementDirection_a03 | 0.34 | 0.08 | 26 |
| HandMovementDirection_a04 | 0.08 | 0.00 | 10 |
| HandMovementDirection_a05 | 0.31 | 0.20 | 20 |
| HandMovementDirection_a06 | 0.03 | -0.40 | 29 |
| HandMovementDirection_a07 | 0.43 | -0.08 | 7 |
| Cricket_a00 | 0.50 | -0.53 | 7 |
| Cricket_a01 | 0.80 | 0.59 | 30 |
| Cricket_a02 | 0.69 | -0.48 | 11 |
| Cricket_a03 | 0.85 | 0.36 | 19 |
| Cricket_a04 | 0.16 | -0.21 | 18 |
| Cricket_a05 | 0.73 | 0.41 | 30 |
| Cricket_a06 | 0.77 | -0.24 | 10 |
| Cricket_a07 | 0.23 | -0.49 | 20 |
| **中位** | **0.46** | **-0.13** | |

### CV 标签（官方网格内 5 折 CV 分数，best-of-3）与 holdout acc 标签的一致性（只在走网格分支的任务）

| 任务 | n_train | 可用配置 | Spearman(CV, acc) 全部 | 只算存活 | CV 标签 failed-等价份额（CV ≤ 探针 CV） |
|---|---:|---:|---:|---:|---:|
| PhonemeSpectra_a01 | 677 | 40 | 0.76 | 0.47 | — |
| PhonemeSpectra_a03 | 371 | 40 | 0.80 | 0.69 | — |
| PhonemeSpectra_a05 | 359 | 40 | 0.79 | 0.51 | — |
| PhonemeSpectra_a07 | 631 | 40 | 0.73 | 0.55 | — |
| HandMovementDirection_a01 | 146 | 40 | 0.14 | -0.02 | — |
| HandMovementDirection_a03 | 86 | 40 | 0.30 | 0.38 | — |
| HandMovementDirection_a04 | 52 | 40 | 0.33 | -0.57 | — |
| HandMovementDirection_a05 | 72 | 40 | 0.58 | -0.05 | — |
| HandMovementDirection_a07 | 156 | 40 | 0.03 | -0.16 | — |
| Cricket_a01 | 89 | 40 | 0.90 | 0.82 | — |
| Cricket_a03 | 73 | 40 | 0.77 | 0.59 | — |
| Cricket_a05 | 65 | 40 | 0.78 | 0.51 | — |
| Cricket_a07 | 132 | 40 | 0.37 | 0.30 | — |
| **中位** | | | **0.73** | **0.47** | |

