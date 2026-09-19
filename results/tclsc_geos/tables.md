# GEO-S 检验 tables：几何子空间（29 靶 × 18 家族，干净基，官方口径）

读数码 **GEOS-HEADROOM-ONLY**（门 {'G-S1': True, 'G-S2': False, 'G-S3': False}）

## S1 干净净空：家族 oracle − 固定最好家族 **+0.0422**（超 2σ 靶 0.52，σ 中位 0.009）；− 先验家族 0.0422；− GGS 家族 0.0556；固定最好家族 ('triplet', 'dot', 'none')；两大家族占冠军 0.31；目录冠军一致率 0.10

冠军家族分布：{"('triplet', 'dot', 'none')": 6, "('info_nce', 'dot', 'l2')": 4, "('triplet', 'euclidean', 'none')": 4, "('info_nce', 'euclidean', 'l2')": 2, "('info_nce', 'dot', 'none')": 1, "('info_nce', 'euclidean', 'none')": 3, "('triplet', 'cosine', 'l2')": 1, "('triplet', 'cosine', 'layer_norm')": 1, "('triplet', 'euclidean', 'layer_norm')": 1, "('info_nce', 'cosine', 'l2')": 2, "('triplet', 'dot', 'layer_norm')": 1, "('info_nce', 'euclidean', 'layer_norm')": 1, "('info_nce', 'dot', 'layer_norm')": 2}

| 靶 | 冠军家族 | oracle − 固定 | σ | 最好 acc |
|---|---|---:|---:|---:|
| StandWalkJump | ('info_nce', 'dot', 'layer_norm') | +0.200 | 0.094 | 0.600 |
| DuckDuckGeese | ('info_nce', 'euclidean', 'l2') | +0.160 | 0.000 | 0.660 |
| FingerMovements | ('info_nce', 'euclidean', 'none') | +0.160 | 0.046 | 0.590 |
| Handwriting | ('info_nce', 'dot', 'l2') | +0.116 | 0.017 | 0.551 |
| MotorImagery | ('info_nce', 'euclidean', 'none') | +0.090 | 0.014 | 0.590 |
| SelfRegulationSCP2 | ('info_nce', 'dot', 'layer_norm') | +0.078 | 0.008 | 0.600 |
| SelfRegulationSCP1 | ('info_nce', 'euclidean', 'layer_norm') | +0.058 | 0.005 | 0.901 |
| EthanolConcentration | ('triplet', 'cosine', 'l2') | +0.049 | 0.011 | 0.312 |
| Cricket | ('info_nce', 'dot', 'l2') | +0.042 | 0.010 | 0.986 |
| HandMovementDirection | ('triplet', 'cosine', 'layer_norm') | +0.041 | 0.024 | 0.378 |
| PEMS-SF | ('triplet', 'euclidean', 'none') | +0.040 | 0.018 | 0.792 |
| FaceDetection | ('info_nce', 'euclidean', 'l2') | +0.040 | 0.007 | 0.557 |
| LSST | ('triplet', 'euclidean', 'layer_norm') | +0.030 | 0.008 | 0.580 |
| RacketSports | ('triplet', 'euclidean', 'none') | +0.026 | 0.007 | 0.875 |
| BasicMotions | ('triplet', 'euclidean', 'none') | +0.025 | 0.009 | 1.000 |
| Libras | ('info_nce', 'cosine', 'l2') | +0.022 | 0.029 | 0.900 |
| NATOPS | ('triplet', 'dot', 'layer_norm') | +0.022 | 0.010 | 0.944 |
| Heartbeat | ('info_nce', 'dot', 'l2') | +0.010 | 0.040 | 0.741 |
| EigenWorms | ('info_nce', 'dot', 'none') | +0.008 | 0.022 | 0.817 |
| PhonemeSpectra | ('info_nce', 'cosine', 'l2') | +0.004 | 0.006 | 0.232 |
| CharacterTrajectories | ('triplet', 'euclidean', 'none') | +0.002 | 0.002 | 0.995 |
| ArticularyWordRecognition | ('triplet', 'dot', 'none') | +0.000 | 0.005 | 0.980 |
| AtrialFibrillation | ('info_nce', 'dot', 'l2') | +0.000 | 0.047 | 0.333 |
| ERing | ('triplet', 'dot', 'none') | +0.000 | 0.005 | 0.904 |
| Epilepsy | ('info_nce', 'euclidean', 'none') | +0.000 | 0.008 | 0.942 |
| JapaneseVowels | ('triplet', 'dot', 'none') | +0.000 | 0.003 | 0.984 |
| PenDigits | ('triplet', 'dot', 'none') | +0.000 | 0.001 | 0.990 |
| SpokenArabicDigits | ('triplet', 'dot', 'none') | +0.000 | 0.004 | 0.983 |
| UWaveGestureLibrary | ('triplet', 'dot', 'none') | +0.000 | 0.012 | 0.925 |

## S2 E 预测家族（LODO 岭，81 号同式）

| 臂 | n | 命中 (猜多数) | 增益 vs 固定 | 胜:负 | p_gain | p_hit |
|---|---:|---:|---:|---|---:|---:|
| et60 | 29 | 0.00 (0.21) | **-0.0356** | 4:21 | 0.798 | 1.000 |
| et290 | 29 | 0.00 (0.21) | **-0.0247** | 6:18 | 0.516 | 1.000 |
| phi | 23 | 0.04 (0.22) | **-0.0395** | 2:15 | 0.959 | 0.863 |
| et60+phi | 23 | 0.04 (0.22) | **-0.0216** | 6:15 | 0.500 | 0.824 |

## S3 SH6-3-3 截点模拟（29 靶）：E 排序 0.7273 vs 固定顺序 0.7214（Δ +0.0059，p 0.611，5:9）；先验优先 0.7214；oracle 0.7438

C0：{'rows': 580, 'anchor_seed_diff_median': 0.012500000000000011}

预测：{'P1': {'desc': 'G-S1', 'hit': True}, 'P2': {'desc': 'G-S2', 'hit': False}, 'P3': {'desc': 'G-S3', 'hit': False}, 'P4': {'desc': '两大家族占冠军 ≥ 0.6', 'value': 0.3103448275862069, 'hit': False}, 'P5': {'desc': '目录冠军家族一致率 ≤ 0.6', 'value': 0.10344827586206896, 'hit': True}, 'P6': {'desc': '净空集中在 EEG / 传感器类靶', 'value': (0.06949480799590105, 0.0317885612524821), 'hit': True}}

