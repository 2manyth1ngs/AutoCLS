# GEO-E 检验 tables：E_T → 损失几何家族（29 个官方靶）

读数码 **GEO-E-DEAD**（门 {'G-G1': False, 'G-G2': False}）

家族 8 个；冠军家族分布 {"('info_nce', 'euclidean', 'none')": 11, "('triplet', 'dot', 'none')": 10, "('info_nce', 'dot', 'layer_norm')": 3, "('info_nce', 'dot', 'none')": 1, "('triplet', 'euclidean', 'none')": 1, "('info_nce', 'euclidean', 'layer_norm')": 1, "('triplet', 'cosine', 'layer_norm')": 1, "('triplet', 'euclidean', 'l2')": 1}；猜多数 0.38；固定最好家族（全数据）['info_nce', 'euclidean', 'none']，家族 oracle − 固定最好 +0.0226；先验 top-1 家族 regret -0.0373

| 特征臂 | n | 命中率 (猜多数) | p_hit | mean regret（选） | regret（固定最好） | 增益 vs 固定 | 胜:负 | p_gain | EEG 类增益 / 其余 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| et60 | 29 | 0.14 (0.38) | 0.969 | -0.0484 | -0.0226 | **-0.0258** | 3:9 | 0.784 | -0.0050 / -0.0338 |
| et290 | 29 | 0.21 (0.38) | 0.860 | -0.0530 | -0.0226 | **-0.0304** | 2:8 | 0.849 | 0.0000 / -0.0420 |
| phi | 23 | 0.30 (0.39) | 0.550 | -0.0320 | -0.0220 | **-0.0099** | 4:4 | 0.320 | 0.0144 / -0.0185 |
| et60+phi | 23 | 0.17 (0.39) | 0.849 | -0.0517 | -0.0220 | **-0.0296** | 4:8 | 0.645 | -0.0165 / -0.0343 |

## et60 逐靶

| 靶 | 冠军家族 | 选 | 固定 | regret（选） | 增益 vs 固定 | σ |
|---|---|---|---|---:|---:|---:|
| RacketSports | triplet/dot/none | triplet/dot/none | info_nce/euclidean/none | +0.000 | +0.013 | 0.013 |
| CharacterTrajectories | triplet/dot/none | triplet/dot/none | info_nce/euclidean/none | +0.000 | +0.001 | 0.002 |
| PenDigits | triplet/cosine/layer_norm | triplet/euclidean/none | info_nce/euclidean/none | -0.001 | +0.001 | 0.001 |
| ArticularyWordRecognition | info_nce/euclidean/none | triplet/dot/none | info_nce/euclidean/none | +0.000 | +0.000 | 0.002 |
| BasicMotions | info_nce/euclidean/none | info_nce/euclidean/none | info_nce/euclidean/none | +0.000 | +0.000 | 0.014 |
| ERing | triplet/dot/none | info_nce/euclidean/none | info_nce/euclidean/none | -0.044 | +0.000 | 0.006 |
| EigenWorms | info_nce/euclidean/none | info_nce/euclidean/none | info_nce/euclidean/none | +0.000 | +0.000 | 0.059 |
| Epilepsy | info_nce/dot/layer_norm | info_nce/euclidean/none | info_nce/euclidean/none | +0.000 | +0.000 | 0.004 |
| EthanolConcentration | info_nce/dot/none | info_nce/euclidean/none | info_nce/euclidean/none | -0.027 | +0.000 | 0.029 |
| FaceDetection | triplet/dot/none | info_nce/euclidean/none | info_nce/euclidean/none | -0.056 | +0.000 | 0.013 |
| FingerMovements | triplet/dot/none | info_nce/euclidean/none | info_nce/euclidean/none | -0.030 | +0.000 | 0.025 |
| HandMovementDirection | triplet/dot/none | info_nce/euclidean/none | info_nce/euclidean/none | -0.068 | +0.000 | 0.068 |
| JapaneseVowels | info_nce/euclidean/none | triplet/dot/none | info_nce/euclidean/none | +0.000 | +0.000 | 0.002 |
| Libras | triplet/dot/none | info_nce/euclidean/none | info_nce/euclidean/none | -0.050 | +0.000 | 0.023 |
| PEMS-SF | info_nce/euclidean/layer_norm | info_nce/euclidean/none | info_nce/euclidean/none | -0.087 | +0.000 | 0.093 |
| PhonemeSpectra | info_nce/dot/layer_norm | info_nce/euclidean/none | info_nce/euclidean/none | -0.030 | +0.000 | 0.006 |
| SelfRegulationSCP1 | triplet/dot/none | info_nce/euclidean/none | info_nce/euclidean/none | -0.089 | +0.000 | 0.025 |
| SelfRegulationSCP2 | info_nce/dot/layer_norm | info_nce/euclidean/none | info_nce/euclidean/none | -0.067 | +0.000 | 0.043 |
| SpokenArabicDigits | triplet/dot/none | info_nce/euclidean/none | info_nce/euclidean/none | -0.012 | +0.000 | 0.003 |
| UWaveGestureLibrary | triplet/euclidean/l2 | info_nce/euclidean/none | info_nce/euclidean/none | -0.025 | +0.000 | 0.012 |
| LSST | info_nce/euclidean/none | triplet/euclidean/none | info_nce/euclidean/none | -0.008 | -0.008 | 0.006 |
| Heartbeat | triplet/dot/none | triplet/euclidean/l2 | info_nce/euclidean/none | -0.020 | -0.015 | 0.005 |
| Cricket | info_nce/euclidean/none | triplet/dot/none | info_nce/euclidean/none | -0.028 | -0.028 | 0.014 |
| NATOPS | info_nce/euclidean/none | triplet/cosine/layer_norm | info_nce/euclidean/none | -0.028 | -0.028 | 0.020 |
| MotorImagery | triplet/euclidean/none | triplet/dot/none | info_nce/euclidean/none | -0.090 | -0.040 | 0.047 |
| AtrialFibrillation | info_nce/euclidean/none | info_nce/dot/layer_norm | info_nce/euclidean/none | -0.067 | -0.067 | 0.115 |
| DuckDuckGeese | info_nce/euclidean/none | triplet/dot/none | info_nce/euclidean/none | -0.120 | -0.120 | 0.020 |
| Handwriting | info_nce/euclidean/none | triplet/dot/none | info_nce/euclidean/none | -0.127 | -0.127 | 0.021 |
| StandWalkJump | info_nce/euclidean/none | triplet/euclidean/none | info_nce/euclidean/none | -0.333 | -0.333 | 0.102 |

预测：{'P1': {'desc': 'G-G1', 'hit': False}, 'P2': {'desc': 'G-G2', 'hit': False}, 'P3': {'desc': 'et60 增益优于 phi', 'hit': False}, 'P4': {'desc': '增益集中在 EEG / 传感器类靶', 'value': (-0.0050000000000000044, -0.03378785834158879), 'hit': True}}

