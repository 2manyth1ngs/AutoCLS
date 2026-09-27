# S3-Q 表（160 任务 / 17 源 LOSO；regret 为 84.regret_at_k 口径）

## 目录层 regret@3（任务均值 / 源均值）

| E | BLIND | KNN | KNN10 | KNN-X | ORACLE | regret@1 BLIND / KNN |
|---|---|---|---|---|---|---|
| E_pt | 0.0213 / 0.0179 | 0.0114 / 0.0107 | 0.0092 | 0.0235 | 0.0048 | 0.0347 / 0.0264 |
| E_sh | 0.0213 / 0.0179 | 0.0192 / 0.0168 | 0.0245 | 0.0219 | — | 0.0347 / 0.0384 |
| E_A | 0.0213 / 0.0179 | 0.0233 / 0.0184 | 0.0230 | 0.0251 | — | 0.0347 / 0.0368 |
| E_TF | 0.0213 / 0.0179 | 0.0118 / 0.0119 | 0.0134 | 0.0232 | — | 0.0347 / 0.0305 |

## 门（源加权差，精确符号翻转）

| E | G1 BLIND−KNN | p（Holm4） | 源 +/− | G2 KNNX−KNN | p | G3 Sp KNN−BLIND | p | 读数码 |
|---|---|---|---|---|---|---|---|---|
| E_pt | +0.0072 [-0.0009, +0.0260] | 0.176（0.702） | 9/7 | +0.0096 | 0.034 | -0.053 | 0.031 | S3Q-RETR-CONTENT |
| E_sh | +0.0011 [-0.0041, +0.0092] | 0.726（1.000） | 6/9 | +0.0017 | 0.609 | -0.039 | 0.161 | S3Q-RETR-NULL |
| E_A | -0.0005 [-0.0101, +0.0045] | 0.882（1.000） | 8/8 | +0.0039 | 0.198 | -0.088 | 0.017 | S3Q-RETR-NULL |
| E_TF | +0.0060 [-0.0004, +0.0218] | 0.183（0.702） | 12/3 | +0.0079 | 0.052 | -0.044 | 0.040 | S3Q-RETR-NULL |

## 目录外（random-20，Spearman 任务均值 / 中位 / 前一半；代理 shared regret@3；前 3 高估）

| E | SUR-BLIND | SUR-KNN | SUR-X | 前一半 BLIND / KNN | 代理 regret@3 BLIND / KNN | 高估 BLIND / KNN |
|---|---|---|---|---|---|---|
| E_pt | 0.140 / 0.157 | 0.088 / 0.094 | 0.069 | 0.047 / -0.022 | 0.0203 / 0.0103 | +0.256 / +0.221 |
| E_sh | 0.140 / 0.157 | 0.109 / 0.102 | 0.087 | 0.047 / 0.005 | 0.0203 / 0.0186 | +0.256 / +0.509 |
| E_A | 0.140 / 0.157 | 0.054 / 0.079 | 0.056 | 0.047 / -0.020 | 0.0203 / 0.0218 | +0.256 / +0.419 |
| E_TF | 0.140 / 0.157 | 0.105 / 0.109 | 0.076 | 0.047 / 0.039 | 0.0203 / 0.0147 | +0.256 / +0.401 |

## E_pt 逐源（regret@3 源均值差 BLIND−KNN；k；近邻来源前 3）

| 源 | BLIND−KNN | KNNX−KNN | k | 近邻来源 |
|---|---|---|---|---|
| Handwriting * | +0.0016 | +0.0019 | 10 | FordA 0.53, PTB-XL 0.24, SleepEDF 0.06 |
| UniMiB-SHAR | +0.0043 | +0.0098 | 10 | FordA 0.29, Handwriting 0.15, Cricket 0.15 |
| PhonemeSpectra * | -0.0015 | +0.0040 | 10 | PTB-XL 0.39, FordA 0.24, LargeKitchenAppliances 0.16 |
| UWaveGestureLibrary * | +0.0046 | +0.0027 | 10 | ShapesAll 0.54, NonInvasiveFetalECGThorax1 0.14, SelfRegulationSCP1 0.12 |
| Cricket * | +0.0015 | -0.0050 | 10 | SleepEDF 0.36, UniMiB-SHAR 0.14, CinCECGTorso 0.12 |
| PTB-XL | -0.0060 | -0.0042 | 10 | SleepEDF 0.34, FordA 0.25, Handwriting 0.21 |
| SleepEDF | +0.0195 | +0.0179 | 10 | FordA 0.44, PTB-XL 0.36, Handwriting 0.06 |
| PAMAP2 | -0.0009 | -0.0025 | 5 | CinCECGTorso 0.23, Cricket 0.15, UWaveGestureLibrary 0.15 |
| NonInvasiveFetalECGThorax1 | -0.0004 | +0.0102 | 10 | ShapesAll 0.42, Adiac 0.15, EthanolConcentration 0.11 |
| ShapesAll | +0.0000 | +0.0036 | 10 | UWaveGestureLibrary 0.46, NonInvasiveFetalECGThorax1 0.34, Adiac 0.11 |
| LargeKitchenAppliances | +0.0288 | +0.0288 | 5 | SemgHandMovementCh2 0.9, PTB-XL 0.07, PhonemeSpectra 0.03 |
| Adiac | +0.0034 | +0.0117 | 10 | ShapesAll 0.26, NonInvasiveFetalECGThorax1 0.2, SelfRegulationSCP1 0.16 |
| FordA | -0.0071 | +0.0036 | 10 | SleepEDF 0.41, Handwriting 0.33, PTB-XL 0.21 |
| SemgHandMovementCh2 | +0.0812 | +0.0905 | 10 | LargeKitchenAppliances 0.62, EthanolConcentration 0.11, PhonemeSpectra 0.09 |
| EthanolConcentration * | -0.0001 | +0.0007 | 5 | ShapesAll 0.33, NonInvasiveFetalECGThorax1 0.28, UWaveGestureLibrary 0.1 |
| CinCECGTorso | +0.0002 | -0.0061 | 20 | SleepEDF 0.17, Cricket 0.11, NonInvasiveFetalECGThorax1 0.09 |
| SelfRegulationSCP1 * | -0.0071 | -0.0050 | 5 | UWaveGestureLibrary 0.34, ShapesAll 0.23, CinCECGTorso 0.12 |

（* = UEA 折源）

## 预测

- P1: ✗
- P2: ✓
- P3: ✗
- P4: ✓
- P5: ✓
- P6: ✗
- P7: ✓
