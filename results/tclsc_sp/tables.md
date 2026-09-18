# SP1 检验 tables：剪空间 S⁺

读数码 **SP-DEAD**（门 {'G-S1': False, 'G-S2': False, 'G-S3': True}）

标签：{'rows': 1827, 'slices': 87, 'configs': 881, 'evaluable_slices': ['Cricket_01', 'Cricket_03', 'EthanolConcentration_02', 'HandMovementDirection_03', 'HandMovementDirection_04', 'Handwriting_00', 'Handwriting_01', 'Handwriting_02', 'Handwriting_03', 'Handwriting_04', 'Handwriting_05', 'PhonemeSpectra_00', 'UWaveGestureLibrary_00', 'UWaveGestureLibrary_01', 'UWaveGestureLibrary_02', 'UWaveGestureLibrary_05', 'UniMiB-SHAR_00', 'UniMiB-SHAR_01', 'UniMiB-SHAR_02', 'UniMiB-SHAR_03', 'UniMiB-SHAR_04', 'UniMiB-SHAR_05'], 'zbad_rate': 0.17624521072796934, 'zbad_rate_random': 0.2471264367816092, 'strict_rate_evaluable': 0.31601731601731603}

| 数据集 | n | z-bad 率 | 可评切片 | strict 率 |
|---|---:|---:|---:|---:|
| Cricket | 252 | 0.17 | 2 | 0.36 |
| EthanolConcentration | 252 | 0.16 | 1 | 0.81 |
| HandMovementDirection | 252 | 0.17 | 2 | 0.12 |
| Handwriting | 252 | 0.21 | 6 | 0.23 |
| PhonemeSpectra | 252 | 0.19 | 1 | 0.43 |
| SelfRegulationSCP2 | 63 | 0.17 | 0 | — |
| UWaveGestureLibrary | 252 | 0.14 | 4 | 0.27 |
| UniMiB-SHAR | 252 | 0.19 | 6 | 0.38 |

## LODO pooled

| 口径 | n | AUC | Spearman | kept | z-bad 率 | strict 率 (n) | 好配置召回 (n) |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 1764 | 0.785 | 0.446 | 0.76 | 0.093 | 0.184 (347) | 0.906 (668) |
| random | 840 | 0.709 | 0.357 | 0.64 | 0.153 | 0.261 (138) | 0.796 (255) |

## 逐折（random 口径）

| 折 | τ | AUC | Spearman | kept | z-bad | strict (n) | 召回 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Handwriting | -0.374 | 0.767 | 0.493 | 0.77 | 0.103 | 0.134 (97) | 0.951 |
| UniMiB-SHAR | -0.175 | 0.839 | 0.597 | 0.67 | 0.024 | 0.151 (86) | 0.897 |
| PhonemeSpectra | -0.240 | 0.752 | 0.441 | 0.69 | 0.069 | 0.231 (13) | 0.867 |
| UWaveGestureLibrary | -0.212 | 0.786 | 0.543 | 0.74 | 0.038 | 0.156 (64) | 0.939 |
| EthanolConcentration | -0.826 | 0.432 | -0.254 | 0.86 | 0.157 | 0.789 (19) | 0.812 |
| Cricket | -0.351 | 0.757 | 0.543 | 0.75 | 0.074 | 0.194 (31) | 0.930 |
| HandMovementDirection | -0.554 | 0.550 | 0.109 | 0.85 | 0.160 | 0.108 (37) | 0.941 |

## 全数据模型：τ -0.384，S⁺ 占空间 0.653（100000 样本），OOF kept 0.78 / z-bad 0.100；信任域在 S⁺：{'4f7d4135': True, 'cffb5503': True, '8c4aa283': True, 'f48e5da1': True, 'a1f67db4': True, 'e490b88e': True, '9092e36e': True, '65627517': True, '20be4f74': False, '8b87d66f': True, 'c92058f2': True}

| τ 分位 | τ | 标签内 kept | z-bad 率 | 好配置召回 |
|---|---:|---:|---:|---:|
| 0.0 | -1.920 | 1.00 | 0.176 | 1.000 |
| 0.1 | -0.943 | 0.90 | 0.136 | 0.969 |
| 0.2 | -0.462 | 0.80 | 0.105 | 0.923 |
| 0.3 | -0.207 | 0.71 | 0.087 | 0.856 |
| 0.4 | -0.083 | 0.60 | 0.073 | 0.804 |
| 0.5 | +0.156 | 0.50 | 0.060 | 0.717 |
| 0.6 | +0.341 | 0.40 | 0.053 | 0.606 |
| 0.7 | +0.429 | 0.30 | 0.065 | 0.427 |
| 0.8 | +0.490 | 0.21 | 0.064 | 0.288 |

字段置换重要性（Spearman 下降，held-out 均值）：{'temperature': 0.193, 'sim_func': 0.109, 'norm': 0.09, 'rescale': 0.083, 'emb_mask': 0.007, 'emb_jitter': 0.007, 'freq_mask': 0.005, 'kernel': 0.004, 'crop': 0.003, 'jitter': 0.002, 'order': 0.002, 'adj_neighbor': 0.002, 'pool': 0.001, 'resize': 0.0, 'instance': 0.0, 'cross_scale': -0.0, 'point_mask': -0.002, 'loss_type': -0.005, 'temporal': -0.007}

ridge 对照（random）：{'auc': 0.6625878736277397}

预测：{'P1': {'desc': 'AUC ≥ 0.85（random 口径）', 'value': 0.7093784612602566, 'hit': False}, 'P2': {'desc': 'S⁺ 占空间 20–50%', 'value': 0.65341, 'hit': False}, 'P3': {'desc': '置换重要性前 2 = temperature, sim_func', 'value': ['temperature', 'sim_func', 'norm'], 'hit': True}, 'P4': {'desc': 'G-S2', 'hit': False}, 'P5': {'desc': '好配置召回 ≥ 0.9', 'value': 0.905688622754491, 'hit': True}, 'P6': {'desc': 'LODO Spearman（random）≥ 0.5', 'value': 0.3574663003488861, 'hit': False}}

