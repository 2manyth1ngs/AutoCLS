# S2-CORPUS 读数（S2-CORPUS-GO+PROB0）

G-A2′ 真实份额 0.736（门 ≥ 0.5；净占比 0.612、MS_noise 0.211）→ ✓

| 试用源 | 行 | ρ_seed ≥ 0.5 | h_t ≥ 0.10 | 判定 |
|---|---:|---:|---:|---|
| HandMovementDirection | 8 | 1 | 0 | 剔除 |
| SelfRegulationSCP2 | 8 | 2 | 0 | 剔除 |

| 反向源 | 行 | ρ_seed ≥ 0.5 | h_t ≥ 0.10 | 判定 |
|---|---:|---:|---:|---|
| SemgHandMovementCh2 | 16 | 15 | 16 | 16 片 |
| EthanolConcentration | 16 | 7 | 4 | 降 8 片 |
| CinCECGTorso | 16 | 15 | 16 | 16 片 |
| SelfRegulationSCP1 | 16 | 10 | 16 | 16 片 |

门后源池 17：['Handwriting', 'UniMiB-SHAR', 'PhonemeSpectra', 'UWaveGestureLibrary', 'Cricket', 'PTB-XL', 'SleepEDF', 'PAMAP2', 'NonInvasiveFetalECGThorax1', 'ShapesAll', 'LargeKitchenAppliances', 'Adiac', 'FordA', 'SemgHandMovementCh2', 'EthanolConcentration', 'CinCECGTorso', 'SelfRegulationSCP1']

| 源 | 行 | ρ_seed 中位 / 最低 | h_t 中位 / 最低 | failed(诊断) | 轴得分 |
|---|---:|---|---|---:|---:|
| Handwriting | 8 | 0.94 / 0.23 | 0.340 / 0.038 | 0.29 | 0.95 |
| UniMiB-SHAR | 8 | 0.90 / 0.72 | 0.410 / 0.205 | 0.35 | 2.03 |
| PhonemeSpectra | 8 | 0.88 / 0.52 | 0.067 / 0.034 | 0.57 | 2.44 |
| UWaveGestureLibrary | 8 | 0.85 / 0.17 | 0.686 / 0.242 | 0.30 | -0.06 |
| Cricket | 8 | 0.87 / 0.68 | 0.799 / 0.399 | 0.57 | 0.10 |
| PTB-XL | 8 | 0.79 / 0.43 | 0.249 / 0.104 | 0.55 | 0.47 |
| SleepEDF | 8 | 0.79 / 0.39 | 0.139 / -0.000 | 0.56 | 0.97 |
| PAMAP2 | 8 | 0.72 / 0.48 | 0.593 / 0.163 | 0.65 | 0.98 |
| NonInvasiveFetalECGThorax1 | 8 | 0.88 / 0.69 | 0.637 / 0.429 | 0.38 | 0.44 |
| ShapesAll | 8 | 0.94 / 0.78 | 0.692 / 0.449 | 0.40 | 1.26 |
| LargeKitchenAppliances | 8 | 0.81 / 0.35 | 0.315 / 0.247 | 0.83 | -0.63 |
| Adiac | 8 | 0.78 / 0.55 | 0.669 / 0.500 | 0.28 | 2.20 |
| FordA | 8 | 0.83 / 0.51 | 0.309 / 0.225 | 0.51 | 2.14 |
| SemgHandMovementCh2 | 16 | 0.83 / 0.39 | 0.350 / 0.186 | 0.51 | -3.31 |
| EthanolConcentration | 16 | 0.44 / -0.10 | 0.065 / 0.006 | 0.61 | -2.57 |
| CinCECGTorso | 16 | 0.79 / 0.41 | 0.703 / 0.236 | 0.49 | -2.09 |
| SelfRegulationSCP1 | 16 | 0.58 / -0.10 | 0.299 / 0.252 | 0.44 | -1.71 |
| HandMovementDirection | 8 | 0.23 / -0.14 | 0.011 / -0.006 | 0.67 | -2.06 |
| SelfRegulationSCP2 | 8 | 0.29 / 0.06 | 0.033 / 0.004 | 0.67 | -1.53 |

轴：PC1 占比 0.34；源得分 vs UCR-ADM 事后 4 行版 Spearman 0.87；反向源前 8 行投影 {SemgHandMovementCh2: -3.62, EthanolConcentration: -2.55, CinCECGTorso: -2.34, SelfRegulationSCP1: -1.33}
H_cat（目录内；LODO，seed 1 评；任务加权）：先验@1 0.043 / 前 3 0.024 / SH6-3-3 0.017 / SH10-4-3 0.006；regret@3 中位 0.006
**H_rand_adj（主指标；oracle 扩到 shared ∪ random，扣 κ_n·σ_i）**：先验@1 0.048 / **前 3 0.030** / SH6-3-3 0.022 / SH10-4-3 0.012；regret@3 中位 0.011；H_rand_raw 前 3 0.048；oracle 扩展：raw 份额 0.72 / 均值 0.024，adj 份额 0.20 / 均值 0.005（κ 均值 1.86、σ 中位 0.023）；判读：H_rand 只能给 GO（§1.5）
random-20 z-bad 均值 0.00；配对产量中位 640；failed 按 τ {'0.01': 0.43132411067193677, '0.1': 0.4633152173913043, '1.0': 0.7018633540372671}；单格中位 48.8 s（14631 格，282.7 GPU-h）；error 12；标签 9115
