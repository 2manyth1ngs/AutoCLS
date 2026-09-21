# S1″ 第 3 步读数（ACTS-S1PP-GO；H：HEADROOM-NA）

G-A0 regret 门：29 靶均值 0.006（门 ≤ 0.01）、最大 0.055、名义丢冠军 8 → ✓

| 源 | 任务 | (i) 前一半 seed ρ | (ii) 净占比 | 真实份额 | (iii) IQR/σ_mean | 判定 |
|---|---:|---:|---:|---:|---:|---|
| Handwriting | 4 | 0.86 | 0.02 | 0.63 | 7.6 | 拒 net_share |
| UniMiB-SHAR | 4 | 0.71 | 0.02 | 0.45 | 4.1 | 拒 net_share |
| PhonemeSpectra | 4 | 0.56 | 0.14 | 0.64 | 2.8 | 拒 net_share |
| UWaveGestureLibrary | 4 | 0.44 | 0.18 | 0.68 | 2.7 | 拒 net_share |
| EthanolConcentration | 4 | 0.25 | 0.26 | 0.61 | 1.7 | 拒 rho_top,iqr_sigma |
| Cricket | 4 | 0.45 | 0.32 | 0.75 | 1.6 | 拒 iqr_sigma |
| HandMovementDirection | 4 | 0.11 | 0.21 | 0.53 | 1.5 | 拒 rho_top,net_share,iqr_sigma |
| SelfRegulationSCP2 | 4 | -0.13 | 0.23 | 0.51 | 1.0 | 拒 rho_top,net_share,iqr_sigma |
| SelfRegulationSCP1 | 4 | 0.13 | 0.20 | 0.57 | 1.6 | 拒 rho_top,net_share,iqr_sigma |
| PTB-XL | 4 | 0.61 | 0.21 | 0.70 | 3.0 | 拒 net_share |
| SleepEDF | 4 | 0.57 | 0.14 | 0.61 | 2.8 | 拒 net_share |
| PAMAP2 | 4 | 0.12 | 0.04 | 0.36 | 1.1 | 拒 rho_top,net_share,iqr_sigma |
| Opportunity | 4 | 0.17 | 0.05 | 0.34 | 1.7 | 拒 rho_top,net_share,iqr_sigma |

源池 []（底线 7，不足）

| 试点集 | 任务 | 净占比 | 毛占比 | 真实份额 | MS_noise | failed(诊断) shared | 两种子都超地板 中位 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PhonemeSpectra | 8 | 0.14 | 0.39 | 0.45 | 0.245 | 0.57 | 10 |
| Cricket | 8 | 0.41 | 0.55 | 0.77 | 0.117 | 0.57 | 12 |
| UWaveGestureLibrary | 8 | 0.36 | 0.58 | 0.69 | 0.198 | 0.30 | 20 |

合并真实份额 0.648（门 ≥ 0.5，逐集过 2/3）；净占比过门 2/3（门 ≥ 0.25）
余量 H_same regret@1 nan（中位 nan，regret@3 nan，0 任务）；H_cross nan → **HEADROOM-NA**
failed（探针地板口径）按 τ：{'0.01': 0.4303977272727273, '0.1': 0.5078125, '1.0': 0.7455357142857143}；配对产量中位 319 / 任务；标签 2323，双缺 2

| 门 | 结果 |
|---|---|
| G-A0 | ✓ |
| G-A1n | ✓ |
| G-A2 | ✓ |
