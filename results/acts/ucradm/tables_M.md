# UCR-ADM 读数（UCR-ADM-GO+SAME）

准入 10/10：['NonInvasiveFetalECGThorax1', 'CinCECGTorso', 'SemgHandMovementCh2', 'EOGHorizontalSignal', 'ShapesAll', 'MixedShapesSmallTrain', 'LargeKitchenAppliances', 'Adiac', 'FiftyWords', 'FordA']；参照池（S1″ v3′）11 源

| 源 | 任务 | ρ_t（v3′） | pass | h_t（护栏） | pass_h | v2 前一半 ρ | 净占比 | 真实份额 | IQR/σ_mean | failed(诊断) | 都超地板 中位 | ρ vs 池均值 | 判定 |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NonInvasiveFetalECGThorax1 | 4 | 0.91 0.69 0.87 0.84 | 4/4 | 0.738 0.429 0.664 0.740 | 4/4 | 0.66 | 0.10 | 0.68 | 4.2 | 0.36 | 16 | 0.73 | 准入 |
| CinCECGTorso | 4 | 0.84 0.82 0.82 0.93 | 4/4 | 0.741 0.628 0.737 0.716 | 4/4 | 0.48 | 0.28 | 0.75 | 2.8 | 0.54 | 12 | 0.49 | 准入 |
| SemgHandMovementCh2 | 4 | 0.95 0.81 0.83 0.76 | 4/4 | 0.506 0.281 0.467 0.512 | 4/4 | 0.73 | 0.18 | 0.82 | 4.6 | 0.52 | 14 | 0.22 | 准入 |
| EOGHorizontalSignal | 4 | 0.92 0.95 0.37 0.86 | 3/4 | 0.720 0.627 0.242 0.581 | 4/4 | 0.61 | 0.16 | 0.68 | 3.4 | 0.67 | 12 | 0.87 | 准入 |
| ShapesAll | 4 | 0.98 0.86 0.94 0.95 | 4/4 | 0.831 0.514 0.700 0.807 | 4/4 | 0.87 | 0.32 | 0.92 | 5.3 | 0.37 | 17 | 0.80 | 准入 |
| MixedShapesSmallTrain | 4 | 0.86 0.89 0.88 0.88 | 4/4 | 0.651 0.537 0.616 0.659 | 4/4 | 0.47 | 0.19 | 0.69 | 2.5 | 0.67 | 8 | 0.83 | 准入 |
| LargeKitchenAppliances | 4 | 0.93 0.67 0.79 0.92 | 4/4 | 0.359 0.287 0.282 0.492 | 4/4 | 0.50 | 0.08 | 0.51 | 2.0 | 0.86 | 3 | 0.59 | 准入 |
| Adiac | 4 | 0.89 0.81 0.94 0.76 | 4/4 | 0.715 0.664 0.703 0.676 | 4/4 | 0.58 | 0.10 | 0.61 | 2.1 | 0.30 | 22 | 0.74 | 准入 |
| FiftyWords | 4 | 0.93 0.86 0.88 0.94 | 4/4 | 0.585 0.367 0.515 0.617 | 4/4 | 0.69 | 0.24 | 0.82 | 4.2 | 0.52 | 12 | 0.79 | 准入 |
| FordA | 4 | 0.88 0.77 0.85 0.88 | 4/4 | 0.331 0.286 0.332 0.338 | 4/4 | 0.45 | 0.03 | 0.38 | 2.4 | 0.45 | 14 | 0.75 | 准入 |

多样性：+DIVERSE 计数 1（ρ < 0.3）；合池 SS_between 0.60（参照池 0.59，UCR 池 0.57）
failed（探针地板口径）按 τ：{'0.01': 0.44772727272727275, '0.1': 0.44583333333333336, '1.0': 0.7821428571428571}；单格耗时中位 52.3 s（2400 格，47.4 GPU-h）；error 0；标签 1200

| 源 | 合池 agree（与其余源平均序的 Spearman） |
|---|---:|
| Handwriting | 0.86 |
| UniMiB-SHAR | 0.89 |
| PhonemeSpectra | 0.65 |
| UWaveGestureLibrary | 0.88 |
| EthanolConcentration | 0.02 |
| Cricket | 0.78 |
| SelfRegulationSCP1 | 0.39 |
| PTB-XL | 0.84 |
| SleepEDF | 0.75 |
| PAMAP2 | 0.63 |
| Opportunity | 0.63 |
| NonInvasiveFetalECGThorax1 | 0.89 |
| CinCECGTorso | 0.74 |
| SemgHandMovementCh2 | 0.32 |
| EOGHorizontalSignal | 0.88 |
| ShapesAll | 0.93 |
| MixedShapesSmallTrain | 0.92 |
| LargeKitchenAppliances | 0.68 |
| Adiac | 0.78 |
| FiftyWords | 0.89 |
| FordA | 0.75 |
