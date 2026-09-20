# S1′ 阶段 A 读数（ADM / BUD）

| 候选源 | 任务 | 存活 | ρ(seed0, seed1) | failed s0 / s1 | 地板 | 发表差 |
|---|---|---:|---:|---|---:|---:|
| Handwriting | a01 | 13 | 0.885 | 0.30 / 0.30 | 0.323 | nan |
| Handwriting | a07 | 14 | 0.899 | 0.30 / 0.30 | 0.455 | nan |
| UniMiB-SHAR | a01 | 10 | 0.927 | 0.50 / 0.50 | 0.693 | nan |
| UniMiB-SHAR | a07 | 10 | 0.745 | 0.45 / 0.50 | 0.569 | nan |
| PhonemeSpectra | a01 | 8 | 0.643 | 0.60 / 0.60 | 0.145 | nan |
| PhonemeSpectra | a07 | 7 | nan | 0.60 / 0.60 | 0.101 | nan |
| UWaveGestureLibrary | a01 | 10 | 0.378 | 0.40 / 0.50 | 0.770 | nan |
| UWaveGestureLibrary | a07 | 11 | 0.897 | 0.45 / 0.35 | 0.850 | nan |
| EthanolConcentration | a01 | 3 | nan | 0.80 / 0.85 | 0.346 | 0.057 |
| EthanolConcentration | a07 | 9 | 0.700 | 0.45 / 0.45 | 0.466 | 0.057 |
| Cricket | a01 | 9 | 0.628 | 0.45 / 0.50 | 0.897 | nan |
| Cricket | a07 | 0 | nan | 0.85 / 0.85 | 0.861 | nan |
| HandMovementDirection | a01 | 1 | nan | 0.75 / 0.75 | 0.298 | -0.067 |
| HandMovementDirection | a07 | 4 | nan | 0.65 / 0.60 | 0.304 | -0.067 |
| SelfRegulationSCP2 | a01 | 7 | nan | 0.50 / 0.55 | 0.527 | 0.078 |
| SelfRegulationSCP2 | a07 | 1 | nan | 0.80 / 0.85 | 0.537 | 0.078 |
| SelfRegulationSCP1 | a01 | 12 | 0.628 | 0.30 / 0.15 | 0.790 | 0.381 |
| SelfRegulationSCP1 | a07 | 10 | 0.135 | 0.45 / 0.35 | 0.750 | 0.381 |

ADM：准入 ['Handwriting', 'UniMiB-SHAR', 'UWaveGestureLibrary']；拒 ['PhonemeSpectra', 'EthanolConcentration', 'Cricket', 'HandMovementDirection', 'SelfRegulationSCP2']；fallback ['SelfRegulationSCP1']；源池 ['Handwriting', 'UniMiB-SHAR', 'UWaveGestureLibrary', 'SelfRegulationSCP1']（底线 7，不足）；试点集 ['PhonemeSpectra', 'Cricket', 'UWaveGestureLibrary']

| Cricket 任务 | n | failed 1× | failed 3× | 3× 存活 | ρ(1/3, 1) | ρ(2/3, 1) | mean 1× / 3× |
|---|---:|---:|---:|---:|---:|---:|---|
| Cricket_a00 | 20 | 0.55 | 0.40 | 12 | 0.568 | 0.807 | 0.460 / 0.450 |
| Cricket_a02 | 20 | 0.75 | 0.90 | 2 | nan | nan | 0.844 / 0.821 |
| Cricket_a01 | 20 | 0.45 | 0.50 | 10 | 0.842 | 0.942 | 0.853 / 0.834 |
| Cricket_a04 | 20 | 0.35 | 0.50 | 10 | 0.522 | 0.837 | 0.504 / 0.478 |
| Cricket_a03 | 20 | 0.85 | 0.85 | 3 | nan | nan | 0.701 / 0.698 |
| Cricket_a05 | 20 | 0.50 | 0.45 | 11 | 0.411 | 0.723 | 0.894 / 0.860 |
| Cricket_a06 | 20 | 0.70 | 0.70 | 6 | 0.045 | 0.883 | 0.871 / 0.860 |
| Cricket_a07 | 20 | 0.85 | 1.00 | 0 | nan | nan | 0.755 / 0.728 |

BUD：failed 降 -0.037（门 ≥ 0.15）、存活区 ρ(1/3, 1) 中位 0.522（门 < 0.5）→ **1x**
