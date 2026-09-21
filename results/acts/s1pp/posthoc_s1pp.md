# S1″ 事后读数（协议外探索；正式读数见 tables_C.md）

## 1. v3′ 准入（只看可靠性：任务级 Spearman(seed0, seed1) 全 30 条 ≥ 0.5，任务过半 → 源准入）

| 源 | 任务 | 各任务 ρ | 过 | 判定 |
|---|---:|---|---|---|
| Handwriting | 4 | 0.96 0.97 0.96 0.95 | 4/4 | 准入 |
| UniMiB-SHAR | 4 | 0.95 0.94 0.88 0.96 | 4/4 | 准入 |
| PhonemeSpectra | 8 | 0.65 0.94 0.91 0.91 0.76 0.86 0.52 0.90 | 8/8 | 准入 |
| UWaveGestureLibrary | 8 | 0.17 0.81 0.89 0.91 0.58 0.89 0.81 0.89 | 7/8 | 准入 |
| EthanolConcentration | 4 | 0.66 0.73 0.65 0.38 | 3/4 | 准入 |
| Cricket | 8 | 0.68 0.88 0.94 0.94 0.86 0.85 0.92 0.74 | 8/8 | 准入 |
| HandMovementDirection | 4 | 0.47 0.53 0.33 0.43 | 1/4 | 拒 |
| SelfRegulationSCP2 | 4 | 0.61 0.32 0.36 0.25 | 1/4 | 拒 |
| SelfRegulationSCP1 | 4 | 0.73 0.81 0.68 0.46 | 3/4 | 准入 |
| PTB-XL | 4 | 0.93 0.85 0.88 0.75 | 4/4 | 准入 |
| SleepEDF | 4 | 0.88 0.94 0.73 0.84 | 4/4 | 准入 |
| PAMAP2 | 4 | 0.96 0.75 0.73 0.84 | 4/4 | 准入 |
| Opportunity | 4 | 0.86 0.80 0.39 0.71 | 3/4 | 准入 |

v3′ 源池 = [np.str_('Handwriting'), np.str_('UniMiB-SHAR'), np.str_('PhonemeSpectra'), np.str_('UWaveGestureLibrary'), np.str_('EthanolConcentration'), np.str_('Cricket'), np.str_('SelfRegulationSCP1'), np.str_('PTB-XL'), np.str_('SleepEDF'), np.str_('PAMAP2'), np.str_('Opportunity')]（11 个）

## 2. 余量 H（诚实：oracle seed 0 选 seed 1 评；任务盲底在其他数据集上按 seed 0 选）

### 全部 64 任务（64 任务，13 源）

| 底 | regret 均值 | 中位 | 档 |
|---|---:|---:|---|
| 先验@1 任务加权 | 0.023 | 0.013 | HEADROOM-MID |
| 前3全量实测 任务加权 | 0.018 | 0.003 | HEADROOM-B |
| SH6-3-3 任务加权 | 0.011 | 0.000 | HEADROOM-B |
| SH10-4-3 任务加权 | 0.009 | 0.000 | HEADROOM-B |
| 先验@1 源等权 | 0.023 | 0.013 | HEADROOM-MID |
| 前3全量实测 源等权 | 0.021 | 0.008 | HEADROOM-MID |
| SH6-3-3 源等权 | 0.014 | 0.000 | HEADROOM-B |
| SH10-4-3 源等权 | 0.009 | 0.000 | HEADROOM-B |

逐源 regret@1（先验@1，任务加权）：Handwriting 0.092 (4)；HandMovementDirection 0.080 (4)；EthanolConcentration 0.045 (4)；UWaveGestureLibrary 0.031 (8)；SleepEDF 0.022 (4)；SelfRegulationSCP2 0.014 (4)；Cricket 0.012 (8)；SelfRegulationSCP1 0.011 (4)；PhonemeSpectra 0.010 (8)；PTB-XL 0.007 (4)；Opportunity -0.001 (4)；PAMAP2 -0.001 (4)；UniMiB-SHAR -0.002 (4)

### v3′ 池（56 任务，11 源）

| 底 | regret 均值 | 中位 | 档 |
|---|---:|---:|---|
| 先验@1 任务加权 | 0.022 | 0.016 | HEADROOM-MID |
| 前3全量实测 任务加权 | 0.017 | 0.008 | HEADROOM-B |
| SH6-3-3 任务加权 | 0.008 | 0.000 | HEADROOM-B |
| SH10-4-3 任务加权 | 0.006 | 0.000 | HEADROOM-B |
| 先验@1 源等权 | 0.019 | 0.012 | HEADROOM-B |
| 前3全量实测 源等权 | 0.018 | 0.010 | HEADROOM-B |
| SH6-3-3 源等权 | 0.007 | 0.000 | HEADROOM-B |
| SH10-4-3 源等权 | 0.005 | 0.000 | HEADROOM-B |

逐源 regret@1（先验@1，任务加权）：Handwriting 0.092 (4)；EthanolConcentration 0.045 (4)；Cricket 0.027 (8)；SleepEDF 0.022 (4)；UWaveGestureLibrary 0.020 (8)；PAMAP2 0.018 (4)；SelfRegulationSCP1 0.015 (4)；PhonemeSpectra 0.010 (8)；PTB-XL 0.007 (4)；Opportunity -0.001 (4)；UniMiB-SHAR -0.002 (4)

### v3′ 池去 PS（48 任务，10 源）

| 底 | regret 均值 | 中位 | 档 |
|---|---:|---:|---|
| 先验@1 任务加权 | 0.022 | 0.010 | HEADROOM-MID |
| 前3全量实测 任务加权 | 0.018 | 0.007 | HEADROOM-B |
| SH6-3-3 任务加权 | 0.008 | 0.000 | HEADROOM-B |
| SH10-4-3 任务加权 | 0.006 | 0.000 | HEADROOM-B |
| 先验@1 源等权 | 0.019 | 0.012 | HEADROOM-B |
| 前3全量实测 源等权 | 0.019 | 0.008 | HEADROOM-B |
| SH6-3-3 源等权 | 0.008 | 0.000 | HEADROOM-B |
| SH10-4-3 源等权 | 0.005 | 0.000 | HEADROOM-B |

逐源 regret@1（先验@1，任务加权）：Handwriting 0.092 (4)；EthanolConcentration 0.045 (4)；UWaveGestureLibrary 0.031 (8)；SleepEDF 0.022 (4)；Cricket 0.012 (8)；SelfRegulationSCP1 0.011 (4)；PTB-XL 0.007 (4)；Opportunity -0.001 (4)；PAMAP2 -0.001 (4)；UniMiB-SHAR -0.002 (4)

### 试点 3 集 8 行（24 任务，3 源）

| 底 | regret 均值 | 中位 | 档 |
|---|---:|---:|---|
| 先验@1 任务加权 | 0.030 | 0.035 | HEADROOM-MID |
| 前3全量实测 任务加权 | 0.015 | 0.010 | HEADROOM-B |
| SH6-3-3 任务加权 | 0.007 | 0.000 | HEADROOM-B |
| SH10-4-3 任务加权 | 0.008 | 0.000 | HEADROOM-B |
| 先验@1 源等权 | 0.030 | 0.035 | HEADROOM-MID |
| 前3全量实测 源等权 | 0.015 | 0.010 | HEADROOM-B |
| SH6-3-3 源等权 | 0.007 | 0.000 | HEADROOM-B |
| SH10-4-3 源等权 | 0.008 | 0.000 | HEADROOM-B |

逐源 regret@1（先验@1，任务加权）：PhonemeSpectra 0.038 (8)；Cricket 0.027 (8)；UWaveGestureLibrary 0.023 (8)

跨域（external ↔ UEA，v3′ 池）先验@1 regret 均值 0.022（56 任务）→ HEADROOM-MID

## 3. 源间偏好差异（池级）：配置的源级平均秩（seed 0）方差分解——源间 / 源内任务 / 残差

SS 源间 0.48 / 源内任务 0.52（总 = 1，配置主效应已随秩去掉均值在总里）；

| 源 | 与其余源平均序的 Spearman | 该源前 3 的 τ |
|---|---:|---|
| Handwriting | 0.79 | ['0.1', '0.1', '0.1'] |
| UniMiB-SHAR | 0.87 | ['0.1', '0.1', '0.01'] |
| PhonemeSpectra | 0.65 | ['0.01', '0.1', '0.1'] |
| UWaveGestureLibrary | 0.79 | ['0.1', '0.1', '0.01'] |
| EthanolConcentration | -0.07 | ['0.01', '0.01', '0.01'] |
| Cricket | 0.90 | ['0.01', '0.1', '0.01'] |
| HandMovementDirection | -0.22 | ['0.1', '0.1', '0.01'] |
| SelfRegulationSCP2 | -0.12 | ['0.01', '1.0', '0.01'] |
| SelfRegulationSCP1 | 0.35 | ['0.1', '1.0', '0.01'] |
| PTB-XL | 0.80 | ['0.01', '0.1', '0.1'] |
| SleepEDF | 0.66 | ['0.01', '0.1', '0.01'] |
| PAMAP2 | 0.71 | ['0.01', '0.1', '0.01'] |
| Opportunity | 0.68 | ['0.1', '0.1', '0.01'] |

## 4. 成本 / error

plan_C 4,243 格全部在缓存；格级累加 预训练 59.5 h + 评估 13.5 h = 73.0 GPU-h（含复用的 S1′ 1,627 格）；error 4：[('PhonemeSpectra_a07', '8c482976', 0, 'EvalTimeout: checkpoint eval exceeded 480s'), ('PhonemeSpectra_a07', '8c482976', 1, 'EvalTimeout: checkpoint eval exceeded 480s'), ('PAMAP2_a01', '8c482976', 0, 'EvalTimeout: checkpoint eval exceeded 480s'), ('PAMAP2_a01', '8c482976', 1, 'EvalTimeout: checkpoint eval exceeded 480s')]

