# UCR-ADM 事后：余量 H（零 GPU，规则 = S1″ PROTOCOL §1.8 / posthoc §2）

诚实口径：任务 oracle 按 seed 0 选、seed 1 评；任务盲底按 seed 0 在其他源（LODO）上选、seed 1 评；SH 的截点淘汰全按 seed 0。决策阈值（S1″）：先验@1 任务加权 ≥ 0.03 GO / < 0.02 落点 B / 中间 MID。

### v3′ 11（复现 S1″ posthoc）（56 任务，11 源）

| 底 | regret 均值 | 中位 | 档 |
|---|---:|---:|---|
| 先验@1 任务加权 | 0.022 | 0.016 | HEADROOM-MID |
| 前3全量实测 任务加权 | 0.017 | 0.008 | HEADROOM-B |
| SH6-3-3 任务加权 | 0.008 | 0.000 | HEADROOM-B |
| SH10-4-3 任务加权 | 0.006 | 0.000 | HEADROOM-B |
| 先验@1 源等权 | 0.023 | 0.018 | HEADROOM-MID |
| 前3全量实测 源等权 | 0.017 | 0.010 | HEADROOM-B |
| SH6-3-3 源等权 | 0.007 | 0.004 | HEADROOM-B |
| SH10-4-3 源等权 | 0.006 | 0.003 | HEADROOM-B |

逐源 regret@1（先验@1，任务加权）：Handwriting 0.092 (4)；EthanolConcentration 0.045 (4)；Cricket 0.027 (8)；SleepEDF 0.022 (4)；UWaveGestureLibrary 0.020 (8)；PAMAP2 0.018 (4)；SelfRegulationSCP1 0.015 (4)；PhonemeSpectra 0.010 (8)；PTB-XL 0.007 (4)；Opportunity -0.001 (4)；UniMiB-SHAR -0.002 (4)

### v3′ 11 + UCR 10 = 21（主）（96 任务，21 源）

| 底 | regret 均值 | 中位 | 档 |
|---|---:|---:|---|
| 先验@1 任务加权 | 0.031 | 0.018 | HEADROOM-GO |
| 前3全量实测 任务加权 | 0.013 | 0.000 | HEADROOM-B |
| SH6-3-3 任务加权 | 0.009 | 0.000 | HEADROOM-B |
| SH10-4-3 任务加权 | 0.003 | 0.000 | HEADROOM-B |
| 先验@1 源等权 | 0.032 | 0.018 | HEADROOM-GO |
| 前3全量实测 源等权 | 0.014 | 0.004 | HEADROOM-B |
| SH6-3-3 源等权 | 0.009 | 0.003 | HEADROOM-B |
| SH10-4-3 源等权 | 0.003 | 0.002 | HEADROOM-B |

逐源 regret@1（先验@1，任务加权）：SemgHandMovementCh2 0.216 (4)；Handwriting 0.092 (4)；LargeKitchenAppliances 0.054 (4)；EthanolConcentration 0.045 (4)；NonInvasiveFetalECGThorax1 0.039 (4)；EOGHorizontalSignal 0.034 (4)；UWaveGestureLibrary 0.031 (8)；Cricket 0.027 (8)；SleepEDF 0.022 (4)；PAMAP2 0.018 (4)；Adiac 0.018 (4)；ShapesAll 0.017 (4)；FiftyWords 0.016 (4)；CinCECGTorso 0.014 (4)；SelfRegulationSCP1 0.011 (4)；PhonemeSpectra 0.010 (8)；FordA 0.007 (4)；PTB-XL 0.007 (4)；Opportunity -0.001 (4)；MixedShapesSmallTrain -0.001 (4)；UniMiB-SHAR -0.002 (4)

### 护栏后 9 + UCR 10 = 19（去 PS / Ethanol）（84 任务，19 源）

| 底 | regret 均值 | 中位 | 档 |
|---|---:|---:|---|
| 先验@1 任务加权 | 0.024 | 0.015 | HEADROOM-MID |
| 前3全量实测 任务加权 | 0.014 | 0.000 | HEADROOM-B |
| SH6-3-3 任务加权 | 0.011 | 0.000 | HEADROOM-B |
| SH10-4-3 任务加权 | 0.003 | 0.000 | HEADROOM-B |
| 先验@1 源等权 | 0.024 | 0.016 | HEADROOM-MID |
| 前3全量实测 源等权 | 0.014 | 0.004 | HEADROOM-B |
| SH6-3-3 源等权 | 0.011 | 0.004 | HEADROOM-B |
| SH10-4-3 源等权 | 0.003 | 0.002 | HEADROOM-B |

逐源 regret@1（先验@1，任务加权）：Handwriting 0.092 (4)；SemgHandMovementCh2 0.082 (4)；LargeKitchenAppliances 0.054 (4)；NonInvasiveFetalECGThorax1 0.039 (4)；EOGHorizontalSignal 0.034 (4)；UWaveGestureLibrary 0.031 (8)；SleepEDF 0.022 (4)；Adiac 0.018 (4)；ShapesAll 0.017 (4)；FiftyWords 0.016 (4)；CinCECGTorso 0.014 (4)；Cricket 0.012 (8)；SelfRegulationSCP1 0.011 (4)；FordA 0.007 (4)；PTB-XL 0.007 (4)；Opportunity -0.001 (4)；MixedShapesSmallTrain -0.001 (4)；PAMAP2 -0.001 (4)；UniMiB-SHAR -0.002 (4)

### UCR 10（40 任务，10 源）

| 底 | regret 均值 | 中位 | 档 |
|---|---:|---:|---|
| 先验@1 任务加权 | 0.021 | 0.000 | HEADROOM-MID |
| 前3全量实测 任务加权 | 0.014 | 0.000 | HEADROOM-B |
| SH6-3-3 任务加权 | 0.005 | 0.000 | HEADROOM-B |
| SH10-4-3 任务加权 | 0.000 | 0.000 | HEADROOM-B |
| 先验@1 源等权 | 0.021 | 0.006 | HEADROOM-MID |
| 前3全量实测 源等权 | 0.014 | 0.002 | HEADROOM-B |
| SH6-3-3 源等权 | 0.005 | 0.000 | HEADROOM-B |
| SH10-4-3 源等权 | 0.000 | 0.000 | HEADROOM-B |

逐源 regret@1（先验@1，任务加权）：SemgHandMovementCh2 0.170 (4)；LargeKitchenAppliances 0.017 (4)；FiftyWords 0.015 (4)；EOGHorizontalSignal 0.008 (4)；CinCECGTorso 0.007 (4)；Adiac 0.005 (4)；NonInvasiveFetalECGThorax1 0.003 (4)；ShapesAll 0.000 (4)；MixedShapesSmallTrain -0.004 (4)；FordA -0.012 (4)

## 跨域（只作上界）

- 三组（uea / external / ucr）按组 LODO：先验@1 0.025（uea 0.028 / external 0.011 / ucr 0.028）；前 3 0.014；SH6-3-3 0.010；SH10-4-3 0.009
- 非 UCR 11 源选 → UCR 40 任务评：先验@1 0.028 / 前 3 0.017 / SH6 0.011 / SH10 0.010
- UCR 10 源选 → 非 UCR 56 任务评：先验@1 0.023 / 前 3 0.011 / SH6 0.010 / SH10 0.005
- **H_cross（两向均值，先验@1）= 0.025 → HEADROOM-MID**

## 池大小曲线（主池 21 源 96 任务；训练侧随机抽 k 个其他源，40 次均值）

| k | 先验@1 | 前3全量实测 | SH6-3-3 |
|---:|---:|---:|---:|
| 2 | 0.028 | 0.015 | 0.010 |
| 4 | 0.028 | 0.015 | 0.009 |
| 6 | 0.028 | 0.015 | 0.010 |
| 8 | 0.027 | 0.014 | 0.009 |
| 10 | 0.027 | 0.014 | 0.009 |
| 12 | 0.027 | 0.014 | 0.009 |
| 15 | 0.028 | 0.013 | 0.009 |
| 20 | 0.031 | 0.013 | 0.009 |

## 稳健性（协议外，读 0.031 必需；`headroom_H_robust.json`）

| 池 | H_same（先验@1 任务加权） | 去一源 jackknife 范围 | 去 Semg | 种子互换（seed 1 选 / seed 0 评） | 源 bootstrap 5–95% | P(≥ 0.03) | P(< 0.02) |
|---|---:|---|---:|---:|---|---:|---:|
| 21（v3′ 11 + UCR 10） | 0.031 | 0.020–0.035 | 0.020 | 0.022（前 3 0.009 / SH6 0.007 / SH10 0.002） | 0.018–0.050 | 0.56 | 0.11 |
| 19（护栏后 9 + UCR 10） | 0.024 | 0.020–0.034 | 0.024 | 0.029（前 3 0.011 / SH6 0.008 / SH10 0.003） | 0.015–0.034 | 0.16 | 0.28 |

机制：21 池的 LODO 全局冠军在 80/96 任务是 018d72f2、16 任务是 2a667428（两条都 τ 0.01，全池近乎平局，训练侧换几个源就翻）；Semg 的 4 个任务恰好抽到 2a667428，而它在 Semg 上是灾难（0.40–0.54 vs oracle 0.49–0.85 → regret 0.17–0.32），去 PS / Ethanol 后冠军回到 018d72f2、Semg regret 降到 0.082。**0.031 过 GO 线是"顶部两条配置平局 + 一个源对平局敏感"给的，不是任务感知余量整体抬高**：去 Semg 0.020、种子互换 0.022、bootstrap 只有 56% 落在 ≥ 0.03。三个测量式底（前 3 / SH6-3-3 / SH10-4-3）在所有池、所有稳健性变体下都 < 0.015（B 档）。

## 读法

- 按 S1″ 规则：主池 21 的 H_same = 0.031 → 名义 **HEADROOM-GO**，但落在阈值上且不稳（见上）；UCR-ADM 自己的准入尺子（v3′ ∧ 护栏）下一致的池是 19 源 → 0.024 → **HEADROOM-MID**（效率门）；与 S1″ v3′ 池 0.022 / MID 同档。
- 扩池 11 → 21 没有改变余量的量级：任务感知相对单条默认 ≈ 0.02–0.03、相对三条实测 ≈ 0.01、相对 SH ≈ 0.003–0.01（与 S1″ / MX-4 → K3 同形）。池大小曲线在 k = 2–20 平坦（0.027–0.031），加源既不抬也不压余量。
- 跨域 H_cross 0.025（MID）：非 UCR → UCR 0.028、UCR → 非 UCR 0.023，两个方向都与同域 LODO 同量级——UCR 十源与现池的偏好序同向（REPORT §4 的 +SAME）在余量上的表现。
- 落点：与 S1″ 的 H 决策一致，"0.02 ≤ H < 0.03 → 铺 S2 但只报效率门，任务感知门作探索读数"；若按 21 池名义 GO 立项，须把"顶部平局"写进 S2 协议的门（regret@1 对平局敏感，主指标改用 regret@3 或去掉平局对）。
