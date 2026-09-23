# WP17-ACTS S2-CORPUS 中期报告（2026-09-23 04:35 UTC，跑到 ~50%）

协议 `PROTOCOL.md`（锁 ff7bcfa，修订 1 = 0520fc4）｜ 实现 119ada9 ｜ 实验室 8k-4090 GPU 3/4/6/5 四进程，09-22 10:13 UTC 起，墙钟 18.4 h
**本文件是进度快照，不是读数**：门（G-P / G-R / G-A2′）按协议只在全部格跑完、`--analyze-s2` 出 `analysis_S.json` 后才判；下表的可靠性 / 护栏数只是"哪些源已经跑完、数字大致落在哪"，**不作准入结论**。

## 1. 进度

| 项 | 值 |
|---|---|
| plan_S 总格 | 14,643（含复用 S1″ + UCR-ADM 的 5,683） |
| 已有 | 10,133（69.2%）＝ 种子 5,683 + **本轮新跑 4,450** |
| 剩余 | 4,510（shared 2,977 / random 1,533） |
| 本轮成本 | 80.8 GPU-h，单格中位 **45.8 s**、均值 65.5 s |
| 剩余估计 | 按逐源实测单格成本 ≈ **78 GPU-h**，四进程并行 ≈ **19.5 h** → 预计 09-24 00:00 UTC 前后跑完 |
| 异常 | Traceback 0；OOM 10 条（全在 PTB-XL_a06，10 个不同 key、无重复，收官 verify 会补跑）；error 行 11 条（EvalTimeout 480 s：PAMAP2_a01 5 / PS_a07 2 / UniMiB_a07 2 / PTB-XL 2，占 0.11%，与 S1″ 同形） |
| 重启 | 0 次（守护未触发；共享盘写满问题已由根盘 symlink 规避） |

**剩余集中在反向源与试用源**（它们每源 16 / 8 行 × 60 格，且起步晚）：CinCECGTorso 1,040、SelfRegulationSCP1 1,040、EthanolConcentration 1,007、SemgHandMovementCh2 513、HMD 400、SCP2 400；13 个正向源里 8 个已 100%、其余 5 个各剩 ≤ 40 格。

## 2. 已完成任务上的中间量（只作进度参考，**不是门**）

| 源 | 完成任务 | ρ_seed 中位 | ρ ≥ 0.5 行 | h_t 中位 | h ≥ 0.10 行 |
|---|---:|---:|---:|---:|---:|
| Handwriting | 8 | 0.95 | 7 | 0.340 | 7 |
| ShapesAll | 8 | 0.94 | 8 | 0.692 | 8 |
| UniMiB-SHAR | 8 | 0.90 | 8 | 0.410 | 8 |
| NonInvasiveFetalECGThorax1 | 8 | 0.88 | 8 | 0.637 | 8 |
| PhonemeSpectra | 8 | 0.88 | 7 | 0.067 | 2 |
| Cricket | 8 | 0.87 | 8 | 0.799 | 8 |
| UWaveGestureLibrary | 8 | 0.85 | 7 | 0.686 | 8 |
| FordA | 8 | 0.83 | 8 | 0.309 | 8 |
| **CinCECGTorso**（反向，跑了 4/16） | 4 | 0.83 | 4 | 0.726 | 4 |
| LargeKitchenAppliances | 8 | 0.81 | 7 | 0.315 | 8 |
| **SemgHandMovementCh2**（反向，10/16） | 10 | 0.79 | 8 | 0.330 | 10 |
| PTB-XL | 8 | 0.79 | 7 | 0.249 | 8 |
| Adiac | 8 | 0.78 | 8 | 0.669 | 8 |
| SleepEDF | 8 | 0.78 | 7 | 0.139 | 6 |
| PAMAP2 | 8 | 0.72 | 7 | 0.593 | 8 |
| **SelfRegulationSCP1**（反向，4/16） | 4 | 0.69 | 3 | 0.311 | 4 |
| **EthanolConcentration**（反向，4/16） | 4 | 0.65 | 3 | 0.075 | 1 |
| **HandMovementDirection**（试用，4/8） | 4 | 0.46 | 1 | 0.011 | 0 |
| **SelfRegulationSCP2**（试用，4/8） | 4 | 0.34 | 1 | 0.033 | 0 |

读法（谨慎）：
- 正向 13 源在新跑的 a00 / a02 / a04 / a06 行上与 S1″ / UCR-ADM 的 4 行同形，没有"新行特别吵"的迹象。
- **两个试用源（HMD / SCP2）目前 4/8 行里 ρ ≥ 0.5 各 1 行、h_t ≥ 0.10 各 0 行**，离 G-P 的 5/8 + 5/8 很远。按协议这是"跑完再判"，但方向与预注册先验（各 30%）一致；若最终不过，按协议自动剔除、标签留作靶。
- 反向源里 CinC / Semg 稳（ρ 0.79–0.83、h 0.33–0.73），SCP1 0.69 / Ethanol 0.65 偏低且 Ethanol 的 h_t 中位只 0.075——G-R 的 10/16 门要等 16 行跑完才有定论。
- PhonemeSpectra 的 h_t 0.067（2/8 行过 0.10）再次显示它是 chance 型；它在 S1″ 就靠 v3′ 进池、被护栏拒过。这不改变它在 S2 的角色（正向源，不受 G-P / G-R 约束），只是提醒 S3 的 E 消融要注意它。

## 3. 下一步
跑完 → `--verify --phase S`（补 10 条 OOM）→ `--analyze-s2`（G-P / G-R / G-A2′ + H_cat / H_rand_adj + 轴稳定性 + random z-bad + 配对产量）→ REPORT → PROTOCOL §5 / CLAUDE.md / memory。
