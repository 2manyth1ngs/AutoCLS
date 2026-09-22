# S2 修订 1 零 GPU 复算：S1″ 试点 24 任务的 H_cat vs H_rand（`pilot_hrand.py`，83 新函数）

口径 = `cmd_analyze_s2`：frames（shared 两种子终点、random 单种子）、M = 两种子终点 + q25 / q50 截点、LODO 先验只在其他数据集上、oracle_cat 按 seed 0 选 seed 1 评；
H_rand_raw = max(oracle_cat, max_j R0_j)，H_rand_adj = max(oracle_cat, max_j R0_j − κ_n·σ_i)（κ_n = n 个标准正态最大值期望、σ_i = shared 两种子差 sd/√2）。
试点 = S1″ 有 random-20 的 24 任务（PS / Cricket / UWave 各 8 行；random 有效条数中位 17，单种子）。

| 底 | H_cat | H_rand_raw | H_rand_adj |
|---|---:|---:|---:|
| 先验@1 | 0.0295 | 0.0472 | 0.0322 |
| 前 3 全量实测 | 0.0152 | 0.0329 | 0.0179 |
| SH6-3-3 | 0.0069 | 0.0246 | 0.0096 |
| SH10-4-3 | 0.0078 | 0.0255 | 0.0105 |
| regret@3 中位 | 0.0104 | 0.0196 | 0.0137 |

oracle 扩展：random 原始最好抬高目录 oracle 的任务 **0.67**（均值 +0.0177）；扣 κ_n·σ_i 后 **0.17**（均值 +0.0027）；κ 均值 1.78、σ 中位 0.0255。
逐任务 H_rand_adj − H_cat（regret@3）> 0 的任务 4/24，最大 +0.046。

全部 64 任务版（无 random 的 40 任务 H_rand = H_cat）：先验@1 0.0234 / 0.0300 / 0.0244，前 3 0.0179 / 0.0246 / 0.0189，SH6-3-3 0.0114 / 0.0180 / 0.0124。

读法：
- H_cat 与 S1″ REPORT §4.2 试点列逐位同（先验@1 0.030、SH6 0.007）= 接线正确。
- raw 与 adj 的差（0.67 → 0.17、+0.018 → +0.003）就是单种子 random-20 的赢家诅咒量级（κ·σ ≈ 0.045）：真值在两者之间，adj 是保守下界。
- 试点上 H_rand_adj − H_cat 在 regret@3 上 +0.003（P9 < 0.005 成立）；但这只是 3 个源 × 均匀 10–20 条，全语料 184 任务 × random-20 的 H_rand 在 S2 analyze 时按 §1.5 读，且 H_rand 只能给 GO。
