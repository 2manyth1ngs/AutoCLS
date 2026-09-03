# MX-4 WP2 协议：字段结构化代理 vs T-CLSC（零 GPU，LODO）

日期：2026-09-03 ｜ 脚本：`scripts/37_surrogate_loo.py` ｜ 模块：`autocls/search/surrogate.py`（已入库，`tests/test_surrogate.py`）｜
数据：`results/mx3_pq/{rprime_cache.jsonl, evo_cache.jsonl, delta_calibration.json, rankings.json}`（只读）。

## 0. 诚实声明

同 WP1：探索读数已看过（scalar 特征 HGB 配对 acc 0.788 / one-hot 0.780；ridge 0.765–0.773；进化 CLS 预测 z +0.9 vs 真 +0.05）。
本协议把它固化成正式口径并补三件探索里没有的东西：分档 one-hot 编码（`CLSEncoder(mode="binned")`）、字段主效应表、
以及"信任域 / LCB 能否消除目录外高估"的判定。判读线在正式跑之前写死。

## 1. 问题

| 问 | 内容 | 检验 |
|---|---|---|
| Q1 | 字段结构化代理的 LODO 配对准确率 / regret 是否优于 T-CLSC | S1 / S2 |
| Q2 | 哪些 CLS 字段决定质量（论文分析章：普适的设计规则） | 主效应表（描述性） |
| Q3 | 代理对目录外候选（进化产物）是否高估；信任域 / LCB 能否修正 | S3 |

## 2. 口径

- 训练：7 折 LODO，训练侧 = 其余 6 + SCP2 的全部 (切片, CLS, 预算) 行，目标 = within-(切片, 预算) z 分数；
  特征 = 编码 ⊕ b/290。四个臂：`ridge_scalar`（32 维）、`ridge_binned`（一律按档 one-hot）、`hgb_scalar`、`hgb_binned`；
  外加 `hgb_binned_bag10`（bootstrap 10 成员，给 std）。超参 = `surrogate.py` 默认（ridge α=1；hgb 300 树 / lr 0.05 / 深 4 / l2 1）。
- 评估实例：126 = 7 靶 × 6 切片 × 3 预算；候选 = 目录 20 条。
  - 配对 acc：|ΔR'| ≥ δ(数据集, 预算)（`delta_calibration.json`）的无序对上，代理得分方向 = 真值方向的比例；
  - top-10 内 acc：两端都在真值 top-10 的 δ 存活对；
  - regret@1 / @3、Spearman（20 条）。
  - T-CLSC 对照：`rankings.json` 的 `main` 排名 → 得分 = −名次，同口径算配对 acc / regret（REPORT 报的 0.724 是双向对的模型输出口径，
    本口径下的 T-CLSC 数字以本脚本算出的为准，两者并列）。
- 字段主效应：`ridge_binned` 在全 8 数据集上拟合（分析用，不做 LODO）的每档系数（同字段内减均值）；
  `hgb_binned` 的**按字段分组置换重要性**（整段 one-hot 一起置换，ΔMSE），全数据。
- 目录外诊断：`evo_cache.jsonl` 378 行（21 实例 × 18 条进化 CLS）。每实例：用该折 LODO 模型（不含目标数据集）预测进化行；
  真 z 用**目录 20 条**的 mean/std 归一（进化行是同 (切片, 预算) 的额外候选）。读数：
  预测 z 均值 − 真 z 均值（高估量）、进化 ⊕ 目录合并 Spearman、进化行到训练集的字段汉明距离分布；
  修正臂：LCB = mean − κ·std（bag10，κ=1）与信任域（汉明 ≤ r 之外的候选降为 −∞），看合并 Spearman 是否上升、高估量是否下降。

## 3. 判读（先写死）

| 门 | 检验 | 先验（已看探索读数） |
|---|---|---|
| S1 | `hgb_binned` 配对 acc > T-CLSC（同口径）于 ≥5/7 折 | 75% |
| S2 | `hgb_binned` regret@1 ≤ T-CLSC regret@1（pooled 126，配对 Wilcoxon 单侧 α=0.05）| 55%（探索：0.056–0.063 vs 0.074，但 @290 gbdt1 0.086 更差） |
| S3 | 进化行 LCB 修正后合并 Spearman（21 实例均值）> 未修正，且高估量下降 ≥ 50% | 40% |

读数码：**SUR-WIN** = S1 ∧ S2；**SUR-RANK** = S1 ∧ ¬S2（排序更准但单选不赢）；**SUR-DEAD** = ¬S1。S3 独立报（决定开放空间搜索能否立项）。

## 4. 预测

- P1：`hgb_binned` pooled 配对 acc ∈ [0.76, 0.81]（75%）。
- P2：主效应绝对值最大的三个字段 ⊂ {temperature, freq_mask, jitter, resize, crop, sim_func}（70%）；
  {temporal, cross_scale, pool, adj_neighbor, loss_type} 的分组重要性都在末 8 位（60%）。
- P3：进化行高估量（预测 z − 真 z）≥ +0.5（80%）；进化行到训练集的字段汉明距离中位 ≥ 4（60%）。
- P4：LCB 修正后合并 Spearman 提升 < 0.10（55%）——不确定性估计救不了系统性外推偏差。

## 5. 不做的事

- 不重训 T-CLSC、不动 rankings.json；不改 space.yaml；不用目标数据集任何行拟合（铁律 6 同口径）。
