# MX-1c 协议：task-feature 先验收缩——看门层策略层的功能价值检验（判读带先于计算写死）

日期：2026-08-30 ｜ 脚本：`scripts/18_prior_shrinkage.py`（零 GPU，纯模拟）｜
单测：`tests/test_prior_shrinkage.py` ｜ 前作：MX-1/MX-1b 判负、勿复活条款
生效（学习型分布预测器关闭）。本协议**不是**复活预测主张：deliverable =
**测量效率**（金丝雀预算下毒性估计误差），task-feature 先验只是估计器的
一个部件；判负则看门层用全局先验，先验类 reframe 一并关闭（不再有 MX-1d）。

## §1 问题与主张

金丝雀看门层对未见数据集花 j 条测量估计毒性。问：MX-1b 幸存的**序信号**
（预测破坏率 Spearman 0.555，p=0.026，暗示性）经**秩校准 + 收缩**后，能否
在同等预算下降低估计误差？主张若成立 = "task-feature 引导的探针估计"，
task-feature 的岗位从预测器改为**先验**。

## §2 偷看披露

已见：MX-1/MX-1b 全部结果（含 bm 各折 LOO 预测值与逐折 err）、320 格
label 全表。先验条件于此。**未做过**：任何收缩/校准估计器的计算。

## §3 数据与估计器（公式先于计算写死）

真值：数据集 d 的破坏率 p_d = 其 20 格 label 的 destroy 频率
（`results/mx1_feasibility/labels.jsonl`）。主指标限破坏率单维（看门层的
头条量；三态分布由测量本身覆盖，先验只管毒性水平）。

**先验（皆按折构造，无泄漏）**：
- prior_g（对照）= 该折 15 个训练数据集 300 格的全局破坏率；
- prior_f（被试）= **秩校准 bm 先验**：取 MX-1b 各数据集的 LOO 预测破坏率
  （每个都出自其自身折 = 全部 out-of-sample，`mx1b analysis.json
  per_fold[*].pred_rates.bm`），以 15 个训练数据集的 (LOO 预测值 → 观测
  破坏率) 对为映射表（按预测值排序线性插值，端点截断），把留出数据集的
  LOO 预测值映射为先验。只用**秩**、不信 bm 的绝对标定——这正是 ρ=0.555
  能支撑而 err_TV 不能支撑的用法。

**估计器（Dirichlet 收缩，伪计数 m）**：
- raw：j 格实测破坏频率 p̄_j；
- shrink_g：(j·p̄_j + m·prior_g)/(j+m)；
- shrink_f：(j·p̄_j + m·prior_f)/(j+m)；
- 诊断：prior_g / prior_f 单独（j=0）。

**模拟**：每折每预算 j ∈ {2,3,5,8}，从 20 格无放回抽 j 格，R=500 次
（种子按 (折, j) 派生写死）；误差 = |估计 − p_d|，对 draws 取均值 →
逐折值。**主判读点：j=3、m=4**（写死）；m ∈ {2,8} 与其余 j 做敏感性报告。

## §4 判读带（先于计算写死）

- **D1c**：j=3, m=4 下 err(shrink_f) < err(shrink_g)，16 折配对 exact
  Wilcoxon 单侧 p<0.05。——隔离"特征先验超出全局收缩"的净值。
- D2c（次，不做门）：prior_f 单独 < prior_g 单独（秩校准是否已胜全局中心）。
- D3c（次，不做门）：shrink_g < raw（收缩本身有益的健全性）。

| 条件 | 判定 |
|---|---|
| D1c | **成立**：task-feature 先验有功能价值，看门层策略层立项（金丝雀协议含 prior_f） |
| ¬D1c 且 p<0.15 且方向在 ≥3 个 j 档一致 | 边缘：交拍板 |
| 其余 | **不成立**：看门层用全局先验；task-feature 先验类主张关闭（不再 reframe） |

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1c | D1c 成立 | 45% |
| P2c | D2c（prior_f 单独胜 prior_g） | 40% |
| P3c | D3c（收缩胜 raw @j=3） | 75% |
| P4c | D1c 方向（不论显著）在全部 4 个 j 档一致 | 50% |

## §6 执行

零 GPU 纯 numpy；协议+tests+实现 → 协议锁 commit → 跑数 → REPORT。
产物：`results/mx1c_prior_shrinkage/{PROTOCOL.md, analysis.json, REPORT.md}`。
与 MX-2a（选源）并行跑，收官后两路对比择优（用户 2026-08-30 拍板）。
