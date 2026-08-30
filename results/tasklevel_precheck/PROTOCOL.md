# 任务级脆弱度存在性预检协议（第 0 步，零 GPU；判读带先于计算写死）

日期：2026-08-30 ｜ 脚本：`scripts/21_tasklevel_precheck.py` ｜ 单测：
`tests/test_tasklevel_precheck.py` ｜ 定位：任务级条件化（宽任务面板）
立项前的免费先验检——**同一数据集内，三态脆弱度是否随任务切片变化超出
可交换噪声**。阳性 → 宽面板 GPU 拍板先验升；阴性 → 拍板先验降
（口径折扣见 §3，阴性不封 @290）。金丝雀 j=3 已判 invalid（决策树上游）。

## §1 数据（现成缓存，与 scripts/14 Panel B 逐字同口径）

- trained：`results/labels_workpoint/rprime_cache.jsonl`（k=1 canonical，
  HW@60 12 任务 × shared 10 ＋ UM@30 16 任务 × shared 10 = 280 cell）；
- 基线：`results/iters_workpoint/{uea,candidates}/baseline_cache.jsonl`
  （iters=0，≥3 种子），带 = 2·max(σ, 1/n_eval)；
- 配对校验逐字复用 `collect_baselines/collect_trained`（classes/split/iters）；
- 三态 flag = `tri_state`（scripts/15，@工作点口径——**勿与 @290 混**）。

## §2 统计量与检验（先于计算写死）

每数据集一张 10 CLS × T 任务的三态矩阵：
- **T2 统计量** = Σ_状态 Σ_任务 (p_{t,s} − p_s)²（各状态的任务间离散度和；
  p_{t,s} = 任务 t 内状态 s 的比例，p_s = 数据集全局比例）。
- **零假设置换**：每条 CLS 行独立随机置换其 cell 的任务归属——保 per-CLS
  状态计数（CLS 主效应完整保留），只杀"哪个任务"的结构。任务间真实异质
  以外的一切（CLS 主效应、数据集水平）都在零假设内。
- N_PERM = 10000，种子写死；p = (1 + #{T2_perm ≥ T2_obs}) / (N_PERM + 1)。
- 逐状态分解（T2 的 improve/inert/destroy 分量）随判读一并出（诊断）。

## §3 判读带（先于计算写死）

| 条件 | 判定 |
|---|---|
| 任一数据集 p < 0.025（Bonferroni ×2） | **阳性**：任务级脆弱度变化在工作点口径存在 → 宽任务面板立项先验升，GPU 拍板项呈交 |
| 双数据集均 ≥ 0.025 | **阴性**：该口径无证据 → 宽面板拍板先验降 |

**口径折扣（写死）**：工作点是低破坏 regime（HW 0.01 / UM 0.14，破坏是
iters 的函数）；阴性**不封** @290 口径的可能性，只降先验——宽面板若立项
必须用 @290。HW 的信号预计主要在 improve 态（工作点 A 层 56↑）。
本预检只判存在性，不涉 E（E 条件化是宽面板协议的第二道闸）。

## §4 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1f | 任一数据集阳性 | 50% |
| P2f | HW 阳性 | 40% |
| P3f | UM 阳性 | 35% |
| P4f | 若 HW 阳性，T2 的 improve 分量占比最大 | 60% |

## §5 执行

零 GPU；协议+tests+实现 → 协议锁 commit → 跑数 → REPORT。产物：
`results/tasklevel_precheck/{PROTOCOL.md, analysis.json, REPORT.md}`。
