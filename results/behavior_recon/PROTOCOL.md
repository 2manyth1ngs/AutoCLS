# 行为条件化侦察协议：测量条件化（转导补全）零 GPU 选拔场（判读带先于计算写死）

日期：2026-08-31 ｜ 脚本：`scripts/28_behavior_recon.py`（零 GPU，只读既有缓存）｜
单测：`tests/test_behavior_recon.py` ｜ 用户 2026-08-31 拍板：**A 行为条件化侦察先行、
C（k 放大小面板）备升级；侦察→过闸再锁确认协议；任务级+数据集级两级都测。**

## §0 地位声明（选拔场）

本目录是**选拔场**。任何读数（含正读数）不作论文确证证据；过闸只解锁一件事——
"测量条件化确认协议"的预注册（新格子、判据先于见数锁 commit）。选拔/确认严格分离
（教训出处：`results/gate2_pilot_b` 选拔场 strong → `results/wide_panel` 确认场 D1w 判负）。

## §1 新主张家族声明 + 不翻旧案条款

**主张家族 = 测量条件化（transductive/转导式）**：条件化输入是目标任务/数据集
**自己的 j 个实测行为格**（(task, cls) → 三态），经行为空间邻居/低秩结构补全其余
未观测格。**没有任何"固有特征 → 行为"的学习映射**；不使用 E / P / G / 元数据特征。

本协议**不翻**下列已关闭判定（引用 `results/source_pool_v2/PROTOCOL.md` §勿复活名单）：

- 任务级条件化（P/E 特征核插值）：`results/wide_panel/REPORT.md` D1w 方向 − p=0.9944，维持判负；
- 数据集级学习可行性预测器：`results/mx1b_feasibility/PROTOCOL.md` §1 norevive 条款，维持关闭
  （本协议无特征输入、无跨数据集参数化映射训练，不属该家族）;
- 毒哨（全局最毒 CLS 哨兵）：`results/canary_validation` invalid，本协议观测格一律**均匀随机抽**；
- E 主链 / 细粒度 task-aware 排序：维持关闭。

若本侦察判负，"测量条件化"家族在 @290 + 共享 10 CLS 口径下关闭；是否立 norevive 条款交用户。

## §2 偷看披露

已见读数（全部来自既有 REPORT/analysis，本协议判据锁定前未算过任何新指标）：

- 宽面板：UWave T2=1.050 p=0.0001 / UM 0.790 p=0.0007（存在性 2/5）；D1w 方向 − p=0.9944；
  P5w 双条件前瞻 3/3（全面板 5/5 同向）；
- ds×cls 交互份额 0.45（`results/canary_validation`，毒哨死因）；
- C2 尸检：数据集内任务两两排名 ρ=0.574 / 跨数据集 0.240（`results/decisive_e_ablation/REPORT.md`）；
- 金丝雀：j=5 毒哨 ρ=0.732 / 随机+收缩 bacc 0.809；R1 单种子 rep 间 ρ 0.21/0.24/0.50；
- mx1b P6b 破坏率序信号 Spearman 0.555 p=0.026。

**声明**：第 0 步的 κ / 画像 ρ 从未计算过；第 1/2 步各臂 Brier 从未计算过。杀门三档与
过闸判据在其揭晓前锁定（协议锁 commit 为凭）。`--audit` 的装配计数（行数/满格性/拒收数）
不属指标，先于 step0 允许查看。

## §3 数据装配（写死；全部 @290 iters，缓存只读）

**共享 10 CLS**：`load_probe_cls(results/route_a_probe/probe_cls_catalog.json, 10)`，
列序 = sorted(cls_key)。

**基线（band）路径**（喂 `dm.collect_baselines`，exclude_plus_c=True；band=2·max(σ,1/n_eval)，
≥3 canonical seed）：

1. `results/baseline_probe/baseline_cache.jsonl`
2. `results/candidate_survey/baseline/baseline_cache.jsonl` —— **装载时先丢弃
   dataset ∈ {PTB-XL, NonInvasiveFetalECGThorax1} 的行**（双源污染钉源）
3. `results/candidate_survey/baseline_crop256/baseline_cache.jsonl` —— **只取上述两个数据集的行**
4. `results/wide_panel/baseline_cache.jsonl`

7 份副本缓存（iters_workpoint / full_iters_probe / workpoint_followup / workpoint_confirm 各处）
一律不喂，防重复计数。`data/uea_cache/results/stage1_20260803/` 旧快照不碰。

**@290 格路径**（喂 `dm.collect_trained`，先到先得，路径序即优先序）：

1. `results/stage1_rerun_20260804/results/labels_route_a/rprime_cache.jsonl`（13 UEA seed0 canonical）
2. `results/baseline_probe/trained_fill.jsonl`
3. `results/candidate_survey/baseline/trained_fill.jsonl`（同上钉源：丢弃 PTB-XL/NIFECG 行）
4. `results/candidate_survey/baseline_crop256/trained_fill.jsonl`（只取 PTB-XL/NIFECG）
5. `results/wide_panel/rprime_cache.jsonl`

**k3 均值**：`dm.collect_trained_k3(results/stage1_rerun_20260804/results/step0_snr_gate/noise_cache.jsonl)`
（13 UEA，seed 1–3 恰 3 条均值）。

**排除（写死）**：PenDigits（无 iters=0 基线 + L=8 判"只产噪声 label"，CLAUDE.md）；
`results/route_a_probe/rprime_cache.jsonl`（任务定义/classes/split 与注册表切片不一致，整体排除）；
一切 iters≠290 缓存。

**数据集级行（第 1 步）**：行 = 数据集；**只用 `_00/_01` 两任务**（全行统一粒度）。
任务格值：13 UEA 用 k3 均值（`scripts/15` 先例），其余用 k1 canonical。任务格 → `fl.tri_state`
三态 → **(ds, cls) 格值 q ∈ Δ³ = 两任务三态率向量**（取值 {0,.5,1}³），行向量 30 维。
入行判据（机械规则，`--audit` 出名单为准）：10 CLS × 2 任务满格 + 两任务基线齐。预期 n=25 行；
n<20 视为装配异常，停下排查。行内混粒度（13 行 k3、其余 k1）如实声明，由敏感性 S1/S3 兜底。
UM 为正常行（试点 B 的选拔角色只涉任务级闸 2，此处无 P 特征无循环性）。

**任务级面板（第 2 步）**：宽面板 5 数据集 × 8 任务（`_00.._07`）× 10 CLS，k=1 canonical
三态码（0/1/2），与 `results/wide_panel` analyze 同口径。UM 补充报告不入主判（惯例沿用）。

**第 0 步信度粮草**：13 UEA × 2 任务 × 10 CLS = 260 格 × 4 观测（seed0 = labels_route_a
canonical；seed1–3 = noise_cache）。**口径声明**：noise seed 1–3 用非 canonical split
（`split:{s}` 派生），故跨 seed 一致性度量的是**权重+划分联合噪声 = 部署态 k=1 单测噪声**
（先例：`collect_trained_k3` 同样不查 split）。per-seed 标签 = `tri_state(r'_seed,
canonical_mean, canonical_band)`。

## §4 第 0 步：k=1 三态标签重测信度杀门（先行，先于第 1/2 步）

指标（先于计算写死）：

- **M1** = 6 个 seed 对的格级 Cohen's κ（260 格池化，逐对算 κ 后取均值）；
- **M2** = 任务内 10-CLS r' 画像跨 seed Spearman ρ 的**中位数**（26 任务 × 6 对 = 156 值）；
- 辅助只报不判：逐状态一致率、seed0(k1) vs k3 均值标签一致 κ（估 12 个 k1 行的有效信度）。

三档判读（数值先锁；推理链：n=25/32 折配对 Wilcoxon 在 α=0.025/0.05 下功效 0.8 需
dz≈0.58/0.45；真效应可信带上锚 = C2 尸检数据集内 ρ=0.574；行为签名随格级信度近似线性衰减）：

| 档 | 条件 | 动作 |
|---|---|---|
| **杀** | M1 < 0.35 **或** M2 < 0.45 | 侦察停：REPORT 记杀因，转 §7 C 升级路径，第 1/2 步不跑 |
| **琥珀** | M1 ∈ [0.35, 0.50) 或 M2 ∈ [0.45, 0.60)（未触杀） | 继续，但主判读点改 **j=7**，过闸加严：p 线之外还要求折胜率 ≥60% |
| **绿** | M1 ≥ 0.50 且 M2 ≥ 0.60 | 按原设计，主判读点 **j=5** |

参考锚：金丝雀 R1 rep 间 ρ 0.21–0.50 → 杀/琥珀是活风险（先验见 §8 PB1）。

## §5 第 1 步：数据集级 LOO（预期 25 折；第 0 步非杀档才跑）

每折留出 1 行。j ∈ {3, 5, 7}，每档 R=100 次确定性抽观测列集 J
（`run_seed_for(f"br1:{held_ds}", f"j{j}:rep{r}")`，无放回均匀抽——毒哨勿复活，禁按毒性排序选格）。
四臂对**同一未观测列集**（10−j 列）预测三态率向量：

- **(a) 全局列先验**：训练 24 行 q 的逐 CLS 均值（任务盲对照）；
- **(b) 平移臂（最强对照，必须战胜）**：λ_s = mean_{c∈J}[q_held(c)_s − p_a(c)_s]，
  p_b(c) = normalize(clip(p_a(c)+λ, 0, 1))；退化（clip 后全零）回退 p_a。只调水平不调
  per-CLS 结构（金丝雀"随机格+全局收缩"的近亲，协议注明）；
- **(c) 行为 kNN**：距离 = J 列上 q 的均方 L2（mean_{c∈J}‖q_held(c)−q_row(c)‖²）；
  k=3 等权取邻居未观测列均值（并列距离按数据集名字典序断）；敏感性 k ∈ {1, 5}；
- **(d) 低秩补全**：训练 24 行按列（30 维）去均值 → SVD 取 rank=2；留出行以 3j 维观测
  ridge（λ=1e-3）回归主成分得分 → 重构未观测列 + 列均值 → clip [0,1] → 逐 CLS 归一化；
  敏感性 rank ∈ {1, 3}。

**指标** = 未观测格软目标 Brier：mean_{c∉J} Σ_s (pred(c)_s − q(c)_s)²（`masked_brier`，
one-hot 情形退化为 `profile_brier` 语义）。每折对 R 取均值 → 折间配对
`paired_wilcoxon_less`。

**D1r（主判据）**：主判读点 j 档上，(c) 或 (d) **同时**胜 (a) 与 (b)，各比较
p < **0.025**（两行为臂 Bonferroni ×2），且该臂对两对照的方向（中位差符号）在
j ∈ {3,5,7} 三档一致。琥珀档另加：主 j 档两比较折胜率均 ≥60%。

**折隔离**：列先验/SVD 基/列均值只用训练 24 行；留出行只贡献 J 列。
**敏感性（不改判读带）**：S1 = 只留 13 个 k3 行重跑（n=13）；S3 = 13 UEA 行改用 seed0 k1
重建（部署粒度稳健性）；S2 = 5 面板数据集用 8 任务率向量重算（描述性）。

## §6 第 2 步：任务级宽面板 LOTO（32 折主判 + UM 8 折补充）

4 个 primary 数据集（CT/UWave/PS/HW）各 8 任务逐折留出（数据集内 LOTO，`scripts/24` 闸 2
骨架照搬，**P 特征距离换成行为距离**）。每 rep（R=200）：

- 抽 J（`run_seed_for(f"br2:{held_tid}", f"j{j}:rep{r}")`，j ∈ {3,5,7}）；
- 抽 k=3 锚（`gp.sample_anchors`，seed=`run_seed_for(f"br2:{held_tid}", f"anchor:j{j}:rep{r}")`）；
- **行为距离** = J 列三态码不一致率（只用观测列）；σ = 其余 7 任务在同一 J 上的成对距离
  中位数（下限 1e-6；全零距离时核权自然退化为等权）；
- `kernel_w` + `anchor_predict` 预测，`masked_brier`（one-hot 目标）只评未观测 10−j 列；
- **两个对照**（同锚/同 J 配对）：等权锚（D1w 赢家）+ 数据集内全局列先验（其余 7 任务
  画像均值，限未观测列）。

**D2r（主判据）**：32 折 pooled（4 数据集 × 8 任务），行为核对两对照各
`paired_wilcoxon_less` p < **0.05**（单行为臂），且方向在 j 三档一致。琥珀档另加折胜率 ≥60%。
per-dataset 分层照 `wide_panel/REPORT.md` 格式描述性报告。UM 8 折同法单独报，不入主判。

**探索性变体（不入闸，只报）**：40 任务池（5×8）跨数据集 kNN——留出任务的邻居限**其他
数据集**的任务，k=5 等权，距离同上；对照同款全局列先验（39 任务，限未观测列）。

## §7 过闸判据与两条后路

- **过闸** ⇔ D1r 或 D2r 至少一个完整达标（含琥珀加严项，若适用）。
- **过闸后**（本协议只概述，确认协议另立预注册、见新数据前锁 commit）：确认场 = 全新格子——
  首选搭 `results/source_pool_v2` 待跑 283 格 GPU 补格便车（SAD/NATOPS/Ethanol/HMD/Cricket
  ×6 任务、SCP2 ×3：全新数据集行 + 全新 `_02+` 任务行，与本侦察零重叠）；备选专列扩格。
- **不过闸**：REPORT 如实判负，测量条件化家族在本口径下关闭（norevive 与否交用户）。
- **C 升级路径（第 0 步杀档触发时）**：UWave（强异质 p=0.0001）+ Handwriting（弱对照
  T2=0.083）× 8 任务 × 10 CLS × 补 4 seed（噪声流派生 `{key}:{s}` / `split:{s}`，s=4..7
  与既有 1..3 不撞）= 640 格 @290 ≈ 3.5 h @RTX 5090 单卡（**4060 8GB 在 UWave L=315
  resize 重尾段禁用**，硬件标签铁规）。产出 k=5 标签 → 信度天花板重估 → 另立协议重开。

## §8 预测表（先验主动下调；教训：`results/anchor_sim/REPORT.md` §5——对小预算测量的
功效先验一贯过于乐观）

| # | 预测 | 先验 |
|---|---|---|
| PB1 | 第 0 步绿档 / 琥珀 / 杀 | 40% / 30% / 30% |
| PB2 | D1r 过（数据集级行为臂双胜两对照） | 20% |
| PB3 | D2r 过（任务级行为核双胜两对照） | 25% |
| PB4 | 总过闸（PB2 ∨ PB3） | 35% |
| PB5 | 跨数据集任务池变体相对数据集内 kNN 有可见增量 | 10% |

## §9 执行

顺序：协议锁 commit → tests（红）→ 实现 → tests（绿）→ `--audit`（装配对账）→ commit →
`--step0`（杀门判读；杀档即收 REPORT 停）→ `--step1` → `--step2` → `--analyze` → REPORT +
memory + commit。全程零 GPU、零新 R'、缓存只读；预计总计算 <10 min（CPU）。
产物：`results/behavior_recon/{audit.json, step0_reliability.json, step1_dataset_loo.json,
step2_task_loto.json, analysis.json, REPORT.md}`。所有判定引用带 **@290 + n 口径**。
