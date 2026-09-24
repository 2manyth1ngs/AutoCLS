# WP17-ACTS S3-A PROTOCOL：低秩交互模型 + 零 GPU 特征预测轴坐标 a(t)（止损段，零 GPU）

**会话定位**：S2-CORPUS-GO+PROB0（`results/acts/s2/REPORT.md`）之后，按方案 `docs/AUTOCTS_STRICT_REPRO_PLAN.md` §3.3 的 S3 三段止损（用户 2026-09-24 拍板，memory `wp17-acts-s3-design`）执行第一段 **S3-A**。问题：S2 的任务 × 配置交互集中在一条源级轴上；**不训练任何网络、不用 GPU**，只用已缓存的标签与探针，能否从便宜特征预测留出源的轴坐标 a(t)，并让留出源的 regret@3 优于任务盲？S3-A 不过门不写 S3-B / S3-C 代码。
产物 `results/acts/s3a/`；代码 `scripts/84_acts_s3a.py`（`--analyze`）；单测 `tests/test_acts_s3a.py`；输入只读 `results/acts/s2/{plan_S.json, catalog.json, runs.jsonl, probe0.jsonl, analysis_S.json}` + `configs/acts.yaml profiles.s2`。

## §0 已看过的读数（如实声明）

1. S2 全部读数（REPORT / analysis_S / tables_S），含每源 ρ_seed、h_t、failed、轴得分、H 三层。
2. UCR-ADM `posthoc_axis.md` §3：**23 源轴得分与多数类 / oracle / 中位 acc / 探针地板 / k / C 的相关 |ρ| ≤ 0.28，只有 T（−0.39）与 n（+0.36）弱相关**——即组 ① / ② 的大部分量在源级已被看过、预期弱。同文 §4 提出"轴两端两条配置 @25% 各跑一次读方向"的测量式替身（= 本协议组 ④ 的来源）。
3. **上一会话（2026-09-24，memory `wp17-acts-s3-design`）的四组仪器核查数**：训练池 160 任务 × 30 配置去配置主效应后残差 MS 0.801 / MS_noise 0.225 = 信噪 3.6×；秩 1/2/3 样本内解释 22 / 35 / 44%、LOSO（b 训练侧学、a 留出侧最小二乘）解释留出残差 18 / 29 / 37%（折噪 ≈ 25 / 40 / 52%）；任务级 a(t) 按源均值 vs S2 轴得分 ρ 0.98、源内方差占 a(t) 20%；`clip(ρ_seed,0,1)` 权重使反向源份额 0.350 → 0.305、1/σ² 使 CinC 独占 45%。
4. **本会话锁前复算（`--analyze` 同一数据装配，S2 标签尺度 = `clip_z(shared ∪ random 两种子均值, 0.5)` 取 shared 30 列）**：样本内（只去配置主效应）22.1 / 35.7 / 44.7%（与第 3 条一致）；LOSO 留出解释 17.7 / 30.4 / 38.8%（一致）；任务级 a₁ 源均值 vs S2 轴 ρ **0.988**（一致）；**但** 双向去均值后 MS_int 0.754 / MS_noise 0.175（信噪 4.3×，第 3 条为 0.801 / 0.225 = 3.6×；0.799 是 184 任务全语料的 MS_int，第 3 条可能混了全语料口径，未能逐位复现）、**a₁ 源内方差份额 0.27**（第 3 条 0.20）。本协议以第 4 条（可复现）为准；有效 n 仍 = 17 源（源级 73%）。
5. **没看过**：任何特征组的 LOSO 预测结果；任务盲 μ(c) 与 H_cat 先验在这 160 任务上的 regret@3；oracle-a 天花板；K 的内层选择结果；HMD / SCP2 的 â。

## §1 设计

### 1.1 数据
- **训练池** = `analysis_S.json: pool_after_gates`（17 源），EthanolConcentration 只用 a00–a07 → **160 任务**（正向 13 × 8 + Semg / CinC / SCP1 × 16 + Ethanol × 8）。HMD / SCP2（各 8 任务）只作 §1.6 表征读数，不进任何拟合。
- **模型目标**：Z(t, c) = S2 标签尺度的钉底 z（`clip_z(frame_vals, 0.5)`，shared 两种子均值 ∪ random 单种子一起定尺度），取 shared-30 列 → 160 × 30。
- **regret 真值**（= S2 H_cat 口径）：shared-30 原始终点 acc 两种子 end_0 / end_1（`_raw_seed_frames` + `_fill_pairs`）；regret@k(t) = end_1[t, argmax end_0[t]] − end_1[t, pick]，pick = 按模型分排前 k 条里 end_0 最高者（"前 k 全量实测"）。任务加权均值。
- **@25% 截点**：shared-30 seed 0 第 0 个截点 acc（`_ckpt(..., seed=0, idx=0)`，缺 → `_fill_single`）。探针地板 = probe0 三次 iters=0 均值。

### 1.2 低秩交互模型（每个 LOSO 折 s ∈ 17 源独立拟合，留出源不参与任何拟合）
- μ(c) = 训练任务 Z 的列均值；R = Z − μ 再按行去均值（去任务主效应，不改任务内排序）。
- R_train 的 SVD 前 K 个右奇异向量 B_K（30 × K，正交）；训练任务 a(t) = R_train(t) B_K。
- 留出任务的**真** a(t) = R_test(t) B_K（正交投影 = 最小二乘，用留出标签，只作真值 / 天花板）。
- 预测分 ŝ(t, c) = μ(c) + â(t) B_Kᵀ；任务盲分 = μ(c)。
- 符号：每折 b₁ 与全池 SVD 的 b₁ 同向（点积 ≥ 0，否则翻转 b₁ 与 a₁），仅为 ρ 读数对齐。
- K ∈ {1, 2, 3}：**每个特征组、每个外层折，用训练 16 源上的内层 LOSO 选 K**（最小化内层任务加权 regret@3，平局取小 K）。另报 K 固定 1 / 2 / 3 的全表。

### 1.3 特征组（每组一个 ridge：标准化特征 → a(t) 各分量；α ∈ {0.1, 1, 10, 100, 1000} 由内层 LOSO 最小化 a 的 MSE 选）
| 组 | 特征（每任务） | 目标侧成本（全跑当量） | 可部署 | 进门 |
|---|---|---:|---|---|
| ① meta | log n_train、log T、log C、log k、majority、切片档位 c / w / n（tag 三位） | 0 | ✓ | ✓ |
| ②′ stat25 | 探针地板 − majority；shared-30 @25% seed 0 的：前一半中位 − majority、≤ 地板份额、IQR | 7.5（+ 3 次 iters=0 评估） | ✓ | ✓ |
| ② stat-diag | 设计原文：h_t（两种子终点，chance_guard 定义）、failed（终点 ≤ 地板份额）、σ_task（top_half_sigma） | 60（两种子全部终点） | ✗ | **✗（诊断）** |
| ③ land25 | shared-30 @25% seed 0 的钉底 z 画像，减训练折 @25% 列均值、行去均值后投到 B₃ 的 3 个坐标 | 7.5 | ✓ | ✓ |
| ④ probe | 训练折 μ 前 2 条（并列冠军）+ b₁ 载荷最大 / 最小的 2 条（轴两端）@25% seed 0：两对各自的 acc 差、两对差各除以 max(0.01, 四条均值 − 地板)、四条均值 − majority | 1.0 | ✓ | ✓ |

- ② 按设计原文保留，但它的三个量是全部 60 个终点的函数——**部署时等于已经实测完整目录**，且与 regret 真值同源（循环）；只作"a(t) 是否由这些统计量决定"的诊断，不进门。②′ 是同一意图的可部署版（这是本协议对设计的唯一改动，理由即此）。
- 副读数（不进门）：①+②′+③+④ 拼接组 all；oracle-a（真 a 代回，低秩模型天花板）。

### 1.4 读数与统计
- **ρ_a1**（门）：LOSO 下 160 个留出任务 â₁ vs a₁ 的 Spearman（池化，第一分量与 K 无关）。附：17 源均值的 Spearman、源聚类 bootstrap 95% CI（按源重抽 10,000 次，seed 0）。
- **Δregret@3**（门）= regret@3(任务盲 μ) − regret@3(组，内层选 K)，任务加权；源聚类 bootstrap 95% CI；p = 17 个源均值差的 Wilcoxon 符号秩（双侧）。
- 附：regret@1、K 固定 1/2/3 表、每源 regret（反向源单独列）、H_cat 先验（训练源平均秩，= S2 `_headroom_four` 前 3）在同 160 任务 LOSO 上的值、同成本测量对照 **M25**（直接按 @25% 排 shared-30 取前 3 再实测选 1；与 ②′ / ③ 同成本 7.5）与 **M4**（④ 的 4 条里 @25% 最好 + 任务盲前 2 补足 3 条，成本 ≈ ④）。
- 源内残差（副，只看源内 27%）：(â₁ − 源均值) vs (a₁ − 源均值) 的 Spearman。

### 1.5 门 G-S3A（先于数据；只在"可部署 ∧ 进门"的四组 ①、②′、③、④ 上判）
| 分支 | 条件 | 动作 | 读数码 |
|---|---|---|---|
| GO | 存在一组 ρ_a1 ≥ 0.70 **且同一组** Δregret@3 ≥ 0.010（点估计） | 进 S3-B（per-task TS2Vec + SetTF E，须显著优于该组） | **S3A-GO** |
| FIELD | 不满足 GO，但存在一组 ρ_a1 ≥ 0.50 | 只进 S3-C 的 field 臂 | **S3A-FIELD** |
| STOP | 四组 ρ_a1 全 < 0.50 | 停，写负结果（轴存在但便宜特征读不出） | **S3A-STOP** |
- 设计原文未规定"ρ ≥ 0.7 但 Δ < 0.010"：本协议归 FIELD。
- 四组择优有多重比较，报告时写明"最好组"是四选一；② 诊断组、all 组、oracle-a 无论多好都不改变读数码。

### 1.6 只报不设门
- HMD / SCP2 表征：17 源全训的模型 + 各组特征给出的 â₁ 是否落在轴负端（< 训练任务 a₁ 的下四分位）；真 a₁ 同报。只作表征，不作精度（h_t ≈ chance）。
- 每组的目标侧成本 × Δregret@3 对照表（成本按 §1.3 列）。

## §2 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | oracle-a（K 内层选）Δregret@3 ≥ 0.010（低秩模型本身有兑现空间） | 65% |
| P2 | ① meta ρ_a1 < 0.50（posthoc_axis §3 源级已弱） | 75% |
| P3 | ③ land25 是可部署四组里 ρ_a1 最高的 | 70% |
| P4 | ③ land25 ρ_a1 ≥ 0.70 | 45% |
| P5 | 读数码 = S3A-GO | 30% |
| P6 | M25（同成本直接测量）regret@3 ≤ ③ land25 的 regret@3 | 60% |
| P7 | ② 诊断组 ρ_a1 < 0.50（a 不由难度 / chance / 噪声统计决定） | 65% |
| P8 | 内层选 K 在 ≥ 9/17 折上选 K = 1 | 40% |

## §3 执行
1. 本文件 + `tests/test_acts_s3a.py`（先红）→ 协议锁 commit。
2. 实现 `scripts/84_acts_s3a.py`（纯函数 + `--analyze`）；单测绿；`--analyze` 零 GPU 本地跑 → `analysis.json` / `tables.md` → REPORT → commit → CLAUDE.md / memory。
3. 过门才立 S3-B / S3-C 协议；不过门不写它们的代码。

## §4 铁律对照
3（只读缓存，不产新 R′）；5（先红、§0 声明）；6（每折 μ / B / ridge / K / α 只在训练源拟合，留出源只考；HMD / SCP2 不进拟合）；7（纯矩阵函数不含数据集概念，源分组只在脚本层）；8（regret 用 acc 越大越好，分类单一 task_type，经脚本层显式处理）。

## §5 执行台账（跑后填写）
