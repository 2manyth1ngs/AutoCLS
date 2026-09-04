# MX-4 WP6-TF 协议锁：任务特征对零样本 CLS 排序的影响——现构造（打分器 + 成绩单）上的零 GPU 验证

日期：2026-09-04 ｜ 分支 `wp6-tf`（自 wp5-lc 9a8121f）｜ 脚本 `scripts/40_task_feature_scorer.py` ｜ 单测 `tests/test_task_feature_scorer.py` ｜
数据（只读）：`results/mx3_pq/{rprime_cache.jsonl, noise_probe.jsonl, delta_calibration.json, tasks.json}`、
`results/mx4_deploy/{runs.jsonl, candidates.json}`、`results/protocol_alignment/runs.jsonl`（audit_iters0）、
`results/baseline_probe` + `results/candidate_survey/baseline`（iters=0 基线）｜ 对照：`results/mx4_prior_sh/analysis.json`（WP1）｜
计划：`~/.claude/plans/ts2vec-e-quirky-bubble.md`（用户 2026-09-04 拍板：家族 A + B；起跑名单可自适应、淘汰规则不动；
主判据 = oracle 入榜率 + regret@3；只偷看阶段 0 再锁本协议）。

## 0. 诚实声明（与盲预注册的区别）

本协议**不是盲预注册**。写判读线之前已看过（全部落盘于 `stage0.{json,md}`，脚本 `--stage0`）：
- C0：本脚本重算 prior3 / gbdt3 / hybrid3（126 实例）与 SH6-3-3_prior（@290）与 WP1 逐位相同（max|Δ| = 0，均值 0.018443）。
- oracle ∉ hybrid 前 6 @290：16/42，其中 **15 个 oracle 是 random 候选**、1 个 shared；前 6 内最好的 mean regret 0.0118（= WP5 不可约部分）；
  按数据集 Ethanol 5 / HMD 3 / PS 3 / Cricket 2 / UM 2 / UWave 1 / HW 0。
- 去偏置 s0z（shared 先验 + random 任务盲 GBDT，无 +10）：入前 6 @290 19/42、regret@3 0.0300（vs hybrid 0.0313，p=0.30）、SH 0.0325——
  任务盲 GBDT 高估 random，单纯去偏置不行。
- C1 数据集身份上界 `c1_pure_z`（shared = 同数据集其余切片 z 均值；random = hgb ⊕ onehot(dataset)）：regret@3 0.0141 vs hybrid 0.0313
  （p=5.7e-08，59 胜/21 负）、入前 6 @290 29/42（+3）、SH6-3-3 起跑名单换上界 @290 0.0133 vs 0.0184（p=0.023）；
  捞回 Ethanol×3 + HMD×2，丢 HW_01 与 HMD_04。计划 §4.2 参考线（+4/42 或 regret@3 改善 ≥0.005 且 p<0.05）过。
  带偏置的 `c1_pure_hyb` 入前 6 不变（26/42）、SH 0.0181——**收益必须靠让 random 进前 6**。
- 0b：字段效应跨数据集方差 / 切片 bootstrap 方差之比前列 = cross_scale 6.0、pool 5.6、norm 4.0、emb_mask 3.8、sim_func 3.8、freq_mask 3.7、temperature 3.4、resize 3.4；
  逐数据集 shared-10 z 均值 vs LODO 先验 Spearman：UM 0.87 / UWave 0.84 / Cricket 0.83 / HW 0.37 / HMD 0.30 / SCP2 0.22 / PS 0.20 / Ethanol −0.06；
  嵌套方差分解 @290：任务 0.726 / CLS 0.132 / 交互 0.143，交互中数据集级 79% / 切片级 21%，MS 数据集级交互/σ² = 25.8（切片粒度 df 63）、切片级 1.3×、σ = 0.037。
- 计划阶段的探索代理另算过一次"@290 漏网 15/16 是 random"与"SH6-3-3 成本 = 1050、SH6+2-3-3 = 1170 = SH8-3-3"，与阶段 0 一致。

**A/B 任何臂的任何读数都未算过**（φ 未计算、残差模型未拟合、B 未模拟、官方口径未重放）。§3 臂表、§4 判读、§5 预测在正式跑之前写死。
0b 的读数只用于本协议 §2.3 的特征机制配对说明与 `ridge_int_top4` 的字段选择（预注册规则：取 0b 前 4 = cross_scale / pool / norm / emb_mask）。

## 1. 主张家族声明与不翻旧案条款

主张家族 = **数据集级任务特征进入零样本起跑名单打分器**，两种进入方式：A 预测型（跨数据集学 φ(d) × CLS 字段交互，LODO）、
B 转导型（目标自己的成绩单第一级读数当 φ，不跨数据集学 φ 的语义）。标签 = within-(切片, 预算) 连续 z（不经 δ 过滤）；杠杆 = 起跑名单（SH 淘汰规则不动，关键决策 8）。

本协议**不翻**下列已关闭判定：切片级 / 细粒度 task-aware 排序（配对 δ 标签口径，CLAUDE.md 判死 (a)）、T-CLSC 配对比较器与 E 主链、
可行性预测器（mx1*）、P/E 特征核插值（wide_panel）、k=1 三态行为锚插值（behavior_recon）、逐格零训练代理（proxy_conditioning）、
学习曲线外推（关键决策 8）、开放空间零样本外推（判死 (b)）。本协议与它们的差异：连续 z 标签、数据集级单位、起跑名单杠杆、
对照 = 现行 hybrid 与同成本的先验第 7–8 名。与 behavior_recon 的差异写在 §2.5。

## 2. 数据与口径（铁律 2 同口径：iters 计，缓存 seconds 换算）

### 2.1 真值与实例
- `rprime_cache.jsonl`：45 切片 × 20 目录 CLS（shared-10 + random-10；GGS 目录外单列）× 截点 {60,120,290} 同一轨迹，seed 0。
  搜索实例 = (靶数据集, 切片, b_final)；oracle = 目录 20 条 @b_final 的 max（同分 key 升序）；regret = oracle − 臂。
- 折：7 折 leave-one-dataset-out（`pq_common.fold_split`）；训练侧 = 其余 6 + SCP2。铁律 6：先验 / 任务盲 GBDT / 残差模型 / φ 的标准化统计量只用训练侧。
- A：126 实例（7 × 6 × 3 预算）；B：42 实例（@290）。新鲜种子复核：`noise_probe.jsonl`（切片 00/01 × shared-10 × seed 1/2）。

### 2.2 现行打分器 s0（与 36 号逐位同，C0 已过）
shared → 训练侧先验 z 均值；random → 任务盲 `FieldSurrogate("hgb")` 于 `CLSEncoder(scalar)` 32 维 ⊕ b/290；
hybrid = s0 + HYBRID_BIAS(10)·[shared]（现行名单）；s0z = 无偏置版。

### 2.3 任务特征 φ(d)（家族 A，12 维，全部零 GPU、只用目标训练划分）
| 组 | 分量 | 机制伙伴（按 0b 交互字段排名） | 来源 |
|---|---|---|---|
| φ_meta (4) | log10 T、log10 C_slice、类数 k、log10 N_train | cross_scale/pool × T、norm × C、sim_func/temperature × k | `tasks.json`（classes/channels/n_train）、注册表 `l_enc`（Ethanol/Cricket 取 crop_len） |
| φ_base (2) | iters=0 基线 R' 5 种子均值、band = 2·max(σ, 1/n_eval) | 全部字段 × 净空；temperature × 可分性 | `pq_common.baseline_bands`（两份 baseline_cache，45/45 齐） |
| φ_sig (6) | 谱集中度（去 DC 后前 10% 频点能量占比）、高频能量比（上半频带）、自相关衰减时间 τ/T（首次 ACF<1/e，无则 1）、超额峭度、通道方差 CV、样本方差 CV | freq_mask × 谱集中度；jitter/point_mask × 高频比；cross_scale/pool/resize/crop × τ；rescale × 峭度；norm/emb_mask × 尺度异质 | `load_pq_datasets` → `materialize_task` → `split_task_train_eval(eval_ratio=0.3, canonical seed)` 训练划分；逐 (样本, 通道) 算后取均值；池内等长、crop 为前缀截断 |

标准化：HGB 臂原值；ridge 臂用**训练侧** mean/sd z-score。φ 拼在 `36:305 features()` 输出的**最尾部**（`field_hamming` 只看字段列，模块零改动）。

### 2.4 模型形态：残差修正（隔离 φ 的贡献）
- 残差目标 r = z − ŝ0_z，**交叉拟合**：训练侧内层 LODO（按数据集，7 次）——每个训练数据集的 ŝ0 用其余训练数据集拟合（shared 先验 / random 任务盲 hgb）。
- 残差模型 g(CLS_binned 116 ⊕ b/290 ⊕ φ) → r̂；**最终分 s = ŝ0_z + r̂，无偏置**（§0：收益必须靠 random 进前 6；带偏置副臂只作描述）。
- φ 置换对照：跨数据集 derangement 打乱 φ（切片按下标对供体切片数取模；5 种子）。

### 2.5 家族 B（转导型）
- 观测：靶切片 s 上该折 hybrid top-6（全 shared）的 R'@60（缓存同轨迹截点）→ φ_B ∈ R⁶，**6 条内 z**（部署时只有这 6 个数；sd=0 → 零向量）。
  源切片（训练侧 39 个）在同 6 列上同法算 φ_B（列语义 = 该折先验第 i 名，逐折定义）。
- `B_resid`：§2.4 残差模型加 φ_B（训练行**剔除** cls_key ∈ top-6 的行——其 z@60 就在 φ_B 里，自泄漏），在 b=290 上给 14 条未起跑候选打分，取前 2（无偏置）。
- `B_cf`：4 条未起跑 shared 的协同过滤——源切片矩阵列 = [top-6 的 6 个 6 条内 z@60] ⊕ [10 个 shared 的 z@290]，kNN(k=3，L2 于 6 观测列)预测目标 4 列，取前 2。
- 规则固定：总是追加 m=2 → 8 条 @60 → 按实测留 3 → @120 留 3 → @290 取最好。成本 8·60 + 3·60 + 3·170 = **1170 = SH8-3-3**。
- 与 behavior_recon 的差异：连续 z（非 k=1 三态码 κ 0.47）、观测列由先验挑（前 6）、问题收窄为"追加 2 条选谁"、对手 = 先验第 7–8 名（同成本）。

### 2.6 官方口径重放（阶段 2，描述性）
`results/mx4_deploy/{candidates.json, runs.jsonl}`（6 靶，目录 shared-10 + GGS + 提名 10，截点 25/50/100%）。
A：s0 = candidates.json 的 prior（shared+GGS）/ 提名 pred_z；φ 在官方 TRAIN 上算（φ_base 用 `protocol_alignment/runs.jsonl` audit_iters0 3 种子，band=2·max(σ,1/n_test)）；
残差模型 = 该靶 LODO 模型（不含该数据集的切片行）；b 特征 = 1.0；名单前 6 → SH6-3-3 于官方截点。
B：prior top-6 的 25% 截点 acc → φ_B → +2 → SH8-3-3。报每靶 oracle 是否入榜、冠军 acc、HW 上 GGS 是否进名单。HAR 不做（无 iters=0 基线，特征需 imputed）。

## 3. 臂表

**A（每实例 126）**：`hybrid`（现行，= WP1 逐位）、`s0z`、`resid_nophi`（残差模型不给 φ：第二级修正本身）、
**`hgb_phi_all`（主臂，12 维）**、`hgb_phi_meta`（meta+base 6 维）、`hgb_phi_slice`（只用切片级变化的 k / N / base / band / sig：排除 T、C——LODO 下数据集常量在测试折从未出现）、
`ridge_int_top4`（ridge on CLS_binned ⊕ φ ⊕ {cross_scale, pool, norm, emb_mask} one-hot × φ）、`hgb_phi_shuffle_{0..4}`、副臂 `hgb_phi_all_hyb`（带偏置，描述）。
指标：oracle 入前 6 率（主）、regret@3（主）、regret@1、前 6 内最好、δ 存活对配对 acc、SH6-3-3 起跑名单换成本臂的 @290 regret（描述性）、分数据集 @290 表、φ 分组置换重要性（meta/base/sig）。

**B（每实例 42 @290）**：**`B_resid`（主臂）**、`B_cf`、`B_nophi`（残差模型无 φ_B 的前 2）、`B_shuffle_{0..4}`（φ_B 跨数据集 derangement）、
对照 `+2_prior78`（≡ SH8-3-3_prior，同成本 1170）、`+2_gbdt`（未起跑候选按 s0z 前 2）、`+2_random`（C(14,2)=91 对精确期望）、
参考 `SH6-3-3_prior`（1050）、`SH10-4-3_prior`（1350）。指标：oracle 入 8 率（主）、SH6+2-3-3 @290 regret（主）、成本、新鲜种子复核（仅最终 pick ∈ shared 的实例，描述性）。

## 4. 判读（先于正式跑写死；配对单侧 Wilcoxon α=0.05，n 如实报；A/B 两族各自判读，不做跨族校正但读数码要求族内两门联立）

| 门 | 检验 | n |
|---|---|---:|
| **A1**（名单质量） | `hgb_phi_all` oracle 入前 6 @290 ≥ hybrid + 3（即 ≥ 29/42）∧ regret@3 pooled < hybrid3（p<0.05） | 42 / 126 |
| **A2**（归因于 φ） | `hgb_phi_all` regret@3 < `resid_nophi`（p<0.05）∧ 真增益(vs hybrid) ≥ 2 × 五个 shuffle 臂增益均值 | 126 |
| **B1**（名单质量） | `B_resid` oracle 入 8 @290 ≥ `+2_prior78` + 3 ∧ SH6+2-3-3 regret < `+2_prior78`（p<0.05） | 42 |
| **B2**（归因于 φ_B） | `B_resid` regret < `B_nophi`（p<0.05）∧ 真增益(vs +2_prior78) ≥ 2 × 五个 shuffle 臂增益均值 | 42 |
| **B3**（经济性，描述性） | SH6+2-3-3_B ≤ SH10-4-3_prior + 0.005（成本 1170 vs 1350） | 42 |

读数码：**TF-WIN** = (A1 ∧ A2) ∧ (B1 ∧ B2)；**TF-A-WIN** = A1 ∧ A2 且 B 不过；**TF-B-WIN** = B1 ∧ B2 且 A 不过；
**TF-DEAD** = 两族都不过（上界 C1 非空 → "信号在，但这些特征 / 6 个实测数在 n=7 下辨识不出"）。任一门不过如实入 REPORT，不 reframe。
B3、副臂、阶段 2、新鲜种子复核只报不判。

## 5. 预测（先验写死；阶段 0 已看过）

| # | 预测 | 先验 |
|---|---|---|
| P2 | A1 ∧ A2（TF-A-WIN 条件） | 25% |
| P3 | B1 ∧ B2（TF-B-WIN 条件） | 45% |
| P4 | B3：SH6+2-3-3_B 与 SH10-4-3_prior 同带或更好 | 50% |
| P5 | 官方口径 HW 上 B 让 GGS 进 8 条起跑名单 | 40% |
| P7 | `resid_nophi` 单独已显著优于 hybrid3（第二级修正效应） | 40% |
| P8 | φ 分组置换重要性 sig > meta | 55% |
| P9 | `hgb_phi_slice` 与 `hgb_phi_all` regret@3 之差 < 0.003（数据集级常量 T/C 无增量） | 60% |

## 6. 成本
零 GPU。φ 计算 ≈ 3–4 min（8 次 .ts 解析）；A 7 折 × (7 内层 + 十余臂) HGB ≈ 5 min；B ≈ 2 min；阶段 2 ≈ 2 min；单测 < 30 s。

## 7. 不做的事
不改 `36/37/39` 与 `autocls/search/surrogate.py`、`cls_encoder.py`（铁律 7：模块只见向量）；不产新 R'、不跑 GPU；不动 SH 淘汰规则（关键决策 8）；
不重训 T-CLSC；不做 HAR 重放（无基线）；不做 WP4 画像与 E 对照（用户拍板不选）。

## 8. 跑后修订记录（如实登记，不改判读线）

实现细节澄清（协议正文未写死、实现时定，判读线未动）：
1. φ 置换对照（A 与 B）实现为**测试时替换**：模型用真 φ 训练，评估时把靶切片的 φ 换成数据集级 derangement 供体切片的 φ（切片按下标对供体切片数取模），5 种子；不重训。
2. B 的残差模型训练行剔除 cls_key ∈ top-6 后为 1638 行/折；源切片 39 个全部有同 6 列 @60。`B_cf` 只给 4 条未起跑 shared 打分（kNN k=3，目标列 = 源切片 z@290）。
3. `+2_random` 用 C(14,2)=91 对的精确期望（regret 与 oracle 入 8 率），不抽样。
4. 阶段 2：源切片 φ_B 需要 GGS 列时从 `rprime_cache.jsonl` 的 ggs 行补入（`load_inputs` 的 cat 不含 ggs 行）；官方 φ 的 T 取全长、k 取全类、base 取 `audit_iters0` 3 种子，与切片 φ 口径有差，阶段 2 只作描述。
5. φ 分组置换重要性在训练侧样本内计算（7 折均值），只作 P8 描述。
读数：**TF-DEAD**（A1 ✗ A2 ✗ B1 ✗ B2 ✗，B3 ✓），见 REPORT.md。
