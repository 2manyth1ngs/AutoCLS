# WP16 SP1 检验 PROTOCOL：剪空间 S⁺——任务盲的"明显错误配置"分类器，定义 AutoCTS++ 式全空间搜索的采样域（步骤 1，零 GPU）

**会话定位**：用户 2026-09-18 重申主张：E + T-CLSC 必须是 AutoCTS++ 式的——候选从搜索空间里由比较器 + E 搜出来（不是信任域 11 条里排名），部署超任务盲先验，cross 消融可见；接受"避免明显错误配置的空间"。
EV1 的机理读数：训练标签的分布（random-10 全空间均匀，65% 落在破坏区、好区域每切片 ~3.5 条）≠ 搜索的分布（好区域几万条），比较器在无标签处高估、进化精确找到那里。AutoCTS++ 的对应做法是先按领域规则剪掉退化结构、再在剩余空间均匀采样打标签。
本协议 = 四步计划的第一步：**用现有源侧标签学一个任务盲的空间剪枝器，定义 S⁺（非破坏区域），验证它可迁移、纯、且够大**；过门才请示步骤 2（S⁺ 上均匀补标签，pod ≈ $12）。产物 `results/tclsc_sp/`；脚本 `scripts/78_tclsc_sp.py`；单测 `tests/test_tclsc_sp.py`。

## §0 已看过的读数（写协议前为定口径查过的描述量；**未看**任何分类器的表现）

- 源侧 @290 行：87 切片 × (shared-10 + random-10) + GGS = 1827 行、881 个不同配置；空间大小 4.73×10¹²。
- 未训练基线（iters=0，5 种子）只覆盖原 45 切片，扩展 42 切片无。strict 破坏（R' < 基线均值 − 带）在 45 切片上：shared 29% / random 42% / GGS 16%；**GGS 在 Ethanol 83%、SCP2 33%、UniMiB 17% 被判"破坏"**——近 chance 的数据集上"最好的 shared"也在基线带内或以下（Ethanol 6 切片 margin −0.078…+0.044、HMD 3/6 负、Cricket 3/6 负、SCP2 2/3 负），strict 判据在这些切片上量的是噪声不是破坏；HW（margin 0.17–0.34）/ UniMiB / UWave / PS 可评。
- WP2：字段结构化代理（HGB on binned）LODO 配对 acc 0.78、对目录外候选高估（S3 不过）；解耦实验：另一语料上随机配置 65% 破坏性；字段主效应 temperature ≫ sim_func > norm > freq_mask。
- EV1：比较器"避坏"成立（提名均值 vs 随机 +0.06），"找更好"不成立。

## §1 定义

- **训练目标（主）**：within-(切片, 290) z 分数（`zscore_within`），任务盲代理 `FieldSurrogate("hgb")` on `CLSEncoder(binned)` 116 维（WP2 同构；扩展字段 input_mask 不在标签里，S⁺ 按基础字段定义，掩码开关两版都在 S⁺）。
- **"明显错误"的两种量法**：(a) z-bad：z < −1（切片内底部）；(b) strict：R' < 基线均值 − 带，只在**可评切片**上算（可评 = 最好 shared − (基线均值 + 带) ≥ 0.02；按 §0 约 20–30 个原切片）。
- **S⁺** = {c ∈ 空间 : ẑ(c) ≥ τ}，τ 在**训练折**上取使 kept 配置的 z-bad 率 ≤ 10% 的最小值（曲线整体报）；采样器 = 均匀采样 + 拒绝（接受率 = S⁺ 占空间比例的估计）。
- 划分：LODO 7 折（SCP2 只训练）；每折报 (i) held-out 全部配置、(ii) held-out 的 random 配置（训练里从未出现的配置）两种口径。

## §2 判读（先于数据）

| 门 | 定义 |
|---|---|
| C0 | 代理逐位确定（同 seed 重训预测相同）；采样器输出全部合法且 ẑ ≥ τ；z / strict 标签单测钉死 |
| **G-S1 学得到** | LODO held-out **random 配置**上：z-bad 的 AUC ≥ 0.85 且 Spearman(ẑ, z) ≥ 0.5（7 折 pooled） |
| **G-S2 纯度** | τ 只用训练折选；held-out kept 配置 z-bad 率 ≤ 10%（pooled），可评切片上 strict 破坏率 ≤ 10% |
| **G-S3 覆盖** | S⁺ 占空间 ≥ 20%（10 万条均匀样本估计，用全数据拟合的代理）；held-out 好配置（z ≥ +0.5）召回 ≥ 0.90 |

**读数码**：G-S1 ✓ ∧ G-S2 ✓ ∧ G-S3 ✓ → **SP-GO**（请示步骤 2）；G-S1 ✓ ∧ G-S2 ✓ ∧ S⁺ < 20% 或召回 < 0.9 → **SP-TIGHT**（剪枝器能用但域太小 / 剪掉好配置，报数字待拍板）；G-S1 ✗ 或 G-S2 ✗ → **SP-DEAD**（破坏区域跨数据集不一致、任务盲剪不了 → 剪枝只能在目标上测量做）。
只报：逐折；字段置换重要性；按数据集的破坏率；τ–纯度–覆盖曲线；ridge 对照；strict 口径 vs z-bad 口径一致性。

## §3 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | G-S1 AUC ≥ 0.85 | 70% |
| P2 | S⁺ 占空间 20–50% | 55% |
| P3 | 置换重要性前 2 = temperature、sim_func | 70% |
| P4 | G-S2 纯度过 | 60% |
| P5 | 好配置召回 ≥ 0.90 | 65% |
| P6 | LODO Spearman（random 配置）≥ 0.5 | 55% |

## §4 执行

1. 本文件 + `tests/test_tclsc_sp.py`（先红）→ 协议锁 commit；2. `scripts/78_tclsc_sp.py`（--labels / --fit / --analyze，零 GPU）→ 绿 → 跑 → REPORT → commit → memory；3. SP-GO 时写步骤 2 协议（S⁺ 上每切片 20 条均匀标签 + 扩展切片基线）再请示 pod。

## §5 铁律对照

铁律 1（采样只经 space.sample_random + validate）；6（LODO 按数据集；τ 只用训练折）；7（代理 / 采样器只见向量，不见数据集概念——脚本层做 z 归一化与折划分）；8（z 由 R' 按分类 is_better 方向定义：高为优）。

## §6 执行台账（跑后填写）

- 2026-09-18 本地零 GPU：`78 --all` 28 s；6 例单测绿；**SP-DEAD**（G-S1 ✗ random AUC 0.709 / ρ 0.357、G-S2 ✗ kept z-bad 0.153 / strict 0.261（random）、G-S3 ✓ S⁺ 0.653 / 召回 0.906；预测 2/6）。五个非 chance 集 AUC 0.75–0.84 / kept z-bad 2–10%，Ethanol 0.43 / HMD 0.55 拉低 pooled；random-10 实测 z-bad 25%（非 65%），剪枝对步骤 2 的覆盖增益只 1.2–1.3×。REPORT.md。
