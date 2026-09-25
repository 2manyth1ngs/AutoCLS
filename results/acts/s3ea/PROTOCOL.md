# WP17-ACTS S3-EA PROTOCOL：S3-E 搜索增益的归因对照（任务盲集成 + 换源 E）

**会话定位**：S3-E（S3E-CATALOG-PAR）里，★ 臂（field-on-L4）5 种子集成 + LCB 做全空间搜索，最终 acc 0.618：
- 对同成本随机 +0.019（p 0.016）；
- 与目录任务盲前 3（0.617）持平；
- 对 S3-C 严格复现的 T-CLSC（A 0.608）+0.010，不显著。

但它的排序读数没有变好（换源 −0.003），而且 30 次训练里有 21 次在 epoch 0 早停。所以这 0.618 可能来自"集成 + LCB + L4 标签 + field 头的任务盲部分"，而不是 E。

用户 2026-09-26 拍板先做归因对照，八卡并行。本协议只回答一个问题：**S3-E 搜索里的增益，有多少来自 E？** 不改 S3-E 的任何设置，只加两个对照臂：
- **B（任务盲集成）**：去掉 E 的同款比较器。field 头去掉任务加性项后，logit = s₀(l_a) − s₀(l_b)；没有 SetTF、没有 field_w、没有辅助损失。标签（L4）、优化、早停、5 种子、搜索流程、λ 全部与 ★ 相同。
- **X（换源 E）**：用 S3-E 已训好的 ★ 5 模型，只把目标任务的 E 换成另一个源同 OA 行任务的 E（`88.cross_maps` 偏移 1，与 S3-B / S3-C 的换源同一指派），其余不变。

产物 `results/acts/s3ea/`；代码 `scripts/92_acts_s3ea.py`（复用 91 / 89 的标签、配对、Round-Robin、分块筛选、进化）；单测 `tests/test_acts_s3ea.py`；服务器 `scripts/pa_pod/acts_lab_run_s3ea.sh`。

## §0 已看过的读数（如实声明）
- S3-E 全部读数（REPORT / tables / analysis.json）。其中：
  - E-lcb 0.618、E-mean 0.612，逐源表；
  - ★ 集成排序 regret@3 0.019、random-20 Spearman 0.13；
  - ★ 臂最好 epoch 分布：21/30 为 0；逐种子留出 regret 0.011–0.019。
- S3-C 的 D（任务盲、concat 头、L0 标签、单模型）：搜索 0.597，训练约 10 s、最好 epoch 多在 10 以后。
- **没看过**：field 头任务盲集成（L4、5 种子）的任何排序或搜索读数；换源 E 下的搜索提名。

## §1 设置（与 S3-E 相同之处不再重复）
- 6 个 UEA 折、训练 / 验证 / 留出划分、L4 标签与权重、BCE、Adam 1e-4、batch 64、≤ 100 epoch、按验证源损失早停（patience 5）：全部同 S3-E §1–§2。
- **B 臂训练**：`TCLSC(cls_dim, None, task_branch="none", head="field")`，seed 0–4 × 6 折 = 30 次；`torch.manual_seed(seed)` 后构造。
- **搜索**（24 个源侧任务，同 S3-C / S3-E）：Ks 20,000（同一采样种子）→ 40 块 × 500 → 400 → 10 → 进化 10 代（p1 0.8 / p2 0.2），适应度 f = mean_m w_m − λ·std_m w_m。
  - B 臂出三组：**B-mean**（λ = 0）、**B-lcb**（λ = 1）、**B-lcb-div**（λ = 1，家族分散）。
  - X 臂只出 **X-lcb**（λ = 1）。
  - 采样与进化随机种子与 S3-E 逐位相同（`run_seed_for(t, "s3c_ks")` / `"s3c_evo"`），候选池因此相同。
- **真值**：提名配置按 `ts2vec_official` cv5 跑 seed 0 / 1（83 phase X）；seed 0 选、seed 1 评。S2 / S3-C / S3-E 已跑过的格直接复用。

## §2 读数与门
### 2.1 搜索（24 个任务；主读数）
- 每臂的最终 acc；配对差用任务级 Wilcoxon，另报 6 个源里同号的源数与源级 bootstrap CI。
- **主对比 Δ_E = E-lcb − B-lcb**：
  - **EA-E-CONTRIB**：Δ_E ≥ +0.010，且 p < 0.05，且 ≥ 4/6 源为正；
  - **EA-E-HURT**：Δ_E ≤ −0.010，且 p < 0.05；
  - **EA-E-NULL**：其余情况。
- **后缀**：
  - **+CONTENT**：Δ_X = E-lcb − X-lcb ≥ +0.010，且 p < 0.05，且 ≥ 4/6 源为正（E 的内容在搜索里被用上）；
  - **+BLIND-PAR**：B-lcb − 目录任务盲前 3（0.617）≥ 0（点估计；任务盲集成也不输目录）。
- 描述项（不进门）：
  - B-lcb − B-mean（任务盲下 LCB 的作用）；
  - B-lcb − S3-C D（0.597；field 头 + L4 + 集成 + LCB 对任务盲单模型的总增益）；
  - B-lcb vs 随机 R / 目录 oracle；逐源表。

### 2.2 排序（56 个留出任务；描述项，不进门）
- B 臂 5 种子集成（胜场均值）的 regret@3（shared-30，seed 0 选、seed 1 评）与 random-20 中位 Spearman，与 ★ 集成（0.019 / 0.13）逐任务配对，报源级 Wilcoxon。
- B 臂最好 epoch 分布。

### 2.3 读法（先于数据）
- **E-NULL + BLIND-PAR**：S3-E 的搜索增益来自集成 + LCB + L4 / field 的任务盲部分，E 在搜索里没贡献。下一步先让 E 起作用（修训练动力学），再做 L3。
- **E-CONTRIB + CONTENT**：E 的内容在搜索里有用，按原计划修训练 + L3。
- **E-CONTRIB，无 CONTENT**：E 通路起的是结构 / 正则作用，不是内容。
- **E-HURT**：E 通路在搜索里有害，任务盲集成更好。

## §3 预测（先验）
| # | 预测 | 先验 |
|---|---|---|
| P1 | 读数码主体 = EA-E-NULL | 60% |
| P2 | +BLIND-PAR（B-lcb ≥ 0.617） | 40% |
| P3 | B-lcb − S3-C D ≥ +0.010 | 55% |
| P4 | B-lcb > B-mean | 55% |
| P5 | 不带 +CONTENT | 75% |
| P6 | B 集成留出 regret@3 ≤ ★ 集成 0.019 | 55% |
| P7 | B 臂最好 epoch 中位数 ≥ 3 | 65% |

## §4 成本（共享 4090；按空闲显存分槽）
- B 臂训练 30 次，每次约 10–30 s（无 SetTF）。
- 搜索：B 臂 6 折 × 4 任务 × 5 模型；X 臂 6 折 × 4 任务 × 5 个 ★ 模型。约 5 min。
- 真跑 ≤ 4 组 × 24 任务 × 3 条 × 2 种子 = 576 格，去重后预计 350–450 格，单格中位约 50 s，约 5–6 GPU-h。
- 八卡墙钟约 1 h。

## §5 执行
PROTOCOL + 单测先红 → 协议锁 → 实现 92（`--train` / `--search` / `--plan-runs` / `--analyze` / `--smoke`）→ 本地冒烟 → 服务器全自动流水（B 训练 → 搜索 → 计划 → 真跑 → 打包）→ 拉回 → analyze → REPORT。

## §6 铁律对照
3（runs.jsonl + resume）；5（先红、§0 声明）；6（训练、早停只用训练源与验证源；留出源只考；X 臂的换源 E 来自其他源，不引入留出源标签）；7；8。

## §7 执行台账（跑后填写）
