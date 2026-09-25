# WP17-ACTS S3-E PROTOCOL：提升 T-CLSC（头 + E 辅助监督 + 连续加权标签 + 集成不确定性搜索）

**会话定位**：
- S3-C（严格复现 T-CLSC）：排序 S3-E-INERT（换源 Δ ≈ 0），搜索 SEARCH-DEAD+CATALOG-LOSS。
- S3-C-D（部署口径）：A 0.605 ≈ 任务盲 D 0.596，都不如目录内做法。
- 用户 2026-09-25 定调：主架构 = T-CLSC + 全空间搜索，放弃一切探针思路；授权直接上实验室服务器、八卡并行。
- 本协议是改进方案的**第一阶段**，不需要新标签：
  - **L1**：diff / field 头 + E 辅助监督；
  - **L4**：两种子连续加权配对标签；
  - **L2**：5 种子集成 + 胜场层面的不确定性惩罚 + 家族分散的前 3。
- 第二阶段（L3 源侧对抗扩充）视本阶段读数另立协议。

产物 `results/acts/s3e/`；代码 `scripts/91_acts_s3e.py`（复用 89 的数据、标签、Round-Robin、分块选择）；单测 `tests/test_acts_s3e.py`；服务器 `scripts/pa_pod/acts_lab_run_s3e.sh`。

## §0 已看过的读数（如实声明）
- S3-C / S3-C-D 全部读数，以及逐源表。与本设计直接相关的：
  - concat 头下换源 Δ ≈ 0；SetTF 集内余弦 0.49；名单外 Spearman 0.22；
  - 搜索提名全部来自进化代；D 在 HW 的提名为 0.36–0.38，A 在 Cricket 有一条 0.458；
  - 源侧目录任务盲前 3 0.617、oracle 0.635。
- 历史机制读数：L3（concat + 硬标签 → 任务向量退化为偏置，diff 头 / 连续标签可解开）；S3-B″（field 结构下 E_sh 内容被用上，p 0.003）。
- **没看过**：任何 diff / field 头 T-CLSC、辅助监督、加权标签、集成搜索在 S2 语料上的读数。

## §1 共同设置（同 S3-C，除下文列出的改动）
- 数据、折、E：6 个 UEA 折（HW / PS / UWave / Cricket / Ethanol / SCP1），160 任务训练池，每折留出一源、另一源作验证源（同 S3-C）；E = E_pt 窗口集合（S3-B 抽取）+ TaskSetEncoder(256, 256) 联合训练；32 窗子采。
- 优化：BCE，Adam 1e-4，batch 64，≤ 100 epoch，按验证源损失早停（patience 5），保留最好权重。

## §2 筛选（L1 × L4，单种子 seed 0）
2 × 2 × 2 = 8 臂 × 6 折 = 48 次训练：

| 因子 | 取值 |
|---|---|
| 头 | **diff**：logit = s(l_a, t) − s(l_b, t)；**field**：logit = s₀(l_a) − s₀(l_b) + ⟨W(t), x_a − x_b⟩（`TCLSC(head=...)`，t = set_fc(E)） |
| E 辅助监督 | **off** / **on**：on 时 SetTF 输出 E 另接 Linear(256 → 3)，回归该任务在折内低秩轴上的坐标 a(t)（`84._fold` 在训练任务上的投影，按训练侧标准化）；总损失 = BCE + 1.0 × MSE；只用训练任务 |
| 标签 | **L0**（AutoCTS++ 原样，同 S3-C）：seed-0 best-of-3，y = 1[r_a > r_b]，权重 1；**L4**：shared 用两种子终点均值、random 用 seed-0 终点，y 同上，权重 w = min(\|Δ\| / (2σ_pair), 1)，σ_pair = σ_task·√(1/n_a + 1/n_b)，σ_task = 该任务的 `top_half_sigma`，n = 种子数 |

- 配对方式与 S3-C 相同（linear 动态配对、自配对跳过、不加反向对）；L4 用加权 BCE。验证源损失与训练同口径。
- **选臂规则（先于数据）**：每臂在 6 折的**验证源**任务上算 shared-30 regret@3（Round-Robin 前 3 → seed 0 终点选 → seed 1 终点评），取 6 折均值，最小者为 **★ 臂**。平局按 diff 先于 field、on 先于 off、L4 先于 L0 的顺序取。验证源不是留出源，选臂不看留出源。

## §3 集成与搜索（L2）
- ★ 臂补训 seed 1–4（6 折 × 4 = 24 次），与 seed 0 组成 5 模型集成。
- **集成排序**：每个模型各自对候选集做 Round-Robin 胜场 w_m(c)；适应度 f(c) = mean_m w_m(c) − λ·std_m w_m(c)，λ = 1；按 (−f, key) 排序。λ = 0 即集成均值。
- **搜索**：流程与 S3-C 相同（Ks 20,000，同一采样种子；40 块 × 500 → 400 → 10；进化 10 代，p1 0.8 / p2 0.2），只把每一步的排序换成上面的 f。交叉 / 变异 / 合法性复用 `autocls.search.evolution` 的 `_make_offspring`。
- **三个搜索臂**（24 个源侧搜索任务，同 S3-C）：
  - **E-mean**：λ = 0，最终种群前 3；
  - **E-lcb**：λ = 1，最终种群前 3；
  - **E-lcb-div**：λ = 1，按最终排序依次取，跳过与已选配置同一 (loss_type, sim_func, norm) 家族的，取满 3 条。
- **真值**：提名配置在切片上按 `ts2vec_official` cv5 跑 seed 0 / 1（83 phase X，与 S3-C 同一执行体）；seed 0 选、seed 1 评。已跑过的格（S2 或 S3-C）直接复用。

## §4 读数与门
### 4.1 排序（56 个留出任务，★ 臂的 seed 0 单模型与 5 模型集成均报；另报全部 8 臂的单模型读数）
- regret@3（shared-30）；16 偏移换源 Δ；名单外 random-20 中位 Spearman；SetTF 集内 / 跨集余弦。
- 对照：S3-C 的 A（concat，0.012）/ D（0.015），任务盲 μ（0.016）。
- **排序门（★ 臂集成）**：
  - **R1 内容**：换源 Δ ≥ 0.005，且 6 个源同号（源级 Wilcoxon 最小 p 0.031）；
  - **R2 价值**：regret@3 比 S3-C A 低 ≥ 0.003（点估计）；
  - **R3 名单外**：random-20 中位 Spearman ≥ 0.35。

### 4.2 搜索（24 个任务）
- 最终 acc（seed 0 选、seed 1 评）；对照：S3-C 的 A（0.608）/ D / 随机 R（0.599）、目录任务盲前 3（0.617）、目录 oracle（0.635）。
- **搜索门（主臂 = E-lcb）**：
  - **S1 胜随机**：− R ≥ +0.010，任务级 Wilcoxon p < 0.05，且 ≥ 5/6 源同号；
  - **S2 不输目录**：− 目录任务盲前 3 ≥ 0（点估计）；
  - **S3 胜旧 T-CLSC**：− S3-C A ≥ +0.010，任务级 p < 0.05。
- **读数码**：S1 ∧ S2 ∧ S3 → **S3E-SEARCH-WIN**；S2 ✓ 但 S1 或 S3 ✗ → **S3E-CATALOG-PAR**；S2 ✗ → **S3E-SEARCH-LOSS**。排序门作后缀（**+R1** / **+R2** / **+R3**，各自过则加）。

## §5 预测（先验）
| # | 预测 | 先验 |
|---|---|---|
| P1 | ★ 臂的头是 diff | 50% |
| P2 | ★ 臂辅助监督 = on | 55% |
| P3 | R1 ✓ | 30% |
| P4 | R3 ✓ | 15% |
| P5 | E-lcb 最终 acc > E-mean | 55% |
| P6 | S2 ✓（不输目录任务盲前 3） | 30% |
| P7 | S1 ✓ | 25% |

## §6 成本（共享 4090；八卡按空闲显存分槽）
- 筛选 48 次 + 集成 24 次训练，每次 12–25 min，约 20–25 GPU-h。
- 搜索 3 臂 × 6 折（每折 4 任务、5 模型），约 1 GPU-h。
- 真跑 ≤ 3 × 24 × 3 × 2 = 432 格，单格中位约 49 s，约 6 GPU-h。
- 八卡墙钟约 4–6 h。

## §7 执行
PROTOCOL + 单测先红 → 协议锁 → 实现 91（`--train` / `--select` / `--search` / `--plan-runs` / `--analyze` / `--smoke`）→ 本地冒烟 → 服务器全自动流水（筛选 → 选臂 → 集成 → 搜索 → 计划 → 真跑 → 打包）→ 拉回 → analyze → REPORT。

## §8 铁律对照
3（runs.jsonl + resume）；4（TCLSC 头已注释 AutoCTS++ 对应项）；5（先红、§0 声明）；6（训练、辅助目标、选臂只用训练源与验证源；留出源只考）；7；8。

## §9 执行台账（跑后填写）
- 协议锁 09d21b7（6 例先红）→ 实现 `scripts/91_acts_s3e.py` + `scripts/pa_pod/acts_lab_run_s3e.sh`（全自动流水，按各卡空闲显存分槽）；单测 6 例绿；本地 `--prep` 生成 axis_targets.npz（6 折轴坐标目标）；本地冒烟 OK（合成窗口：diff-on-L4 / field-off-L0 × 1 折 × 2 epoch + seed 1；两模型集成搜索 Ks 1,000 × 2 代给出 mean / lcb / lcb-div 各 3 条）。
