# MX-4 WP9-P1 协议：官方口径 6 靶候选池扩到 100 条，零 GPU 模拟"学来的名单 + 淘汰赛"与标准 HPO 的同成本对照

日期：2026-09-06 草案 / 2026-09-07 拍板锁定 ｜ 状态：**协议锁**（判读带 §4 与预测 §5 在任何新格与任何基线读数之前写死；协议锁 commit 后才开 pod）｜
分支 `wp9-p1`（= wp8-bo 3a6fc85 + wp10-uea30 0c2ed6c 合并 54ac004；WP6 掩码字段与 UEA30 靶外部署已并入）｜
脚本 `scripts/46_p1_pool.py`（薄驱动，importlib 复用 35/36/39/43/45，零改动）+ 真源 `configs/p1.yaml` + 单测 `tests/test_p1_pool.py`（16 项，先红后绿）｜
产物目录 `results/mx4_p1/`。场地 = WP3 官方口径 6 个 UEA 靶（`configs/train.yaml: protocols.ts2vec_official`，HW iters 200 / 其余 600，截点 25/50/100%，
截点评估 SWA 当前平均权重，官方划分 test acc）。所有 task-aware / 数据集特征 / 学习曲线外推 / 开放空间搜索的线已关闭，本包不碰。

## 0. 已看过的读数（如实声明，铁律 5）

写协议前已看过并且直接决定了本包数字下界的读数：
- WP3（`results/mx4_deploy/REPORT.md`）：每靶目录 21 条（PS 20，一格 SVM 超时）三截点齐；SH6-3-3_prior seed 0 = HW 0.419 / UWave 0.931 /
  PS 0.251 / EC 0.274 / Cricket 1.000 / HMD 0.378；目录 oracle 0.546 / 0.931 / 0.251 / 0.300 / 1.000 / 0.486；SH10-4-3 在 HW 选中 GGS 0.546；
  25%→100% 排名 Spearman 中位 0.71；3 种子冠军 0.431 / 0.918 / 0.256 / 0.241 / 0.986 / 0.450。
- WP6（`results/mx4_mask/s1_sweep.md` 的 `top10_off_3lvl` 行 = SH10-4-3_prior 掩码关，目录 21 子池 seed 0）：HW 0.546 / UWave 0.931 / PS 0.251 /
  EC 0.297 / Cricket 1.000 / HMD 0.378；pick c92058f2 / 8c4aa283 / 4f7d4135 / cffb5503 / c92058f2 / 9092e36e。3 种子六靶追平 TS2Vec 6/6。
- WP8（`results/mx4_bo/REPORT.md`、`stage0.json` 的 SH10-4-3_prior 策略行）：每靶 bo_ucb / rand_nbr 提名各 15 条完整轨迹（PS bo 14 + 1 error，
  EC rand 有 1 条与 bo 重键）；51 条池 oracle = HW 0.549 / HMD 0.486 / UWave 0.934 / Cricket 1.000 / EC **0.437**（单跑；3 种子 0.373）/ PS 0.251；
  池内 ρ(@25%, @100%) 0.22–0.79；BO-SH / RAND-SH（7.5）3 种子 vs SH6-3-3 均值差 −0.011。WP8 行在 **4060** 上跑（WP3 在 4090）。
- WP10（`results/uea30/REPORT.md`）：SH10-4-3_prior 靶外 23 靶 3 种子追平 TS2Vec 23/23，故本包头条档定 5.0。
- 由此**在跑任何新格之前就已知**：SH10-4-3_prior 相对 51 条池 oracle 的 seed-0 regret 下界 = HW 0.003 / HMD 0.108 / UWave 0.003 / Cricket 0 /
  EC 0.140 / PS 0，六靶均值 ≥ 0.042；SH6-3-3_prior 同口径 0.130 / 0.108 / 0.003 / 0 / 0.163 / 0，均值 ≥ 0.067。100 条池只会让它更大或不变。
  这个数字不是本包要"发现"的，而是要如实报告并解释的（HMD / EC 的名单失败：目录外提名 0.486 / 0.437 从未起跑）。
- 协议锁前已跑 `--simulate --c0`（`c0.json`，只出我们的臂，不出基线列）：SH6-3-3_prior 六靶与 WP3 `analysis.json` 逐位、SH10-4-3_prior 与 WP6 扫臂 /
  WP8 stage0 逐位、代理对 WP3 提名 10 条的 pred_z 复现到 1e-9（`test_c0_*` 三项 PASS）。51 条池 oracle 与 WP8 同（HW 0.549 / HMD 0.486 /
  UWave 0.934 / Cricket 1.000 / EC 0.437 / PS 0.251）。
- 其余读数（WP1/WP2/WP5/WP6-TF/WP7 的切片协议数字、`results/mx4_source_size`）与本包的门无关，只进消融主表。
- **未看过、本包要回答的**：任何 HPO 基线（Hyperband / 随机搜索 / 随机起点 SH / TPE）在这些池上的读数；随机补足的 297 格的任何读数；
  UCR 84 语料上"UEA 先验 + SH"的 regret；3 种子核过的池 oracle。**在协议锁之前不跑基线模拟**（`--simulate` 只以 `--c0` 模式跑过）。

## 1. 固定项（协议锁）

| 项 | 值 |
|---|---|
| `configs/p1.yaml` md5（去 CR） | 见协议锁 commit 的 `analysis.json: config_md5`；`pool.json / c0.json` 里同一字段必须一致 |
| 池定义 | `results/mx4_p1/pool.json`（协议锁产物，pod 上不重建）：每靶 100 条 = 已见 51（EC 50 / PS 51 含 2 error）+ 随机 49（EC 50 / PS 51），共 **297** 新格 |
| 随机采样 | `autocls.labeling._sample_unique(space, default_rng(4500 + order), n, taken)`，taken = 该靶三源已见全部键（含 error）；`sample_random` 天然 `input_mask=none`；前缀稳定 |
| 执行顺序 | order 0..5 = Handwriting → HandMovementDirection → UWaveGestureLibrary → Cricket → EthanolConcentration → PhonemeSpectra（便宜先） |
| 新格身份 | stage `s1_pool`、arm `random_space`、seed 0、`39.checkpoints_for(iters)`；`35.cache_key` 含 arm → 与三源旧键零碰撞（单测） |
| 代理 | `FieldSurrogate("hgb", seed = 0 + nominee_idx)`（nominee_idx = `35.TARGETS` 序，与 WP3 `cmd_nominate` 同 seed），拟合于 `43.source_rows(d)`（LODO 源池 z 行），特征 `CLSEncoder(binned) ⊕ b/290`；对池成员以 `X ⊕ 1.0` 预测；WP8 `bo_state.json` 的回灌 pred_z **不用** |
| 三源合池 | `load_pool_rows`：三源同靶 seed-0 目录行的 `protocol / iters / checkpoints / options` 逐字段相同否则 SystemExit；同 cls_key 多行按源序 wp3 < wp8 < new、同源按 arm 升序取第一条（完整行优先），其余记 `duplicates`（EC 1 条） |
| 模拟 | `N_DRAWS = 200`，种子 rand_sh 4600 / hb 4700 / tpe 4800 各 + order；成本 = 0.25·k0 + 0.25·k1 + 0.5·k2 全跑当量（FPE） |
| 数据装载 | `39.execute` 默认 `load_arrays_any` → `35.load_arrays`（与 WP3 / WP8 同一份数组）；pod 侧 6 靶 `.ts` 原始字节 md5 对账 |
| 评估超时 | Linux SIGALRM 480 s（39）；Windows 走 43 号 loky 子进程版；超时整格 error、不补采 |

## 2. 候选池（每靶 100 条，seed 0 完整 25/50/100% 轨迹）

| 来源 | 条数 | 状态 | origin |
|---|---:|---|---|
| shared-10 + GGS | 11（PS 10） | 已有（WP3 S1，4090） | shared / ggs |
| WP3 一次性代理提名 | 10 | 已有（WP3 S1） | nominated_wp3 |
| WP8 bo_ucb 提名 | 15（PS 14） | 已有（WP8，4060） | bo_ucb_wp8 |
| WP8 rand_nbr 提名 | 15（EC 14 去重） | 已有（WP8，4060） | rand_nbr_wp8 |
| **空间均匀随机** | **49（EC 50 / PS 51）** | **新跑 297 格** | random_space |

- 三种零样本序（pool.json）：`prior_order`（11 条先验降序，其余按 key 垫底）、`hybrid_order`（11 条置顶 + 其余 89 条按 pred_z）、`surrogate_order`（100 条按 pred_z）。
- 随机格若截点评估超时记 error 行、不补采；池 = 三截点齐全者，实际条数 ≤ 100，REPORT 报 error 数。error 格对基线是"免费跳过"（基线不会抽到它），
  对基线有利，作 limitation 写明。
- **公平性说明（先写）**：基线从池均匀抽样，而池的 51% 是策划过的配置（目录 + 邻域提名），比真实空间（解耦实验：随机空间 65% 破坏性）好得多——
  这有利于基线。另报只在 **随机 49 条子池** 与 **策划 51 条子池** 上抽样的基线行（§3.3）。
- **硬件混合 limitation**：WP8 的 30 条在 4060、WP3 与本包在 4090。成本主口径用 iters 全跑当量（硬件无关）；秒只按来源带硬件标签描述。

## 3. 零 GPU 模拟（`--simulate`；全部在 pool.json + 缓存上）

复用不重写：`36.sh_path / sh_pick / path_cost_iters / path_cost_seconds / successive_halving / expected_random_sh / rank_by / expected_max_of_k`、
`39.trajectory_from_rows / deploy_arms / finals_worklist / execute`、`43.source_rows / three_seed`、`45.drop_repeated_oom`。淘汰规则不动（关键决策 8）。

### 3.1 我们的臂（确定性，单条路径）
| 臂 | 定义 | 成本 |
|---|---|---:|
| prior1 / prior3 | 先验前 1 / 前 3 全跑取最好 | 1.0 / 3.0 |
| SH6-3-3_prior（3.75 档） | 先验前 6 @25% → 3 @50% → 3 @100% 取最好 | 3.75 |
| **SH10-4-3_prior（5.0 档，头条 = 部署主臂）** | 先验前 10 → 4 → 3 | 5.0 |
| **hybrid16-8-3（7.5 档主臂）** | hybrid_order 前 16（= 11 条 shared+GGS + 代理 pred_z 前 5）@25% → 8 → 3 | 7.5 |
| prior6+sur10-8-3（7.5 档次臂） | 先验前 6 + 代理 pred_z 前 10（去重）→ 8 → 3 | 7.5 |
| SH16-8-3_surrogate | surrogate_order 前 16 → 8 → 3（纯代理，不用先验） | 7.5 |
| 描述性 | oracle_pool / oracle_catalog（21）/ oracle_curated（51）/ oracle_random（49）；SH100-8-3（28.5） | — |
C0（已过）：SH6-3-3_prior 与 WP3 `analysis.json` 逐位；SH10-4-3_prior 与 WP6 `s1_sweep.md` / WP8 `stage0.json` 逐位；regret 分列：对池 oracle / 对目录 oracle。

### 3.2 HPO 基线（随机性臂：N_DRAWS = 200 次固定种子抽样，报期望、中位、q25/q75，以及"抽样击败头条臂的比例" p_beat）
| 臂 | 定义 | 成本 |
|---|---|---:|
| RS-n | 不放回均匀抽 n 条各全跑取最好；期望 `expected_max_of_k`、分位与 p_beat 由次序统计量**解析**（P(max ≤ v(i)) = C(i,k)/C(n,k)） | n ∈ {1,…,8,10,12,14}；对照档 **RS-5 / RS-8** |
| rand-SH6-3-3 / 10-4-3 / 12-6-3 / 16-8-3 | 随机起点、同淘汰规则（与 `36.expected_random_sh` 同抽样序） | 3.75 / **5.0** / **6.0**（对基线有利 +1.0，明写）/ 7.5 |
| rand-SH8-4-2 | η=2 单支架 | 4.0（曲线点） |
| **Hyperband**（η=2，R=100%，r_min=25% → s_max=2） | HB-7：s=2 4@25%→2@50%→1@100%（2.0）+ s=1 3@50%→1（2.0）+ s=0 3@100%（3.0）；**不放回**抽 10 条（对 HB 有利）；终选 = 到达 100% 的 5 条里最高 | **7.0** |
| HB-5 | s=2 ∪ s=0（抽 7 条，4 条到 100%） | **5.0** |
| HB-lite / HBx2 | s=2 ∪ s=1（7 条）/ 两轮 HB-7（20 条） | 4.0 / 14.0（曲线点） |
| **TPE-n**（Optuna 类目 TPE 的离散池版） | n_startup = 2 随机起点全跑（Optuna 默认 10 > 预算，如实声明）；γ(n) = min(⌈0.1n⌉, 25)（n ≤ 10 时 good 集只 1 条 → 实为围绕现任最优的局部搜索）；19 字段各自 Laplace 类目 Parzen (count+1)/(n_active+m)；门控字段按 Optuna 条件参数语义（候选里 None 的字段不入乘积，密度只在激活观测上估）；每步在**未评估的池成员**里取 argmax l/g（并列 key 升序；对 TPE 有利）；全保真 | **TPE-4 / TPE-5 / TPE-8** |
| BOHB | **不报数**：KDE 要求最高保真 ≥ dim+1 = 20 个观测才启用模型，7.5 当量内任一保真 ≤ 5 → 按定义退化为 Hyperband。REPORT 写明 | — |
| 手工默认 | GGS（1.0）、TS2Vec 论文数字（外部锚） | 1.0 |
成本曲线：每靶 regret 对 FPE（RS 解析 11 点、rand-SH 5 点、HB 4 点、TPE 3 点、我们 1.0 / 3.0 / 3.75 / 5.0 / 7.5 五点）。

### 3.3 子池行（描述性）
同 3.2 在 (a) 随机 49 条子池、(b) 策划 51 条子池 上各报一遍——(a) 是"在真实空间从零搜索"的代理，(b) 是"只在好配置里选"的上界（HB-7 需 ≥ 10 条，子池 ≥ 12 条才报）。

### 3.4 3 种子核验（GPU，`--finals`；复用三源已有终评行）
每靶：SH10-4-3_prior 冠军、hybrid16-8-3 冠军、池 seed-0 前 3 条 → 补 seed 1/2 到 100%。复用表按 (dataset, cls_key, seed) 与 WP3 `s2_finals` /
WP8 `bo_finals` / WP6 `s2_finals`（只取 `input_mask == none`）非 error 行去重（`finals_reuse.json`）。
产出：冠军 3 种子（与 TS2Vec 论文并排）；**3 种子核过的 oracle** = 池前 3 的 3 种子均值最大者（赢家诅咒扣除后的 Q2 答案）。

## 4. 判读带（先写死；n = 6 靶，只报计数与均值差，不做显著性宣称）

regret 主口径 = 池 oracle（seed 0，@100%）− 臂（seed 0，@100%）；均值 = 6 靶等权；基线取期望。
- **E1（学习在哪）**：SH10-4-3_prior regret < E[rand-SH10-4-3]（同规则同成本 5.0）于 ≥ 4/6 靶，且六靶均值差 ≥ 0.01。3.75 档 SH6-3-3 vs rand-SH6-3-3 同法报，不入门。
- **E2（5.0 档 vs HPO）**：SH10-4-3_prior 六靶均值 regret ≤ min{RS-5 (5.0), rand-SH10-4-3 (5.0), rand-SH12-6-3 (6.0), HB-5 (5.0), TPE-5 (5.0)}。HB-7 在 5.0 档表里只作描述列。
- **E3（7.5 档 vs HPO）**：hybrid16-8-3 六靶均值 regret ≤ min{RS-8, HB-7, rand-SH16-8-3, TPE-8}，且 vs HB-7 于 ≥ 4/6 靶不差。
- 读数码：**P1-WIN** = E1 ∧ E2 ∧ E3；**P1-TIER1** = E1 ∧ E2 ∧ ¬E3（便宜档赢、贵档被 HB 追平）；**P1-LIST** = E1 ∧ ¬E2（只赢同规则随机起点）；**P1-DEAD** = ¬E1。
- Q2 不设门：主表报 SH10-4-3_prior / hybrid16-8-3 的 regret 三列（对 seed-0 池 oracle / 对 3 种子核过 oracle / 对目录 oracle）。

## 5. 预测（先验概率，写在数据之前；`configs/p1.yaml: predictions` 同文）

| # | 预测 | 先验 |
|---|---|---:|
| P1 | E1 过 | 60% |
| P2 | E2 过 | 50% |
| P3 | E3 过（HB-7 不放回抽 10 条、池 51% 策划，是强基线） | 45% |
| P4 | 随机 49 条把池 oracle 抬高到 51 条池 oracle 之上（≥ +0.01）的靶 ≤ 1/6 | 75% |
| P5 | 每靶随机 49 条 ≥ 50% 落在池后半区 | 70% |
| P6 | TPE-8 与 RS-8 六靶均值 regret 差 < 0.01（8 个观测撑不起模型） | 65% |
| P7 | HB-7 均值 regret < rand-SH10-4-3；HB-7 < SH10-4-3_prior | 70% / 40% |
| P8 | 3 种子核过的池 oracle 比 seed-0 池 oracle 低 ≥ 0.02 于 ≥ 3/6 靶（赢家诅咒） | 70% |
| P9 | UCR 84：UEA 先验 SH6-3-3 regret@290 均值比 rand-SH6-3-3 低 ≥ 0.01；与 UCR 域内 LODO 先验 SH 差 < 0.005 | 60% / 50% |
| P10 | hybrid16-8-3 在 HW 选中 GGS（0.546） | 70% |

## 6. 附带零 GPU 交付

- **(a) UCR 84 语料表**（`--ucr`，`results/ucr_rprime/rprime_cache.jsonl`）：**口径不同须注明**——切片/legacy R'（bs128、lr 1e-3、无 SWA、eval_ratio 0.3、
  legacy SVM 网格、整数据集 = 一个任务、N ≤ 1024 / T ≤ 512），截点 {60,120,290} 同一轨迹，每数据集池 = shared-10 + random-10 = 20 条，n = 84 × 1 种子。
  先验 = UEA 全 8 源（UCR 不在源池），random-10 由同一代理（全 8 源拟合）打分；域内参照 = UCR 其余 83 集 shared 行的 z 均值（LODO）。
  臂：prior1 / prior3 / SH6-3-3_prior（1050 iters = 3.62 全跑）/ SH10-4-3_prior（1350 = 4.66）/ hybrid16-8-3 / 域内 SH6 / SH10 / rand-SH6 / rand-SH10 / RS-4 /
  Hyperband（rung 60/120/290：530+530+870 = 1930 iters = 6.66）/ TPE-4 / 20 池 oracle。主表全 84；副表剔除贴顶数据集（shared-10 最大 R'@290 ≥ 0.98）。
- **(b) 消融主表**（`--ablation`，从现有 analysis.json 汇总，每行带来源路径 / @b_final / n；块 A–H' 见 `configs/p1.yaml: ablation_sources`，缺文件记 missing）。
- **(c) 成本表**（`--cost`，`cost.md`）：FPE 列 + 4090 秒列（带硬件标签）；**AutoCL**：Table 6（HAR）总时 10.516 h，"Full Pre-train f_E" 变体 85.193 h =
  500 次控制器迭代各一次完整预训练 → 单次 ≤ 0.170 h → AutoCL ≥ 10.516/0.170 ≈ **62 全跑当量**（下界）；先验一次性离线成本 945 条轨迹 ≈ 8.3 GPU-h（5090），
  跨全部未来目标摊销，单列。

## 7. 阶段、成本与排程（4090 pod，卷 tudo4ntlot；开 pod 前另行请示）

| 阶段 | 内容 | GPU | 估时 |
|---|---|---|---|
| S0 锁 | tests 先红后绿（16 项）；`--build-pool`（297）；`--simulate --c0`（PASS）；`--dry-run`；协议锁 commit | 无 | 已完成 |
| S1 随机格 | 297 格 seed 0 三截点，order 序 | 是 | 4.5 h（锚均值）/ 6.3 h（×1.4） |
| S2 模拟 | `--simulate`（基线首次出数）→ `--analyze` 出门、写 picks | 无 | 分钟级 |
| S3 终评 | 每靶 ≤ 5 配置 × seed 1/2 去重复用，≈ 30–50 格 | 是 | ≤ 1 h |
| S4 附带 | `--ucr --ablation --cost`；REPORT、memory、CLAUDE.md、commit | 无 | — |

估时锚（`configs/p1.yaml: anchor_4090` = WP3 目录 21 条单格均值 / 最大秒，含预训练 + 三截点评估）：HW 7.1 / HMD 28.1 / UWave 22.9 / Cricket 48.2 /
EC 62.7 / PS 94.7 s；最大 17.9 / 52.8 / 43.4 / 92.5 / 157.2 / 123.3 s。`--dry-run`：297 格 4.48 h，保守 ×1.4 = 6.3 h（每格 +10 s 启动）；按最大 7.6 h。
风险：随机配置 resize 极端档拉长 T（显存/时间按 T² 走）→ OOM 进 `oom_failures.jsonl`、≥2 次剔除；PS 超时最坏 10 格 × 520 s ≈ +1.4 h。
**pod 合计 ≈ 6–8 h ≈ $4.5–6**（$0.74/h）。HAR 不进主跑。

## 8. 拍板记录（用户 2026-09-07）

1. 硬件：**4090 pod**（卷 tudo4ntlot；resume mvqetmtt1t4azg，无卡则 deploy 同卷；开 pod 前再请示）。
2. 池规模：**每靶 100**（随机 49–51，共 297 格）。
3. 头条档：**5.0（SH10-4-3_prior）**，3.75 / 7.5 保留；7.5 档主臂 hybrid16-8-3，次臂 prior6+sur10-8-3、纯代理 SH16-8-3。
4. S3 终评：做（冠军 2 + 池前 3，复用三源）。
5. HAR 不进主跑。
6. BOHB 只写说明不报数；TPE 报 4 / 5 / 8；N_DRAWS = 200。
7. 分支：wp10-uea30 合入 wp9-p1（54ac004），路径一在 wp9-p1 上做，mx4 不动。

## 9. 命令
```
python scripts/46_p1_pool.py --build-pool                 # 零 GPU：pool.json（协议锁产物）
python scripts/46_p1_pool.py --dry-run                    # 格数 + 估时（4090 锚）
python scripts/46_p1_pool.py --simulate --c0              # 零 GPU：只验我们的臂与 WP3 / WP6 / WP8 逐位
python scripts/46_p1_pool.py --run --resume               # S1：297 格（pod：pa_pod/p1_run.sh）
python scripts/46_p1_pool.py --simulate                   # S2：全部臂 + 基线 → simulate.json / tables.md
python scripts/46_p1_pool.py --analyze                    # 门 E1–E3、预测 P1–P10、picks.json → analysis.json
python scripts/46_p1_pool.py --finals --resume            # S3：冠军 + 池前 3 × seed 1/2
python scripts/46_p1_pool.py --ucr --ablation --cost      # 附带零 GPU 交付
```
pod 三件套：`scripts/pa_pod/p1_bundle.sh`（本地打包）→ scp → `p1_setup.sh`（依赖 / md5 / 单测 / C0 逐位 / 冒烟 / dry-run）→ `nohup p1_run.sh`。

## 10. 不做的事
不改 35/36/39/43/45/autocls；不动 SH 淘汰规则；不按结果调池构成、N_DRAWS、HB 支架、TPE 参数、门阈值（要调另开预注册）；不用 WP8 回灌代理分；
不跑 BOHB 伪实现；不重提 task-aware / 学习曲线外推 / 开放空间搜索；不合 wp9-p1 回 mx4（用户另拍）。
