# MX-4 WP6 协议：第 20 字段 `input_mask`（encoder 侧二项时间步掩码）进 CLS 空间 + 交互感知代理

日期 2026-09-04 ｜ 分支 `wp6-mask`（自 mx4 dc27cc4，独立工作树，MASK-DEAD 则代码不合并）｜ 脚本 32（`--twins`）/ 39（`--twins`）/ 40 ｜
数据（只读）：`results/mx3_pq/`、`results/mx4_deploy/`、`results/protocol_alignment/runs.jsonl` ｜ 产物 `results/mx4_mask/`

## 0. 诚实声明（协议锁前已看的读数）
- protocol_alignment 3 种子阶梯（官方口径）：`+input_mask=binomial` 对 TS2VEC_EQUIV HW +0.082 / UWave +0.114 / HMD +0.099，
  PS −0.012 / Ethanol −0.008 / Cricket +0.005；重增强 zs pick UWave 8b87d66f −0.125、HW 三条 −0.014…−0.024；GGS HW −0.013 / UWave +0.032。
- **S0（本协议前算出，`s0_alignment.{json,md}`）**：targets / targets_ladder 两阶段的 36 个干净 (靶, 配置) 对，Δ 均值 +0.005、
  正 17/36、Spearman(增强总强度, Δ) −0.25。效应同时依配置（干净 3c231edc +0.077/+0.114/+0.099 vs 重掩码 8b87d66f −0.125）与
  依数据集（同一 3c231edc 在 PS/Ethanol −0.012/−0.008；同一 9092e36e 在 HMD +0.050、Ethanol +0.029、HW −0.024）。
  → 任务盲代理最多学到"配置 × 掩码"这一半，"数据集 × 掩码"那一半只能靠目标侧测量（K4 的测量臂）。
- WP3 DEPLOY-DEAD：官方口径 SH6-3-3 追平 4/6；HW 0.431 是名单失败（GGS 0.546 被先验排第 8，事后 SH10-4-3 选中）；Ethanol 0.241 噪声带。
- 切片侧头顶空间：SH6-3-3 0.0184，其中名单不含 oracle 的不可约 0.0118；GBDT@1 0.056 输先验 0.046；pair acc 0.780；
  19 字段空间好半区平坦无交互——本 WP 就是给学习组件造一个先验表达不了的交互。
- **C0 回放门已过**（`c0.json`）：加扩展字段后 legacy-only 复算与 WP1（126 实例 prior1 / SH6-3-3_prior）、WP2（7 折 hgb_binned pair acc）、
  WP3（6 靶 SH6-3-3_prior 臂值与 picks）逐位相同；金标单测（采样 RNG 流前 5 键 / random CLS 前缀 / 探针目录哈希 / dim 32）全绿。

## 1. 问题 → 门
| 问 | 内容 | 门 |
|---|---|---|
| Q1 | 加第 20 字段后旧路径是否逐位不变 | C0（已过） |
| Q2 | 切片协议下孪生效应的符号是否与官方口径一致（源侧标签能否承载这个字段） | K1 |
| Q3 | 字段代理能否学到"双重掩码"交互（先验表达不了的东西） | K2 |
| Q4 | 部署绝对精度是否上升（W1 线：≥ TS2Vec − 0.03） | K3 |
| Q5 | 学习臂能否以一半探测预算追平"两版都量"的测量臂 | K4 |

## 2. 口径
- **孪生**：`twin(cls, input_mask="binomial")`；`cls_key` 含扩展字段（旧键逐字节不变）；权重种子随新键派生（`canonical_seeds`），
  故 Δ 含轨迹地板噪声（δ_nd≈0.03），不是共享种子配对。
- **源侧（切片协议 @{60,120,290}）**：孪生行写 `results/mx4_mask/source_twins/rprime_cache.jsonl`，绝不进 `results/mx3_pq`；
  孪生 z 用同 (task_id, iters) 的 legacy 20 条 mean/sd 换算（legacy z 逐位不变）。补标档位（用户拍板）：shared-10 孪生 45 切片、
  GGS 孪生 45 切片、random-10 孪生每数据集前 2 切片 = 655 条轨迹 / 1965 行；S1b = 其严格前缀（HW / UWave / HMD × 切片 00/01 ×
  shared-10 + GGS = 66 条）。δ 沿用 `configs/labels.yaml` 的 (数据集, 预算) 表。
- **源侧臂（7 折 LODO，`40 --source`）**：prior19（= WP1 先验）、prior20（shared ∪ 其孪生的 z 均值）、gbdt19（hgb on binned 19 字段）、
  gbdt20（hgb on binned 20 字段 ⊕ b/290，训练含孪生行）。实例候选 = legacy 20 ∪ 该 (切片, 预算) 已有孪生（切片 00/01 为 40，02–05 为 30；
  GGS 及其孪生不入候选，与 36 号同）。指标：配对 acc（δ 表）、regret@1/@3、SH6-3-3@290、孪生方向准确率（δ 存活孪生对上
  sign(score_t − score_k) 对 sign(R′_t − R′_k)，对照 always-on / always-off）、分组置换重要性（含 input_mask）、交互读数
  （每 legacy 配置的预测 Δz vs 真 Δz，按 point_mask / emb_mask 档位分层）。
- **部署（官方口径 ts2vec_official，`39 --twins`，`40 --deploy`）**：掩码关半边 = WP3 `runs.jsonl`（146 行，只读）；掩码开半边 =
  WP3 目录 21 条各配一格孪生（PS 的 20be4f74 eval 超时格跳过）= 125 格，截点 {25,50,100}%，seed 0；每靶 42（PS 40）目录。
  排名：`prior_off`（= WP3 先验序，C0）、`interleave`（prior_off 每条后紧跟其孪生）、`allon`（只孪生）、`prior20` / `gbdt20`（S2 后按折拟合；
  prior20 = shared 及孪生用源池 z 均值 +偏置置顶、GGS 沿 WP3 先验、提名及其孪生按 gbdt20）。臂：每种排名的 top1 / top3 / SH6-3-3 / SH10-4-3，
  `SH12-3-3_measured`（prior_off 前 6 各带孪生 → 12@25% → 3@50% → 3@100%），oracle42 / oracle_off / oracle_on。
  主臂 `SH6-3-3_gbdt20`（S2 前临时用 `SH6-3-3_interleave`）；基线 `SH6-3-3_prior_off`（= WP3 SH6-3-3_prior，C0）。
  3 种子 = seed 0 + finals seeds {1,2}（WP3 已有的 s2 行复用）；成本 iters 逐级累加 + `checkpoint_results` 秒增量。
- **硬件（用户拍板）**：S1 在 4090 pod（与 WP3 legacy 部署行同机型）；S1b + S2 在 5090 pod（与 mx3_pq legacy 标签行同机型）。
  每个 pod 开启前先请示用户；跑完 `pod_ctl.py stop` + `confirm-stopped 3/3`。

## 3. 阶段与工单（dry-run 计数已核）
| 阶段 | 内容 | 计数 / 成本 |
|---|---|---|
| A（已完成） | 扩展字段核心层 + 32/39/40 + 单测 + C0 | 零 GPU；commit 52fdd1b / 0afeb8f + 本协议锁 |
| S0（已完成） | PA 既有孪生 Δ 表 | 零 GPU；36 对 |
| S1 | `wp6_run.sh`：39 孪生目录 125 格 → `40 --deploy`（第一批 picks）→ finals → `40 --deploy` | 125 格 ≈ 1.13 h（4090 锚）+ finals ≤ 6 靶 × ≤ 6 臂 × 2 种子 |
| S1b | `twins_run.sh` 档 0–1：66 条轨迹 | ≈ 66 × 49 s ≈ 55 min（5090） |
| K1 | 拉回后本地 `40 --k1` | 零 GPU；pod 空转等拍板 |
| S2 | `twins_run.sh CONTINUE=1`：其余 589 条 | ≈ 589 × 49 s ≈ 8 h 上界（5090） |
| S3 | `40 --source`、`40 --deploy`；prior20 / gbdt20 冠军为新配置时第二批 finals（重开 4090） | 零 GPU（+ ≤ 0.5 h） |

## 4. 判读（先写死）
- **C0**：WP1 126 实例 prior1 / SH6-3-3_prior、WP2 7 折 hgb_binned pair acc、WP3 6 靶 SH6-3-3_prior 臂值与 picks 逐位相同（已过）。
- **K1**：官方 Δ_official（孪生 − legacy，seed 0，100% 截点）与切片 Δ_slice（同配置该数据集 S1b 全部孪生格均值，2 切片 × 3 预算）
  在 |Δ_official| ≥ 0.02 的 (数据集, 配置) 对上 sign 一致率 ≥ 0.7 且 n ≥ 15（候选对 = HW / UWave / HMD × (shared-10 + GGS) = 33）。
  n < 15 → 按 0.01 地板复算并**标注为决策点**，不自动放行。不过 → S2 取消，只报测量臂，读数码加后缀 `-NOLEARN`。
- **K2**：gbdt20 孪生方向 acc（留出数据集，δ 存活对合并）≥ 0.70 且 ≥ max(always-on, always-off) + 0.10；且 gbdt20 regret@1 < prior20 regret@1
  （配对 Wilcoxon 单侧 p < 0.05，n = 126 混合候选数主口径；切片 00/01 的 40 候选子集 n = 42 另报）。
- **K3**：`SH6-3-3_gbdt20` 冠军 3 种子 ≥ TS2Vec − 0.03 于 ≥ 5/6 靶（HW 单列报；`SH10-4-3_*` 作次要臂）。
- **K4**：`SH6-3-3_gbdt20` 冠军与 `SH12-3-3_measured` 冠军差 ≤ 0.01 于 ≥ 5/6 且第一档成本比 ≤ 0.75（6/12 = 0.5）。
- **读数码**：MASK-WIN = K2 ∧ K3；MASK-SPACE = K3 ∧ ¬K2（字段有用但只靠测量）；MASK-DEAD = ¬K3；K1 不过加 `-NOLEARN`。
- n = 6 靶只报计数与配对差，不做显著性宣称（与 WP3 同）。

## 5. 预测（先验）
| # | 预测 | 先验 |
|---|---|---|
| P1 | Ethanol 在某个孪生臂（interleave / measured / gbdt20）3 种子追平（≥ 0.278） | 55% |
| P2 | HW 只在 SH10-4-3 臂过线（GGS 0.546 的名单问题不是掩码能解的） | 60% |
| P3 | UWave 冠军仍是掩码关配置（重掩码 pick 开掩码 −0.125） | 85% |
| P4 | `input_mask` 分组置换重要性进前 4（温度 / sim_func / norm 之后） | 60% |
| P5 | K1 一致率 ≥ 0.7（切片协议偏爱重增强，方向可能翻） | 50% |
| P6 | K2 第一子句（gbdt20 方向 acc ≥ 0.70）成立——S0 已显示数据集依赖占一半 | 40% |
| P7 | `SH12-3-3_measured` 追平 ≥ 5/6（含 Ethanol 或 HW 之一） | 50% |

## 6. 已知坑
- 孪生权重种子 ≠ legacy：Δ 含轨迹地板噪声；binomial 掩码用全局 torch RNG，同 seed 下与 legacy 的批次流也分叉。
- 切片 δ（HW 0.16）远大于官方孪生 Δ（≈0.08）→ K1 用官方侧 0.02 地板而非切片 δ 过滤；存活 n 仍可能小（决策点）。
- PS 目录 20（20be4f74 eval 超时）；HMD n_test 74（单种子 ±0.05）；Ethanol 噪声带 0.06 宽，单种子选人是掷骰子。
- 跨机：S1 4090 vs WP3 4090 同型号；S1b/S2 5090 vs mx3_pq 5090 同型号；5090 run-to-run ±1 样本。
- protocol_alignment 的 binomial 行用 legacy 键（S0 按 (dataset, 行内 cls_key, seed) 配对，不用新 cls_key 去连）。
- 切片协议偏爱重增强、官方口径偏爱轻增强（WP3 §4.2）是 K1 的真实风险；K1 只测符号不测量级。
- 源侧 gbdt20 在 S1 之前不存在：第一批 finals 只覆盖 interleave / measured / allon / SH10 的冠军。

## 7. 拍板记录
- 2026-09-04 用户：方向 1 立项；补标档位 = shared-10 孪生全切片 + random-10 孪生每数据集 2 切片（+ GGS 孪生）。
- 2026-09-04 用户：硬件 = S1 4090 / S1b+S2 5090；每阶段出报告；开 pod 前请示。
- 2026-09-04 用户：实验失败不得影响原代码 → 分支 `wp6-mask` 隔离、C0 回放门、MASK-DEAD 不合并。
- `results/mx4_mask/candidates.json` = WP3 逐字节副本，md5 `7f11da18a4f035de8cbf04404aa532cf`。
- 协议锁 commit：见 git log（本文件随该 commit 生效；开跑后判读线不改）。
