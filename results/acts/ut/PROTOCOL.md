# WP17-ACTS UT PROTOCOL：方法固定后在未见 UEA 数据集上部署评估（ME / UNI / BLIND 三臂 + 手工模型）

**会话定位**：
- 之前 6 个部署靶上，ME 系列三次（S3-MD / XS / XE）都在 0.626–0.633，BLIND 0.638；ME > UNI 三次同向（+0.018 / +0.011 / +0.007）。
- 这 6 个靶的 TEST 已用九次，0.01 量级的差别读不出来。
- 用户 2026-10-04 拍板：方法固定，换一批没用过的 UEA 数据集做部署评估，按 AutoCTS++ 方式比较消融臂，并与 TS2Vec、InfoTS_s 对比。

产物 `results/acts/ut/`（部署 `results/acts/ut/md/`）；代码 `scripts/118_acts_ut.py`（复用 90 的整集 E_pt、115 的专家 / 搜索、111 的计划 / 选择、116 的合并数据）；单测 `tests/test_acts_ut.py`；服务器流水 `scripts/pa_pod/acts_lab_run_ut.sh`。

## §0 已看过的读数（如实声明）

- S3-MD / XS / XE 全部读数（只涉及 6 个旧靶）。
- **未见靶的历史：**
  - WP10（2026-09-06/07）在这 11 个靶中的大部分上，用旧方法（先验名单 + SH，shared-10 + GGS）跑过官方口径部署，看过它们的 TEST 读数；
  - HMD / SCP2 在 S2 作为试用源切片标注过，FingerMovements / Heartbeat 在 XS 准入时切片标注过；四者都被门剔除、**不在训练池里**，切片标签本协议不使用。
  - 本方法（T-CLSC + 源专家 + 全空间搜索）在这 11 个靶上**从未跑过**。
- **基线数字**：InfoTS 原文（arXiv 2303.11911）附录 Table 9 的 InfoTS_s / TS2Vec 列，与 `108_acts_s3v.py` 已收录的 6 个旧靶逐值一致；TS2Vec 另与 `configs/uea30.yaml`（TS2Vec 原文 Table 10）对账。

## §1 方法（全部固定，不再调）

- **源池**：XS 的 21 源、192 任务（17 旧源 + AWR / ERing / NATOPS / PEMS-SF）。
- **主干**：S3-Z NEW_B 的已训模型，不重训。
  - 未见靶不属于任何折，10 个成员按固定规则取：成员 q（q = 0…9）= 折 F[q mod 6]、种子 q 的模型。F = (HW, PS, UWave, Cricket, Ethanol, SCP1)。
  - 各成员的训练集都不含任何未见靶；成员间折不同，构成多样性。
- **源专家**：每成员对 21 个训练源分别做残差岭回归；(α_w, τ, γ) 在 21 源上内层留一源选择。都与 `115 --fit` 原样相同。
- **E**：E_pt ⊕ E_A（S3-MD / XS 的现用形式）。
  - 靶 E_pt = 官方 TRAIN 上单变量 TS2Vec 窗口表示均值（90 同法）；
  - 靶 E_A = 官方 TRAIN 上的 `response_profile`；
  - TEST 不读。
- **部署**：
  - 11 个搜索者（10 成员 + 集成）、Ks 20,000 + 进化 10 代，每个搜索者取前 2，合并去重得 U；
  - 目标官方 TRAIN 分层 3 折 CV 选 1；完整 TRAIN 3 种子 TEST。
  - 与 S3-Y / Z / MD / XS 逐项相同。
- **数据加载**：
  - 部署靶走 `acts_common.load_target_full`（= `load_uea_merged`，与源加载器读出的数组逐位相同，只是绕开“目标池不得当源”的护栏）；
  - 83 的整集 loader 只在 `tasks_meta.deploy_target = true` 时用它，旧任务行为不变；
  - 靶只用于部署评估，不进任何训练（铁律 6）。

## §2 未见靶（规则先于数据）

- UEA 29 集（InsectWingbeat 按用户 2026-09-06 决定不跑）中，去掉：
  - 21 源池里的 10 个 UEA 源：HW / PS / UWave / Cricket / Ethanol / SCP1 / AWR / ERing / NATOPS / PEMS-SF；
  - 变长序列（切片与 E 按等长实现）：CharacterTrajectories / SpokenArabicDigits / JapaneseVowels；
  - L < 50（源池全部在 L ≥ 50，超出源的覆盖范围）：LSST / Libras / PenDigits / RacketSports；
  - L > 3000（官方 `max_train_length` 分段，ACTS runner 未实现）：EigenWorms。
- 得 **11 个未见靶**：AtrialFibrillation、BasicMotions、DuckDuckGeese、Epilepsy、FaceDetection、FingerMovements、HandMovementDirection、Heartbeat、MotorImagery、SelfRegulationSCP2、StandWalkJump。

| 未见靶 | TS2Vec | InfoTS_s |
|---|---:|---:|
| AtrialFibrillation | 0.200 | 0.267 |
| BasicMotions | 0.975 | 1.000 |
| DuckDuckGeese | 0.680 | 0.600 |
| Epilepsy | 0.964 | 0.993 |
| FaceDetection | 0.501 | 0.525 |
| FingerMovements | 0.480 | 0.620 |
| HandMovementDirection | 0.338 | 0.514 |
| Heartbeat | 0.683 | 0.771 |
| MotorImagery | 0.510 | 0.610 |
| SelfRegulationSCP2 | 0.578 | 0.572 |
| StandWalkJump | 0.467 | 0.467 |
| **均值** | **0.5796** | **0.6308** |

## §3 臂（AutoCTS++ 式消融）

| 臂 | 含义 |
|---|---|
| **ME（主臂）** | E 相似度加权的源专家混合 |
| UNI | 均匀权重源专家（去掉 E） |
| BLIND | γ = 0（去掉源专家，= 任务盲 T-CLSC 搜索） |

三臂共用同一批成员、Ks 与进化种子；已跑格跨臂复用。

## §4 门与读数码（11 个未见靶，3 种子 TEST 均值）

- **主门（ME）：**
  - 均值 > InfoTS_s 0.6308 且 > TS2Vec 0.5796 → **UT-BEATS-BOTH**；
  - 只 > TS2Vec → **UT-BEATS-TS2VEC**；
  - 否则 → **UT-BELOW**。
- **后缀（不要求显著）：**
  - **+E**：ME > UNI；
  - **+EXP**：UNI > BLIND；
  - **+TA**：ME > BLIND，即任务感知整体的作用。
- **描述项：**
  - 逐集表、逐集胜负数（ME 对 TS2Vec / InfoTS_s / UNI / BLIND）；
  - 6 旧靶 + 11 未见靶合并的 17 集均值（旧靶用 XS 的读数，标注 TEST 已多次使用）；
  - U 最好 / 平均、CV 选择损失、选中的 (α_w, τ, γ)、π 前 3。
- 差 < 0.01 视为持平。AtrialFibrillation（TEST 15）/ StandWalkJump（TEST 15）单样本 = 0.067，逐集读数以这个分辨率看。

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | ME > TS2Vec 均值 0.5796 | 55% |
| P2 | ME > InfoTS_s 均值 0.6308 | 25% |
| P3 | ME > UNI | 60% |
| P4 | ME > BLIND | 40% |
| P5 | ME 逐集胜 TS2Vec ≥ 6/11 | 50% |
| P6 | UNI > BLIND | 40% |

## §6 成本、执行与铁律

- **成本：**
  - 靶 E_pt 11 个（GPU）、E_A 11 个（CPU）；
  - 搜索 33 个（3 臂 × 11）；
  - CV 约 11 × 3 × 22 × 4 ≈ 2,900 格上限（跨臂重合者复用），最终约 130 格。
  - FaceDetection（TRAIN 5,890）/ MotorImagery（L 3,000 × 64）/ DuckDuckGeese（C 1,345）单格较慢。
- **服务器**：实验室 8k-4090，四路并行，接管式 fill。预计 15–25 h。
- **铁律：**
  - 2；3（缓存 + resume + 跨臂复用）；5（先红、§0）；
  - 6：未见靶不进任何训练、选择与超参；专家 / E 标准化 / 超参只在 21 源上；靶 E 只来自官方 TRAIN；TEST 只进读数；
  - 7；8。

## §7 执行台账（跑后填写）
- 2026-10-04 实现：`scripts/118_acts_ut.py` + `acts_common.load_target_full` + 83 整集 loader 的 `deploy_target` 分支 + 39 的评估超时环境变量覆盖（默认 480 s 不变，UT 流水设 1800 s）；tests 8 例绿。`--prep`：§2 规则复算 = 协议 11 靶；TS2Vec 与 `configs/uea30.yaml` 对账 10/11 一致。
- 本地冒烟（BasicMotions、2 个随机初始化成员、ks 300 / 1 代）：E_pt / E_A / s0 / fit / 三臂搜索 / plan cv 44 格 / select / final 54 格 / analyze 全通；本机 Windows 有 3 格 EvalTimeout（loky），analyze 改为对 error 行记 NaN 并报缺格数。
- 2026-10-04 02:14 UTC 起跑（实验室 8k-4090，MAXP = 4，HEAD 4496d7b，评估超时 1800 s）：输入核对 / pytest 20 例通过；靶 E_pt 11 / 11 第一轮完成（单靶 15–80 s，无切段）。
- 10-04 02:31 起 CV 第 1 轮（2,872 格）。**负载不均**：fill 按（未完成数降序, 任务名）轮流分配，44 个任务未完成数相同 → 按名字序，j mod 4 = 0 的全是 `*_whole_official`（每靶最重的完整 TRAIN 任务）落到同一进程（GPU 7）；其余 3 个进程 ~20 h 跑完后空等。23:12 UTC 时该进程还剩 137 格（FaceDetection 官方每格约 6.5 min）。
- **操作偏离**：23:13 只停该进程（PID 2918539，丢 1 个在跑格、无行写入），fill 自动进入第 2 轮，剩余 136 格 / 8 任务重新分到 4 卡。结果不变，只改执行顺序。
- **StandWalkJump tcv0 全部 error**：该折训练只有 7 个样本，< batch 8，`cl_trainer` 在 drop_last 下按设计报错（ValueError 写 error 行）；tcv1 / tcv2 正常 → 该靶 CV 只用 2 折。官方 TRAIN 12 ≥ 8，最终 TEST 不受影响。另有 MotorImagery tcv0 2 格 OOM，在第 2 轮补跑。
- 10-05 02:54 UTC：CV verify 2,872 / 2,872 OK（error 65 = SWJ tcv0）→ select → 最终 124 格 → 05:18 打包。
- 读数（REPORT.md）：**UT-BEATS-TS2VEC+E**（无 +EXP / +TA）：ME 0.5826 / UNI 0.5559 / BLIND 0.5914 / TS2Vec 0.5796 / InfoTS_s 0.6308；预测 2/6。
