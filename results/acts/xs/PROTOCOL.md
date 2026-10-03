# WP17-ACTS XS PROTOCOL：UEA 扩源（按偏好代表性挑源）+ 源专家混合部署

**会话定位**：
- S3-MD（`results/acts/s3md/REPORT.md`，S3MD-BEATS-TS2VEC+E）：ME 0.6334 > UNI 0.6154，与 BLIND 0.6377 持平。
  - E 只做相似度时，部署层能看到 E 的贡献；
  - 但 π 很平（最大源权重 0.10–0.18），集中不到偏好近亲上。
- 用户 2026-10-01 拍板“进行扩源的任务”，方式按此前讨论：从 UEA 里挑能代表不同偏好的数据集加入源池，其余留作未见靶，不把 UEA 全部加进来。
- 假设：源池里有代表不同偏好的 UEA 源后，π 有可以集中的对象，ME 应当上升（ME_new > ME_old），并有机会超过任务盲。

产物：
- 准入语料 `results/acts/xsadm/`；补全语料 `results/acts/xs/`；部署 `results/acts/xs/md/`。
- 代码 `scripts/116_acts_xs.py`（复用 83 的切片 / 计划 / 准入 / 门，86 / 101 的 E，115 的专家拟合 / 搜索 / 选择）。
- 单测 `tests/test_acts_xs.py`；服务器流水 `scripts/pa_pod/acts_lab_run_xs.sh`。
- 配置：`configs/acts.yaml` 增 profile `xsadm` / `xs`；`configs/acts_sources.yaml` 增 6 个 UEA 源（`source: uea`，只增不改）。

## §0 已看过的读数（如实声明）

- S3-MD 全部读数。
- **零 GPU 探索**：用 WP10 / WP3 / GEO-S 的官方口径整集 TEST 种子 0 标签（28 条共同配置）做 UEA 残差偏好聚类，`scratchpad/uea_pref_clusters.py`、`uea_src_select.py`。看过以下三类读数：
  - 非靶 16 集的聚类结果。
  - **6 个部署靶各自在 29 集里残差偏好最像的数据集**。这用了靶的标签，下面的选源规则不使用它，但如实列出：
    - Cricket ↔ Handwriting / BasicMotions / SAD；
    - Ethanol ↔ HMD / Heartbeat / PEMS-SF；
    - PS ↔ Epilepsy / LSST / DuckDuckGeese；
    - SCP1 ↔ RacketSports / FingerMovements / DuckDuckGeese；
    - UWave ↔ AWR / CharacterTrajectories / RacketSports；
    - HW ↔ Cricket / SAD / Heartbeat。
  - 可行集（§1.1）在官方标签上的聚类：{AWR, ERing}、{Heartbeat, NATOPS, PEMS-SF}、{FingerMovements}。另有官方分辨率（最好 − 中位）：AWR 0.027 / ERing 0.106 / FingerMovements 0.080 / Heartbeat 0.024 / NATOPS 0.039 / PEMS-SF 0.058。
- **没看过**：6 个候选在 ACTS 切片上的任何标签（从未跑过）；扩源后的任何 LOSO 或部署读数。

## §1 选源（靶盲，预注册）

### 1.1 可行集（规则先于数据）

从 UEA 29 集（InsectWingbeat 按用户 2026-09-06 决定不跑）里依次排除：

| 排除条件 | 排除的数据集 |
|---|---|
| 6 个部署靶 | 6 个靶本身 |
| S2 G-P 已剔除 | HandMovementDirection、SelfRegulationSCP2 |
| `source_pool.yaml` 的 target_pool 成员（铁律 6 结构护栏：加载即拒） | Epilepsy、FaceDetection |
| L < 50（encoder 适用边界） | JapaneseVowels、LSST、Libras、PenDigits、RacketSports |
| L > 1751（现源池最长序列 = Ethanol） | EigenWorms、MotorImagery、StandWalkJump |
| N < 200（切片样本数轴需 ≥ 4 档 × n_lo 80） | AtrialFibrillation、BasicMotions、DuckDuckGeese |
| 变长（切片器按等长实现） | CharacterTrajectories、SpokenArabicDigits |

→ 可行集 6：**ArticularyWordRecognition、ERing、FingerMovements、Heartbeat、NATOPS、PEMS-SF**。

### 1.2 阶段 A 准入（与 UCR-ADM 同一把尺子）

- 每源用 S1″ 切片器切 8 行（OA 前 8 行，前缀稳定），准入用行 1 / 7 / 3 / 5。
- 每行 shared-30 × 种子 {0, 1}，加 iters = 0 探针；`ts2vec_official` 口径，cv5 终点。
- 准入 = v3′ ∧ 护栏：
  - v3′：任务级全 30 条 Spearman(seed 0, seed 1) ≥ 0.5 的行 ≥ 3/4；
  - 护栏：前一半两种子均值中位 − 多数类 ≥ 0.10 的行 ≥ 3/4。
- 两种子齐的 shared 不足 20 条的行不计。

### 1.3 阶段 B 选代表

- 准入数 ≤ 4：全选。
- 准入数 > 4：按偏好残差聚类，每类取一个代表。
  - 画像 = 该源 4 个准入行的 shared-30 两种子均值钉底 z（同 `clip_quantile`）的均值；
  - 残差 = 画像 − **S2 训练池 11 个非靶源**同 4 行画像的均值（不用 6 个部署靶的任何标签）；
  - 距离 = 1 − Spearman；平均连接；`maxclust` 4；
  - 每类取类内平均相关最高者；单源类取其本身；并列按名字序。

### 1.4 阶段 B 补全与门

- 选中源补行 0 / 2 / 4 / 6 的 shared-30 × 2 种子，并给 8 行各补 random-20 × 种子 0（= S2 正向源口径，每任务 80 格）。
- 补齐后过 8 行 G-P 门（= S2 试用源门）：ρ_seed ≥ 0.5 的行 ≥ 5/8 ∧ 护栏行 ≥ 5/8。不过即剔除，不补位。
- 未选中的准入源与可行集其余数据集留作未见靶，本轮不部署。

## §2 并入与部署

- **新任务与旧 160 任务同口径：**
  - L4 标签 = shared 两种子终点均值 ⊕ random 种子 0 终点；
  - 失败 = 不高于探针地板；
  - vec = binned 编码；
  - E_pt = 每任务单变量 TS2Vec 窗口表示均值（86 同参数、同窗口种子）；
  - E_A = `response_profile`（101 同口径：物化切片、按通道 z-score）。
- **主干固定为 S3-Z NEW_B 的 10 个成员（不重训）：**
  - BLIND 臂 = S3-Z NEW_B（0.6377），不变；新源只增加源专家。
  - 不对称（声明）：新源的 s₀ 是样本外的，旧源的 s₀ 是样本内的。
- **源专家、超参与 π：** 与 S3-MD 相同（`115 --fit` 原样）。训练源 = 16 + m 个（每折去掉靶自己），内层留一源选 (α_w, τ, γ)。
- **臂：**
  - 新跑：ME_new（E 相似度加权）、UNI_new（均匀权重）；
  - 复用：BLIND（S3-Z NEW_B）；
  - 参照：ME_old 0.6334、UNI_old 0.6154（S3-MD）。
- **部署与 S3-MD 逐项相同：**
  - 11 个搜索者，同一批 Ks 与进化种子，取并集 U；
  - 目标官方 TRAIN 分层 3 折 CV 选 1；
  - 完整 TRAIN 3 种子 TEST；
  - 已跑格（S3-MD / S3-Z）复用。

## §3 门与读数码

- **阶段 A：** XS-ADM-n（n = 准入数）。n = 0 → **XS-NOSRC**，停在阶段 A。若阶段 B 门后保留 0 源，同样记 XS-NOSRC。
- **主门（ME_new，6 集 3 种子 TEST 均值）：**
  - > InfoTS_s 0.6468 且 > TS2Vec 0.6243 → **XS-BEATS-BOTH**；
  - 只 > TS2Vec → **XS-BEATS-TS2VEC**；
  - 否则 → **XS-BELOW**。
- **后缀（按 AutoCTS++ 方式对比，不要求显著）：**
  - **+SRC**：ME_new > ME_old 0.6334（扩源的作用）；
  - **+BLIND**：ME_new > BLIND 0.6377；
  - **+E**：ME_new > UNI_new。
- **描述项：**
  - 逐集表（并列 ME_old / UNI_old / BLIND / TS2Vec / InfoTS_s）；
  - π 在新源上的总份额、最大源权重、每靶 π 前 3；
  - 选中的 (α_w, τ, γ)；
  - U 最好 / 平均（种子 0）、CV 选择损失；
  - 新源与旧源的残差幅度。
- 差 < 0.01 视为持平；6 个数据集、3 种子，不做显著性声明。

## §4 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | 阶段 A 准入 ≥ 3/6 | 50% |
| P2 | ME_new > ME_old 0.6334 | 50% |
| P3 | ME_new > BLIND 0.6377 | 35% |
| P4 | ME_new > UNI_new | 60% |
| P5 | ≥ 3 / 6 个靶的 π 前 3 含新源 | 55% |
| P6 | ME_new > InfoTS_s 0.6468 | 15% |

## §5 成本、执行与铁律

- **规模：**

  | 阶段 | 格数 |
  |---|---|
  | A | 6 源 × 4 行 × 60 = 1,440 格，加 6 × 8 × 3 个探针 |
  | B | ≤ 4 源 × 400 = 1,600 格 |
  | E_pt | ≤ 32 个任务 |
  | 部署 | 约 1,100 格（与 S3-MD 重合者复用） |

- **服务器：** 实验室 8k-4090（他人占卡），4 路并行（用户 2026-10-01 指定）。执行器沿用 S3-MD 的接管式补跑：每轮清空 `oom_failures.jsonl`，按任务分到空闲显存最多的卡，同时在跑的进程 ≤ 4。预计 1.5–2 天。
- **铁律：**
  - 1：新源走注册表，不硬编码取值；
  - 2；3（缓存 + resume + 复用）；5（先红、§0）；
  - 6：选源规则靶盲；专家 / E 标准化 / 超参只在训练源上拟合；靶的 E 只来自官方 TRAIN；TEST 只进读数；
  - 7；8。

## §6 执行台账（跑后填写）
- 2026-10-01：协议锁 4e70c0f；实现 4323832（tests 6 例绿；本地冒烟：PEMS-SF C=875 一格 4060 实测 44 s，估时模型按 T×C 线性放大高估约 8×；select 在 UCR-ADM 真实准入结果上选 4 个代表；ERing 补全 1 个任务 80 格后 gate / labels / E_pt / E_A / score-new / merge（161 任务）/ fit / search 全通）。
- 20:33 UTC 起跑（实验室 8k-4090，他人占卡，MAXP = 4）：服务器 plan_M 1,440 格，键集与本地逐位相同（md5 c8a9eeab）；首轮 4 卡（3 / 6 / 0 / 7）约 5.4 格 / 分钟，0 OOM。
- 2026-10-02 01:20 UTC：阶段 A 完成（1,440 / 1,440，0 error，0 OOM；墙钟 4.8 h）。**XS-ADM-4**：准入 AWR（v3′ 4/4、护栏 4/4）、ERing（3/4、4/4）、NATOPS（3/4、4/4）、PEMS-SF（4/4、4/4）；拒 FingerMovements（0/4、0/4，h_t 0.02–0.07）、Heartbeat（0/4、0/4，h_t ≈ 0）= chance 型。准入数 ≤ 4 → 4 个全选（§1.3）；四源残差 Spearman AWR–ERing 0.52，其余 −0.10 ~ 0.29。
- 01:20 起阶段 B：plan_S 2,560 格（复用阶段 A 960、新 1,600），32 任务；02:19 完成 554 / 1,600，0 OOM。
- 06:45 UTC：阶段 B 完成（2,560 / 2,560，0 error，0 OOM）；8 行 G-P 门 4 源全保留（v3′ 8 / 7 / 6 / 8 of 8，护栏 8/8）；labels 32 任务（失败占比 0.455）/ E_pt 32 / E_A 32 / score-new / merge（192 任务）/ fit（γ 6/6 折 0.75、τ 0.5、α_w 1）。
- 06:52 搜索 12 → CV 1,044 格（新 1,004）→ 12:41 select → 最终 1,088 格 → 13:01 打包；墙钟合计 16.5 h，0 error，0 OOM。
- 读数（REPORT.md）：**XS-BEATS-TS2VEC+E**（无 +SRC / +BLIND）：ME_new 0.6302 / UNI_new 0.6189 / ME_old 0.6334 / BLIND 0.6377；预测 2/6。事后描述项 `partners.json`（切片口径近亲 vs π）。
- 本机后台监视（等待准入结束）2026-10-01 因本机内存紧张被 Claude Code 回收，服务器流水不受影响；未重启，改为按用户要求手动查询。
