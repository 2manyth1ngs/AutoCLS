# 源池 v2 定稿协议（晋级判据/靶池判据先于补格数据写死；MX-2v2 协议锁前置）

日期：2026-08-31 ｜ 脚本：`scripts/27_source_pool_v2.py` ｜ 单测：
`tests/test_source_pool_v2.py` ｜ 量化依据：2026-08-31 零 GPU δ 存活率筛表
（memory `source-pool-screen-20260831`；本协议 `--screen` 重算落盘对账）。
锚子模拟（`results/anchor_sim/`）已收官，本协议为管线 v2 待办②。

## §1 偷看披露（先于判据声明）

已见读数（全部已揭晓）：

- **n=2 初筛存活率**（stage1 `_00/_01`×10 @290）：SAD 58 / NATOPS 58 /
  Ethanol 54 / HMD 52 / Cricket 51 / ERing 40 / SCP2 36 / SCP1 34 /
  FingerMovements 34 / MotorImagery 31 / AWR 23 / Heartbeat 0（%）。
- **8 任务网格存活率**（宽面板）：HW 72 / UM 68 / PS 55 / UWave 45 / CT 19。
- HW 工作点口径净空（oracle−GGS）0.004、区分度（oracle−中位）0.168/0.124；
  UM 顶部 65% δ 带内送分；stage1 Ethanol max R' ≈0.97；AWR 顶 1.000 /
  CT 顶 0.995 贴顶；Heartbeat 全距 0.04 死平。
- mx2 的 GGS 格是**工作点口径**——**GGS@290 全部未测**，靶池净空判定在
  新数据上做。晋级候选的任务 `_02+` 网格全部未测，确认判定在新数据上做。

**n=2 初筛数字禁止作为晋级依据（写死）**：n 小假阳教训（workpoint n=20
假阳率 2/3），初筛只定"谁进确认轮"，晋级只看确认轮数据。

## §2 固定项（用户拍板，非本协议判定对象）

- **核心源 4**：Handwriting / UniMiB-SHAR / PhonemeSpectra /
  UWaveGestureLibrary（8 任务 × shared 10 @290 网格现成，免确认）。
- **剔除 3**：CharacterTrajectories（19%，顶 0.995 贴顶——从宽面板 primary
  直接掉档）/ ArticularyWordRecognition（23%，顶 1.000）/ Heartbeat（0%，
  死平）。终判时从源池注册表移出（CT/AWR → 退役注释；Heartbeat 出 reserve）。
- **确认轮名单 6**：晋级候选 SpokenArabicDigits、NATOPS、
  EthanolConcentration、HandMovementDirection、Cricket（5 个，n=6 口径）
  + EEG 低权代表候选 SelfRegulationSCP2（n=3 口径，见 §3 例外）。
- 勿复活名单不变（任务级条件化/毒哨/E 主链/细粒度 task-aware）。

## §3 确认口径（先于数据写死）

- **网格**：每晋级候选取注册表确定性切片序前 **6 任务**（`_00.._05`）×
  shared 10 CLS @290 k=1 canonical（种子/切分与 stage1 逐字段同口径；
  `_00/_01`×10 复用 stage1 缓存，classes 校验同 scripts/24）。基线 =
  iters=0 × 5 种子（确认任务缺则补，秒级）。
- **SCP2 例外（写死）**：注册表 planned_tasks=3（唯一真源不动）→ n=3
  口径（3 任务 × 10，135 配对），只竞争低权席位，不竞争晋级席位。
- **存活率** = 数据集内全部确认任务的 C(10,2)=45 配对池化比例，配对存活
  ⇔ |ΔR'| > δ_ds（`configs/labels.yaml: delta_per_dataset` 唯一真源：
  SAD 0.03 / NATOPS 0.06 / Ethanol 0.05 / HMD 0.06 / Cricket 0.08 /
  SCP2 0.05；与初筛同表，确认轮**不重定 δ**——防移动球门。UM 不在该表
  （非路线 A 池），对账表沿用初筛口径 δ=0.05（δ=0.05 复现已见 68%，
  已实测钉死）；UM 是核心源不入确认判）。
- **GGS@290 格**（靶池净空用）：GGS = `scripts/m1_ggs_har.GGS`（唯一真源），
  核心 4 × 8 任务 + SAD/NATOPS/Cricket/Ethanol × 8 任务 + HMD × 6 +
  SCP2 × 3 = **73 格**，k=1 canonical（weight_seed=run_seed_for(tid,
  ggs_key)）。候选的 `_06/_07` GGS 格无网格配对、本轮不入判读，仅为靶池
  路径预置（MX-2v2 补网格后启用）。

## §4 判读带（先于数据写死）

**晋级判（对 5 个候选，确认轮 n=6 数据）**：

- 合格 = 确认存活率 ≥ **40%** ∧ 数据集 max R' < **0.99**（贴顶剔除）∧
  数据集全距（max−min）≥ **0.10**（死平剔除）。
- **锁位**：SAD 合格必入（语音域，面板缺）；Ethanol 合格必入（非 EEG
  长锚，解"长序列×EEG 域共线" limitation）。
- 余位按确认存活率降序从 NATOPS/HMD/Cricket 取（并列按名字典序），
  **晋级总数 ≤ 4**。
- 源池 v2 = 核心 4 + 晋级 0–4 + EEG 低权 0–1（下条）。

**EEG 低权代表判（SCP2，n=3 数据）**：确认存活率 ≥ **30%** ∧ 全距 ≥ 0.10
→ 以**低权身份**入池：只供 label/毒性先验粮草（破坏配置是 avoid-worst
粮草，训练侧不过滤——铁律），**不作 LOO 折、不入靶池**；不合格 → 不留
EEG 代表，域覆盖缺失写 limitation。

**靶池判（核心 4 + 晋级确认者中 planned_tasks ≥ 8 者；网格∩GGS 任务上算）**：

- **可评** = 任务中位 (oracle − 中位 CLS R') ≥ **0.05** ∧ max R' < 0.99
  （oracle = shared 10 网格逐任务最大值）。
- **净空** = 任务中位 (oracle − GGS@290) ≥ **0.05**（GGS 目录外净空可为负，
  如实报——网格 oracle 是 10-CLS 代理下界）。
- 判定：可评 ∧ 净空 → **常规靶**；可评 ∧ ¬净空 → **展示靶**（报表只对
  random/中位并明示净空上限——HW 工作点已见 0.004，预计落此档）；
  ¬可评 → 不入靶池。
- 晋级者入靶池须补 `_06/_07` × 10 网格（预算记 MX-2v2，本轮不跑）。
- HMD（planned 6）与 SCP2（planned 3）构造上不入靶池。

**硬闸**：终判只在确认轮满格数据上做（每候选 6×10、SCP2 3×10、GGS 73 格
齐全且零 null）；缺格 → 先补齐再判，禁止部分数据宣判。

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| PS1 | SAD 确认存活率 ≥40%（n=2 58% 回归收缩后仍过线） | 70% |
| PS2 | Ethanol 确认 ≥40% | 60% |
| PS3 | 晋级总数 ≥2 | 75% |
| PS4 | HW @290 净空 <0.05（与工作点 0.004 同向）→ 展示靶档 | 85% |
| PS5 | SCP2 确认 ≥30% | 60% |
| PS6 | 常规靶 ≥2 个 | 60% |

## §6 执行（pod 开通后；铁规照旧）

- **预算（锚点带硬件标签）**：新格 = 确认 210（5×40+SCP2 10）+ GGS 73 =
  **283 格 @290** + 新基线 ~105 条（秒级）。RTX 5090 短 L（≤400）
  ~20 s/格；长 L（Ethanol 1024 / SCP2 1152）按 resize 重尾 2.3× 预
  45–150 s/格 → **合计 ~2.5–3.5 h 单卡**（4060 8GB 长 L 段不适用）。
- 执行顺序按 l_enc 升序（断点落长 L 端）；OMP_NUM_THREADS=1 + joblib
  threading；逐条落盘 + inflight + `--resume`；OOM 抛 RprimeOOM 不写
  null；device_name 逐条；云环境按 Blackwell 闸门指纹（torch 2.8.0+
  cu128/cuDNN 91002）+ 代码树 md5 校验。
- **收官重验（幂等教训）**：行数/唯一键/零 null/inflight 0 B；勿信中途
  计数。
- 终判后动作：`--analyze` 判读 → 更新 `configs/source_pool.yaml`（唯一
  真源：UM 入 source（npz 源）、晋级者入 source、CT/AWR/Heartbeat 退役、
  SCP2 标低权、Ethanol 出 reserve 若晋级）+ REPORT + memory + commit。
  这是 MX-2v2 协议锁的前置。
- 顺序：本协议 → tests → 实现 → `--screen` 对账（零 GPU，复算 §1 初筛
  表核对机器）→ 协议锁 commit →（pod）`--run --resume` → 收官重验 →
  `--analyze` 终判 → REPORT/memory/commit。

产物：`results/source_pool_v2/{PROTOCOL.md, screen.json, rprime_cache.jsonl,
baseline_cache.jsonl, analysis.json, REPORT.md}`。
