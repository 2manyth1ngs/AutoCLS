# MX-4 WP8-BO 协议：目标侧提名回灌搜索（"第三条"）——信任域提名 + 真评估回灌能否搜到目录外更好的 CLS

分支 `wp8-bo`（自 wp7-ucr HEAD）。脚本 `scripts/43_bo_nominate.py`，单测 `tests/test_bo_nominate.py`（12 项）。
场地 = WP3 官方口径 6 个 UEA 靶（`results/mx4_deploy`，`ts2vec_official`，截点 25/50/100%，HW iters 200 / 其余 600）
+ WP3b UCI-HAR（`results/mx4_deploy_har`，`legacy_full`，2320，截点 580/1160/2320）。计算：本地 RTX 4060 脱离进程。
用户拍板（2026-09-05）：本地 4060 过夜；每靶 3 轮 × 5 条提名；对照臂 = 同成本随机邻域提名（必需）。

## 0. 已看过的读数（如实声明，铁律 5）

**阶段 0（零 GPU，`43 --simulate --label stage0`，落盘 `stage0.json / stage0_tables.md`）**：在 WP3/WP3b 缓存（每靶目录 21 条 =
shared-10 + GGS + 一次性提名 10，三截点齐；PS 一格 error 剔除为 20）上模拟了下表策略。SH6-3-3_prior 与 WP3 `analysis.json`
逐位相同（7/7 靶，C0 ✓）。

| 靶 | 目录 oracle（key，先验名次） | SH6-3-3_prior | SH10-4-3_prior | ONESHOT-SH | MIX-SH | Spearman(pred_z, @100%) 一次性提名 |
|---|---|---:|---:|---:|---:|---:|
| Handwriting | 0.546（c92058f2 = GGS，第 8） | 0.419 | **0.546** | 0.419 | 0.378 | −0.70 |
| UWave | 0.931（8c4aa283，第 6） | 0.931 | 0.931 | 0.931 | 0.831 | −0.26 |
| PhonemeSpectra | 0.251（4f7d4135，第 6） | 0.251 | 0.251 | 0.251 | 0.234 | −0.13 |
| Ethanol | 0.300（f48e5da1，第 8） | 0.274 | 0.297 | 0.285（提名 b8fc4fb7） | 0.293 | +0.38 |
| Cricket | 1.000（c92058f2，第 6） | 1.000 | 1.000 | 1.000 | 0.986 | +0.30 |
| HMD | 0.486（**提名 51b6f528**，第 16） | 0.378 | 0.378 | **0.486** | **0.486** | +0.30 |
| HAR | 0.940（4f7d4135，第 4） | 0.940 | 0.940 | 0.940 | 0.927 | −0.09 |
| 成本（全跑当量） | — | 3.75 | 5.00 | 6.25 | 3.75 | — |

读法（写在数据之前的判读带不受影响，但下面的门是在看过这张表之后写死的，这里如实说明）：
- 一次性提名进起跑名单（ONESHOT-SH）在 HMD 拿到目录 oracle（+0.108）、Ethanol +0.011，其余 5 靶不变——WP3 留下的
  "提名有没有用"的答案是：**有，但只在提名恰好是全场最高的靶上**，代价 +67% 成本。
- 按代理分混排（MIX-SH）在 HMD/Ethanol 同样拿到提名，但 HW/UWave/PS/Cricket/HAR 全部变差（UWave −0.100）：代理对
  目录外候选系统性高估（判死 (b) 的形态在部署口径上复现），一次性提名的 pred_z 与 @100% 真值 Spearman 在 4/7 靶为负。
  → 本工作包**不采用**代理分与先验混排；提名只以"真评估后按 @25% 竞争"的方式进入淘汰赛（§2.3 BO-SH 定义）。
- HW 的头顶空间在目录内（GGS 第 8，SH10-4-3 拿到 0.546），HMD/Ethanol 的在目录外提名里；UWave/PS/Cricket/HAR 目录 oracle 已被 SH6-3-3 拿到。
- 计划文件里 ONESHOT-SH 成本写成 5.5 是算术错误，正确为 16×0.25 + 3×0.25 + 3×0.5 = **6.25**（单测锁定）。

**冒烟（GPU，`results/mx4_bo_smoke`，已删除）**：Handwriting 1 轮 × 每臂 1 条，走完 提名 → 轨迹 → 回灌 → 模拟 全链：
bo_ucb 提名 33da9184（pred z 1.05，汉明 2）@100% 0.406；rand_nbr 提名 9d7ae7be（汉明 1）0.460；两格 12.5 s / 5.7 s。
正式跑的 HW 第 1 轮 bo_ucb 因 seed 相同会再次含 33da9184——这两条读数对下面任何门都无判定力，如实登记。

**未看过**：任何靶的 3 轮回灌结果、BO-SH / RAND-SH 冠军、3 种子均值、HAR 提名。

## 1. 问题

零样本层是任务无关天花板（n=8 六轮 + n=84 一轮判负），任务感知只在目标侧测量里发生（SH）；剩余 regret 在**未起跑**的候选上
（HW：GGS 第 8；HMD：提名 51b6f528 第 16）。判死 (b) 关掉了"代理零样本外推到目录外"，唯一出口是
**代理提名 + 目标侧真评估回灌（BO 式）**。本包问：在官方口径 6 靶上，用靶侧 @25% 真值回灌、在信任域内迭代提名 3 轮，
(B1) 能否找到目录外更好的配置；(B2) 部署策略 BO-SH 是否胜过现行 SH6-3-3_prior；(B3) 代理引导是否优于同成本随机邻域。

## 2. 设计

### 2.1 起点与证据
- 第 0 轮证据 = 目录 21 条的完整轨迹（WP3/WP3b 缓存，零新跑）。**回灌只读 @25% 截点**（部署可得）；@100% 只作真值。
- 代理：`FieldSurrogate("hgb", bagging=10)`，特征 `36.features`（binned 116 ⊕ b/290），训练行 = 源池 z 行（LODO：靶在源池内剔除
  其切片；HAR 用全 8 源，与 WP3/WP3b 同式）∪ 靶侧已测行（within-靶 @25% z，`iters=60` 档，复制 ×10 当权重）。

### 2.2 一轮（每靶每臂，`nominate_round`）
- 父本 = 靶上 @25% 最高的 5 条已测配置（目录或前轮提名，error 格不入）。
- 候选池 2000 次变异：父本改 1–3 个字段（`space.choices` 均匀重采 + `inactive_overrides` + `validate`），去重（已评过 / 已提名 / 目录 / 源池全部排除）。
- 信任域：到源池 ∪ 靶侧任一已见配置的字段汉明 ≤ 3（`field_hamming`）。
- `bo_ucb`：按 `pred + 1.0·std` 取前 5；`rand_nbr`：同池同半径同父本，均匀取 5（抽签在拟合代理前、只依赖 seed）。
- seed = 4300 + 100·靶序 + 10·臂序 + 轮次；每轮提名先落盘 `bo_state.json`（父本、pred/std/汉明、池大小）再跑。
- 每条提名跑**完整三截点轨迹**（`39.run_trajectory_cell`，seed 0，`arm=f"{臂}_r{轮}"` 入缓存键）；两臂各自独立回灌。
- 3 轮后每臂 15 条；候选池 = 目录 21 + 本臂 15，全部有 25/50/100。

### 2.3 部署策略（缓存上模拟；成本 = iters 全跑当量，另报缓存墙钟）
| 策略 | 定义 | 成本 |
|---|---|---:|
| SH6-3-3_prior（现行，须与 WP3 逐位） | 先验前 6 @25% → 3 @50% → 3 @100% | 3.75 |
| SH10-4-3_prior | 先验前 10 → 4 → 3 | 5.00 |
| ONESHOT-SH | 先验前 6 + 一次性提名 10 @25% → 3 → 3 | 6.25 |
| MIX-SH | 先验 z 与提名 pred_z 混排取前 6 → 3 → 3 | 3.75 |
| **BO-SH（主）** | 先验前 6 + bo_ucb 3 轮 × 5 条（全部 @25%）→ 按 @25% 留 3 → @50% 留 3 → @100% 取最好 | 7.50 |
| RAND-SH（对照） | 同上，提名换 rand_nbr | 7.50 |
| 池 oracle | 目录 ∪ 本臂提名 @100% 最大 | — |
淘汰规则本身不动（关键决策 8）：按当前截点分淘汰、最后一级留 3。冠军 3 种子：BO-SH 与 RAND-SH 的冠军补 seed 1、2 到 100%
（与 WP3 冠军同口径；同 cls_key 的已有终评行直接复用）。

### 2.4 门（先写死；在 §0 表之后、任何回灌数据之前）
- **B1 头顶空间**：bo_ucb 15 条提名的 @100%（seed 0）最大值 > 目录 oracle + 0.01，于 ≥ 3/6 靶。
- **B2 部署**：BO-SH 冠军 3 种子均值 ≥ SH6-3-3_prior 冠军 3 种子均值于 ≥ 4/6 靶，且六靶均值差 ≥ +0.01，**且 HW BO-SH ≥ 0.485**（TS2Vec 线）。
- **B3 引导有效**：bo_ucb 提名 @100% 均值 > rand_nbr 于 ≥ 4/6 靶，且池 oracle(bo) ≥ 池 oracle(rand) 于 ≥ 4/6 靶。
- 读数码：**BO-WIN** = B1 ∧ B2 ∧ B3；**BO-HEADROOM** = B1 ∧ ¬B2（找得到、策略没吃到）；**BO-LUCK** = B1 ∧ B2 ∧ ¬B3（随机邻域同样好）；
  **BO-DEAD** = ¬B1；六靶或 3 种子未齐 = BO-PENDING。HAR 描述性：BO-SH 3 种子 ≥ 0.935？
- 噪声提醒（limitation 先写）：单跑 R' sd 0.035–0.048；B1 的 0.01 边际在噪声内，B1 单独通过只说明"池里出现了更高的单跑值"，
  决定性读数是 B2（3 种子）与 B3（bo vs rand 的系统差）。

## 3. 预测（先验概率，写在数据之前）
| # | 预测 | 先验 |
|---|---|---:|
| P1 | B1 过 | 55% |
| P2 | B2 过 | 40% |
| P3 | B3 过 | 45% |
| P4 | HAR BO-SH 3 种子 ≥ 0.935 | 35% |
| P5 | bo_ucb 提名 @100% 均值 < 该靶 shared-10 均值于 ≥ 4/6 靶（提名平均比目录差） | 60% |
| P6 | RAND-SH ≥ SH6-3-3_prior（seed 0）于 ≥ 4/6 靶（多起跑不伤） | 60% |

## 4. 成本与排程
6 靶 × 2 臂 × 15 条 = 180 格全长轨迹 + 冠军 2 策略 × 2 种子 ≈ 24 格；HAR 30 + 4 格。4060 估时（4090 中位 × 2.3；HAR 用本机 312 s 锚）：
HW 0.11 h / HMD 0.48 / UWave 0.40 / Cricket 0.86 / EC 1.03 / PS 1.80 / HAR 2.60 ≈ 7.3 h 不含终评。
顺序：HW → HMD → UWave → Cricket → EC → PS（便宜的先，中断时已完成的靶可独立判读）→ UEA 终评 → HAR 循环 → HAR 终评 → analyze，
一体化脱离进程脚本（`results/mx4_bo/run_all.ps1`），每步 `--resume`。PS 的 SVM 超时在 Windows 无 SIGALRM，卡住靠重启 `--resume`。

## 5. 命令
```
python scripts/43_bo_nominate.py --simulate --label stage0 --datasets <6 靶> HAR   # 阶段 0（已跑）
python scripts/43_bo_nominate.py --loop --resume --datasets <6 靶>                  # 3 轮 × 5 条 × 2 臂
python scripts/43_bo_nominate.py --finals --resume --datasets <6 靶>
python scripts/43_bo_nominate.py --loop --resume --datasets HAR ; --finals --resume --datasets HAR
python scripts/43_bo_nominate.py --analyze --datasets <6 靶> HAR                    # analysis.json / tables.md
```

## 6. 不做的事
不改 35/36/39/autocls；不动 SH 淘汰规则；不用代理分混排；提名只在信任域内且全部真评估；evo_cache（汉明 8）不用；
不按结果调 κ / 半径 / 父本数 / 轮数（若要调，另开预注册）。

## 7. 运行中修订（2026-09-05 14:10，不改判读带）
PhonemeSpectra bo_ucb 第 2 轮首格（d39832ce）的 @25% SVM 网格评估在 Windows 上跑了 > 80 min 未返回（39 号注释已记载的 C=∞ 退化特征不收敛现象；
WP3 在 4090/Linux 以 SIGALRM 480 s 截断、记 1 格 error）。Windows 无 SIGALRM，39 号的上限静默失效。修订：43 号在无 SIGALRM 平台把
`eval_with_timeout` 换成 loky 单 worker 子进程版（`eval_with_timeout_win`，同 480 s、超时杀 worker、整格记 error 行），与 WP3 口径一致；
两项单测。杀掉卡住的进程后 `--resume` 重启 run_all.ps1；此前 160 格结果不受影响（缓存键不变）。
