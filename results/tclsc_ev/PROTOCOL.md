# WP16 EV1 检验 PROTOCOL：源侧进化搜索验证——比较器当适应度、在好区域邻域里真的搜一次，搜出来的配置真跑，回答"搜索净空有没有、E 引导的进化找不找得到"

**会话定位**：NB1 的 G-N1 量的是**随机邻居**（每切片 10 条）的净空，对 2.8 万条邻域的最优几乎没有估计力；"T-AHC 路线被空间封顶"是过度推断（用户 2026-09-18 指出：AutoCTS++ 是训完 T-AHC 后在目标上用进化算法找策略，我们只在 shared / random 提名上排序）。
本协议把 AutoCTS++ 的部署形态（Algorithm 2：采样 → 比较器 Round-Robin 排序 → 交叉 / 变异 → top-3 真训）搬到**源侧切片**上做（那里付得起标签），并按用户要求并行两种 E：转导型 E_T（GGS 探针表示空间响应画像）与 AutoCTS++ 原味 E（共享 TS2Vec + Set Transformer，联合训练）。
产物 `results/tclsc_ev/`；脚本 `scripts/77_tclsc_ev.py`；进化模块钩子 `autocls/search/evolution.py`；SetTF field 臂 `scripts/33` `E_ARMS["es_field_hard"]`；pod 三件套 `pa_pod/ev_{bundle,setup,run}.sh`。

## §0 已看过的读数

- NB1（`results/tclsc_nb/REPORT.md`）：随机邻居 @290 最好邻居 > 最好 shared 占比 0.56、超 δ 0.034、中位 +0.008（HMD +0.038 / Ethanol +0.026 最大）；20 候选上 E_T regret@3 0.0319 vs 任务盲 0.0294（p 0.83）；高估签名任务盲 +0.35 / E_T +0.15；邻居 vs 中心判对率 0.60 / 0.54、"别改"0.65；邻居内配对 acc 0.66 / 0.64。
- K3：信任域内 E_T 0.0151 vs 任务盲 0.0191（p 9e-6）。WP8-BO：Ethanol 目录外 0.373（3 种子，池最大 0.430）> TS2Vec，六靶净空 1/6。D 检验：汉明 8 的提名被一致高估。
- S1E：SetTF E 在 diff 头下不塌缩但学成数据集身份（集内余弦 ≈ 1），cross 消融零（E-INERT）；concat 头下 E 连 zero 都零（七轮）。
- AutoCTS++ 论文设定：p1 0.8 / p2 0.2、种群 kp 10、取 top-3 真训；代码 `random_search.py` 零样本段是采样 + 比较器排序取 top-k。
- **没看过**：任何进化搜索在源切片上的输出；任何用 NB1 标签训的 SetTF 比较器；任何进化提名的 R'。

## §1 设计

### 1.1 适应度（三种引导 + 两种随机对照）

| 臂 | 适应度 / 来源 | 说明 |
|---|---|---|
| `ev_et` | NB1 的 `ft60_hard`（E_T field 头）折模型，seeds 0/1/2 平均 P | 切片所属数据集的 LODO 折模型（该数据集从未参训）；ctx = E_T@60 ⊕ φ12（折内标准化）⊕ 预算 one-hot（@290） |
| `ev_es` | **新臂 `es_field_hard`**：E = 折共享 TS2Vec（只在训练侧 7 集 fit，铁律 6）编码切片窗口 → Set Transformer（与比较器联合训练）；field 头；ctx = 预算 one-hot | 用 NB1 同一份训练标签（shared k3 + 邻居 k1、87 切片）、同 LODO、同超参训 7 折 × seeds 0/1/2；头取 field 而非 concat：concat 头七轮把 E 训成偏置（L3），field 头是 E_T 臂的头——**两臂唯一差别 = E 的来源** |
| `ev_blind` | NB1 的 `blind_diff_hard` 折模型 seeds 0/1/2 | 任务盲引导 |
| `rand_u` | 均匀随机：中心在信任域 11 条里均匀取、h ∈ {1,2,3} 均匀取、NB1 `mutate`（比例档 ±1/±2） | 每切片 3 条新标签；与进化同一搜索域、无任何标签信息 |
| `rand_nb` | NB1 该切片的 6 条切片邻居（已有 k1 标签，零新 R'） | 中心 = 该切片 **k3 真值前 2**（带标签信息的强对照）；取 C(6,3)=20 个三元子集的期望 |

### 1.2 搜索域与进化（Algorithm 2 适配，`evolutionary_search` 加三个可选钩子，缺省逐位不变）

- 信任域 T = shared-10 + GGS（11 条，所有切片相同）；搜索域 D = 到 T 的最小汉明（19 个基础字段）≤ 3 且 `validate` 合法。
- 初始：T 全部 + ks = 200 条域内随机（`rand_u` 同一采样器）→ Round-Robin 胜场 → 种群 kp = 20；G = 10 代；每代产 kp 个后代：p_cross 0.8（两父代逐字段均匀选）否则 p_mut 0.2（一步局部变异 = NB1 `mutate(h=1)`）否则复制；后代必须合法且 ∈ D，否则重试；并入后重排截断；最终 Round-Robin 排全种群。
- 打分：`score_pairs` = 3 个种子模型的平均 P（34 号 `make_score_pairs` 语义），ctx 预算 = 290；SetTF 臂的 E 每模型对该切片窗口前向一次。
- **提名 = 最终排名里前 3 条非信任域配置**；同时记录每条提名的位次、信任域最好一条的位次 → `pred_above_trust`（比较器是否认为它赢过信任域）。
- 切片：14 = 7 个折数据集各 2（`{d}_00` 与 `{d}_x00`，固定规则，不按 NB1 净空挑）。SCP2 无 LODO 模型，不入。

### 1.3 标签（铁律 3 缓存 + --resume + --verify）

- 提名 14 × 3 引导 × 3 = 126 条（跨引导重复的配置只跑一次；已在 NB1 / K3 / mx3 / SX 缓存里的配置直接复用同 canonical 种子的行，标 `reused_from`）+ `rand_u` 42 条 = ≤ 168 条轨迹 × 三截点，canonical 种子（k = 1），32 号执行体；缓存 `rprime_ev.jsonl`。
- 分工：`ev_et` / `ev_blind` / `rand_u` 的进化与标签在本地 4060（模型在本地，≤ 126 条 ≈ 3–4 h，脱离进程）；pod 5090 只做 SetTF 段：`--fit-folds`（7 折 TS2Vec + 87 切片窗口特征）→ `es_field_hard` 21 跑（3 并行）→ `ev_es` 进化 → 提名 ≤ 42 条 → NB1 考卷排名（12 切片 × 20 条，seeds 0/1/2，供 §2.4）→ 打包；估 3.5–4 h ≈ **$3.5–4**。

## §2 指标（零 GPU；真值 k1 @290 为主，@60 / @120 只报）

1. 每 (切片 s, 臂 g)：`best3` = 3 条提名 R' 的最大值，`mean3` = 均值；参照 `ref(s)` = 信任域最好 = max(shared-10 的 k3 均值, GGS 的 k1)（部署会考虑 GGS，取更保守的参照）；副参照 = NB1 口径 shared-10 seed-0 单值。
2. `gap(s,g)` = best3 − ref；超 δ 记 gap > δ_k1(数据集, 290)。占比 `frac_positive` / `frac_over_delta`（n = 14）。
3. 配对（14 切片，单侧 Wilcoxon + 均值差）：`mean3(g)` vs `mean3(rand_u)`；`best3` 同；`ev_et` / `ev_es` vs `ev_blind`；`ev_es` vs `ev_et`；`rand_nb` 期望值同表。
4. 高估：每臂 `pred_above_trust` 占比 − 真超参照占比（R' > ref）；NB1 考卷上 `es_field_hard` 的 regret@3 / hit@3 / M1（seeds 0/1/2 单组，与 NB1 同组数字并排）。
5. 诊断只报：提名到信任域的汉明分布、提名落在 NB1 已标邻居上的比例、进化最终种群里信任域配置占前 3 的切片比例、按数据集分层的 gap。

## §3 判读（先于数据）

| 门 | 定义 |
|---|---|
| C0 | 提名重跑逐位相同；提名 ∉ T、∈ D、合法；复用行的 (切片, key, 截点) 与源缓存逐位相同；`es_field_hard` ArmSpec = E+ctx / settf / fold / budget / field |
| **G-V1 搜索净空**（按臂） | `frac_positive` ≥ 0.50 且 `frac_over_delta` ≥ 0.20（≥ 3/14） |
| **G-V2 引导 > 随机**（按臂） | `mean3(g) − mean3(rand_u)` 均值 ≥ +0.010 且 Wilcoxon p < 0.05 |
| **G-V3 E > 任务盲**（按 E 臂） | `mean3(E 臂) − mean3(ev_blind)` > 0 且 p < 0.05 |
| G-V4 不高估（只报） | E 臂 `pred_above_trust` 占比 − 真超参照占比 ≤ 0.25 |

**读数码**（E 臂 = `ev_et` 或 `ev_es`，任一满足即按其读）：
- G-V1 ✓ ∧ G-V2 ✓ ∧ G-V3 ✓ → **EV-SEARCH-WIN**：比较器引导的进化在源侧找得到超噪声的更好配置且 E 是原因 → 立项目标侧 NB2（官方口径六靶）。
- G-V2 ✓ ∧ G-V3 ✓ ∧ G-V1 ✗ → **EV-GUIDED-FLAT**：E 引导比随机 / 任务盲好，但域内没有超 δ 的东西——搜索有效、空间平。
- G-V1 ✓（任一臂含 `rand_u`）∧ G-V2 ✗ → **EV-HEADROOM-RANDOM**：净空存在但比较器不是找到它的原因 → 价值在目标侧真测搜索，不在 E。
- 其余 → **EV-DEAD**：NB-NO-HEADROOM 的读法成立（随机邻域净空 ≈ 搜索净空 ≈ 0）。
只报：`ev_es` vs `ev_et` 谁强；`rand_nb` 期望；逐数据集。

## §4 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | G-V1 对 `ev_et` | 35% |
| P2 | G-V2 对 `ev_et` | 40% |
| P3 | G-V3 `ev_et` > `ev_blind` | 45% |
| P4 | `ev_es` mean3 ≤ `ev_et` mean3（SetTF E = 身份向量，S1E） | 65% |
| P5 | 高估：`ev_blind` − `ev_et` ≥ 0.10（NB1 签名在搜索下复现） | 60% |
| P6 | 超 δ 的切片只出现在 HMD / Ethanol | 55% |
| P7 | ≥ 50% 切片的最终种群前 3 含信任域配置（邻域标签后比较器偏好已知配置） | 50% |

## §5 执行

1. 本文件 + `tests/test_tclsc_ev.py`（先红）→ 协议锁 commit；2. 实现（进化钩子 / 33 新臂 / 77 号 --evolve / --rprime / --verify / --fit-folds / --train-es / --rank-es / --analyze，pod 三件套）→ 绿 → 本地冒烟 → commit；
3. 本地：`--evolve --guides ev_et ev_blind --random` → `--rprime`（脱离进程，4060）；pod（5090，开前请示一次）：nvidia-smi + CUDA 闸 → HTTP 代理上传 → ev_setup → ev_run（fit-folds → train-es 3 并行 → evolve ev_es → rprime → rank-es → 打包）；3 分钟内确认首条日志、10 分钟无进度必查；4. 拉回 → 合并缓存 → `--verify --analyze` → REPORT → stop + confirm 3/3 → commit → memory。

## §6 铁律对照

铁律 1（变异 / 交叉只用 space.yaml 档位与约束，validate 把关）；2（三截点同一轨迹）；3（缓存 + --resume + --verify，复用行标来源）；5（先红）；6（折模型与折 TS2Vec 都不含考侧数据集；SetTF 特征只前向；E_T 标准化只用训练切片）；7（evolution.py 钩子是纯函数，不见任务概念）；8（is_better 判优）。

## §7 执行台账（跑后填写）

（空）
