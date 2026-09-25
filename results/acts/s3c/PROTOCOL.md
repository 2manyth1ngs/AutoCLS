# WP17-ACTS S3-C PROTOCOL：黑盒比较器 T-CLSC（AutoCTS++ 原样标签）+ 源侧搜索验证（6 个 UEA 折）

**会话定位**：
- S3-A（S3A-GO）；S3-A′ 三项负；S3-B（S3B-E-NOT-BETTER）；S3-B′ / S3-B″ 的 field-E（内容 ✓、价值边缘）。
- S3 线上还**没有训练过任何 T-CLSC**。用户 2026-09-25 拍板直接进入 S3-C，并定下两项设置：
  1. **主臂用 AutoCTS++ 原样标签**；
  2. **加源侧搜索验证**（Ks 采样 + 进化 → 前 3 条真跑），只在 5–6 个 UEA 数据集上做留一折，不做 17 折。
- 本协议对应方案 `docs/AUTOCTS_STRICT_REPRO_PLAN.md` §3.3 的 S3-C 与 §4 G-S3C，并加一段源侧搜索验证。

产物 `results/acts/s3c/`；代码 `scripts/89_acts_s3c.py`；单测 `tests/test_acts_s3c.py`；服务器三件套 `scripts/pa_pod/acts_lab_{bundle,setup,run}_s3c.sh`。

## §0 已看过的读数（如实声明）
1. S2 全部读数；S3-A / S3-A′ / S3-B / S3-B′ / S3-B″ 全部读数（参照数见 §3）。
2. 与本协议直接相关的已知读数：
   - SetTF E 在本语料学成数据集身份（S3-B：换源 Δ ≈ 0，同源余弦约 0.9）；
   - field-E（E_sh 均值池化 + PCA）在 16 次换源检验下内容显著（S3-B″ p 0.003）；
   - 名单外配置排序：φ 模型 Spearman 0.20，field-E 0.13–0.18；
   - EV1（旧语料）：进化把比较器的高估从 +0.35 放大到 +0.81。
3. **臂 C（field-E）是看过 S3-B″ 之后选的**，只作事后对照，不进读数码。
4. **没看过**：S2 语料上的 best-of-3 单种子标签的任何统计；任何 T-CLSC 在 S2 语料上的训练或读数；任何搜索提名及其真值。

## §1 标签与训练（主臂 = AutoCTS++ 原样）
- **折**：训练池里的 6 个 UEA 源轮流留出。
  - 6 个源：Handwriting / PhonemeSpectra / UWaveGestureLibrary / Cricket / EthanolConcentration（8 片）/ SelfRegulationSCP1（16 片），共 56 个留出任务。
  - 每折在其余 16 个源上训练，其中 1 个源作早停验证：池顺序中紧跟留出源的那个，同 S3-B。
  - 训练池、E、任务均与 S3-A 一致（160 任务）。
- **标签 R′**：seed 0 轨迹在 25% / 50% / 100% 三个截点的 5 折 CV acc 取最大（best-of-3，对应 AutoCTS++ 的 best-of-5 epoch；截点来自 S2 缓存，零 GPU 重建）。
  - 每任务的候选 = shared-30 + 该任务有效的 random-20；
  - 单种子（shared 的 seed 1 不进标签）；
  - 不套地板、无死区；error / 缺失格记该任务可用 R′ 的最小值（S1″ 规则，AutoCTS++ 无此情形）。
- **配对**（← AutoCTS++ `AHC_DataLoader_linear.generate_pairs`）：
  - 每个 epoch，每个任务把记录序列与其乱序副本逐位配对；
  - y = 1 若 R′_a > R′_b，否则 0（**平局记 0，不丢弃**）；自配对跳过；
  - 不追加反向对，不做课程（源码主循环无课程）。
- **训练**：
  - BCE，Adam lr 1e-4，batch 64 对，最多 100 epoch；
  - 早停 patience 5（方案 §3.4），监控验证源的 BCE：验证源任务内全部有序对，固定，不随 epoch 变；保留最好权重；
  - 训练时每任务每步子采 32 个窗口（方案 §8-6 的显存偏差，同 S3-B），评估时用全部窗口；
  - 每臂每折训练 1 个种子（seed 0；AutoCTS++ 也只训练一次）。

## §2 臂（全部 `TCLSC`，配置支路 = binned MLP，embedding 128，fz 128）
| 臂 | 任务支路 | 头 | 说明 |
|---|---|---|---|
| **A（主，严格复现）** | E_pt 窗口集合（S3-B 抽取），TaskSetEncoder(256, 256) 联合训练 | concat | AutoCTS++ T-AHC 原样（GIN → MLP 为不可对齐项） |
| B | E_sh 窗口集合（该折共享 TS2Vec），SetTF 联合训练 | concat | 换 TS2Vec 来源 |
| C（事后） | ctx = E_sh 均值池化 → 该折训练任务拟合的标准化 + PCA，r = 4 | field | field-E 搬进配对训练；r = 4 取 S3-B″ 内层最常选的值 |
| D（对照） | 无任务支路 | concat | 任务盲比较器（方案消融"w/o E"） |

## §3 排序读数（56 个留出任务）
- **目录内**：对 shared-30 做全部有序对打分 → 对称化 → Round-Robin 胜场排序 → 前 3 → 按 seed-0 终点选 → 用 seed-1 终点评。regret@3 / regret@1 与 S3-A 同口径，便于同表。
- **名单外**：对 shared ∪ random 做 Round-Robin，取 random 配置的胜场与其 seed-0 终点原始 acc 算 Spearman，逐任务取中位（S3-A′ 口径）。
- **换源检验**（A / B / C）：留出任务的 E 依次换成偏移 k = 1…16 的其他源同行号任务的 E，重算 regret@3，取 16 次均值；Δ = 换源 − 本身。
- **身份诊断**（A / B）：SetTF 输出向量的集内 / 跨集平均余弦。
- **参照**（同 56 任务）：
  - 任务盲 μ、④（4 探针 + 低秩）、③ land25、field-E ridge（S3-B″ E_sh）；
  - 这些参照都来自 17 折 LOSO，而每个留出源对应的训练集相同，所以可以直接比。
- **统计**：配对差报源聚类 bootstrap 95% CI（6 个源）、源均值 Wilcoxon（n = 6，最小双侧 p = 0.031，即 6 个源同号才可能显著）、任务级 Wilcoxon（n = 56，只报）。

### 3.1 排序门（方案 §4 G-S3C；读数码由主臂 A 决定，B / C / D 只报）
| 门 | 定义 |
|---|---|
| C1 身份 | A 的 SetTF 输出集内余弦 < 0.9 |
| C2 内容 | A 的换源 Δregret@3 ≥ 0.010 且源级 p < 0.05 |
| C3 价值 | A 的 regret@3 比 ④ 低 ≥ 0.005 且源级 p < 0.05 |
| C5 名单外 | A 的 random-20 中位 Spearman ≥ 0.50 |

读数码（方案原文）：C2 ∧ C3 → **S3-E-WIN**；C3 ✗ 且 C2 ✓ → **S3-LOWRANK-SUFFICES**；C2 ✗ → **S3-E-INERT**。C1 / C5 作后缀报告。

## §4 源侧搜索验证（用户要求）
- **任务**：6 个折源各取 OA 行 a01 / a03 / a05 / a07（UCR-ADM 的 4 行版，覆盖三轴档位），共 24 个任务。比较器用该折训练好的那个。
- **搜索**（A / B / C / D 四臂，同一任务共用同一份 Ks 样本）：
  1. Ks = 20,000：用与 S2 random-20 同一个采样器（`acts_space` + `validate` + 剪枝四条 + 去重），种子 = `run_seed_for(task, "s3c_ks")`；
  2. **初始选择**（← 源码"分块 nlargest + 块赢家两两计分"，O(Ks²) 全量 Round-Robin 不可行）：分成 40 块 × 500，块内 Round-Robin 取前 10 → 400 条块赢家做 Round-Robin → 前 kp = 10 作初始种群；
  3. **进化**：`evolutionary_search(ks=0, seed_population=初始种群, kp=10, generations=10, p_cross=0.8, p_mut=0.2, accept=validate ∧ ¬剪枝)`，最终 Round-Robin 取前 3；
  4. **随机对照 R**：同一份 Ks 样本的前 3 条（均匀随机，成本相同）。
- **真值**：每条提名配置在该切片上按 `ts2vec_official` cv5 跑 seed 0 与 seed 1（83 同一执行体）；已在 S2 缓存里的配置直接复用。每臂按 seed-0 终点在 3 条里选 1，**报 seed-1 终点**（独立种子，去掉挑选时的赢家诅咒）。
- **目录参照**（零新格）：
  - 目录任务盲前 3（训练源平均秩，= S2 H_cat）；
  - 目录 oracle（shared-30 里 seed 0 最好的一条，用 seed 1 评）；
  - H_rand_adj oracle（S2 修订 1）。
- **读数**：每臂的最终 acc；配对差 A − R、A − D、A − 目录任务盲前 3、A − 目录 oracle；提名配置真值超过目录 oracle 的份额；提名配置的比较器胜场与真值名次的对应（校准）。

### 4.1 搜索门（先于数据；主臂 A）
| 门 | 定义 |
|---|---|
| S1 搜索价值 | A − R ≥ +0.010，且任务级 Wilcoxon p < 0.05（24 任务），且 ≥ 5/6 个源同号 |
| S2 任务感知 | A − D > 0，且任务级 p < 0.05，且 ≥ 5/6 个源同号 |
| S3 不输目录 | A − 目录任务盲前 3 ≥ 0（点估计） |

读数码：S1 ∧ S2 → **SEARCH-WIN**；S1 ∧ ¬S2 → **SEARCH-BLIND**（搜索有用但任务感知没用）；¬S1 → **SEARCH-DEAD**；S3 作后缀（**+CATALOG-OK** / **+CATALOG-LOSS**）。

## §5 C0
- 标签重建：从 S2 缓存重建的终点值须与 S2 `labels.jsonl` 的 seed 值逐位相同；best-of-3 ≥ 终点。
- 参照：任务盲 / ④ / ③ 在 56 任务上的均值与 S3-A 逐任务数组的子集逐位相同。
- 训练冒烟：2 折 × 每臂 2 epoch 能跑通，loss 下降；搜索冒烟：Ks = 2,000、2 代，能产出 3 条合法、未剪枝的配置。

## §6 预测（先验）
| # | 预测 | 先验 |
|---|---|---|
| P1 | A 的 SetTF 集内余弦 ≥ 0.9（身份型，C1 ✗） | 60% |
| P2 | C2 ✓ | 20% |
| P3 | C3 ✓ | 10% |
| P4 | C5 ✓ | 5% |
| P5 | S1 ✓（A 搜索优于同成本随机） | 20% |
| P6 | S2 ✓（A 搜索优于任务盲比较器搜索） | 15% |
| P7 | S3 ✓（A 搜索不输目录任务盲前 3） | 35% |
| P8 | A 的提名真值低于目录 oracle 的任务 ≥ 70%（搜索下高估） | 60% |

## §7 执行
1. PROTOCOL + 单测先红 → 协议锁 commit。
2. 实现 89：
   - `--prep`（本地：标签、折、PCA ctx、参照）；
   - `--train --arm --fold`（服务器：训练 + 留出排序读数，写 json）；
   - `--search --arm --fold`（服务器：24 任务提名）；
   - `--plan-runs`（服务器：提名去重 → 83 计划，phase X）；
   - 83 `--run --phase X`（服务器：真跑）；
   - `--analyze`（本地）。
   
   单测绿 → 本地 4060 冒烟 → commit。
3. **请示用户后**上实验室服务器。
4. 拉回 → `--analyze` → REPORT → commit → CLAUDE.md / memory。

## §8 成本估计（按 S2 / S3-B 实测锚点，共享 4090 可能慢 2–5 倍）
- 比较器训练：6 折 × 4 臂 = 24 次，A / B 臂约 20–40 min / 次，C / D 臂分钟级；约 5–8 GPU-h。
- 搜索：24 任务 × 4 臂，每次约 1–3 min；约 2–4 GPU-h。
- 真跑：24 任务 × 5 组（4 臂 + R）× 3 条 × 2 种子 ≤ 720 格，单格中位约 49 s；约 10 GPU-h。
- 合计约 17–22 GPU-h，4 卡墙钟约 5–8 h。

## §9 铁律对照
- 1：配置只经 `acts_space`，剪枝只在 acts.yaml。
- 2：截点同轨迹，iters 规则显式。
- 3：新格进 runs.jsonl，`--resume`。
- 4：T-CLSC / SetTF 对齐 `ahc_engine.py`，已注释。
- 5：先红；§0 已声明。
- 6：比较器 / PCA / 先验只在训练源；留出源只考；E_sh 用本折编码器。
- 7：`TCLSC` / `evolution` 不含数据集概念。
- 8：acc 越大越好。

## §10 执行台账（跑后填写）
