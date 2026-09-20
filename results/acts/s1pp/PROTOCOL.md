# WP17-ACTS S1″ PROTOCOL：标签与尺子离开地板 + 空间剪枝 + 源准入 v2 + 整集口径对照 + 扩池余量决策

**会话定位**：S1′ 读数 ACTS-S1P-SAME（`results/acts/s1p/REPORT.md`）后，用户 2026-09-20 拍板四步方案（第 0 步重锁协议 → 第 1 步零 GPU 三个数 → 第 2 步条件触发的整集口径对照 → 第 3 步主批 → 第 4 步余量决策），并采纳评审修订：准入规则改成"不依赖地板但也不被失败块抬高"的版本、门的尺度写死、第 3 步成本按每源 4 任务算、第 4 步同域 LODO 定落点。执行机器仍是实验室 8k-4090（部署路径固定 `/home/b311/data/wangwenjie/kejunyu/AutoCLS`，口径声明沿用 S1 PROTOCOL §1.10）。
产物 `results/acts/s1pp/`；真源 `configs/acts.yaml profiles.s1pp`（`extends: s1p`，基线与 s1p 不动）；代码 `scripts/acts_common.py` + `scripts/83_acts_corpus.py --profile s1pp`；单测 `tests/test_acts_s1pp.py`；pod 脚本 `scripts/pa_pod/acts_lab_*_s1pp.sh`。

## §0 已看过的读数（如实声明）

- S1′（`results/acts/s1p/REPORT.md`，960 条试点标签）：G-A2 ✓ 0.553（Cricket 0.734 / UWave 0.715 / PS 0.166），G-A1 ✗ 0.781；failed（探针地板口径）0.51、按 τ 0.34 / 0.46 / 0.62 / 0.78、triplet/euclidean/l2 τ ≥ 1 时 100%；ADM 只准入 HW / UniMiB / UWave + SCP1。
- 后续核算（`followup_s1p{,_addendum}.md`）：诚实任务感知余量 regret@1 0.015（floored）、全局前 3 实测 ≈ 0；剪枝代价（官方 29 靶目录）triplet ∧ τ=10 名义丢 7 冠军、6/7 差 0.011–0.017 在噪声内、FaceDetection 0.055，家族 triplet/euclidean/l2 丢 0，家族 ∧ τ ≥ 1 丢 0；官方冠军 τ=1 18/29（基率 0.47）、τ=0.01 3/29（基率 0.25）；Cricket 被 a07 行（全通道 × 最短窗，探针地板 0.861）卡住；shared 比 random 多失败在抽样 sd 内；地板之下种子一致 ρ 0.62–0.77。
- 第 1 步三个数（`step1_s1pp.md`，本协议第 1 步已在锁前算完）：regret@3 去掉 top-3 侧赢家诅咒 0.001（floored）/ 0.003（raw）；池大小 5 → 20 的 regret@1 0.007 → 0.015 在 m ≈ 10–15 饱和、regret@3 全程 ≈ 0；**温度翻转坐实**：切片 top-1 τ ≤ 0.1 占 PS 8/8、Cricket 6/8、UWave 7/8，官方目录同三集冠军全 τ=1（UWave 官方 τ=1 均值 0.886 vs τ=0.01 0.814）；家族层切片 vs GEO-S 整集 Spearman 0.16 / −0.03 / 0.27 → **第 2 步触发**。
- 无地板准入三条按用户原文试算：PS 进、UWave 出（全量 seed Spearman 被失败块抬高到 0.80–0.86、前一半只 0.42–0.49；净占比随尺度 0.07–0.38；regret 分支反向）。
- **本协议尺度的标定值（S1′ 试点，锁前看过）**：中位钉底 + 前一半噪声（§1.2）下，交互净占比 PS 0.07 / Cricket 0.36 / UWave 0.33，未扣噪声 0.27 / 0.50 / 0.49（→ 真实份额 0.26 / 0.72 / 0.67）；前一半 seed Spearman 0.42 / 0.49 / 0.45；前一半 IQR/σ_mean 中位 1.6 / 2.9 / 2.6；前一半 σ_mean 中位 0.006 / 0.009 / 0.013；净空任务上同域 LODO regret@1（raw）0.038 / 0.018 / 0.014。§1.5 的阈值就是在这三个数上定的（PS ✗、Cricket ✓、UWave ✓），对新源是预注册，对这三集是标定集，如实记。
- **没看过**：任何整集（whole）口径下 shared-20 的标签；任何新 shared 配置的标签；任何外部源（PTB-XL / SleepEDF / PAMAP2 / Opportunity）的标签；旧源 a03 / a05 行的标签；任何按本协议尺度算出的新数据。

## §1 设计

### 1.1 标签 (a′)：不含地板
- R′ = 终点截点的原始 acc；shared 两种子均值、random 单种子；**不套地板**。探针地板（iters = 0 × 3 种子）照跑照记，只作诊断量（failed 份额、存活数按它报，不进标签、不进门、不进准入）。
- 配对规则（S2 比较器训练用，本轮只预注册并报产量）：对 (i, j) 的 gap = |R′_i − R′_j|，sd_gap = σ_task · √(1/n_i + 1/n_j)（n = 种子数），σ_task 见 §1.2；权重 w = 0 若 gap < 2·sd_gap（死区），否则 min(gap / (4·sd_gap), 1)；若 i 或 j 在该任务两种子均值的前四分位，w × 2。产量读数 = 每任务 w > 0 的对数。

### 1.2 门与准入统计量的尺度（写死）
- **中位钉底**：任务内候选向量 v（试点任务 = shared 30 + random；只有 shared 的任务 = shared 30）取 v_c = max(v, median(v))，再在 v_c 上 z 化（均值 / sd 用同一向量）。这是秩定义的"地板"，不依赖探针；失败块的可复现序（§0）不进主效应。
- **噪声**：σ_task = 两种子都有的 shared 里、按两种子均值排前一半的配置的 (seed0 − seed1)² 均值 / 2 开方；σ_mean = σ_task / √2；两种子均值标签的噪声均方 MS_noise = mean over (task, shared) 的 (z0 − z1)² / 4（z0 / z1 = 各种子值钉底 z 化）。
- **交互分解**（任务 × shared 矩阵 Z）：SS_task / SS_cfg / SS_int 两因素分解；**净占比** = (SS_int − MS_noise · |Z|) / (SS_task + SS_cfg + SS_int)；**真实份额** = (MS_int − MS_noise) / MS_int（= S1′ G-A2 量）。
- 所有门（§1.6）、准入（§1.5）、余量（§1.8）都按本节尺度算；REPORT 另报探针地板口径的 failed / 存活数作诊断。

### 1.3 剪枝与 G-A0
- 规则四条（`prune.rules`）：τ = 100；info_nce ∧ τ = 10（S1 两条不变）；**triplet ∧ euclidean ∧ l2 ∧ τ ≥ 1**；**triplet ∧ τ ≥ 10**。
- **G-A0 改 regret 门**：官方 29 靶目录（mx4_deploy + mx4_mask + uea30 的 `s1_catalog` seed 0）每靶 (冠军 acc − 剪后最好 acc) 的均值 ≤ 0.01；单靶最大差与丢冠军名单一并报（FaceDetection 0.055 预记 limitation）。锁前已知：四条规则下均值 0.004、名义丢 7 冠军。
- 剪掉份额（均匀采样）一并报。

### 1.4 候选
- shared-30：同一 rng 流（`shared_seed`）在剪后空间接受制采样；前 17 条 = S1′ shared-20 里未被新规则剪掉的 17 条（同序，标签复用），后 13 条为新配置（单测锁定这一性质）。
- 试点 24 任务的 random-20：复用 S1′ 目录与标签，剪掉的成员移出候选集（≈ 18–19 条 / 任务）。新任务不采 random（准入只用 shared）。
- 试点集不变：PhonemeSpectra / Cricket / UWaveGestureLibrary（PS 预期被准入 v2 拒，仍作 chance 型对照留在门里）。

### 1.5 源准入 v2（不依赖探针地板；每源 ≥ 4 任务，OA 行 1 / 7 / 3 / 5 = `_a01 _a07 _a03 _a05`，shared-30 × 2 种子）
三条**全部**满足才准入（§0 标定：PS ✗✗、Cricket ✓✓✓、UWave ✓✓✓）：
- (i) 可靠性：各任务前一半配置（按两种子均值）上的 Spearman(seed 0, seed 1) 的均值 ≥ 0.40；
- (ii) 结构：该源任务 × shared 的交互净占比（§1.2）≥ 0.25；
- (iii) 分辨率：各任务前一半配置的 IQR / σ_mean 的中位 ≥ 2。
候选 13 = 旧 9（HW / UniMiB / PS / UWave / Ethanol / Cricket / HMD / SCP2 / SCP1）+ 新 4（PTB-XL / SleepEDF / PAMAP2 / Opportunity，`data/candidates/*.npz`，注册表 `configs/candidate_pool.yaml`）。源池底线 7；不足记 limitation，不再补。原 ADM 的"两种子都存活 ≥ 8"取消（地板依赖）。

### 1.6 门（试点 3 集 × 8 任务）
- **G-A1′**：各集交互净占比 ≥ 0.25，且 ≥ 半数试点集过；
- **G-A2**：真实份额合并 ≥ 0.50 且 ≥ 半数试点集单独 ≥ 0.50（同 S1′，换尺度）。
读数码：G-A1′ ✓ ∧ G-A2 ✓ → **ACTS-S1PP-GO**；G-A1′ ✗ → **ACTS-S1PP-SAME**；G-A1′ ✓ ∧ G-A2 ✗ → **ACTS-S1PP-NOISE**。GO 只表示语料可用，铺不铺 S2 由 §1.8 决定。

### 1.7 第 2 步：整集评估口径对照（第 1 步第三个数触发，先于第 3 步）
- Cricket、UWaveGestureLibrary 各自的**整集**（官方 TRAIN+TEST 合并 npz，前 n_train 行 = 官方 TRAIN）上，S1′ shared-20（含被新规则剪掉的 3 条，为了与切片标签一一对应）× 两种评估：
  - `official`：官方 TRAIN 预训练、TRAIN 拟合标准化、TEST 出 acc（= 部署口径，`39.run_trajectory_cell`）；
  - `cv5`：整集预训练 + 分层 5 折 CV（= S1′ 切片口径，`run_trajectory_cell_cv`）；
  × 种子 {0, 1}；探针 iters = 0 × 3 种子 × 两种口径。= 160 格 + 12 探针（PS 不做：整集 6,668 × 39 类的官方网格撞 480 s 上限，且它是 chance 型）。
- 读数：ρ_eval = Spearman(official 两种子均值, cv5 两种子均值)（每集）；ρ_seed = 同口径两种子 Spearman（噪声天花板）；ρ_slice = Spearman(cv5 整集均值, S1′ 切片 8 任务钉底 z 均值)；各口径赢家的 τ 档。
- **决策 W（预注册）**：两集 ρ_eval 均值 < 0.50 → "口径问题"，第 3 步切片标签改回 **holdout**（切片内 70/30 分层、TRAIN 拟合标准化、eval 出 acc，`ac.task_arrays`），已有 cv5 标签不复用、全部重跑；否则维持 cv5。另记 ρ_slice 均值 < 0.50 → "n/L 效应"（不改口径，留给扩池后当交互检验）。

### 1.8 第 3 步主批与第 4 步决策
- 主批（cv5 维持时）：试点 24 任务 × 13 新 shared × 2 种子 = 624；旧源 6（HW / UniMiB / SCP1 / Ethanol / HMD / SCP2）× (a03 / a05 两新任务 × 30 × 2 + a01 / a07 × 13 新 shared × 2) = 1,032；新源 4 × 4 任务 × 30 × 2 = 960；新任务探针 (12 + 16) × 3 = 84。合计 2,616 格 + 84 探针（≈ 25 GPU-h）。holdout 切换时：全部 shared 重跑 = 1,440 + 1,440 + 960 = 3,840（≈ 37 GPU-h）。
- 第 4 步**余量 H**（诚实：任务 oracle 按 seed 0 在 shared-30 里选、seed 1 评；全局冠军按 seed 0 平均排名在**其他**数据集上选、seed 1 评）：
  - 同域 LODO：全部准入源的任务，留一数据集；H_same = regret@1 均值；
  - 跨域：全局冠军在 HAR + ECG 源上选、UEA 源上评，反向再一遍；H_cross = 两向均值（只作上界）；
  - 都只在准入源上算（chance 型不计）；另报 regret@3（前 3 按 seed 0 选一条）与池大小曲线。
- **决策 H（预注册，用 H_same）**：H ≥ 0.03 → 铺 S2，验收两门 = 比较器兑现 oracle 余量 ≥ 50%（同域 LODO regret 上算）+ SH 效率对比（全局排序 SH、随机 SH）；H < 0.02 → **落点 B**（T-AHC 只作 SH 起跑名单排序器，主贡献 = 全局先验 + top-3 实测 + AutoCTS++ 未做的消融）；0.02 ≤ H < 0.03 → 铺 S2 但只报效率门，任务感知门作探索读数。落点 B 与 2026-09-13 主张不同，是用户本轮的主动改判，须同步改 CLAUDE.md。

### 1.9 不变项
OA 设计与 tier 行序、n_lo = max(80, 7k)、每类 ≥ 5、iters 规则、截点 25 / 50 / 100%、剪枝谓词只经 `acts.yaml`、`--check` 跨机器划分自检；random 复用；探针照跑。

## §2 判读（先于数据）

| 门 / 决策 | 定义 | 不过的含义 / 动作 |
|---|---|---|
| C0 | 计划格 key 唯一、截点齐、划分 md5 跨机器一致、探针齐、shared-30 前 17 = S1′ 未剪 17 | 先修 |
| G-A0 | 官方 29 靶 (冠军 − 剪后最好) 均值 ≤ 0.01 | 撤回对应规则 |
| W | 两集 ρ_eval 均值 < 0.50 → 切片口径改 holdout | — |
| ADM v2 | (i) ≥ 0.40 ∧ (ii) ≥ 0.25 ∧ (iii) ≥ 2 | 该源不入池 |
| G-A1′ | 各集净占比 ≥ 0.25、≥ 半数集过 | SAME |
| G-A2 | 真实份额合并 ≥ 0.50 ∧ ≥ 半数集过 | NOISE |
| H | H_same ≥ 0.03 GO / < 0.02 落点 B / 中间 效率门 | — |

只报不设门：探针地板口径 failed 份额（总 / 按 τ / 按家族）与存活数；配对产量；ρ_seed；ρ_slice；τ 赢家分布；成本。

## §3 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | W：两集 ρ_eval 均值 < 0.50（口径问题） | 45% |
| P2 | UWave 整集 official 赢家 τ = 1 | 65% |
| P3 | 新 shared-30 在试点 24 任务的探针地板 failed ≤ 0.45（S1′ 0.51） | 55% |
| P4 | Cricket 过 ADM v2 | 70% |
| P5 | PhonemeSpectra 不过 ADM v2 | 80% |
| P6 | 外部 4 源过 ADM v2 ≥ 2 | 50% |
| P7 | 源池 ≥ 7 | 45% |
| P8 | G-A1′ ✓ | 65% |
| P9 | G-A2 ✓ | 70% |
| P10 | H_same < 0.02（落点 B） | 60%（≥ 0.03：15%） |
| P11 | 结构源上 shared-30 两种子都超探针地板的中位 ≥ 15：UWave | 70%（Cricket 25%） |

## §4 执行

1. 本文件 + `configs/acts.yaml profiles.s1pp` + `tests/test_acts_s1pp.py`（先红）→ 协议锁 commit。
2. 实现（`extends`、剪枝谓词 gte、regret 门、钉底尺度、准入 v2、配对权重、整集官方划分数组、按格分发执行体、`--plan-w / --analyze-w / --plan-c / --analyze-c`、候选池注册表回退）→ 绿 → 本地冒烟 → commit。
3. 第 2 步：bundle → 服务器 setup（`--check`）→ `PHASE=W` 160 格 + 12 探针（≈ 1.5 h）→ 拉回 → `--analyze-w` → 决策 W → 若改 holdout 则修订 profile（`eval.mode`）并重锁。
4. 第 3 步：`--plan-c` → `PHASE=C` → 拉回 → `--analyze-c`（G-A0 / ADM v2 / G-A1′ / G-A2 / H）→ REPORT → commit → CLAUDE.md / memory。
5. 服务器礼仪同 S1′：nice 10、LOKY_MAX_CPU_COUNT=8、每进程一张空闲卡；跑完拉回即删 upload 大包。

## §5 铁律对照

1（档位只经 space.yaml；剪枝谓词只在 acts.yaml，新增比较运算 gte）；2（iters 规则显式；整集官方划分格 iters 按官方 TRAIN 尺寸）；3（runs.jsonl + probe0.jsonl + resume + verify；S1′ 标签复用）；5（先红；标定值 §0 如实声明）；6（外部源只作源；靶数据只在 sanity 读尺寸；SCP1 若入源仍从靶集移出）；7；8。

## §6 执行台账（跑后填写）
