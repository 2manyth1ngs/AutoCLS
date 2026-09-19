# WP17-ACTS S1 PROTOCOL：切片重构（AutoCTS++ 两维四档）+ 试点语料（3 集 × 8 任务 × 40 条，`ts2vec_official` 口径）

**会话定位**：`docs/AUTOCTS_STRICT_REPRO_PLAN.md` §5 的 S1 行。目标只有一个：在 AutoCTS++ 式任务构造（通道子集四档 × 实例内连续时间窗四档、不切类别、不去重）下，用部署口径的标签**先在 3 个源数据集上试点**，读 G-A（同集任务是否真的分化、分化是否超过种子噪声）；过门才铺全语料（S2），不过门止损并记 limitation。本协议逐条落实方案 §3.6 六项（§1.6 有对照表，缺一不锁）。
产物 `results/acts/s1/`；真源 `configs/acts.yaml`；库 `scripts/acts_common.py` + CLI `scripts/83_acts_corpus.py`（S1 / S2 共用；方案里写的 `83_acts_deploy.py` 顺延为 S4 的 85 号，S3 用 84 号）；单测 `tests/test_acts_s1.py`；pod 三件套 `scripts/pa_pod/acts_{bundle,setup,run}.sh`。旧缓存（mx3_pq / k3 / nb / ev / tclsc_slices …）一律不复用为标签；只在 §1.7 的零 GPU 前置对照里**读**它们。

## §0 已看过的读数（如实声明）

- **旧构造下同集切片的偏好一致性**（SX REPORT §4，6 切片 / shared-10 / R′@290，对半分 Spearman）：HW 0.98 / UM 0.94 / PS 0.76 / UWave 0.85 / Cricket 0.91 / Ethanol 0.14 / HMD 0.31。方案 §4 的 G-A 门槛 0.7 就是对着这组数写的；试点 3 集里 PS 0.76、Cricket 0.91 在旧构造下都高于 0.7，HMD 0.31 本来就低——所以 G-A1 对 HMD 几乎必过，读数要看三集均值与 PS / Cricket 各自是否下来。
- **结构性剪枝的三语料核对**（2026-09-19 问答，方案 §3.6-2）：τ = 100 与 info_nce ∧ τ = 10 是三语料（切片 / UCR 84 / 官方 29）都同意的两条；本会话开头在官方口径 29 靶 × {mx4_deploy 21 + mx4_mask 孪生 + uea30 22} 的 seed-0 目录行上**已重算**：冠军保留 **28/29**（唯一丢失 = HMD 冠军 51b6f52899fe，info_nce τ = 100，acc 0.486，WP3 提名配置）；均匀采样 20,000 条的剪掉份额 **0.294**；`kernel = 0` 的份额 **0.187**（= 3/16 理论值）。G-A0（≥ 90%）因此是"已知会过"的形式门，写进来是为了让 S2 扩空间时同一道门继续生效。
- **历史保真度**（25% → 100% 截点 Spearman）：WP3 官方口径六靶目录 21 条中位 0.71，HAR 0.86；方案 §3.6-5 给的历史带 0.57–0.85。**单跑 R′ 噪声** sd 0.035–0.048（k = 3 探针，legacy 口径）；官方口径两锚家族 seed 0 vs 1 |Δ| 中位 ≤ 0.03（GEO-S C0）。
- **成本锚**（RTX 4090，`configs/uea30.yaml anchor_4090`，全量集单格）：HMD 28.1 s、Cricket 48.2 s、PS 94.7 s（其中评估 45 s）；GEO-S 580 格全量集官方轨迹 5090 4.3 h。
- **未训基线**（`results/baseline_probe`，legacy 口径 iters = 0）：贴顶集 NATOPS 0.96 等；试点三集的官方口径未训基线本协议 §1.4 重测，不沿用旧值。
- **没看过**：任何 AutoCTS++ 式构造（通道档 × 时间窗档）下的 R′；任何 `ts2vec_official` 口径下的切片标签；kernel = 0 规范化后的孪生率实测（§1.7 前置对照里算）。

## §1 设计

### 1.1 任务划分（方案 §3.1；`autocls/data/acts_slicer.py::AutoCTSTierSlicer`，新模块，旧 `slice_pool_tasks` 路径逐位不动）

AutoCTS++ `exps/dataset_slice.py` 的两维 = 变量子集 [3, min(300, N)] × 连续时间窗 [7000, min(25000, T)]，各按 25 / 50 / 75 / 100% 四档分层、每格 ⌈n/16⌉ 个、变量 `random.sample` 不放回、窗起点随机、子集间不去重。分类语料的严格对应：

| 维 | 区间 | 四档 |
|---|---|---|
| 通道子集大小 k | [1, C] | 档 i ∈ 0..3 覆盖 [1 + (C−1)·i/4, 1 + (C−1)·(i+1)/4]（前三档右开、末档闭），候选 = 区间内整数；区间内无整数时取区间中点四舍五入。**C = 3 退化为 {1, 2, 2, 3}**（方案原话），C = 6 → {1,2}/{3}/{4}/{5,6}，C = 10 → {1,2,3}/{4,5}/{6,7}/{8,9,10}，C = 11 → {1,2,3}/{4,5}/{6,7,8}/{9,10,11}。源码公式 `variable_s = l + round((i−gap)(r−l))` 在 C 小时给空区间（C = 3 直接崩），故"论文未给全，按 X 语义补全"：同一分位区间、整数候选、末档闭以保证全通道可达。 |
| 实例内连续时间窗长 w | [50, L] | 同上四档；下限 50 = 固定 encoder 适用边界；起点 t0 ~ U[0, L − w]，**全部实例同一窗**（他们全部变量同一段时间）。L 取数据集**原生长度**（不用注册表 crop_len：Cricket 1197 / HMD 400 / PS 217），因为部署靶就是原生长度。 |
| 类别 | 全部 | 不切类别（他们没有标签集合轴；SX §4：类别子集只改标签集合不改信号）。 |
| 样本 | ≤ 1024 | min(1024, N) 个不放回随机样本；N < 32 报错。PS 6668 → 1024（样本轴只在 PS 上产生差异）；HMD 234 / Cricket 180 全取。 |
| 划分 | 切片内 70 / 30 分层 | `split_task_train_eval(eval_ratio 0.3, seed = run_seed_for(task_id, "train_eval_split"))`（canonical 约定）；预训练 + SVM 拟合在 70%，R′ 在 30%。 |

- **档位顺序**（前缀稳定，S2 扩到 16 / 20 时试点 8 个任务逐字段不变）：`tier_order` = (c0,t0),(3,3),(1,2),(2,1),(0,3),(3,0),(1,1),(2,2),(0,1),(1,0),(2,3),(3,2),(0,2),(2,0),(1,3),(3,1)——任意前缀在两维上尽量均衡（前 4 个每档各一次，前 8 个每档各两次），第 17 个起从头循环。
- **RNG**：每任务独立流 `default_rng([slice.seed, run_seed_for(task_id, "acts_slice")])`，抽取顺序 = 通道数 k → 通道集合 → 窗长 w → 起点 t0 → 样本；改 n_tasks 不改任何已有任务（单测钉死 tasks(8) == tasks(16)[:8]）。
- **记录**：`SlicedTask` 加 `time_window: tuple[int, int] | None = None`（缺省 None = 全长，旧对象相等性 / 构造逐位不变）；`materialize_task` 取 `X[idx][:, t0:t1, channels]`。id = `{name}_a{i:02d}`（`dataset_of` 仍剥到数据集名；与注册表流 `_NN`、扩展流 `_xNN` 不撞）；tag = `c{ci}t{ti}`；pool = 数据集名。
- **数据**：`load_source_dataset(name, registry, crop_len=None)`（TRAIN + TEST 合并，全长）；**每切片在自己的 70% 训练部分上按通道重新标准化后套用到 30%**（= 官方 `scaler: train` 语义；合并集的全局 z-score 与切片级 z-score 是同一仿射族，只是均值方差换成切片训练部分的）。

### 1.2 每任务 2L 候选（方案 §3.2；`acts_common.shared_catalog / random_catalog`）

- **空间** = `configs/space.yaml` 19 字段 + `configs/acts.yaml: normalize.constraints`（§1.3 的 kernel = 0 门控）构成的 `acts_space()`（opt-in：默认 `CLSSpace.from_yaml()` 的采样流 / validate / 编码逐位不变，铁律 1 增补同款）。
- **结构性剪枝**（§3.6-2）`configs/acts.yaml: prune.rules`：只剪 τ = 100 与 loss_type = info_nce ∧ τ = 10；其余（τ = 0.01、none + dot、增强 > 0.8）一律不剪。剪枝在**采样与搜索两处**同走（AutoCTS++ 在搜索前剔除无空间 / 时间算子的候选；标签侧多剪一步是省预算，不改语义）。
- **shared L = 20**：`acts_space.sample_random(rng = default_rng(candidates.shared_seed = 0))` 连采 → validate 过 → 未剪 → 按 cls_key 去重，取前 20；**全部任务、全部数据集、S1 / S2 同一批**（= `sample_arch(use_seed=True)`）。不用探针目录、GGS、SP1、信任域。
- **random L = 20 / 任务**：rng = `default_rng([candidates.random_seed, run_seed_for(task_id, "acts_random")])`，同样 validate / 未剪 / 与 shared 及本任务内去重；逐个接受制 → 前缀稳定。
- 每任务 40 条；试点 24 任务 → 960 条。

### 1.3 规范化第二条约束（§3.6-3；`autocls/space/cls_space.py` 约束引擎泛化）

- 语义（`autocls/contrastive/pair_construction.py::total_contrastive_loss`）：`kernel ∈ {None, 0}` → 单尺度，不池化；此时 `pool` 从不被读、`cross_scale` 的分支从不进入 → kernel = 0 下 pool ∈ {avg, max} × cross_scale ∈ {T, F} 四种写法功能相同。
- 约束写法（yaml）：`when_equals: {kernel: 0}` → `gated_fields: [pool, cross_scale]`，`inactive_values: {pool: null, cross_scale: false}`。cross_scale 的无效档取 **false**（而非 null）：这样 temporal = false 时第一条约束（`when_any: [temporal, cross_scale]`）接着触发，kernel / pool / adj_neighbor → null，与旧空间"只有 instance"的规范形逐位相同，不产生第二种写法。
- 引擎改动：`inactive_overrides` 迭代到不动点（约束可链式触发；旧的单约束一趟即定，返回值逐位不变）；`space_size` 改为"自由字段积 × 受约束字段组合的规范形计数"（对旧 space.yaml 仍 = 49 × 自由字段积，既有单测不变）；构造期不再禁止约束间共享字段。三处同走：采样（`sample_random` 内 `inactive_overrides`）/ `validate`（非规范形判非法）/ 编码（`CLSEncoder.decode` 同函数）。
- 协议报数：均匀采样下规范化前后 distinct key 的比值与"被改写的样本份额"（§1.7 前置对照算，预期 ≈ 0.19 被改写）。

### 1.4 轨迹、标签、失败地板、锚（§3.6-1 / -4；方案 §3.4）

- **轨迹** = `ts2vec_official`（`configs/train.yaml protocols`）：batch 8、lr 1e-3、drop_last、SWA、crop_mode fixed、scaler train、svm_grid ts2vec；iters 按 `iters_rule: ts2vec_size`：切片训练部分 N·T·C ≤ 1e5 → 200，否则 600；截点 25 / 50 / 100%（`39.checkpoints_for`）；seed 0。执行体 = `scripts/39_sh_deploy.py::execute`（`run_trajectory_cell` 一字不改），loader = `acts_common.make_task_loader`（任务 id → 切片数组，契约同 `45.make_loader`：`Xtr / ytr / Xte_train / Xte_split / yte`）；评估超时 480 s 沿用（Windows 冒烟用 43 号的子进程版）。**不另写训练循环。**
- **标签** R′_t(c) = **max over 三截点 acc**（best-of-3 = 他们 best-of-5 epoch）；任务 = 切片，无预算 ctx。
- **失败地板**（§3.6-4）：floor_t = max(多数类占比_t（30% 评估部分）, mean over 3 个 seed 的 **iters = 0 探针** acc_t)，探针 = 同口径随机初始化 encoder（`build_encoder`，seed = run_seed_for(task_id, "probe0:{s}")）编码 + 官方 SVM 网格，落 `probe0.jsonl`（3 × 24 = 72 次评估，秒级）。`error` / NaN / best-of-3 ≤ floor_t → **R′ = floor_t**（同任务内并列最末，flag `failed = true`），不剔除；配对时作必输方（与地板并列的对跳过，沿用 `make_pairs` 平局跳过）。
- **锚**（种子噪声）：shared-20 里**按索引顺序前两个 `aug_max ≤ 0.8`** 的配置（aug_max = 六个增强比率的最大值；这是选噪声锚的确定性规则，不是剪枝——避免锚落在破坏区把噪声估成 0），在 24 任务上各加 seed 1 → 48 格。
- **格数**：960 + 48 = **1,008 条官方轨迹** + 72 次探针评估。`plan.json` 先落盘，`--verify` 对账 key 唯一 / 三截点齐。
- **成本估计**（`--plan` 打印）：单格 ≈ iters × (0.006 + 1.0e-5 · T·C) s + 3 × (0.5 + 0.004 · n_train) s（4090 锚粗拟合），5090 × 0.7；预期 ≈ 4–6 h 单进程、2 进程 ≈ 3 h ≈ $3–5。

### 1.5 读数量（零 GPU，跑完 `--analyze`）

- 每任务 40 条 within-task z（含地板行）；shared-20 画像 = 其 z 子向量。
- **对半分 ρ（G-A1）**：每数据集 8 任务里随机抽 6 → 3 / 3 两半（与 SX §4 的 6 切片口径同 half_size = 3，可直接对照 §0 数字），两半各取画像均值，Spearman；200 次抽样取均值；同时报 4 / 4 版本（不设门）。
- **交互 vs 噪声（G-A2）**：每数据集 8 × 20 的 z 表做二因素分解，MS_int = SS_int / (7 × 19)；噪声 MS_noise = 锚的 (z_seed0 − z_seed1)² / 2 在 24 任务 × 2 锚上的均值（z 用该任务 seed-0 的 40 条均值 / sd）；**真实交互份额** = (MS_int − MS_noise) / MS_int，按 3 集合并（格数加权）并逐集报。
- **保真度表**（§3.6-5，旁读数不设门）：逐任务 Spearman(acc@25%, acc@100%) 中位与四分位；recall@3 = |top-3@25% ∩ top-10@100%| / 3 的均值；残差 z@100% − z@25% 按 aug_max 三组（≤ 0.5 / (0.5, 0.8] / > 0.8）的均值；同表在 §1.7 的两个旧语料上先算作参照带。
- **任务相似度**（描述）：24 × 24 shared-20 Spearman 矩阵，同集 / 异集均值（他们 Table 4：0.75 / 0.40）。
- **失败份额**：shared / random 各自 failed 比例；剪枝后仍失败的份额 = "排破坏"仍占标签的多少。

### 1.6 §3.6 六项落实对照

| §3.6 | 本协议 | 单测 |
|---|---|---|
| 1 部署口径标签 + best-of-3 | §1.4：ts2vec_official、39.execute、切片内 70/30、best-of-3 | `test_build_cells_official_options`、`test_best_of_three_and_floor` |
| 2 结构性剪枝只剪两条 + 冠军保留 ≥ 90% | §1.2、§2 G-A0（`--precheck` 在 pod 前算） | `test_prune_rules_only_two`、`test_champion_retention` |
| 3 kernel = 0 门控 + 孪生率 | §1.3、§1.7 | `test_acts_space_kernel0_canonical`、`test_space_size_enumeration` |
| 4 失败配置固定最差标签 | §1.4 floor | `test_best_of_three_and_floor` |
| 5 试点附保真度表 | §1.5 | `test_fidelity_metrics` |
| 6 零 GPU 前置对照 | §1.7 | `test_champion_retention`、`test_split_half_and_interaction_share` |

### 1.7 零 GPU 前置对照（`83 --precheck`，pod 之前跑，结果写 `precheck.{json,md}` 并回填本节）

1. G-A0 冠军保留率（官方 29 靶 × 目录 seed-0 行，来源 mx4_deploy + mx4_mask + uea30）。
2. 剪掉份额 / kernel = 0 改写份额 / 规范化后 distinct key 比值（均匀采样 20,000 条）。
3. 保真度表参照：官方 29 × 22（25 / 50 / 100%）与 UCR 84 × 20（60 / 120 / 290 ≈ 21 / 41 / 100%）。
4. 旧构造对半分（同 §1.5 定义、half_size 3）：mx3_pq 7 集 6 切片 shared-10 best-of-3——特别是 PS / HMD / Cricket 三集，作为 G-A1 的 before 数。
5. **切片分布 sanity（用户 2026-09-19 要求）**：把切出来的子集（试点 24 + 全设计 8 集 × 16，零 GPU）与 29 个官方靶的 (C, L, n_train, 类别数) 画在同一组散点图上（`slice_sanity.png`，对数轴），并给覆盖表：每个靶在四个轴上是否落在子集云的范围内、到最近子集的 log 距离；落在云外的靶（预期：C ≥ 24 的 EEG / PEMS / DuckDuckGeese、EigenWorms L 17984、FaceDetection n 5890）逐个列出，作为 §8-2 源池决定的输入。**这是"子集分布像不像目标分布"的直观检验，不设门。**

### 1.8 §1.7 实测回填（2026-09-19，零 GPU，协议锁 516076b 之后、任何 R′ 之前；`precheck.{json,md}` / `slice_sanity.{json,md,png}` / `plan.json`）

1. **G-A0**：28/29 = 0.966 ✓（丢 HMD 冠军 51b6f52899fe = WP3 提名配置，info_nce τ = 100，acc 0.486）。
2. 均匀采样剪掉份额 0.294；kernel = 0 规范化改写份额 0.186；受约束五字段规范形 49 → 39（空间 −20.4%）。
3. **保真度参照带**：官方 29 × 目录（25% → 100%）ρ 中位 0.568 [q25 0.426, q75 0.742]、recall@3 0.770、残差 z@100% − z@25% 按 aug_max（≤ 0.5 / (0.5, 0.8] / > 0.8）= +0.325 / −0.104 / +0.049；UCR 84 × 20（60 → 290）0.725 [0.544, 0.785]、0.849、+0.177 / +0.020 / −0.041。读法：低增强配置在早截点被系统性低估（残差为正），试点若同带则新构造没有改变截点保真度。
4. **旧构造对半分**（§1.5 同一定义：half 3、best-of-3、within-task z，mx3_pq 6 切片 × shared-10）：PS **0.895**、Cricket **0.774**、HMD 0.487；其余 HW 0.937 / UWave 0.828 / UM 0.770 / Ethanol 0.537。G-A1 的 before 数 = PS 0.895 / Cricket 0.774（SX §4 的 0.76 / 0.91 是 R′@290 单预算口径，不作门）。
5. **切片 sanity**（用户要求）：子集云（试点 24 + 全设计 128）C [1, 11]、L [50, 1676]、n_train [132, 720]、k [2, 39]；29 靶里 **23 个在任一轴云外**（C 9 / L 9 / n 14 / k 0；只看试点 25：C 10 / L 12 / n 14 / k 8）。C 轴云外 = 全部 C ≥ 12 的靶（EEG/MEG 5 个、PEMS-SF 963、DuckDuckGeese 1345、SAD 13、JapaneseVowels 12）——源池最大 C = 11；L 轴云外 = 5 个 L < 50 的短序列靶（encoder 边界，本来不可评）+ 4 个长靶（EigenWorms 17984 / MotorImagery 3000 / StandWalkJump 2500 / Ethanol 1751 vs 云顶 1676）；**n 轴云外 14 = 8 个 n_train < 132 的小靶（ERing 30 / BasicMotions 40 / StandWalkJump 12 / AtrialFibrillation 15 / DuckDuckGeese 50 / Cricket 108 / UWave 120 / EigenWorms 128）+ 6 个 > 720 的大靶（PenDigits / SAD / FaceDetection / PS / LSST / CT）**。根因：本构造的两维（通道、窗长）都不改样本数，每集样本数是常数（PS 1024 → n_train 714–717、HMD 165、Cricket 132）；AutoCTS++ 的连续时间窗轴在预测语料里**同时就是训练样本数轴**，翻译到分类语料时这一维没有对应物。P7 ✓（C 轴 9 ≥ 8）。这是 §8-2（源池）与"要不要加第三维（样本数四档）"的决定输入；S1 本身按锁定设计跑。
6. **计划**：1,008 格（shared 480 / random 480 / anchor 48）+ probe0 72；iters 600: 840 / 200: 168；锚 = shared[14] 8c482976ce21（aug_max 0.80，info_nce / dot / layer_norm τ 0.1）与 shared[18] e3d737685461（0.80，info_nce / dot / layer_norm τ 0.01）——shared-20 里 aug_max ≤ 0.8 的只有这两条（其余 18 条 0.90–0.95：六个比率各 11 档均匀采样下 P(max ≤ 0.8) = (9/11)^6 ≈ 0.30，与 AutoCTS++ 均匀采样同一性质，接受）；估时 4090 4.9 h / 5090 单进程 ≈ 3.4 h、2 进程 ≈ 2 h ≈ $2–3。
7. **冒烟**（本地 4060，HMD 2 任务 × (2 shared + 1 random + 2 锚 seed 1) × 8 iters + probe0 1 个）：10/10 格三截点齐、probe0 2/2；acc 0.275–0.333 ≈ 多数类 0.304（8 iters 只验管线不验学习）；Windows 评估超时走 43 号 loky 子进程版。

## §2 判读（先于数据）

| 门 | 定义 | 不过的含义 |
|---|---|---|
| C0 | 1,008 格 key 唯一、每格三截点齐（error 行计数并列出）；probe0 72 行齐；试点 8 任务 = 全设计 16 任务前缀（逐字段） | 数据不完整 / 前缀不稳，先修 |
| **G-A0** | 官方 29 靶冠军在剪后空间保留率 ≥ 0.90 | 去掉致丢冠军的那条规则再算（零 GPU，pod 前） |
| **G-A1** | 三集对半分 ρ（half 3）均值 ≤ 0.70，且 PS、Cricket 各自低于 §1.7-4 的旧构造 before 数 | 新构造仍造不出集内差异 → 任务轴回到"数据集级"，S2 不铺 |
| **G-A2** | 真实交互份额（3 集合并）≥ 0.50，且 ≥ 2/3 集单独 ≥ 0.50 | 集内差异是种子噪声，不是结构 → 需 k > 1 或换构造，S2 不铺 |

**读数码**：G-A0 ✓ ∧ G-A1 ✓ ∧ G-A2 ✓ → **ACTS-S1-GO**（铺 S2）；G-A1 ✗ → **ACTS-S1-SAME**（记 limitation：两维四档在分类语料上不产生任务级偏好差异）；G-A1 ✓ ∧ G-A2 ✗ → **ACTS-S1-NOISE**；G-A0 ✗ → **ACTS-S1-PRUNE**（先修规则，不进 pod）。
只报不设门：保真度表（与 §1.7-3 参照带同带与否）、孪生率、失败份额、任务相似度、成本、切片 sanity。

## §3 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | G-A1 过（三集均值 ≤ 0.70） | 45% |
| P2 | G-A2 过 | 50% |
| P3 | HMD 对半分最低、Cricket 最高（顺序沿袭旧构造） | 55% |
| P4 | random-20 的 failed 份额 ≥ 0.25（剪枝后仍如此） | 70% |
| P5 | 保真度中位 ρ 落在历史带 [0.57, 0.85] | 70% |
| P6 | 单进程 5090 墙钟 ≤ 6 h | 70% |
| P7 | sanity：29 靶中 ≥ 8 个在 C 轴落在子集云外（源池 C ≤ 11） | 80% |

## §4 执行

1. 本文件 + `configs/acts.yaml` + `tests/test_acts_s1.py`（先红）→ 协议锁 commit。
2. 实现：`cls_space` 约束泛化 → `task_slicer.time_window` → `acts_slicer` → `acts_common` → `83_acts_corpus.py`（--slice / --sanity / --precheck / --plan / --run / --probe0 / --verify / --smoke / --analyze）→ 绿 → 本地冒烟（HMD 2 任务 × 3 条 × 8 iters + probe0）→ commit。
3. `--precheck` + `--sanity`（零 GPU）→ 回填 §1.7 → commit。
4. pod（**须请示后开**，memory `wp-stage-reports-and-pod-consent`）：nvidia-smi + CUDA 闸 → 上传 → `acts_setup` → nohup `acts_run`（2 进程按任务估时分片 → verify 不过补跑 → 打包）；3 分钟内确认首格、10 分钟无进度必查；跑完立刻 stop + confirm 3/3。
5. 拉回 → `--analyze` → REPORT → commit → memory + CLAUDE.md 登记。S1 完成前不写 S2 代码。

## §5 铁律对照

1（档位与合法性只经 space.yaml；acts.yaml 只加约束 / 剪枝，旧路径逐位不变，单测金标）；2（iters 按协议规则显式落盘、三截点同一轨迹）；3（runs.jsonl + probe0.jsonl + resume + verify）；5（先红；补全处注释"论文未给全，按 X 语义补全"）；6（试点三集只作源；LODO 在 S3；靶数据只在 §1.7-5 读尺寸）；7（比较器 / 空间 / 切片核心不引入数据集概念：acts_slicer 只吃 y / C / L）；8（R′ 优劣仍经 is_better）。

## §6 执行台账（跑后填写）
