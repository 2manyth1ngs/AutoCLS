# WP17-ACTS：AutoCTS++ 严格复现方案（2026-09-19，用户拍板"一切设计完全复现 AutoCTS++"，分会话执行）

**范围**：搜索空间暂不谈（= 现 `configs/space.yaml` 19 字段，铁律 1 不变）；改五处：① 任务划分、② 每任务候选 2L 的采样、③ t_i = TS2Vec 窗口表征集合 + 前向内两级 Set-Transformer、④ 标签不设训练死区、⑤ 目标侧按 Algorithm 2 提名 + top-K 真训 + 验证集选 1。
**依据**：论文 `D:/DevelopDoc/AutoCLS/reference/AutoCTS++.txt`（VLDB J 2024）与源码 `D:/DevelopDoc/AutoCLS/reference/AutoCTS_plusplus/`（本方案逐条标注出处；源码与论文不一致处以源码为准并注明）。
**产物约定**：新语料从零开始，独立目录 `results/acts/`、独立真源 `configs/acts.yaml`，旧缓存（mx3_pq / k3 / nb / ev …）一律不复用；每个会话 = PROTOCOL（§0 声明已看读数）+ 红测 + 协议锁 commit → 实现 → 跑 → REPORT → commit → memory + CLAUDE.md 登记；pod 死规定照旧。

---

## 1. AutoCTS++ 到底怎么做（已核）

| 环节 | AutoCTS++ 做法 | 出处 |
|---|---|---|
| 子任务构造 | 每个源数据集切 `sample_num=20` 个子集：变量子集大小 ∈ [3, min(300, N)]、连续时间窗长 ∈ [7000, min(25000, T)]，两维各按 25/50/75/100% 四档分层（16 格，每格 ⌈20/16⌉=2 个），变量 `random.sample` 不放回、窗起点随机、子集之间不去重允许重叠、邻接矩阵按所选变量重建 | `exps/dataset_slice.py` L27–101 |
| 任务 | T = (D, P, Q)；源侧 11 集 × 100 子集 × 2 设置（P12/Q12, P48/Q48）= 200 任务；靶 7 个未见集 × 4 设置 = 28 任务（P24/Q24、P168/Q1 为未见设置） | 论文 §4.1.1、Table 3 |
| 每任务候选 2L | L=20 个 **shared**（固定 seed 采样，所有任务同一批）+ L=20 个 **random**（每任务 `np.random.seed(None)` 独立采）；`sample_arch`：4 个节点，节点 0 取 1 前驱 + 1 算子，节点 i>0 取 1 前驱 + 2 算子（可重复），7 条边；算子集 5 个（skip / dcc_1 / diff_gcn / trans / s_trans） | `exps/generate_seeds.py` L65–103、`NAS_Net/genotypes.py` |
| 超参节点 | 论文 Table 2 六维 {B,C,H,I,U,δ} 挂 "Hyper" 节点；**发布的源码是纯架构版（AHC，无 Hyper 节点）** | 论文 §3.1.2–3.1.3 vs `ahc_engine.py` |
| 标签 R′ | 每个候选短训 k=5 epoch（Adam 1e-3、wd 1e-4、batch 64、MAE 损失），**R′ = 5 个 epoch 里最好的 valid MAE（best-of-5）** | `generate_seeds.py` L98–103、`random_search.py::load_pretrain_noisy_seeds` L135–139 |
| 配对 | 两种 loader：quadratic = 任务内全部有序对（按 loss 排序，(i,j) 正向 1、反向 0）；linear（动态）= 每 epoch 洗牌后第 i 条与乱序第 i 条配对，`a.loss < b.loss → 1 else 0`。**无死区、无 δ、平局给 0 不丢弃** | `random_search.py::generate_task_pairs` L402–412、`utils.py::AHC_DataLoader_linear.generate_pairs` L1047–1056 |
| 课程 | 论文 Algorithm 1：shared 先、random 按 δ 逐轮加入；发布源码主循环无课程 | 论文 §3.2.4 |
| 任务特征 t_i | **每个任务单独 fit 一个 TS2Vec**（固定 seed 301、TS2Vec 默认 iters 200/600 按数据量、repr 256）→ 编码全序列 → 按 seq_len=P 切窗 → **对 N 个变量取均值**（Eq.10）→ 随机采 100 个窗 → 存 (100, P, 256) | `exps/generate_task_feature.py` L110–149、论文 Eq.9–10 |
| T-AHC | 架构支路：算子 embedding → 4 层 GCN/GIN（d=128）→ 节点均值 → graph_fc(256→128)+ReLU；任务支路：intra SetPool(sabPF, 256) 按时间步聚合每窗 → inter SetPool(sabP, 256) 跨窗 → set_fc(256→128)+ReLU；concat → pred_fc(256→128→1) → sigmoid；BCE；**Set-Transformer 与比较器端到端联合训练** | `ahc_engine.py::AHC` L96–182、论文 Eq.11–21 |
| 训练制度 | Adam lr 1e-4（源码默认；论文写 1e-3 / wd 5e-4）、batch 64、≤100 epoch、以**一个留出源任务**（PEMS08 子集）的 valid loss 早停（源码 patience 3，论文 5）；训练集只用 noisy（5-epoch）标签，clean 100-epoch 标签是死代码 | `random_search.py::main` L484–548；`D:/DevelopDoc/AutoCLS/Debug/autocts_alignment_2026_05_20.md` §2 |
| 零样本搜索 | 论文：先剔除无空间或无时间算子的候选 → 采 Ks=300,000 → Round-Robin 排序取 kp=10 初始种群 → 进化（p1=0.8, p2=0.2）→ Round-Robin 取 top-3；源码：采 `sample_scale`=200,000 → 多线程 `heapq.nlargest(top_k, key=cmp_to_key(compare))` → 分块赢家两两计分 → top-5 | 论文 §3.3 / §4.1.4；`random_search.py::main` L550–622 |
| 目标侧最终 | Algorithm 2：D 划 D_train / D_val → TS2Vec 特征 → 搜索排名 → **top-K 在 D_train 全量训练 → 取 D_val 最高者** → 报 test；5 种子 mean±std | 论文 Algorithm 2、§4.2.2 |
| 实验设计 | 基线 = 5 手工 + 3 自动（各自 grid-search 超参）；消融 = w/o TS2Vec（MLP）、w/o Set-Transformer（均值池化）、w/o shared samples；Ks 限量研究（37.5k–600k，300k 后饱和）；效率（搜索 15–27 GPU 分钟）；任务相似度（Fig.6 嵌入聚类 + Table 4：同 200 架构在 PEMS08 子集 vs METR-LA 子集 Spearman 0.75，vs Solar 子集 0.40/0.49） | 论文 §4.1.3–4.2.6 |

---

## 2. 我们现状 vs AutoCTS++（逐项）

| 项 | 现状 | 判定 |
|---|---|---|
| 切片 | 类别子集 × 通道子集（下限 ⌈C/2⌉）× 样本子集，实例全长；每集 6 | **待改**（① 的核心：通道下限、时间窗、类别） |
| 每任务候选 | 10 shared = 手选探针目录 + 10 random；另有 GGS 锚 | **待改**（② shared 改为固定 seed 均匀采样、L=20） |
| 任务 id | 切片@预算（Handwriting_03@120），ctx 拼预算 one-hot | **待改**（任务 = 切片，标签 best-of-3 截点，无 ctx） |
| t_i 集合 | `windows_repr` (100, ≤152, 256) 逐切片存；共享 TS2Vec（训练折源池 fit）；通道当单变量实例编码后跨通道均值 | 集合形态与跨通道均值 **已对齐**（= Eq.10）；**每任务 TS2Vec 待改**（③）；时间轴池化 152 是显存必需项（记 limitation） |
| Set-Transformer | intra sabPF + inter sabP + set_fc，随比较器联合训练（E+ctx 分支） | **已对齐** |
| 架构支路 | binned one-hot → MLP（关键决策 1；L2 已证 GIN 无增益） | **不可对齐**（CLS 不是 DAG；one-hot→Linear ≡ 每算子一张 embedding 表） |
| 融合头 | concat（原样）/ diff / field | 主臂用 **concat**（原样）；diff/field 只作零成本副臂 |
| 标签 | δ 硬过滤 + 平局丢弃 + 动态配对 + 课程 | **待改**（④ δ=0；动态配对与课程保留） |
| 目标侧 | 先验/比较器名单 + SH6-3-3（截点淘汰，最后留 3） | **待改**（⑤ Ks 采样 + 进化 + Round-Robin top-3 + 全量训 + D_val 选 1） |
| 靶 | 7 折 LODO + 23 靶外 UEA 官方口径 | 保留：23 靶外 = 严格未见集（对应论文 7 靶），7 池内集用 LODO |

---

## 3. 五处改动的设计

### 3.1 ① 任务划分（`autocls/data/task_slicer.py` + `slice_strategies.py`，新增字段不破旧路径）

AutoCTS++ 的两维是"变量子集 × 连续时间窗"，分类语料的严格对应：

| AutoCTS++ | 我们 | 规则 |
|---|---|---|
| 变量子集 [3, min(300,N)] 四档 | 通道子集 [1, C] 四档 | 档 i 的大小从 [c_lo, c_hi) 均匀取，c_lo = 1 + round((i−0.25)(C−1))，同 `dataset_slice.py` 公式；C=3 时四档退化为 {1,2,2,3} 照做 |
| 连续时间窗 [7000, min(25000,T)] 四档 | **实例内连续时间窗** [50, L] 四档 | 下限 50 = encoder 适用边界（CLAUDE.md limitation）；窗起点随机、**全部实例同一窗**（他们全部变量同一段时间） |
| 无类别子集 | **不切类别**（保留全部类别） | 他们没有标签集合轴；类别子集只改标签集合不改信号（SX §4 证据） |
| 每格 2 个、每集 20 | 每格 ⌈n/16⌉ 个、每集 n=16 或 20 | 不去重、允许重叠（同他们）；样本上限 1024（成本，= 他们的时间窗长度上限） |
| 训练/验证划分 | 切片内 70/30 分层（现有） | R′ 在切片验证部分评 |

代码：`SlicedTask` 加 `time_window: tuple[int,int]`（缺省 (0, L) 逐位不变），`materialize_task` 取 `X[idx][:, t0:t1, channels]`；新增 `AutoCTSTierSlicer`（通道档 × 窗长档 × 全类别）；`n_channels_range` 由档位表驱动而非 `default_channel_range`。
源池：现 8 集（HW / UM / PS / UWave / Ethanol / Cricket / HMD / SCP2）。**注意**：三集 C=3，通道轴几乎不产生差异，时间窗轴是它们唯一的子任务轴；PS 11 / HMD 10 / SCP2 7 / Cricket 6 通道轴可用。扩池待拍板（§8）。
风险一句话：旧构造下同集切片偏好对半分 ρ 0.85–0.98（SX §4），新构造能否造出集内差异未知，故 S1 先试点过门再铺全语料。

### 3.2 ② 每任务 2L 候选（`scripts/32` 的 build_plan 换采样源）

- shared L：`space.sample_random(rng_seed=0)` 连采、`validate` 过、按 cls_key 去重，取前 L，**所有任务同一批**（= 他们 `sample_arch(use_seed=True)`）。不用探针目录、不用 GGS、不用 SP1 剪枝、不限信任域。
- random L：每任务 rng = hash(task_id)（他们是 `seed(None)` 不可复现，我们改可复现，语义同）。
- L=20（源码 `sample_num=20`）→ 每任务 40 条；论文口径 200 任务 ≈ 8–10k 条标签。
- 轨迹口径（决定项，见 §8）：推荐 **`ts2vec_official`**（batch 8、iters 200/600 按尺寸、截点 25/50/100%、SWA、scaler train、svm_grid ts2vec）——与部署同配方只差训练量，正是 AutoCTS++ "5 epoch 代理 vs 100 epoch 终训"的关系，且绕开硬约束 (c)（batch 128 vs 8 翻转）；成本与 290@128 同量级（GEO-S 全量集 27 s/格，切片更小）。
- 空间合法性剪枝：只用 `validate`（= 他们"剔除无空间/时间算子"这种结构性剔除的对应物）。
风险一句话：全空间均匀随机里破坏性配置多（源池 65%、random-10 z-bad 25%），标签会被"排破坏"主导——AutoCTS++ 同样如此，接受。

### 3.3 ③ t_i 与 E（`autocls/task_embed/` + `scripts/33 --fit-folds`）

- **每任务 TS2Vec**：在该切片的训练部分上 fit（固定 `embed_seed`，iters 按 TS2Vec 默认 200/600、repr 256、`max_train_length` 3000），不再用训练折共享 TS2Vec。已知风险（一句话）：本项目早期诊断"每任务重训的表征基任意、区分力≈0"（`task_embed/__init__.py` 头注释）；AutoCTS++ 靠固定 seed 从同一初始化出发把这个问题压住，我们照做并把共享 TS2Vec 留作消融臂（论文的 w/o TS2Vec 用 MLP，我们多一档）。
- 窗：分类实例 = 窗（`task_windows` 分发不变）；通道当单变量实例各自编码再跨通道均值（= Eq.10，已对齐）；采 100 窗；时间轴 >152 步池化到 152（显存；记 limitation）。存 `(100, T′, 256)` fp16 npz，逐任务。
- 模型：`TCLSC(task_branch="E+ctx", ctx_dim=0)` = 只吃 E；intra sabPF + inter sabP + set_fc 联合训练（现有）；**concat 头**（原样）。binned MLP 架构支路（不可对齐项，写入 REPORT）。
- 显存：EV1 时 87 切片 × 100 窗单进程 16–19.5 GB；200 任务须做（i）每步对每任务随机子采 32 窗进前向（集合子采样，AutoCTS++ 无此项，记偏差）或（ii）梯度检查点；S3 立项时二选一并冒烟。
- 目标侧 t_target：只用官方 train 划分（Algorithm 2 line 1–2）。

### 3.4 ④ 标签与训练制度（`autocls/comparator/tclsc_engine.py`）

- R′_t(cls) = **max over 三个截点**（best-of-3 = 他们 best-of-5 epoch）；任务 = 切片；不再有预算 ctx。
- 配对：`dynamic_pairs(delta=0)`（每 epoch 洗牌逐位配对 = linear loader；反向对照加）；平局：他们给 0，我们 `make_pairs` 跳过——采用跳过并在 PROTOCOL 注明（平局在连续 acc 上极少）。
- 课程：Algorithm 1（shared 先、random 逐轮加入），现有 curriculum 开关。
- 优化：BCE、Adam lr 1e-4（源码默认，我们现值同）、batch 64、≤100 epoch；**早停 = 留出一个源数据集的全部任务作 valid**（他们留 PEMS08 子集），patience 5（论文）；靶永不参与。
- 评估配对（考卷）：任务内全部有序对，无 δ；真值可用 3 种子均值作副读数（K3 方式），主读数按 AutoCTS++ 单种子。
风险一句话：单跑 R′ 噪声 sd 0.035–0.048，无 δ 后近半配对是掷币；K3 证据说训练标签 k1→k3 只差 0.001，所以预期无 δ 不伤排序但配对 acc 绝对值会低（读数按 regret / 部署 acc，不按配对 acc）。

### 3.5 ⑤ 目标侧（新脚本 `scripts/83_acts_deploy.py`，复用 `autocls/search/evolution.py` 与 `scripts/39` 的 execute/loader）

1. 目标官方 train 划分 → 分层 80/20 = D_train / D_val（Algorithm 2 line 1；决定项 §8）。
2. 目标 TS2Vec 特征（只用 D_train）→ 100 窗。
3. 采样 Ks（默认 20,000；Ks 研究 2k / 10k / 50k）→ Round-Robin（`_pairwise_wins`）→ kp=10 初始种群 → 进化 p1 0.8 / p2 0.2（代数论文未给，用 EV1 的 10 代，PROTOCOL 固定）→ Round-Robin top-3。
4. top-3 各在 D_train 上按 `ts2vec_official` 全量训练 → D_val acc 选 1 → 报 test；3 种子。
5. 同口径对照臂（同一 D_train/D_val、同一全量训练）：任务盲先验前 3、同 Ks 随机 top-3（同成本）、GGS、TS2Vec 论文数；SH6-3-3 主臂在同 val 口径下重跑一次（≈ $2）以便同表。
搜索时间预算：Ks=20k 的 Round-Robin 是 O(Ks²) 不可行——按源码做分块 `heapq.nlargest`（比较器当 cmp key，O(Ks log k)）再对块赢家两两计分；进化阶段种群 10 × 10 代两两计分可行。

---

## 4. 实验设计（对齐论文 §4）与预注册门

- **源 / 靶**：源 = 8 池集（SCP2 只训练）；靶 = 23 靶外 UEA（严格未见，对应论文 7 靶）+ 7 池内集 LODO；全部官方口径。
- **主表**：29 靶 test acc（3 种子 mean±std）：ACTS（本方案）vs 先验 SH6-3-3 / SH10-4-3（同 val 口径重跑）vs 同成本随机 top-3 vs GGS vs TS2Vec 论文数。
- **消融表**（同一语料，零额外标签成本）：w/o E（任务盲比较器同管线）；cross-E（换他靶特征）；zero-E；w/o Set-Transformer（均值池化）；w/o shared samples（只 random）；shared-TS2Vec 代替每任务 TS2Vec；副臂 diff / field 头。
- **Ks 限量研究**：2k / 10k / 20k / 50k。
- **任务相似度**：shared-L 的 R′ 跨任务 Spearman 矩阵（他们 Table 4 同集 0.75 / 异集 0.40）+ E 的集内 / 跨集余弦与聚类图（他们 Fig.6）。
- **门（先于数据写进各 PROTOCOL）**：
  - G-A（S1 试点）：新构造下同集任务 shared-L 画像对半分 ρ ≤ 0.7（旧构造 0.85–0.98），且集内任务对 ρ 的方差超过种子噪声（2 条锚配置 seed 1）；不过门 → 记 limitation，全语料不铺。
  - G-B（S3）：E 不是身份：集内余弦 < 0.9 且 cross-E 使留出任务配对 acc 下降 ≥ 0.02（p<0.05）。
  - G-R1（候选来自空间）：按构造成立（Ks 均匀采样 + 进化，无信任域）。
  - G-R2（部署超任务盲先验）：29 靶配对 ACTS − 先验 SH6-3-3 ≥ +0.005 且 p<0.05；副门 vs 同成本随机 top-3 ≥ +0.01。
  - G-R3（cross 可见）：cross-E / zero-E 部署均值下降 ≥ 0.01 且 p<0.05。
  - 读数码：ACTS-GO（R2 ∧ R3）/ ACTS-RANK-ONLY（G-B ∧ ¬R2）/ ACTS-FLAT（¬G-B）。

---

## 5. 分会话执行计划

| 会话 | 内容 | GPU | 门 |
|---|---|---|---|
| **S1 切片重构 + 试点语料** | `configs/acts.yaml`；SlicedTask.time_window + AutoCTSTierSlicer + 单测；32 号换采样源（shared-L 固定 seed / random-L）+ best-of-3 + `ts2vec_official` 切片口径；试点 3 集（PS / HMD / Cricket）× 8 任务 × 40 条 + 2 锚 seed 1 ≈ 1,000 条 | pod ≈ $5 | G-A |
| **S2 全语料** | 8 集 × 16–20 任务 × 40 条 ≈ 5,100–6,400 条；分两批、两 pod 并行；verify 满格 | pod ≈ $30–40 | 满格 + 噪声锚 |
| **S3 T-AHC 严格训练** | 每任务 TS2Vec + 窗集合（≈ 160–200 次 fit）；concat 主臂 + 全部消融臂；留出源集早停；LODO 7 折 + all-8 | pod ≈ $5–8 | G-B |
| **S4 目标侧部署** | 29 靶：特征 → Ks → 进化 → top-3 全量训 → D_val 选 1，3 种子；对照臂同口径；Ks 研究 | pod ≈ $8–10 | G-R2 / G-R3 |
| **S5 REPORT / 论文表** | 主表、消融表、相似度图、成本表；CLAUDE.md / memory 收口 | 0 | — |

每会话开头先读本文件 + 上一会话 REPORT；S1 完成前不写 S2 代码（G-A 不过则止损）。

---

## 6. 成本估算（5090 $0.99/h；单条官方口径切片轨迹按 15–20 s 估，GEO-S 全量集实测 27 s）

| 项 | 量 | 估 |
|---|---|---|
| S1 试点标签 | ≈ 1,000 条 | ≈ 5 h ≈ $5 |
| S2 全语料 | 5,100–6,400 条 | 28–36 h ≈ $30–40（两 pod 并行 ≈ 1.5 天墙钟） |
| 每任务 TS2Vec | 160–200 次 × ≈ 30 s | ≈ 1.5 h ≈ $1.5 |
| 比较器训练 + 消融 | ≈ 10 臂 × 8 折 | ≈ 4–6 h ≈ $5 |
| 目标侧 | 29 靶 × (特征 + 搜索 + 3 全跑) × 3 种子 + 对照臂 | ≈ 8–10 h ≈ $9 |
| **合计** | | **≈ $50–60**（缩到每集 12 任务 ≈ $40） |

对照：AutoCTS++ 论文口径 200 任务 ≈ 10k 标签；我们 160–200 任务 × 40。

---

## 7. 明确不做的事（严格复现的代价，写进 REPORT 的 limitation）

不用 δ 死区；不用 E_T / E_A / φ / 指纹；不用先验、信任域、SP1 剪枝、input_mask 孪生；不做 SH 淘汰（目标侧改 top-3 全量训 + D_val 选）；不复用旧缓存。已知的三条硬约束里：(a) 标签里的数据集级信号——改由 ① 的构造 + ② 的 shared-L 提供；(b) 低秩——不再约束，交给 T-AHC 学；(c) 口径一致——由 3.2 的 `ts2vec_official` 切片口径解决。历史判负只作风险一句话，不作为不做的理由。

---

## 8. 需要用户拍板的决定

1. **切片轨迹口径**：`ts2vec_official`（推荐，与部署同配方）还是沿用 290@128。
2. **源池**：只用现 8 集，还是扩（候选：UCR 84 单变量集只有时间窗轴；或把 23 靶外 UEA 里若干拿进源池、相应减少靶）。
3. **L**：20（源码）还是 10（省一半）。
4. **目标侧验证集**：官方 train 分层 80/20（严格 Algorithm 2；对照臂同口径重跑）还是沿用现行截点 test-acc 口径。
5. **副臂**：是否保留 diff / field 头与 E_T 作零成本对照（不影响主臂）。
6. **显存策略**（S3）：每步 32 窗子采样（记偏差）还是梯度检查点。
