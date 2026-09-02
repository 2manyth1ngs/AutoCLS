# MX-3-PQ 协议锁：AutoCTS++ 同构的多源池 zero-shot CLS 搜索（任务 = 切片 × 预算）

日期：2026-09-02 ｜ 判据 / 划分 / 臂 / 阈值 / 预测**先于数据**写死 ｜ 执行：R' 阶段 RunPod 5090（32 GB），比较器阶段同 pod ｜
脚本：`scripts/32_pq_generate_rprime.py`（R'）、`scripts/33_pq_comparator_loo.py`（折特征 + 比较器 + 消融 + 判读）、`scripts/34_pq_search.py`（搜索）；公共 `scripts/pq_common.py` ｜
计划：`C:\Users\29466\.claude\plans\autocts-e-autocts-autocts-e-autocts-refactored-moler.md`（用户拍板 2026-09-02） ｜ 管线简报：https://claude.ai/code/artifact/92c20ca7-b216-4990-92cc-8934a7fa043f

## 0. 定位与冲突声明

- 第一次在 **8 个源数据集** 上训练 T-CLSC + E 并做 shuffle 消融；此前所有 E 判负（M3、C2、实验 1/2）都在 1–2 个数据集上。
- 对标 AutoCTS++（`reference/AutoCTS_plusplus`、`reference/AutoCTS++.pdf`）：任务 = 数据子集 × 设定；20 shared 架构跑遍所有任务；SetTF 端到端。
  设定 P/Q 的对比学习对应物 = **预训练预算 b ∈ {60, 120, 290}**；原文 P/Q 经窗口长度隐式进 E，我们的预算显式作为任务设定向量拼入
  （铁律 5 注释：论文未给全，按 X 语义补全）。原文**没有** shuffle/zero-E 消融，本实验比原文严格。
- 与 CLAUDE.md "细粒度 task-aware 排序判死勿复活"的关系：不复活切片级协议；测的是**数据集级 + 预算设定级**。

## 1. 问题

| 问 | 内容 | 检验 |
|---|---|---|
| Q1（决定性） | 未见数据集上比较器是否用到数据集级 E | 目标配对上 cross（E 换成训练数据集随机切片的 E，b 不变），7 折 |
| Q2 | 是否用到预算描述符 b | budget（切片内 b 错位） |
| Q3 | E 在 b 之外有无增量 | A0（E⊕b）vs A6（仅 b）normal acc，7 折配对 Wilcoxon |
| Q4 | label 里有没有数据集级信号 | S2 切分上 one-hot 数据集身份 oracle（A8）及其 cross |
| Q5 | AutoCTS++ 式 E vs 铁律 6 E | A1 vs A0 |
| Q6 | 端到端 | 126 目录实例 regret（D1 vs random、D2 vs A6）；进化搜索 top-3 真评估（D3） |

## 2. 数据与口径

- **数据集 8**（源池 v2 定稿）：Handwriting、UniMiB-SHAR、PhonemeSpectra、UWaveGestureLibrary、EthanolConcentration、Cricket、HandMovementDirection（7 个可评，轮流当目标）+ SelfRegulationSCP2（低权，仅训练）。
- **切片**：各数据集注册表前 6 个切片（`slice_source_tasks(...)[1][:6]`，SCP2 = 3）→ 45 切片；定义与既有缓存一致（`_00/_01` 对 stage1 tasks_meta 校验 classes）。
- **CLS 20/切片**：10 shared（`results/route_a_probe/probe_cls_catalog.json` 全部 10 条）+ 10 random（`sample_random_cls(task_id, seed=42)`）+ GGS 附加。
- **预算**：一条轨迹 bs 128 / lr 1e-3 到 290 iters，截点 60 / 120 / 290 各做一次下游评估（max-pool → SVM-RBF C∈{1,10,100} cv 3）。评估不消耗训练随机流。
- **种子**：weight_seed = run_seed_for(切片, cls_key)，split_seed = run_seed_for(切片, "train_eval_split")；k=1 canonical。
- **决定：900 条全部重跑**于同一硬件（旧 @290 行改作复现校验，§8）。
- **噪声探针**：每数据集切片 00/01 × 10 shared × reps {1,2}（run_seed_for(切片, f"{key}:noise{rep}")）→ 320 轨迹。
- **δ(数据集, b)** = max(P95 同 CLS 重复对 |ΔR'|, 2×量子) 上取整到 0.005（2c 公式）；副读数 δ_nd（剔 strict 破坏 cell）。回填 `configs/labels.yaml: pq.delta_per_dataset_budget`。
- **iters=0 基线**：`results/baseline_probe` / `results/candidate_survey/baseline`（strict 带 = 均值 − 2·max(σ, 量子)）。
- **配对**：只在同一 (切片, b) 内两两，|ΔR'| ≥ δ 才给 y（铁律 8 via is_better），双向出样，每 epoch 动态重配；curriculum 前 40 epoch 只用 shared–shared。

## 3. 划分与臂

**S1** = 7 折 LOO：训练 = 其余 6 + SCP2（三预算全用）；测试 = 目标 6 切片 × 3 预算。每折 TS2Vec fit 池、label、δ 均不含目标（铁律 6，meta 断言）。
**S2**（仅 oracle 上界）= 8 数据集切片 00–03 训 / 04–05 测（SCP2 00–01 / 02），单折，TS2Vec fit 于全 8。

| 臂 | 任务支路 | E 来源 | 种子 × 折 |
|---|---|---|---|
| A0 `main` | SetTF(E) ⊕ onehot(b) → set_fc | 折内 TS2Vec | 3 × 7 |
| A1 `autocts_style` | 同 A0 | 每数据集自身无标签数据 fit（含目标；non-compliant，单独报） | 1 × 7 |
| A2 `wo_ts2vec` | 同 A0 | 未训练随机 TS2Vec | 1 × {HW, UM, Cricket} |
| A3 `wo_settf` | 均值池化 ⊕ b | 折内 TS2Vec | 1 × 7 |
| A4 `wo_shared` | A0 | 折内 TS2Vec，只用 random-10 训 | 1 × {HW, UM, Cricket} |
| A5 `cls_only` | 无 | — | 1 × 7 |
| A6 `cls_b` | 仅 onehot(b) | — | 1 × 7 |
| A7 `soft` | A0 + 软 label（Student-t per (数据集, b)） | 折内 TS2Vec | 1 × {HW, UM, Cricket} |
| A8 `onehot_oracle` | onehot(数据集) ⊕ onehot(b) | —（S2） | 1 × 1 |
| A0-S2 `main_s2` | A0 | 全 8 TS2Vec（S2） | 1 × 1 |

超参：`configs/pretrain_tclsc.yaml`（lr 1e-4、bs 64、200 epoch、curriculum 40、embedding 128、fz 128）。E：折内单变量 TS2Vec 600 iters、bs 32、cap 4096；每切片 `settf_windows`（默认 100）个窗口 × T 封顶 152 × 256；SetTF nz 256 联合训练；`set_encoder_init.pt` 全臂共用。
**预注册旋钮**：`settf_windows` 100 → 32 当且仅当折 HW 的 A0 s0 单跑 > 25 min（先标定后全臂统一）。

## 4. 消融（不重训，只改测试输入；每档 5 个 assignment 种子）

- `within`：目标内切片错位（同一切片置换作用于全部预算），b 不变——任务级。
- `cross`：E 换成训练数据集随机切片的 E，b 不变——**域级（决定性）**。
- `budget`：切片内 b 错位（3 预算只有 2 个 derangement：种子 0/1 各取一个；种子 ≥2 为随机非恒等排列，逐行标注 is_derangement）——P/Q 对应物。
- `zero_E`、`zero_ctx`。
- 指标：δ 过滤单向对 acc（双向报告）、McNemar 单侧；档位：pass = Δ̄ ≥ 0.02 且 ≥4/5 种子 p<0.05；detectable = Δ̄ ≥ 0.005 且 ≥3；zero = Δ̄ < 0.005 且 ≤1；其余 ambiguous。
- 按预算分层 acc；`robust_pooled_delta`（测试对改用训练数据集 δ 中位数）作稳健读数。

## 5. 判读表（先于数据写死；A0 每折取 3 种子中 within Δ̄ 的中位种子；pooled = 7 折 (acc_normal − acc_cross) 单侧 Wilcoxon）

| 码 | 条件 | 读法 |
|---|---|---|
| **E-ALIVE** | A0 cross 在 ≥5/7 折 pass ∧ pooled p<0.05 ∧ Q3（A0 − A6 ≥ 0，7 折 Wilcoxon p<0.05） | 数据集级 E 成立 → 主张 ②/③ 正翼 |
| **E-DETECT** | A0 cross 在 ≥4/7 折 ≥ detectable，未达 E-ALIVE | 弱数据集级信号，如实报 |
| **LABEL-BLIND** | A0 cross 在 ≥5/7 折 zero ∧ A8 cross zero（S2） | label 在 b 之外无数据集特异信号 → E 线关闭 |
| **E-PATHWAY** | A0 cross 在 ≥5/7 折 zero ∧ A8 cross pass | 信号在、E 通路失职 → 下一步特征侧修复 |
| **B-DEAD** | A0 budget 在 ≥5/7 折 zero | 模型连任务设定都不用 → 先查实现 |
| AMBIGUOUS | 其他 | 列出未满足条件 |

Q6 门：**D1** regret@1(A0) < regret@1(random@1)，126 实例配对单侧 Wilcoxon p<0.05；**D2** regret@1(A0) < regret@1(A6)，同法；**D3** 进化 top-3 最佳 R'@b ≥ 目录 oracle@(切片 00, b) − 0.02 于 ≥11/21 实例。

## 6. 预测（先于数据；带先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | A0 pooled normal acc ∈ [0.70, 0.85] | 70% |
| P2 | A0 cross 档位 zero 于 ≥5/7 折（pass 于 ≥5/7：10%） | 60% |
| P3 | A0 budget 档位 pass 于 ≥5/7 折 | 65% |
| P4 | Q3：A0 − A6 ≥ 0.01 且 p<0.05 | 20% |
| P5 | A1 cross ≥ detectable 于 ≥4/7 折 | 15% |
| P6 | A8 cross pass（S2） | 45% |
| P7 | D1 过 85%；D2 过 20% | — |
| P8 | D3 过 | 35% |
| P9 | δ@290 ≥ δ@120 于 ≥6/8 数据集 | 65% |
| P10 | A4 比 A0 低 ≥0.03 于 ≥5/7（覆盖折内） | 60% |

## 7. 成本（5090 单进程锚点 `results/source_pool_v2`：HW 7.1 s / UM 6.6 / PS 8.5 / UWave 11.3 / Cricket 18.5 / HMD 31.4 / Ethanol 70.5（尾 144）/ SCP2 75–90 per trajectory @290；每截点 +1 s SVM）

| 阶段 | 量 | 串行 | 墙钟 |
|---|---|---:|---|
| 32 主网格 | 900 轨迹 | 7.4 h | ≈ 4 h（Ethanol → SCP2 串行为关键路径） |
| 32 噪声探针 | 320 轨迹 | 3.0 h | ≈ 2 h |
| 32 GGS + 34 进化真评估 | 45 + 126 轨迹 | 1.5 h | ≈ 1 h |
| 33 折特征 | 7 折 + 8 每数据集 | ≈ 1 h | 与 32 重叠 |
| 33 比较器 | 37 SetTF 跑 + ~20 快跑 | 4–12 h | 2 进程 2–6 h |
| 34 搜索/分析 | 126 + 21 实例 | < 0.5 h | — |
| **合计** | ≈ 1 391 轨迹 / 4 173 R' | ≈ 17 h | **13–16 h** |

退路：剔 SCP2 −2.5 h（用户已选保留）。硬件：R' 阶段必须 32 GB（4090 Ethanol/SCP2 6/283 格 OOM 不可恢复）。

## 8. 校验与工程红线

- 单测先行；`--smoke`（2 数据集 × 1 切片 × 2 CLS，截点 4/8/12）本地跑通 32 → 33 → 34 → analyze。
- pod：`scripts/pq_pod/setup.sh`（依赖 pin + torch 指纹 + md5 清单 LF 归一化 + `--no-same-owner`）；同一硬件出全部 R'（regret 同机规则）；OMP_NUM_THREADS=1；Ethanol/SCP2 1 worker 串行；`--resume` 贯穿；整条轨迹三截点原子写入。
- **复现校验（预注册容差）**：新 @290 行 vs 旧行同 (task_id, cls_key, weight_seed)：5090 对 5090（source_pool_v2 的 GGS 45 格、Cricket/Ethanol/HMD 02–05 × 10、SCP2_02 × 10）≥90% 在 0.02 内且配对中位 |Δ| ≤ 0.01；混机 wide_panel（HW/PS/UWave/UM）≥85% 在 0.03 内。不过 → 停下诊断再进比较器阶段。噪声：P95 |ΔR'|@290 vs stage1 k3 同数据集在 ±30%。
- `shared_ts2vec.meta.json` 永不含目标名（代码断言 + `--verify`）。
- 判读只从 `analysis.json: verdict.code` 读；REPORT.md 引本表 + 预测对账；阈值跑完不改。
