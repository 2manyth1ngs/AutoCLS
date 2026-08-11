# 决定性实验（第二段）协议锁：工作点造 label + T-CLSC + 两档 E 消融

日期：2026-08-11 ｜ 本机 RTX 4060 Laptop ｜ 判据/划分/δ 公式/预测**先于数据**写死。
用户三项拍板已入协议：① "full"=per-dataset 工作点训练量；② 划分方案 C（任务留出
主判定 + 数据集互换副判定）；③ 两档 shuffle E（任务级 derangement + 域级对调）
**升级为主判据**——2 数据集下 domain-aware 是最可能的假阳性，必须显式区分。

前置：第一段排名可迁移性通过（results/workpoint_rank_transfer/，commit 125716b）：
HW 60↔120 好半区 ρ=0.821 CI[0.63,0.91]；label 语义 = 工作点训练量下的优劣。

## 0. 定位

机制判定实验，**非 zero-shot 证据**（与路线 B 同定位声明）：留出任务与源任务共享
数据集原始数据（共享 TS2Vec fit 池含其所在数据集），只验证"E 是否被用于任务间
分辨"。目标池（UCI-HAR/Epilepsy/FaceDetection）零参与（铁律 6）。
本实验回答一个问题：**28 任务、2 数据集的干净+脏 label 组合下，T-CLSC 是否
task-aware**——答案决定扩池是否有意义（用户预注册的三行解读，见 §6）。

## 1. 数据与口径

- **源池 28 任务**：Handwriting 12（configs/source_pool.yaml，[8,14] 类档）+
  UniMiB-SHAR 16（configs/candidate_pool.yaml，[7,14] 类档），各自注册表切片
  （与 confirm/followup 逐字段同口径）。
- **iters（labels.yaml: workpoint.iters_per_dataset，用户拍板①）**：
  Handwriting 60（20ep，confirm A 层从容坐实）、UniMiB-SHAR 30（5ep，A 踩线）。
  iters 入缓存键（铁律 2/3）；train.yaml 的全局 early/full 口径不动。
- **CLS 20/任务**（route_a 口径）：shared 10 = 探针目录前 10（canonical 复用）；
  random 10/任务 = sample_random_cls(task_id, seed=42)。
- **种子**：canonical (weight, split) = (run_seed_for(tid,key), run_seed_for(tid,'train_eval_split'))。
- **既有记录幂等种入**：workpoint_confirm HW@60（60 条）、workpoint_followup/
  iters_workpoint UniMiB@30（60 条）按键 (task_id, cls_key, iters) 增补 append-only，
  种入前逐条校验 classes 与 split_seed 与当前切片一致，不一致即拒收报数。
- 新 R' 记录带 device_name；OOM 走 guarded_run 不写 null；--resume 全程可续。

## 2. δ 工作点重标定（公式先于数据）

- **噪声探针**：每数据集取任务 00–01 × 10 shared CLS，各加跑 2 个额外 weight seed
  （run_seed_for(tid, f"{key}:noise{r}"), r∈{1,2}；split 仍 canonical），加canonical
  共 k=3。新记录落 noise_probe.jsonl（键含 seed，与主缓存分开）。
- **δ 公式（写死）**：对每 (task, cls) 的 3 个重复 R' 取两两 |ΔR'|（3 个/组，
  2 任务 × 10 CLS = 60 个差值/数据集），
  **δ_dataset = max( P95(差值池), 2 × max_task(1/n_eval) )**，向上取整到 0.005。
- 报告：该 δ 下配对存活率、破坏参与率（strict/loose）、δ±50% 敏感性。
- 标定值回填 configs/labels.yaml workpoint 段（δ 唯一真源原则）后才允许配对。

## 3. 任务嵌入 E

- 共享 TS2Vec 只在 {Handwriting, UniMiB-SHAR} 原始数据池 fit（600 iters、每数据集
  封顶 4096 单变量实例、embed_seed=0），落盘 + fit 池指纹校验（2b 同款）。
- SetTF init 落盘一份（set_encoder_init.pt），28 任务特征 windows_repr+E 逐任务 npz。

## 4. 划分（seed 先于数据写死）

- **留出任务（预注册）**：
  - Handwriting：`np.random.default_rng(0).choice(12, 3, replace=False)` → **06, 07, 08**
  - UniMiB-SHAR：`np.random.default_rng(1).choice(16, 4, replace=False)` → **06, 07, 11, 15**
  - 训练 21 任务（HW 9 + UniMiB 12），测试 7 任务。附注：HW 留出三个全在 confirm
    未覆盖的后 6 任务段，属 rng 巧合，如实记录，无挑选。
- **8 个训练跑**（超参全部沿用 configs/pretrain_tclsc.yaml，lr 1e-4/bs 64/
  200ep/curriculum 40）：
  | # | 训练任务 | 测试任务 | 种子 | 角色 |
  |---|---|---|---|---|
  | R1–R3 | mixed 21 | 留出 7 | 0/1/2 | **主判定**（判据取 3 种子中位数） |
  | R4 | HW 9 | HW 留出 3 | 0 | 用户要求的"仅 HW"诊断 |
  | R5 | UniMiB 12 | UniMiB 留出 4 | 0 | 用户要求的"仅 UniMiB"诊断 |
  | R6 | HW 全 12 | UniMiB 全 16 | 0 | 副判定：数据集互换（exploratory） |
  | R7 | UniMiB 全 16 | HW 全 12 | 0 | 副判定：数据集互换（exploratory） |

## 5. E 消融（主判据；不重训只改测试输入）

每跑四档，shuffle 类各 5 个 assignment 种子（0–4）：

1. **normal**：各任务用自己的 E。
2. **within**（任务级）：**数据集内** derangement——每数据集留出任务各自错位，
   数据集身份不变。杀死任务级信号、保留域级信号。
3. **cross**（域级）：每个留出任务的 E 换成**另一数据集**随机任务的 E（从对方
   全部任务里等概率抽，种子派生）。域级信号被杀死。
4. **zero**：E 置零（辅助档，set_fc 常数支路仍在，不参与判定）。

统计口径：acc 在双向评估对上报告（路线 B 同口径）；显著性用**单向对** McNemar
（normal 对 vs 消融对的不一致格 b/c，单侧 binomtest(b, b+c, 0.5)，方向=消融更差）。
R4 的 within 只有 3 任务、R5 只有 4 任务，derangement 合法（n≥2）但功效低，如实报。

**判定量（写死）**：Δ_within = acc_normal − mean₅(acc_within)；Δ_cross 同理。
**达标线**：Δ ≥ 0.02 **且** 5 个种子的 McNemar 单侧 p<0.05 至少 4 个。

## 6. 解读表（先于数据写死；判据取 R1–R3 中位数种子的结果）

| 情形 | 条件 | 结论与行动 |
|---|---|---|
| **A. task-aware（任务级）** | Δ_within 达标（无论 Δ_cross） | 项目活了，扩池值得；主张保持"task-aware CLS 排序" |
| **B. 仅 domain-aware** | Δ_within 不达标 且 Δ_cross 达标 | task-aware 最低档成立**但主张降级**："domain-aware CLS 排序"。2 数据集下这可能只是二值查表——扩池的论证要按"增加域数"重写，且 B 情形下先跑 R6/R7 看域信号是否可跨域迁移 |
| **C. 零下降** | 两者都不达标 | 走用户预注册分支：看 R4 |
| C1 | C 且 R4 的 Δ_within(HW) 达标 | UniMiB 破坏 label 稀释信号 → 剔 UniMiB 或只用干净 label 重训再判 |
| C2 | C 且 R4 也不达标 | "干净信号缺乏跨任务多样性"坐实，**扩池救不了**；诚实调整主张（task-aware 排序 → 跨任务 CLS 可行性预测），不再烧算力 |

对照数字（路线 B，2026-07-12）：留出 acc 0.866、shuffle Δ=+0.0000（5 种子全同）、
zero Δ=+0.0211。本轮 shuffle 类若仍全同为 0.0000，直接 C。

## 7. 预测（先于数据；吸取"点预测必偏、区间才有效"教训，带宽给足）

1. δ_workpoint：两数据集都落 [0.02, 0.10]。
2. label 产量（δ 过滤后无序对）：HW ∈ [700, 1900]、UniMiB ∈ [350, 1500]。
3. R1–R3 留出配对准确率（normal）∈ [0.72, 0.92]（路线 B 0.866 参照；UniMiB 脏
   label 或拉低）。
4. 情景先验：C2 40%、B 25%、C1 15%、A 20%。
5. UniMiB 侧 label 的 strict 破坏参与 ∈ [40, 80]%（confirm 66% 参照）；HW ∈ [0, 20]%。

## 8. 成本预算（4060 同机锚点；估时配方=同机+内插+保守下限）

| 项 | 点估 | 保守 |
|---|---:|---:|
| R' fill（HW 180×7.1s + UniMiB 260×5.6s） | 45 min | 74 min（p90） |
| 噪声探针 80 条 | 9 min | 14 min |
| TS2Vec fit + 28 任务特征 | 5 min | 15 min |
| T-CLSC 8 跑（路线 B 123s/跑，记录量 ×6 → ~12min/跑上限） | 40 min | 100 min |
| **合计** | **~100 min** | **~200 min**（预算上限 4 h） |

## 9. 工程红线（既有事故教训全部生效）

- .ps1 必须带 BOM（GBK 吞行事故）；长跑用脱离进程（Start-Process）+ 日志轮询；
- 种子写入幂等（append-only 按键增补，收官前重验行数/重复键）；
- OMP_NUM_THREADS=1 + threading 后端；8GB 显存不并发两作业；
- iters 入缓存键；铁律 6 泄漏自检照跑；R4–R7 与消融全部复用已训缓存零新增 R'。
