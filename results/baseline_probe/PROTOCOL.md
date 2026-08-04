# 随机初始化基线探针 —— 协议锁（先于任何新数据）

日期：2026-08-04 ｜ 本机 4060 ｜ 不上云 ｜ 判据与阈值写死于本文件，跑完不调

## 0. 问题

解耦实验（`results/decouple_random_cls/REPORT.md`）证明：AWR/CT/ERing/NATOPS 的
**随机初始化基线贴顶**（0.90–0.99），常规 CLS 训练增益只有 ±0.013 → R' 测不出学习
质量，只能测破坏程度。本探针把"哪些数据集能用"从有争议的统计判定（SNR 闸 / gate v2，
两者都对探针 CLS 构成敏感，Pearson 0.72–0.75），换成**有机制解释的客观量**：

> **基线离顶的距离 = 学习质量的可测空间上界。**

## 1. 口径（与阶段1 严格一致，逐项已验证）

| 项 | 取值 | 一致性依据 |
|---|---|---|
| encoder | `build_encoder(input_dims=len(task.channels))`，`configs/encoder.yaml` 默认 10 层膨胀 CNN | 与 `evaluate_rprime` 同一行代码 |
| 训练 | `pretrain_encoder(..., max_iters=0)` → `while step < 0` 不进循环，权重停在随机初始化 | `cl_trainer.py:65` |
| CLS | **iters=0 时完全不参与**（增强/损失都在循环体内）→ 传入哪条 CLS 不影响结果，测试断言 | `tests/test_baseline_probe.py` |
| 输入处理 | `load_source_dataset`（TRAIN+TEST 合并 + 注册表 crop）→ `materialize_task` | 与 step0/2b 同一入口 |
| 任务切分 | `slice_source_tasks(entry, registry, y)`，注册表默认档，切**全部 planned_tasks** | step0 与 2b 唯一共用切片入口 |
| train/eval 划分 | `split_task_train_eval(..., eval_ratio=0.3, seed=canonical)` | `labels.yaml: eval_ratio` |
| 下游评估 | max-pool 表征 + SVM-RBF（C∈{1,10,100}，cv=3），R'=ACC | `downstream/linear_eval.py`，与阶段1 同函数 |
| 调用路径 | 直接调 `autocls.labeling.evaluate_rprime(..., iters=0)` | 不 fork 第二份评估逻辑 |

**种子口径**：`split_seed` 恒为 canonical `run_seed_for(task_id, "train_eval_split")`
——与阶段1 canonical R' 用的是**同一个划分**，故基线是阶段1 R' 的严格零训练对照；
`weight_seed = run_seed_for(task_id, f"baseline:{s}")`，s=0..4 共 **5 个随机初始化**。
只变初始化不变划分：本探针要测的是"随机表征能拿多少分"，划分方差会混进来。

## 2. 测哪些

- **源池 13 个全测**（含 gate v2 判 fail 的 MotorImagery/SCP1/SCP2/HandMovementDirection）
- **reserve 3 个全测**（EthanolConcentration/Heartbeat/FingerMovements）
- **目标池 3 个绝不测**（UCI-HAR/Epilepsy/FaceDetection，铁律 6）——脚本内断言
  `load_source_dataset` 对其 fail-fast

任务数：13 源 121 + 3 reserve 16 = **137 任务 × 5 种子 = 685 条**。

## 3. 判据（写死，先于数据）

### 3.1 三档阈值

| 档 | baseline | headroom = 1−baseline | 理由 |
|---|---|---|---|
| **贴顶** | >0.90 | <0.10 | headroom < 2× 典型 δ（0.05）。实测常规 CLS 增益 ±0.013 ≈ 0.26δ → 配对差值几乎全被 δ 过滤弃样（解耦实验 R-reg 产量 0.02–0.14 实证） |
| **中等** | 0.70–0.90 | 0.10–0.30 | headroom 2–6× δ，有空间但需看是否真被用上 |
| **离顶** | <0.70 | >0.30 | headroom ≥6× δ，常规 CLS 之间足以拉开 δ 以上差距 |

δ 取 `labels.yaml: delta_per_dataset` 的典型值 0.05（跨数据集 0.03–0.08）。
阈值 0.90/0.70 即 headroom = 2δ / 6δ。

### 3.2 headroom 是**上界**，不是可用空间——必须配 realized gain

MotorImagery 这类 2 类数据集 baseline≈0.51（≈随机猜测 0.50），headroom 名义 0.49，
但若 290 iters 训练后 R̄ 仍是 0.51，则这 0.49 一分也没被用上。故同时报：

- **realized gain** = R̄(阶段1 canonical，290 iters) − baseline（同 task_id、同 cls_key、
  同 split 的严格配对；数据来自 `results/stage1_rerun_20260804/.../rprime_cache.jsonl`）
- **best gain** = max_cls R' − baseline（这个任务上最好的 CLS 能拉多少）
- **chance** = 1/n_classes，**lift over chance** = baseline − chance
  （区分"数据集简单"与"表征好"：26 类随机猜测 0.038 而随机表征到 0.85 = 任务本身太易）

### 3.3 结论判定（四象限，写死）

| headroom | realized/best gain | 结论 |
|---|---|---|
| 小（<0.10） | 任意 | **ceiling**：只能产破坏排序 label（已被解耦实验证实） |
| 大（≥0.10） | best gain ≥ 0.05 | **usable**：有可测的学习质量空间 |
| 大（≥0.10） | best gain < 0.05 | **inert**：空间在，但固定 encoder 用不上（伪离顶——低分不等于有信号） |

`best gain` 门槛 0.05 = 典型 δ：低于 δ 的增益产不出存活 label。

## 4. 产出

`results/baseline_probe/`：`baseline_cache.jsonl`（逐条，`--resume`，铁律 3）、
`analysis.json`、`REPORT.md`。交叉表：本探针四象限 × gate v2 三档。

## 5. 预测（先于数据，跑完对账）

1. AWR/CT/ERing/NATOPS 落 **ceiling**（解耦实验已直接测过其中 8 个任务，此处只是
   扩到全部 planned_tasks——若不落 ceiling 说明任务子集选择有强异质性）。
2. Handwriting（"唯一全口径稳的源"，26 类、gate v2 A=0.96/Y=0.81）落 **usable**：
   它是唯一 R̄=0.699 明显离顶的 pass 数据集。
3. MotorImagery/SCP2 落 **inert**（baseline≈chance≈0.5 且 R̄≈0.51–0.53，
   realized gain≈0）——即它们的 fail 不是"噪声大掩盖了信号"，是**根本没有信号**。
4. 若 1–3 全中：源池困境不是"选错数据集"，而是**任务口径**（切片太易 / 固定 encoder
   在难数据集上学不动），出路在换口径而非换数据集。
