# 换源池调研 —— 协议锁（判据/阈值/预测先于任何基线测量）

日期：2026-08-05 ｜ 本机 RTX 4060 ｜ 不上云 ｜ **本文件写死的阈值跑完不调**

## 0. 问题与本次的定位

协议探针（`results/protocol_probe/REPORT.md`，commit 3b8f377）证伪了"改下游评估协议"
这条出路：分类器族是空旋钮（RBF→linear 差 0.003），样本量（few-shot）虽能压低多数
数据集但 AWR 免疫（每类 5 样本、从未训练的随机权重、25 类仍 0.967）。根因判定是
**随机初始化的 10 层膨胀 CNN 在这批 UEA 上已经是个够好的特征提取器**。

剩下的出路是**换数据集**。本调研执行它。

### 0.1 上一轮选错 UEA 的教训 → 本轮的筛选顺序

当初选 UEA 的理由（多领域覆盖、与 TS2Vec/TS-TCC 评测传统一致、有加载工具）都成立，
但**漏了最关键的一条：随机初始化基线是否贴顶**——那条标准是踩坑之后才有的。
本轮该标准是现成的、工具是现成的（`scripts/baseline_probe.py`，685 条 2.5 min），
因此顺序必须是 **先筛后选**：统计初筛 → 基线 → 训练能拉动 → 才谈入池。

### 0.2 目标画像（唯一的正样本）

UCI-HAR 是全项目唯一验证过"CLS 差异可见"的数据集（M1: GGS ACC 0.9372；baseline
TS2Vec 0.930 / TS-TCC 0.904；AutoCL 搜出 0.963——差异清晰可辨）。形态：
**N=10299、L=128、C=9、K=6、人体活动传感器**。UEA 里失败的那些恰恰相反：
样本几百、序列几十、类别 20–39。

**铁律 6 提醒**：UCI-HAR / Epilepsy / FaceDetection 是目标池，本调研**全程不加载、
不测基线**——哪怕只是"验证画像假设"也不行，那是用目标池数据做源池选择的决策。
本文件里 UCI-HAR 的数字全部来自**历史里程碑记录**（M1 验收），不是本轮新测。

---

## 1. 数据口径（预注册；已固化在代码里，先于任何基线测量）

实现：`autocls/data/candidates.py`（21 条单测）+ `scripts/prepare_candidates.py`。

| 项 | 取值 | 理由 |
|---|---|---|
| **窗口长度** | `WINDOW_LEN = 128` 步 @**原生采样率** | = UCI-HAR 的 L（唯一正样本）。按**步**而非按秒对齐：encoder 感受野约束是步数（10 层膨胀 CNN，dilation 到 2^9），与物理时长无关 |
| **窗口步进** | `WINDOW_STRIDE = 64`（50% 重叠） | UCI-HAR 原始协议同 |
| **窗内标签** | 纯标签窗才保留；混合窗丢弃 | HAR 文献通用；跨活动边界窗的标签本身没定义 |
| **null/transient** | 逐步标签置 −1，触及即丢窗 | PAMAP2 的 activityID=0、Opportunity 的 null 不是活动类 |
| **缺失值** | 含 NaN 的窗丢弃，**不插值** | 插值造假值，而被测量恰恰是"表征能学到什么" |
| **连续段** | 按 (受试者, 记录, 标签连续区间) 分段后各自开窗 | 跨段拼接会造出物理上不存在的过渡窗 |
| **剔稀类** | `MIN_CLASS_COUNT = 64`，整类剔除后重编号 0..k-1 | registry `min_samples=32` 的一倍余量；稀类会让类别子集切片抽到不成任务的类 |
| **规模上限** | `MAX_SAMPLES = 30000`，按类分层子采样，seed=0 | 切片每任务只取 1024；30k 足够且 npz <1 GB |
| **标准化** | 按通道 z-score（float64 统计 → float32 落盘） | 与 `uea.py` / `datasets.py` 同口径 |
| **已切好样本的档案** | `segmented: True`，保留原生长度（UniMiB 151 / SleepEDF 3000 / PTB-XL 1000 / UCR） | 档案自身定义了什么是一个样本，重新开窗会破坏其标签语义（如 30 s 睡眠分期 epoch 切成 1.28 s 后标签无意义） |

**这三项（窗口/剔稀类/规模上限）直接决定统计表里的 N 与 K，而初筛判据吃这两个数**，
故写死在驱动里而非各 adapter 里——8 个 adapter 各写一遍必然发散。

---

## 2. 初筛硬条件（步骤 1，全部来自实证，缺一不可）

| # | 条件 | 阈值 | 依据 |
|---|---|---|---|
| **H1** | 序列长度 | **L ≥ 50** | 固定 encoder 适用边界（CLAUDE.md limitation，路线 A 探针实证：PenDigits L=8 的 CLS spread SNR 0.39，比正常长度窄 ~20×） |
| **H2** | 样本量 | **N ≥ 1000** | 推导：R' 噪声 ≈ 0.5/√n_eval，要 ≤ 典型 δ(0.05) 需 **n_eval ≥ 100**；`eval_ratio=0.3` → 任务需 ≥334 样本；类别子集档取约半数类 → N ≥ 1000 是其粗代理。**步骤 4 按各候选实际切片档逐任务核算 n_eval ≥ 100，以那个为准** |
| **H3** | 类别数 | **K ≥ 5**（K=2–4 记 reserve） | 类别子集是任务多样性主力（task_slicer 定稿实测：类别 ≈0.65 强 / 通道 ≈0.92 弱）。K<5 的数据集只能靠通道/样本子集造多样性——UEA 的 2 类 EEG 源（SCP1 只切出 2 任务、SCP2 3 个、MotorImagery 4 个）全部判 inert |
| **H4** | 目标池排除 | UCI-HAR / Epilepsy / FaceDetection 及其派生（UCI HAPT id 341 是 UCI-HAR 扩展版） | 铁律 6 |

**H2 的诚实说明**：N≥1000 这个数字让 Handwriting（N=1000，UEA 里唯一无争议可用的源）
**刚好通过**。这不是巧合——阈值由 n_eval≥100 推导而来，而 Handwriting 的实际任务
n_eval≈127。但需明说：Handwriting 之所以可用，靠的是信号大（best_gain +0.397 ≫ 噪声
0.044），不是噪声低。**H2 是噪声下限的卫生条件，不是充分条件**；真正的闸是 §3/§4。

**基线测量范围 ≠ 准入范围**：K=2–4 的 reserve 候选**照样测基线**（每条 0.2 s，几乎免费），
因为它们能独立回答"贴顶是不是普遍现象"。测了不等于能入池。

---

## 3. 基线判据（步骤 2：随机初始化基线）

口径与 `results/baseline_probe/PROTOCOL.md` §1 **逐项一致**（否则新旧数字不可比）：

- `evaluate_rprime(..., iters=0)` → `pretrain_encoder` 的 while 循环不进入，权重停在随机
  初始化，**CLS 完全不参与**（有单测断言）
- 下游协议 **A**：max-pool 表征 + SVM-RBF（C∈{1,10,100}，cv=3），R'=ACC
  ——即 AutoCL 附录 B.1 原设定（协议探针 §3.1 已证 RBF↔linear 差 0.003，换头无意义）
- **5 个随机初始化种子**，`split_seed` 恒为 canonical（只变初始化不变划分）
- joblib **threading** 后端（Windows 上比 loky 快 ~100× 且逐位同值，有回归测试）

### 3.1 分档阈值（沿用 baseline_probe PROTOCOL §3.1，不新造）

| 档 | baseline | headroom | 依据 |
|---|---|---|---|
| **贴顶** | >0.90 | <0.10 | headroom < 2×典型 δ(0.05)；解耦实验实测常规 CLS 增益 ±0.013 ≈ 0.26δ → 配对几乎全被 δ 过滤弃样 |
| **中等** | 0.70–0.90 | 0.10–0.30 | headroom 2–6δ |
| **离顶** | <0.70 | >0.30 | headroom ≥6δ |

**G1（必要条件）**：`baseline ≤ 0.90` 才进步骤 3。贴顶的直接淘汰，不做任何补救尝试
——加难类别档（baseline_probe §5，7 个数据集降幅一律 0.011–0.040）与换下游协议
（protocol_probe，全表零效应）两条补救路都已被证伪。

---

## 4. "训练能拉动"判据（步骤 3）—— 本轮最重要的一条

**基线探针最贵的一课：headroom 大 ≠ 可用。** MotorImagery / SCP2 / Ethanol / Heartbeat
四个都是低基线但训不动（基线≈随机猜测，290 iters 训完还在原地）。上一版判据只用
`best_gain ≥ 0.05`，把这类放行了；零假设诊断是**事后**补的。本轮**把诊断写进判据**。

### 4.1 测量口径（与 baseline_probe `--fill-trained` 逐字段同口径，保证可比）

- **10 条 shared CLS**（`results/route_a_probe/probe_cls_catalog.json` 前 10）
  **× 前 2 个任务 × 1 种子 = 每候选 20 条**，`iters=290`（= `train.yaml: early_iters`）
- `weight_seed = run_seed_for(task_id, cls_key)`，`split_seed` = canonical（与基线同划分
  → 严格配对相减）

### 4.2 判据（三条**全满足**才算 `usable`）

| # | 条件 | 阈值 |
|---|---|---|
| **T1** | 基线不贴顶 | `baseline ≤ 0.90`（= §3 的 G1） |
| **T2** | 幅度够 | `best_gain = max_cls R' − baseline ≥ 0.05`（= 典型 δ；低于 δ 的增益产不出存活 label） |
| **T3** | **显著且不是选择偏差** | 训练后 R' 跑出随机初始化 ±2σ 带的条数：**超出 ≥ 总条数的 50%**（即 20 条里 ≥10）**且 超出 ≥ 2× 跌破** |

σ 取该任务 5 个基线种子的 std，下限用 ACC 量子 1/n_eval 兜底（与 `summarize_task` 同实现）。

**T3 的阈值怎么定的**：用 UEA 的 16 个已测数据集回标（那是**先验数据**，不是本轮新数据）：

| 数据集 | 超出↑/跌破↓/总 | T3 | 与报告结论对照 |
|---|---|:--:|---|
| Handwriting | 15/4/20 | ✅ | "唯一无争议的" |
| UWaveGestureLibrary | 10/3/20 | ✅ | "站得住" |
| SelfRegulationSCP1 | 5/2/20 | ❌ | "站得住但产量会低"（幅度仅 +0.092）——本判据判否，**从严** |
| PhonemeSpectra | 2/7/20 | ❌ | "判 usable 但诊断不支持" |
| Cricket | 1/8/20 | ❌ | 同上 |
| HandMovementDirection | 1/0/20 | ❌ | "表面增益最大，但只有 1/20 超出自身噪声带" |
| NATOPS | 0/16/20 | ❌ | "训练只会变差" |

即：T3 复现了报告 §6.1 的"站得住"分类，并把边缘的 SCP1 判否。**从严是本轮的立场**
——上一轮的教训全部来自把边缘案例放行。

### 4.3 `inert` 与 `ceiling` 的处置

- `ceiling`（baseline>0.90）：直接出局，不进步骤 3（省算力）
- `inert`（baseline≤0.90 但 T2/T3 不过）：出局，但**要在报告里单列**——"低基线但训不动"
  这个类别的规模本身是结论（若新候选里它占多数，说明问题不在数据集而在固定 encoder）

---

## 5. 源池准入与总量闸（步骤 4）

1. 通过 §4 全部三条 → 候选入源池
2. 逐候选按其 (N, K, C) 定切片档（`n_classes_range` / `planned_tasks`），**核算每任务
   实际 n_eval ≥ 100**（H2 的精确版）
3. **总任务数闸**：`task_count_gate` floor = **100**（AutoCTS++ 用 200）。不足则如实报告
   "候选不够"，**不许降 floor 凑数**——若确实不够，那意味着要重议项目范围
   （降 floor / 或把主张从"task-aware CLS 排序"调整为"跨任务 CLS 可行性预测"）
4. 目标池调整**只报告建议、不擅自改**。当前目标池 UCI-HAR + Epilepsy + FaceDetection：
   UCI-HAR（M1 anchor）与 Epilepsy（AutoCL 原用）保留；若新源池没有 EEG/MEG 支撑，
   FaceDetection 的域距离问题要在报告里点明

---

## 6. 预测（先于数据，跑完对账）

1. **UCR 单变量整体比多变量更贴顶**：UCR 候选里 **≥60% 基线 >0.90**。
   机制：320 维随机卷积 + max-pool 对单通道的信息冗余度更高。
2. **大 N 多受试者的流式 HAR（WISDM / PAMAP2 / MHEALTH / USC-HAD / MotionSense）
   基线落 0.70–0.90 中等档**，即不贴顶也不离顶。机制：128 步窗只覆盖 1.28–6.4 s，
   跨受试者变异大、类内方差高。
3. **类别数多的多传感器数据集（Opportunity 5 类 / REALDISP 33 类）基线 <0.70。**
4. **SleepEDF 基线 <0.70 但判 `inert`**（训不动）——UEA 上 EEG/MEG 域 5 个数据集
   全军覆没（MotorImagery/SCP1/SCP2/FingerMovements/HandMovementDirection 无一 usable）。
5. **至少 3 个候选同时过 T1+T2+T3。** 若不足 3 个 → 换数据集这条路也走不通，
   必须重议项目范围（这是本调研可能得到的**否定结论**，须如实报告）。

预测 5 是本调研的**核心风险**：若它错，项目要改主张而不是继续换数据。

---

## 7. 成本模型（步骤 2/3 的算力估算口径）

实测锚点（本机 4060，threading 后端）：

| 环节 | 实测 | 出处 |
|---|---|---|
| 基线 `iters=0` | 685 条 / 2.5 min = **0.22 s/条** | baseline_probe REPORT §7.3 |
| 特征 dump | 683 条 / 1.2 min = 0.11 s/条 | protocol_probe REPORT §6.6 |
| 训练 `iters=290` | 60 条 / 65 min = **65 s/条** | baseline_probe `--fill-trained` |

- 步骤 2 成本 = (候选数 × planned_tasks × 5 种子) × 0.22 s
- 步骤 3 成本 = (**过 G1 的**候选数 × 20 条) × 65 s
- 长序列修正：`iters=290` 的耗时随 L 近似线性（encoder 前向），`iters=0` 的耗时由
  SVM 主导、几乎与 L 无关。SleepEDF（L=3000）按 **×3** 计，PTB-XL（L=1000）按 ×1.5 计。
