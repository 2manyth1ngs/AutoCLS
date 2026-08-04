# 下游评估协议探针 —— 协议锁（先于任何新数据）

日期：2026-08-04 ｜ 本机 RTX 4060 ｜ 不上云 ｜ 判据、阈值、预测写死于本文件，跑完不调

上游：`results/baseline_probe/REPORT.md`（贴顶是源池主流，5/16 基线 >0.90；
无争议可用源只剩 Handwriting 12 任务，任务闸 floor 是 100）
＋ `results/decouple_random_cls/REPORT.md`（贴顶数据集上存活 label 的 94.7–100% 涉及破坏性 CLS）

---

## 0. 问题与本探针要回答的事

基线探针的因果链止于一句话：

> 这批 UEA 数据集对「随机卷积特征 + 时间维 max-pool + RBF SVM」这个**下游评估协议**
> 本身就太容易（AWR 25 类、随机猜测 0.040，**零训练**表征拿到 0.990）。

`n_classes_range` 旋钮已被证伪（全类别档只降 0.011–0.040）。本探针测**剩下的那个旋钮**：
在其余口径逐项不变的前提下，只换下游评估协议，能否同时做到

1. 把天花板打下来（**空间大**），且
2. 让训练后的表征真的跑赢随机初始化（**空间真被用上**），且
3. 不破坏当前唯一无争议可用的源（Handwriting）。

三者缺一不可 —— 基线探针最贵的一课就是 MotorImagery：基线 0.511、名义 headroom 0.489，
290 iters 训完还是 0.514，**headroom 一分没用上**。

---

## 1. ⚠️ 事实更正：AutoCL 附录 B.1 **不是**线性协议

`results/baseline_probe/REPORT.md §6.3` 与本次实验的初始提议都写了「线性探针而非 RBF
SVM —— AutoCL 附录 B.1 本身就是线性协议」。**核对原文，这句是误读。**

`reference/AutoCLS.pdf` p.10，B.1 Evaluation Metrics 原文：

> Our evaluation follows [53]. For each dataset, we first pre-train an encoder based on a
> CLS, and then train a downstream model, which is evaluated on the test data.
> **For classification, the downstream model is the SVM classifier with RBF kernel, and the
> evaluation metrics are accuracy (ACC) and F1.** For forecasting, the linear regression
> model with 𝑙2 norm is used, and the evaluation metrics are Mean Squared Error (MSE) and
> Mean Absolute Error (MAE). …

- B.1 里的 "linear" 只属于**预测任务**（linear regression with l2 norm）。
- B.3 里的 "MLPs are linear layers with a softmax activation" 说的是 **controller 网络**，
  与下游评估器无关。
- [53] = TS2Vec（Yue et al.），AutoCL 声明 follow 它；`autocls/downstream/linear_eval.py`
  的模块 docstring 记录的也是同一件事。

**结论：当前实现（`fit_svm`，SVC(kernel="rbf")，C∈{1,10,100}，cv=3）才是照 B.1 实现的。**
换线性探针是**主动偏离所复现的论文**，不是回归原始设定。

### 1.1 那么改动的正当性是什么（重新落地，且独立于结果）

不能再说「回到 AutoCL 原文」。能站住的理由只有两条，两条都先于数据：

1. **它测的量不同，而我们要的正是那个量。** RBF SVM 在 320 维特征上是核方法，能解
   「表征里没有任何被学到的结构」的任务；线性探针测的是**线性可分性**，这是自监督
   表征评估的通行约定（SimCLR/MoCo/BYOL 一脉的 linear probe）。本项目 R' 的语义是
   「这条 CLS 学出的表征有多好」，线性可分性比核可分性更贴近这个语义。
2. **协议是评估器的属性，不是结论的属性。** 阈值、判据、预测全部先于数据写死（§4/§5），
   且主候选与备选一并预注册（§3），不存在跑完挑一个能过的。

### 1.2 偏离的代价（必须在论文里认领，不因它「有效」而免除）

- **R' 不再是 AutoCL 报告的那个指标。** M1 在 HAR 上复现 AutoCL 量级（GGS ACC 0.9372）
  是在 SVM-RBF 下取得的；换协议后所有下游数字必须显式标注协议，M1 的复现声明只对
  协议 A 成立。
- **`configs/labels.yaml: delta_per_dataset` 全表作废**：13 源 + 3 reserve 的 δ 都是在
  SVM-RBF 的 R' 分布上标定的（3 个实测 + 13 个 N/C 阶梯推测）。协议一换必须重标定。
- 已有 260 条 canonical R'（`results/stage1_rerun_20260804/.../rprime_cache.jsonl`）
  与 640 条解耦实验 R' 全部是协议 A 口径，只能作为协议 A 的对照，不能与新协议混用。

---

## 2. 口径（除评估协议外与基线探针逐项一致，否则不可比）

| 项 | 取值 | 与基线探针一致性 |
|---|---|---|
| encoder | `build_encoder(input_dims=len(task.channels))`，10 层膨胀 CNN，hidden 64 / out 320 | 同 `configs/encoder.yaml`，不动 |
| 表征 | encoder 逐时间步输出在**时间维 max-pool** → (N, 320) | **不动**（本探针只换分类器，不换 pooling） |
| 输入处理 | `load_source_dataset`（TRAIN+TEST 合并 + 注册表 crop）→ `materialize_task` | 同一入口 |
| 任务切分 | `slice_source_tasks(entry, registry, y)` 注册表默认档，全部 `planned_tasks` | 同一入口；**不用 `+c` 加难档** |
| train/eval 划分 | `split_task_train_eval(..., eval_ratio=0.3, seed=canonical)` | 同一划分 |
| 基线 | `iters=0`，权重停在随机初始化，CLS 完全不参与 | 同 `pretrain_encoder` 的 `while step < 0` |
| 训练 | `iters=290`（`configs/train.yaml` early 口径，铁律 2） | 同阶段1 canonical |
| 种子 | `split_seed = run_seed_for(task_id,"train_eval_split")`（canonical，不变）；<br>基线 `weight_seed = run_seed_for(task_id, f"baseline:{s}")`, s=0..4；<br>训练 `weight_seed = run_seed_for(task_id, cls_key)` | **逐字沿用**，故新旧协议在同一批权重上对比 |
| 加速 | joblib `threading` 后端、`OMP_NUM_THREADS=1` | 同基线探针（已验证 R'/f1/C 逐位相同） |
| 铁律 6 | 目标池 UCI-HAR / Epilepsy / FaceDetection **全程不加载**，脚本内断言 fail-fast | 同基线探针 |

**唯一变量 = 320 维特征之后的那个分类器（及其训练样本量）。** 表征本身逐位相同。

---

## 3. 协议变体（全部先于数据登记；主/次角色写死）

所有变体消费**同一份 dump 的特征**（§6），故它们看到的表征逐位相同。

| 代号 | 角色 | 定义 |
|---|---|---|
| **A** | **对照（现状 = AutoCL B.1）** | `SVC(kernel="rbf")`，C∈{1,10,100}，`GridSearchCV(cv=3)`，全量 train 特征拟合 |
| **B** | **主候选** | `StandardScaler`(fit on train) → `LogisticRegression(penalty="l2", max_iter=2000)`，C∈{1e-3,1e-2,1e-1,1,1e1,1e2}，`GridSearchCV(cv=3)` |
| **B′** | 归因对照 | `SVC(kernel="linear")`，C∈{1,10,100}，cv=3，**不加 scaler** |
| **C** | **预注册次候选** | few-shot probe：每类抽 k∈{5,10,20} 个 train 样本拟合**协议 A 的分类器**，eval 侧不变；每个 k 用 5 个抽样种子（`run_seed_for(task_id, f"fewshot:{k}:{m}")`），报均值与 std |

设计说明（先于数据）：
- **B 的超参口径**：C 网格以 sklearn 默认 C=1 为中心跨 5 个数量级；cv=3 与 A 对齐；
  `StandardScaler` 是 sklearn 线性模型的常规做法，避免结论变成特征尺度的假象。
  **这套取值一次写死，跑完不调**。
- **B′ 的作用**：B 相对 A 同时改了三件事（核 → 线性、hinge → 交叉熵、加了标准化）。
  B′ 只改「核」这一件，用来把因果归到具体旋钮上。它免费（特征已 dump），故一并跑。
- **C 独立于 B 的机制**：B 换的是分类器族，C 换的是**下游样本量**。见 §5 预测 —— 我
  预计 B 单独不够，理由是理论性的，不是拍脑袋。
- **C 的基线与 B 同在闸门 1 出数**（用户拍板）。特征 dump 一次后，A/B/B′/C 全部变体
  离线评估，边际成本≈0；分两轮反而要多跑一次训练。故闸门 1 的产出是**一张
  16 数据集 × 6 个协议变体的基线表**，不是只有 B 一列。
- **多重比较的防火墙**：B 是主判据。若 **B 不过而 C 过**，只记为**次要发现**，
  且必须在其余 12 个数据集上独立复现后才允许采纳 —— 不得在本探针内改口径。
  同理：C 在闸门 1 出数**不代表它可以顶替 B 通过 G1**，判定表（§4.4）只读 B 那一列。

---

## 4. 判据（写死，先于数据；阈值来源全部独立于本次结果）

### 4.1 测哪些

**闸门 1（天花板，iters=0，无训练）**：16 个候选数据集全部 137 个任务 × 5 种子 = **685 条**。
（成本近似为零，见 §6；顺带直接回答「贴顶数据集数量降到几个」。）

**闸门 2（空间是否被用上，iters=290）**：代表性数据集 × 2 任务 × 10 shared CLS。
**分 2a / 2b 两批（用户拍板）**：MotorImagery 不参与判定却占本闸 55% 算力，推迟到
2a 通过后再补。

| 批 | 数据集 | 角色 | 旧协议基线 | 为什么选它 |
|---|---|---|---:|---|
| **2a** | ArticularyWordRecognition | **天花板判据** | 0.990 | 最极端贴顶（25 类、随机猜测 0.040）。它不掉，任何贴顶数据集都不会掉 |
| **2a** | NATOPS | 变差型 | 0.972 | 贴顶且 `realized_gain` −0.109（20 条里 16 条显著低于随机初始化） |
| **2a** | Handwriting | **对照（不该被破坏）** | 0.555 | 唯一无争议可用源（15↑/4↓，最好 +0.397） |
| **2b** | MotorImagery | 诊断（**不参与判定**） | 0.511 | 学不动型。看更弱的分类器下是否反而显出差异 |

2a = 60 条，2b = 20 条。**判定表（§4.4）只需 2a**；2b 只影响 §5 预测 4 的对账，
故 2a 若判 ❌（无效 / 假解决 / 改坏了）则 2b 不跑，并在报告里注明「预测 4 未测」。

### 4.2 零假设诊断（沿用基线探针 §2.2，这是上次的关键教训）

对每个 (task, 协议)：由 5 个基线种子算 σ，**以 ACC 量子 1/n_eval 兜底**，得 ±2σ 带；
统计训练后 20 条 R' 中 **超出↑ / 跌破↓ / 带内** 的条数。「只看基线降低」不算数。

### 4.3 判据（阈值 + 理由）

| # | 判据 | 阈值 | 理由（独立于本次结果） |
|---|---|---|---|
| **G1** | 天花板打下来 | **AWR 新基线 < 0.70** | 0.70 = 基线探针 §3.1 预注册的「离顶」阈值（headroom > 6δ，δ 典型 0.05）。**逐字沿用，非新造** |
| **G1b** | 贴顶面收窄（读数，非判定） | 16 个里基线 >0.90 的个数（现 5 个） | 0.90 = 同一份 PROTOCOL 的贴顶阈值（headroom < 2δ） |
| **G2** | 空间真被用上 | AWR **与** NATOPS 均满足：↑ ≥ 6/20 **且** ↑ ≥ ↓ | 6/20 = 旧协议下 Handwriting（唯一无争议）15/20 的 40%，取「至少要到无争议样本的四成」；「↑≥↓」排除「只是让训练更容易破坏表征」 |
| **G3** | 对照不被破坏 | Handwriting：↑ ≥ **12/20** 且 best_gain ≥ **+0.10** | 12/20 = 旧协议 15/20 的 80%（换协议必然抬噪声，允许适度下降）；+0.10 = 2× 典型 δ，旧协议实测 +0.397，留 4× 余量 |
| **G4** | 幅度可用 | 过 G2 的数据集 best_gain ≥ **0.05** | 沿用基线探针 §3.3 的 `usable` 门槛（= 典型 δ；低于 δ 的增益产不出存活 label） |

### 4.4 判定表（先写死）

| G1 | G2 | G3 | 结论 | 行动 |
|:--:|:--:|:--:|---|---|
| ✅ | ✅ | ✅ | **协议有效** | 推广到 16 个数据集全量；重标定 per-dataset δ；重跑任务闸 |
| ✅ | ❌ | — | **假解决**：天花板降了但 headroom 用不上（MotorImagery 型） | 不推广。B 不是出路 |
| ❌ | — | — | **无效**：这批 UEA 对任何合理下游协议都太容易 | 如实报告；出路只剩换数据集 / 调整项目主张 |
| ✅ | ✅ | ❌ | **改坏了**：造出假的可测空间却毁了唯一可用的源 | 不推广，如实报告 |

- MotorImagery **不参与判定**：它旧协议下基线≈随机猜测且训完不动。新协议若让它显出
  差异 → 记为新增发现（需独立复现）；若仍不动 → 确认「学不动」结论。
- G1 用 AWR 单点判定是**故意的**：它是最强的贴顶反例，取它作 necessary condition
  比取「4 个的均值」更难被侥幸通过。

---

## 5. 预测（先于数据，跑完逐条对账）

**我预测主候选 B 不足以单独通过 G1。** 机制理由（非事后解释）：

> 时间维 max-pool 后的 320 维随机卷积特征 + 线性头，本身就是**随机特征核近似**
> （Rahimi & Recht 2007 的 random features）。把 RBF SVM 换成线性头并没有移除核，
> 只是把它换成了这 320 个随机特征张成的近似核。因此天花板应当**下降但不崩塌**。
> 真正攻击「随机表征就能解」的旋钮是**下游样本量**（协议 C），因为随机特征核机器的
> 泛化误差随训练样本数下降，而与「表征是否被学过」无关的部分会先被样本量掐死。

| # | 预测 | 判读 |
|---|---|---|
| 1 | AWR 协议 B 基线落在 **0.80–0.95**，即 **G1 不过** | 若 <0.70 则我预测错，B 比理论预期强 |
| 2 | Handwriting 协议 B 基线降到 **0.40–0.50**，且 ↑ 条数仍 ≥12/20（G3 过） | 协议改动不该破坏它 |
| 3 | 协议 C 对贴顶的压制强于 B：**k=5 时 AWR 基线 < 0.85** | 若 C 也压不动 → 见预测 5 |
| 4 | MotorImagery 四个协议下基线均在 0.51 ± 0.03，↑ 条数 ≤ 5/20 | 「学不动」是表征侧的问题，换分类器救不了 |
| 5 | 若 1、3 同时成立（B 不过、C 也只到 0.85） → **根因不在下游协议**，而在
    「10 层膨胀 CNN 的随机初始化在这些数据集上已经是个够好的特征提取器」，
    出路只剩换数据集或调整项目主张 | 这是本探针可能给出的最有价值的否定结论 |

---

## 6. 成本与执行顺序（两道闸，先便宜后贵）

### 6.1 现状调查结论（决定成本量级）

- **表征提取与下游评估在代码上已解耦**：`autocls/downstream/linear_eval.py` 的
  `encode_features()`（@torch.no_grad，产出 (N,320)）与 `fit_svm()` 是两个函数。
- **但表征与权重都没有落盘**：`autocls/labeling.py::evaluate_rprime` 建 encoder →
  `pretrain_encoder` → `evaluate_classification` → 只把 `{acc,f1,svm_c}` 写进缓存，
  encoder 出作用域即销毁。既有 260 条 canonical R' **无法直接换协议重算**。
- **但训练是可复现的**：`torch.manual_seed(weight_seed)` + `pretrain_encoder(seed=weight_seed)`，
  且缓存逐条记录了 `weight_seed`/`split_seed`。同机（4060）同种子重跑得到同一份权重
  → 同一份特征。跨机不保证（`results/gate_5090_20260801`：4060 vs 5090 88/90 逐位相同，
  2 条差 ±1 个样本）。既有 4 个目标数据集的 canonical R' 来自 5090，故**协议 A 复算
  与缓存的比对允许 ±1 个 eval 样本的差**，超出即视为口径错配、停机排查。

**因此：一次重跑 + dump 特征，可一次性喂饱 A/B/B′/C 全部变体，且以后再加协议零成本。**
这是本探针的执行核心 —— 把「换协议」从「重训练」降级为「重评估」。

### 6.2 闸门 1：天花板（iters=0，**不训练**）

685 条，纯 `build_encoder` + `encode_features` + 分类器。
基线探针实测本机 **825 条 / 0.316 s 每条**（含加难档），故：

| 项 | 估算 |
|---|---:|
| 685 条特征 dump（协议无关） | **≈ 3 min** |
| A/B/B′/C×3k×5seed 离线评估（读 .npz，无 GPU；**C 的基线在这一步一并出**） | **≈ 2–5 min** |
| 特征落盘体积（(N_train+N_eval)×320×float32） | ≈ 450 MB（`results/protocol_probe/feats/`，**加 .gitignore**） |
| **闸门 1 合计** | **≈ 10 min** |

**G1 在这一步就能判。若 AWR 新基线 ≥ 0.70 → 直接停，不进闸门 2。**

### 6.3 闸门 2：空间是否被用上（iters=290，**必须重训练**）

80 条 = 4 数据集 × 2 任务 × 10 shared CLS（与阶段1 canonical 同 CLS 同种子）。

5090 上阶段1 实测单条均值：AWR 13.5 s / NATOPS 15.5 s / Handwriting 15.0 s /
MotorImagery **54.0 s** → 5090 合计 20×(13.5+15.5+15.0+54.0) ≈ **33 min**。

本机 4060 与 5090 的倍率**没有干净锚点**（两台机器上没有同一数据集的 290-iters 配对）：
- iters=0 段（纯评估、CPU-bound）实测中位倍率 **1.04**（8 个共同任务，AWR/CT/ERing/NATOPS）；
- iters=30 段（`gate_5090_20260801` 90 条配对）中位 **1.2**，但两个数据集分别是 1.93 / 0.63，
  说明该段被开销支配、当时 pod 有争用，**不可外推到 290 iters**。

故按 **1.2–3× 区间**给：

| 项 | 估算 |
|---|---:|
| **2a**：60 条 @290 iters（AWR+NATOPS+Handwriting，含特征 dump） | **25 – 45 min** |
| **2b**：MotorImagery 20 条（推迟；不参与判定） | ~20 – 55 min |
| 离线多协议评估 | < 1 min |
| **闸门 2a（本次执行范围）** | **≈ 25 – 45 min** |
| 2a+2b 合计（若需要） | ≈ 1 – 1.7 h |

**倍率标定（执行的第一件事，约 5 min）**：先跑 4 条（每数据集 1 条 shared CLS），
比对协议 A 复算值与 `stage1_rerun_20260804` 缓存：
① 差值 ≤ 1 个 eval 样本 → 口径无错配，同时得到真实倍率，据此重排 ETA；
② 差值更大 → 停机排查，不继续。

### 6.4 总账

| 场景 | 墙钟 |
|---|---:|
| B 无效（G1 不过，只跑闸门 1） | **≈ 10 min** |
| **本次计划范围：闸门 1 + 2a** | **≈ 40 – 60 min** |
| 若 2a 全过再补 2b | +20 – 55 min |

参照物：基线探针 `--fill-trained` 60 条 @290 iters 本机实测 **65 min**（Cricket L256 /
PhonemeSpectra L217 / SpokenArabicDigits L93，单条 28–109 s）。本探针的 4 个数据集里
3 个比它们短（L 51–152），只有 MotorImagery（L512、64 通道）更长 —— 与 §6.3 的区间自洽。

---

## 7. 产出

`results/protocol_probe/`：
- `feats/{task_id}__{cls_key}__it{iters}__w{weight_seed}.npz`（train/eval 特征 + 标签，**gitignore**）
- `protocol_cache.jsonl`（逐条 `{task_id, cls_key, iters, protocol, r_prime, f1, hp, ...}`，
  `--resume`，铁律 3）
- `analysis.json`、`REPORT.md`（含 §4.4 判定表填写与 §5 预测逐条对账）

## 8. 已知 limitation（先于数据登记）

1. **本探针只换分类器与下游样本量，不换 pooling、不换表征维度。** §6.3 提到的
   「限制表征维度」旋钮未测。
2. 闸门 2 每数据集只有 2 个任务 × 10 条 shared CLS —— 与基线探针 §8.1 同一局限，
   `best_gain` 是「这 2 个任务上 10 条取最大」，天然向上偏。
3. 10 条 shared CLS 不是随机样本（探针目录，跨 HAR 好坏挑选）；解耦实验证明随机样本
   含破坏性 CLS 更多（65% vs 40%），故 ↑/↓ 计数**低估**破坏占比。
4. 协议 B/B′/C 的超参网格是一次性写死的合理默认值，未做敏感性分析。
5. **偏离 AutoCL B.1 的代价见 §1.2**，无论结论如何都必须在论文里认领。
