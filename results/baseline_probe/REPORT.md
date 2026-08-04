# 随机初始化基线探针 —— 结果

日期：2026-08-04 ｜ 本机 RTX 4060 ｜ 协议锁：`PROTOCOL.md`（判据与预测先于数据）
｜ 数据：`baseline_cache.jsonl`（685 条默认档 + 140 条加难档）、`trained_fill.jsonl`（60 条）
｜ 分析：`scripts/baseline_probe.py --analyze` → `analysis.json`

---

## 0. TL;DR（四句话）

1. **贴顶不是那 4 个数据集的毛病，是源池主流**：16 个候选里 5 个基线 >0.90
   （AWR 0.990 / CT 0.981 / NATOPS 0.972 / ERing 0.948 / **SpokenArabicDigits 0.912**）。
   其中 SpokenArabicDigits 是**探针实测的"可靠骨干"之一**——δ 已填死、从没被怀疑过。
   gate v2 判 pass 的 6 个里有 4 个在这一档。
2. **"离顶"绝大多数是假的**：按预注册判据有 7 个 `usable`，但补上零假设诊断
   （训练后 R' 是否跑出随机初始化自身的 ±2σ 带）后只剩 3 个站得住，**只有
   Handwriting 一个是无争议的**（20 条 CLS 里 15 条显著跑赢随机初始化，最好 +0.397）。
   MotorImagery/SCP2/Ethanol/Heartbeat 的低分是**固定 encoder 学不动**，不是有信号
   被噪声埋了——基线≈随机猜测，训练 290 iters 后还在原地。
3. **加难任务口径救不了贴顶**：把 AWR 从 6–12 类子集切到全 25 类（随机猜测
   0.111→0.040，难度 2.8×），基线只从 0.990 掉到 0.973；7 个数据集降幅一律 0.011–0.040。
   类别档这个旋钮已经拧到头，**"换更难的类别口径"这条出路不成立**。
4. **16 个候选里 13 个的训练平均而言在变差**（`realized_gain` 为负），NATOPS 达 −0.109。
   在没有 headroom 的地方，CLS 之间唯一能拉开的差距就是破坏程度。

---

## 1. 基线表（核心产出）

iters=0（encoder 停在随机初始化，CLS 完全不参与），5 个初始化种子，canonical split，
口径与阶段1 逐项一致（PROTOCOL.md §1）。137 个任务 × 5 = **685 条，零失败零 OOM，
墙钟 2.5 min**。

| 数据集 | 角色 | 任务 | 类别 | L | 域 | 基线均值 | 跨任务 std | 任务 min/中位/max | **离顶** | 随机猜测 | 超猜测 | 档 |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---|
| HandMovementDirection | source | 6 | 4 | 400 | EEG/MEG | 0.443 | 0.084 | 0.357 / 0.423 / 0.568 | **0.557** | 0.375 | 0.068 | 离顶 |
| EthanolConcentration | reserve | 8 | 4 | 1024 | Spectro | 0.508 | 0.133 | 0.317 / 0.491 / 0.679 | **0.492** | 0.406 | 0.102 | 离顶 |
| MotorImagery | source | 4 | 2 | 512 | EEG/MEG | 0.511 | 0.039 | 0.468 / 0.513 / 0.547 | **0.489** | 0.500 | 0.011 | 离顶 |
| PhonemeSpectra | source | 18 | 39 | 217 | Audio | 0.541 | 0.090 | 0.414 / 0.527 / 0.721 | **0.459** | 0.162 | 0.379 | 离顶 |
| SelfRegulationSCP2 | source | 3 | 2 | 1152 | EEG/MEG | 0.547 | 0.026 | 0.518 / 0.556 / 0.567 | **0.453** | 0.500 | 0.047 | 离顶 |
| Handwriting | source | 12 | 26 | 152 | HAR | 0.555 | 0.088 | 0.435 / 0.559 / 0.707 | **0.445** | 0.091 | 0.464 | 离顶 |
| FingerMovements | reserve | 4 | 2 | 50 | EEG/MEG | 0.557 | 0.026 | 0.534 / 0.552 / 0.589 | **0.443** | 0.500 | 0.057 | 离顶 |
| Heartbeat | reserve | 4 | 2 | 405 | Audio | 0.720 | 0.023 | 0.698 / 0.717 / 0.748 | 0.280 | 0.500 | 0.220 | 中等 |
| SelfRegulationSCP1 | source | 2 | 2 | 896 | EEG/MEG | 0.808 | 0.018 | 0.795 / 0.808 / 0.821 | 0.192 | 0.500 | 0.308 | 中等 |
| Cricket | source | 10 | 12 | 256 | HAR | 0.876 | 0.073 | 0.694 / 0.888 / 0.955 | 0.124 | 0.120 | 0.756 | 中等 |
| UWaveGestureLibrary | source | 10 | 8 | 315 | HAR | 0.879 | 0.040 | 0.821 / 0.884 / 0.946 | 0.121 | 0.179 | 0.700 | 中等 |
| SpokenArabicDigits | source | 16 | 10 | 93 | Audio | 0.912 | 0.031 | 0.852 / 0.910 / 0.978 | 0.088 | 0.128 | 0.783 | **贴顶** |
| ERing | source | 6 | 6 | 65 | HAR | 0.948 | 0.043 | 0.873 / 0.949 / 0.996 | 0.052 | 0.233 | 0.715 | **贴顶** |
| NATOPS | source | 10 | 6 | 51 | HAR | 0.972 | 0.022 | 0.942 / 0.975 / 1.000 | 0.028 | 0.238 | 0.734 | **贴顶** |
| CharacterTrajectories | source | 16 | 20 | 182 | Motion | 0.981 | 0.014 | 0.937 / 0.986 / 0.997 | 0.019 | 0.119 | 0.863 | **贴顶** |
| ArticularyWordRecognition | source | 8 | 25 | 144 | Motion | 0.990 | 0.011 | 0.966 / 0.994 / 1.000 | **0.010** | 0.111 | 0.879 | **贴顶** |

（"类别"= 数据集总类数；"随机猜测"= 各任务 1/n_classes 的均值——任务用的是类别**子集**，
故猜测线高于 1/总类数。种子内 std 见 `analysis.json` 的 per-task `baseline_std`，
典型 0.01–0.04。）

### 1.1 任务级分布：同一数据集内部差异很大

任务级 137 个的分档是 **贴顶 58 / 中等 24 / 离顶 55**，均值 0.764、中位 0.865。
数据集均值会掩盖内部结构：

- **Handwriting**：12 个任务从 0.435（13 类）到 0.707（9 类）——按数据集均值算"离顶"，
  但 `Handwriting_02` 单独看已经进中等档。
- **UWaveGestureLibrary**：10 个任务 0.821–0.946，其中 3 个单独看已经贴顶
  （`_04` 0.912、`_06` 0.905、`_08` 0.946）。

含义：**"数据集能不能用"这个粒度本身偏粗**，真正的生产单位是任务。

---

## 2. 分档与阈值理由

| 档 | baseline | headroom | 数量 | 依据 |
|---|---|---|---:|---|
| **贴顶** | >0.90 | <0.10 | 5 | headroom < 2× 典型 δ(0.05)。解耦实验实测常规 CLS 增益 ±0.013 ≈ 0.26δ → 配对差值几乎全被 δ 过滤弃样（R-reg 产量实测 0.02–0.14） |
| **中等** | 0.70–0.90 | 0.10–0.30 | 4 | headroom 2–6× δ |
| **离顶** | <0.70 | >0.30 | 7 | headroom ≥6× δ |

δ 取 `labels.yaml: delta_per_dataset` 的典型值 0.05（跨数据集 0.03–0.08）。阈值 0.90 /
0.70 即 headroom = 2δ / 6δ，**先于数据写死**（PROTOCOL.md §3.1）。

### 2.1 headroom 只是上界——必须配"实际用上多少"

这是本探针最重要的一条。2 类 EEG 数据集 baseline≈0.51≈随机猜测，headroom 名义 0.49，
可 290 iters 训完还是 0.51 —— 这 0.49 一分没用上。所以判据是**四象限**：

| 象限 | 定义 | 数据集 |
|---|---|---|
| **ceiling**（5） | headroom<0.10 | AWR, CT, NATOPS, ERing, SpokenArabicDigits |
| **usable**（7） | headroom≥0.10 且最好 CLS 增益 ≥0.05 | **Handwriting**, UWaveGestureLibrary, SelfRegulationSCP1, Cricket, PhonemeSpectra, HandMovementDirection, FingerMovements |
| **inert**（4） | headroom≥0.10 但最好增益 <0.05 | MotorImagery, SelfRegulationSCP2, EthanolConcentration, Heartbeat |

### 2.2 零假设诊断：7 个 `usable` 里只有 3 个站得住

`best_gain` 是"10 条 CLS 取最大"，天然向上偏。补一个不参与判定的诊断：训练后的 R'
有多少条真的跑出**随机初始化自身的 ±2σ 带**（σ 用 ACC 量子 1/n_eval 兜底）。

| 数据集 | 超出↑ | 跌破↓ | 带内 | 最好增益 | 实得增益 | 读法 |
|---|---:|---:|---:|---:|---:|---|
| **Handwriting** | **15** | 4 | 1 | **+0.397** | **+0.161** | 又显著又大——**唯一无争议的** |
| CharacterTrajectories | 10 | 4 | 6 | +0.016 | −0.004 | 显著但**幅度只有 0.016**（贴顶所致，2σ 带只有 0.0065） |
| UWaveGestureLibrary | 10 | 3 | 7 | +0.077 | +0.012 | 显著，幅度勉强过 δ |
| SpokenArabicDigits | 8 | 4 | 8 | +0.053 | −0.057 | 贴顶；提升与破坏并存，均值为负 |
| ERing | 7 | 4 | 9 | +0.048 | −0.028 | 幅度不到 δ |
| Heartbeat | 7 | 0 | 13 | +0.048 | +0.012 | 稳定小幅提升，卡在门槛下 0.002 |
| SelfRegulationSCP1 | 5 | 2 | 13 | +0.092 | +0.014 | 中等，站得住 |
| MotorImagery | 4 | 1 | 15 | +0.049 | −0.003 | 2 类任务上 0.049 ≈ 6/114 个样本 |
| SelfRegulationSCP2 | 3 | 7 | 10 | +0.044 | −0.013 | 破坏多于提升 |
| **PhonemeSpectra** | **2** | **7** | 11 | +0.070 | −0.042 | **判 usable 但诊断不支持**：跌破的比超出的多 3.5× |
| FingerMovements | 2 | 4 | 14 | +0.055 | −0.000 | 刚过门槛 0.005，判得脆 |
| **Cricket** | **1** | **8** | 11 | +0.093 | −0.059 | **同上**：8 条在破坏，1 条超出 |
| **HandMovementDirection** | **1** | 0 | 19 | +0.169 | +0.007 | **表面增益最大，但只有 1/20 超出它自己的噪声带**（基线 std 0.044–0.114，全表最大） |
| ArticularyWordRecognition | 0 | 6 | 14 | +0.014 | −0.053 | 没有一条 CLS 跑赢随机初始化 |
| EthanolConcentration | 0 | 8 | 12 | +0.033 | −0.049 | 同上，且多数在破坏 |
| **NATOPS** | **0** | **16** | 4 | +0.013 | **−0.109** | **20 条里 16 条显著低于随机初始化**——训练只会变差 |

**必须同时看"显著"和"幅度"**：CT 有 10 条显著超出，但幅度只有 0.016（贴顶时 2σ 带
只有 0.0065，一点点提升就"显著"）；HandMovementDirection 幅度 0.169 看着大，可它自己
的初始化抖动就有 ±0.1。

按预注册规则判定表（§2.1）不因这个诊断改动，但要如实标注可信度：

- **站得住**：Handwriting（15↑/4↓）、UWaveGestureLibrary（10↑/3↓）、SelfRegulationSCP1（5↑/2↓）
- **判得脆**：PhonemeSpectra（2↑/7↓）、Cricket（1↑/8↓）、HandMovementDirection（1↑/0↓）、
  FingerMovements（2↑/4↓）——它们的 `usable` 主要由"10 条取最大"的选择偏差撑着
- **门槛边缘**：Heartbeat 的 `inert` 只差 0.002

### 2.3 训练平均而言在变差

`realized_gain`（10 条 CLS 的均值 − 基线）在 16 个数据集里 **13 个为负**：
NATOPS **−0.109**、Cricket −0.059、SpokenArabicDigits −0.057、AWR −0.053、
EthanolConcentration −0.049、PhonemeSpectra −0.042、ERing −0.028 …
只有 Handwriting(+0.161)、SCP1(+0.014)、UWave(+0.012)、Heartbeat(+0.012)、
HandMovementDirection(+0.007) 为正。

这与解耦实验"随机 CLS 空间 65% 是破坏性配置"直接吻合——**在没有 headroom 的地方，
CLS 之间唯一能拉开的差距就是破坏程度**。

---

## 3. 交叉对比：本探针 × gate v2

| 数据集 | 基线 | 离顶 | 最好增益 | 本探针象限 | gate v2 | v2 A | v2 Y | gate v1 | 阶段1 R̄(290) |
|---|---:|---:|---:|---|---|---:|---:|---|---:|
| HandMovementDirection | 0.443 | 0.557 | +0.169 | usable(脆) | fail_noisy | 1.000 | 0.100 | fail_unstable | 0.425 |
| EthanolConcentration | 0.508 | 0.492 | +0.033 | inert | fail_noisy | 1.000 | 0.133 | fail_mismatch | 0.439 |
| MotorImagery | 0.511 | 0.489 | +0.049 | inert | fail_unreliable | 0.410 | 0.944 | fail_mismatch | 0.514 |
| PhonemeSpectra | 0.541 | 0.459 | +0.070 | usable(脆) | 未过闸（骨干） | — | — | — | — |
| SelfRegulationSCP2 | 0.547 | 0.453 | +0.044 | inert | fail_noisy | 1.000 | 0.044 | fail_mismatch | 0.526 |
| **Handwriting** | 0.555 | 0.445 | **+0.397** | **usable** | **pass** | 0.959 | 0.811 | **pass** | 0.699 |
| FingerMovements | 0.557 | 0.443 | +0.055 | usable(脆) | fail_unreliable | 0.747 | 0.967 | fail_mismatch | 0.546 |
| Heartbeat | 0.720 | 0.280 | +0.048 | inert | fail_noisy | 1.000 | 0.044 | fail_mismatch | 0.726 |
| SelfRegulationSCP1 | 0.808 | 0.192 | +0.092 | usable | fail_noisy | 1.000 | 0.022 | fail_mismatch | 0.830 |
| Cricket | 0.876 | 0.124 | +0.093 | usable(脆) | 未过闸（骨干） | — | — | — | — |
| UWaveGestureLibrary | 0.879 | 0.121 | +0.077 | usable | pass | 1.000 | 0.522 | pass | 0.858 |
| SpokenArabicDigits | 0.912 | 0.088 | +0.053 | **ceiling** | 未过闸（骨干） | — | — | — | — |
| ERing | 0.948 | 0.052 | +0.048 | **ceiling** | pass | 1.000 | 0.422 | fail_unstable | 0.872 |
| NATOPS | 0.972 | 0.028 | +0.013 | **ceiling** | pass | 0.957 | 0.533 | fail_unstable | 0.855 |
| CharacterTrajectories | 0.981 | 0.019 | +0.016 | **ceiling** | pass | 1.000 | 0.400 | fail_unstable | 0.981 |
| ArticularyWordRecognition | 0.990 | 0.010 | +0.014 | **ceiling** | pass | 0.972 | 0.411 | pass | 0.929 |

（"未过闸（骨干）"= `tested=true` 的 3 个探针骨干，从未进 step0 SNR 闸/gate v2。）

### 3.1 用户问的三件事，答案

**① 之前判 fail 的（MotorImagery 等）是不是"可救的"？——基本不是。**

MotorImagery / SCP2 / FingerMovements 三个 2 类 EEG 数据集：基线 0.511 / 0.547 / 0.557，
随机猜测 0.500 —— **随机初始化表征几乎不带类别信息（超猜测 0.011–0.057）**，而 290 iters
训完的 R̄ 是 0.514 / 0.526 / 0.546，**还在原地**。它们的 fail 不是"噪声大掩盖了信号"，
是**根本没有信号**：固定 encoder 在这些任务上学不动。换 δ、加种子、加 CLS 都救不了。
EthanolConcentration（0↑/8↓）与 Heartbeat（7↑但幅度 0.048）同理。

唯一的例外候选是 **HandMovementDirection**：基线 0.443、最好 CLS +0.169、且它是全表
唯一"有超出、零跌破"的数据集。但 20 条里只有 1 条跑出它自己的初始化噪声带（其基线
std 0.044–0.114 是全表最大）。**结论是"存疑，需加种子重测"，不是"可救"。**

**② 之前判 pass 的是不是靠破坏排序撑的？——4/6 是。**

gate v2 的 6 个 pass 里，AWR / CT / NATOPS / ERing 基线 0.948–0.990，本探针判 ceiling。
它们的 label 只能来自"破坏 vs 非破坏"的对比——这与解耦实验在同样这 4 个数据集上的
直接测量（δ* 处存活 label 的 94.7%–100% 涉及破坏性 CLS）完全一致，**而本探针是从
完全不同的角度（根本不训练）独立得到的**。剩下 2 个 pass 是 Handwriting（真 usable）
与 UWaveGestureLibrary（headroom 0.121，最好增益 +0.077 刚过 δ，10↑/3↓ 站得住）。

**③ Handwriting 是不是关键线索？——是，而且是唯一的。**

基线 0.555（12 个任务 0.435–0.707），headroom 0.445，最好 CLS +0.397，实得 +0.161，
20 条里 15 条显著跑赢随机初始化。它是全表**唯一**同时满足"空间大"和"空间真被用上"的
数据集。gate v2 称它"唯一全口径稳的源"是从 δ/产量/正确率角度说的，本探针从机制角度
独立指向同一结论。

**额外发现：SpokenArabicDigits 也贴顶（0.912）。** 它是 `tested=true` 的骨干、δ=0.03
已填死、从未被怀疑。它的 16 个任务此前被当作源池最可靠的一部分。

---

## 4. 与随机猜测基线的关系：三种"低分"要分开

用户提的这个对照把"数据集简单"和"表征好"分开了。加上 realized gain，实际上是**三分**：

| 类型 | 特征 | 数据集 | 含义 |
|---|---|---|---|
| **任务太易** | 超猜测 0.70–0.88，基线贴顶 | AWR(+0.879) / CT(+0.863) / SpokenArabicDigits(+0.783) / Cricket(+0.756) / NATOPS(+0.734) / ERing(+0.715) / UWave(+0.700) | **随机卷积特征 + SVM 就解决了**，不是 encoder 的功劳，更不是 CLS 的功劳 |
| **学不动** | 超猜测 0.011–0.102，训练后不动 | MotorImagery(+0.011) / SCP2(+0.047) / FingerMovements(+0.057) / HandMovementDirection(+0.068) / Ethanol(+0.102) | 低分是**难**，但固定 encoder 在 290 iters 内提取不出信息 |
| **真有学习空间** | 超猜测中等，训练显著拉开 | **Handwriting(+0.464)** | 随机表征已解决一部分，留下的部分能被好 CLS 拿到 |

最极端的对照：**ArticularyWordRecognition 25 类、随机猜测 0.040，随机初始化表征配
SVM 拿到 0.973。** 这不是表征好，是这个任务用任意表征都能解。

PhonemeSpectra 超猜测 +0.379 看着像第三类，但它 2↑/7↓、实得 −0.042 —— 更接近第二类。

---

## 5. 加难任务口径能不能救贴顶？——不能（子实验）

用户 §5 提出的分支："若所有候选都贴顶，出路是换更难的任务口径而非换数据集。"
直接测了：把类别子集档拧到最难（**全类别**，`--n-classes-range 99 99`，任务 id 带 `+c`
档标、与默认档缓存共存不互撞），每数据集 4 个任务 × 5 种子 = 140 条。

| 数据集 | 总类数 | 默认档 | 默认基线 | **最难档基线** | 降幅 | 最难档随机猜测 | 仍贴顶 |
|---|---:|---|---:|---:|---:|---:|:---:|
| ArticularyWordRecognition | 25 | [6,12] | 0.990 | **0.973** | 0.017 | 0.040 | 是 |
| CharacterTrajectories | 20 | [6,10] | 0.981 | **0.964** | 0.017 | 0.050 | 是 |
| ERing | 6 | [3,6] | 0.948 | **0.937** | 0.011 | 0.167 | 是 |
| NATOPS | 6 | [3,6] | 0.972 | **0.932** | 0.040 | 0.167 | 是 |
| SpokenArabicDigits | 10 | [6,10] | 0.912 | **0.877** | 0.034 | 0.100 | 否（中等下沿） |
| UWaveGestureLibrary | 8 | [4,8] | 0.879 | **0.855** | 0.024 | 0.125 | 否 |
| Cricket | 12 | [6,10] | 0.876 | **0.836** | 0.039 | 0.083 | 否 |

**类别档这个旋钮已经拧到头**：7 个数据集降幅一律 0.011–0.040。把 AWR 从 6–12 类子集
加难到全 25 类（猜测线从 0.111 降到 0.040，难度提升 2.8×），基线只掉 0.017，仍是 0.973。
4 个贴顶的里没有一个跌出贴顶档。

含义：**"换更难的类别口径"不是出路**——至少 `n_classes_range` 这个旋钮不是。贴顶的
根因不是"切得太容易"，而是**这些 UEA 数据集对"随机卷积特征 + max-pool + RBF SVM"
这个下游协议本身就太容易**。要造出可测空间，得动**下游评估协议**或换数据集，
而不是类别子集大小。

---

## 6. 结论

### 6.1 哪些数据集在当前口径下有学习质量的可测空间

- **无争议：Handwriting**（1 个）。基线 0.555、headroom 0.445、最好增益 +0.397、
  实得 +0.161、15/20 条显著超出零假设带。
- **站得住但产量会低：UWaveGestureLibrary、SelfRegulationSCP1**（2 个）。显著超出
  10/20 与 5/20，但幅度只有 +0.077 / +0.092，**刚够一个 δ**。
- **判 usable 但诊断不支持：PhonemeSpectra、Cricket、HandMovementDirection、
  FingerMovements**（4 个）。前两个跌破的比超出的多几倍（7↓/2↑、8↓/1↑），
  它们的 `best_gain` 主要是"10 条取最大"的选择偏差。
- **只能产破坏排序：AWR、CT、NATOPS、ERing、SpokenArabicDigits**（5 个）。
- **无信号：MotorImagery、SCP2、EthanolConcentration、Heartbeat**（4 个）。

### 6.2 源池是否应该按基线重排？（建议，未动手改）

**应该，但重排后源池撑不起阶段2。** 按本探针，能产"学习质量 label"的任务数是：

| 口径 | 数据集 | 任务数 |
|---|---|---:|
| 只要无争议的 | Handwriting | **12** |
| + 站得住的 | +UWave(10) +SCP1(2) | **24** |
| + 判得脆的 | +PhonemeSpectra(18) +Cricket(10) +HMD(6) +FingerMovements(4) | **62** |

而 `task_count_gate` 的 floor 是 **100**。gate v2 拿到的 106 里，
AWR(8)+CT(16)+NATOPS(10)+ERing(6)+SpokenArabicDigits(16) = **56 个任务来自贴顶数据集**
——那 56 个产的是破坏排序 label。

所以建议是：**不要按基线重排源池然后照常开跑**，重排后的池子不够。真正的选择是 §6.3。

### 6.3 根因与出路

数据支持的因果链：

> UEA 这批数据集对"随机卷积特征 + max-pool + RBF SVM"太容易（§4）
> → 基线贴顶（§1）→ 常规 CLS 的增益被压进 ±0.013（解耦实验）
> → δ 过滤把常规配对全弃样 → 存活 label 全是"破坏 vs 非破坏"（解耦实验 §7）
> → T-CLSC 学的是全局破坏检测、与任务无关 → **路线 B "E 消融零下降"的机制解释**

§5 证明加难类别口径解不开第一环。剩下两条路（都需另行设计与验证，本探针不背书）：

1. **改下游评估协议**，把"随机表征就能解"这件事堵掉。可考虑的旋钮：减少下游训练
   样本（few-shot probe，让表征质量真正吃重）、换更弱的分类器（**线性探针而非 RBF
   SVM——AutoCL 附录 B.1 本身就是线性协议**）、或限制表征维度。
   **这是最小改动且直击根因的方向**；代价是 R' 噪声上升、δ 需重标定。
2. **接受"破坏检测"作为阶段2 的实际任务**，把项目主张从"task-aware CLS 排序"调整为
   "跨任务的 CLS 可行性预测"。65% 的空间确实该避开，不是没价值，但测不到 task-aware
   那一层。

（探索性）本探针另提示一条：**任务比数据集更适合做筛选单位**（§1.1）。Handwriting
内部 0.435–0.707、UWave 内部 0.821–0.946，按任务级 headroom 挑选可以在不换数据集的
前提下筛出 55 个"离顶任务"——但其中多数属于 §4 的"学不动"类，真正可用的仍以
Handwriting 的 12 个任务为主。

### 6.4 预测对账（PROTOCOL.md §5，先于数据）

| # | 预测 | 结果 |
|---|---|---|
| 1 | AWR/CT/ERing/NATOPS 落 ceiling | ✅ 全中（0.948–0.990） |
| 2 | Handwriting 落 usable | ✅ 中，且是唯一无争议的 |
| 3 | MotorImagery/SCP2 落 inert | ✅ 全中（基线≈猜测，实得 −0.003 / −0.013） |
| 4 | 若 1–3 全中 → 困境在任务口径而非数据集选择 | ⚠️ **前提成立但结论要修正**：§5 证明类别档口径**已到头**，出路在下游评估协议，不在切片难度 |

4 条预测 3 条命中、1 条方向对但机制猜错（我原以为"任务口径"= 切片难度，实测表明
是**下游协议**）。这一条按预注册规则记为"部分错"。

---

## 7. 方法学校验

1. **iters=0 时 CLS 完全不参与**：`pretrain_encoder` 的 `while step < max_iters` 不进
   循环。两条差异极大的 CLS（温和 vs 破坏性）同种子下 R' 逐位相同，
   `tests/test_baseline_probe.py::test_cls_is_inert_at_zero_iters` 断言。
2. **5 个种子够不够**：解耦实验的 `baseline_cache.jsonl` 在 AWR/CT/ERing/NATOPS 的
   8 个任务上有 **20 个独立初始化**（同 canonical split）的同一个量。逐任务核对
   classes 与 split_seed **8/8 完全一致**，均值差 0.0002–0.0233（中位 ~0.005），
   远小于 0.10 的档宽。**5 种子足够定档。**
3. **换 joblib 后端不改数值**：Windows 上 `GridSearchCV(n_jobs=-1)` 每条要 spawn 一个
   loky 进程池（8–15 s，占单条 99%）。改 threading 后端后 NATOPS/ERing/MotorImagery
   三例的 R'、f1、选中的 C **逐位相同**，有回归测试。这是 685 条能在 2.5 min 跑完的
   原因（5090 上 0.13 s/条的数字在 Windows 上不成立，是 fork vs spawn 之别）。
4. **阶段1 配对校验**：`realized/best gain` 只接受 classes 与当前切片逐字段相同、
   且 split_seed 等于 canonical 划分的历史记录。命中 **320 条 / 32 个任务**；
   拒收 classes 不符 64、split 不符 176、任务不在当前切片 120 —— 全部来自
   `results/route_a_probe` 那份**早于 `source_pool.yaml` 定稿**的缓存。**没有静默错配。**
   3 个骨干因此补跑了同口径 canonical R'（`--fill-trained`，60 条，墙钟 65 min）。
5. **铁律 6**：目标池 UCI-HAR / Epilepsy / FaceDetection 全程未加载，脚本内断言
   探针范围 ∩ 目标池 = ∅ 且目标池走源加载路径 fail-fast。

## 8. Limitations

1. **`best_gain` 的配对参照只有每数据集 2 个任务 × 10 条 shared CLS**（阶段1 缓存
   口径），不是全部 planned_tasks。数据集级 `best_gain_max` 是"这 2 个任务上最好的"。
2. **10 条 shared CLS 不是随机样本**（探针目录，跨 HAR 好坏挑选）。解耦实验证明随机
   样本含破坏性 CLS 更多（65% vs 40%），故这里的 `n_trained_below_null` 计数
   **低估**了破坏占比。
3. **基线只变初始化、不变 split**（PROTOCOL.md §1），为与阶段1 canonical R' 严格配对
   相减；划分方差未计入。§7.2 的解耦对照（s≥1 同时变 split）显示两者量级相当。
4. 2 类 EEG 数据集任务数少（SCP1 只有 2 个、SCP2 3 个），数据集级统计的 n 很小。
5. **§5 的加难档每数据集只做 4 个任务**，且只测了 `n_classes_range` 一个旋钮；
   通道子集档、样本量档未测。
6. 本探针回答"有没有可测空间、有没有 CLS 用得上"，**不回答"非破坏性 CLS 之间能否
   互相排序"**——那是解耦实验 R-reg 口径的问题，在贴顶数据集上实测≈0。
   **Handwriting 上这个量尚未测过，是下一步最该做的事**（它是唯一 headroom 与
   实得增益都够大、值得花这个算力的数据集）。
