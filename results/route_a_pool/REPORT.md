# 路线 A 源池重排报告 — L≥50 剔除后重新排池 + 任务数验证

日期 2026-07-16 | 只做分析与规划（不跑训练、不造 label、不上云）| 依据 `results/uea_survey/` + `results/route_a_probe/`

**背景**：探针（commit 070d2b7）定死两条：① encoder 适用边界 **L≥50**（10 层膨胀 CNN
dilation 到 2^9=512 为 L≥100 设计，L=8 上 padding 主导、CLS 零分辨，SNR 0.39）；
② δ 改 **per-dataset**（0.02 是 HAR 特调值，UEA 噪声随 n_eval/离天花板 4× 跨度）。
L≥50 剔除打穿了调研 §4 的 12-源池设计——**头号切片主力 PenDigits(L8) 与核心
LSST(L36) 双双出局**，本报告据此重排。**结论先行**：重排后 **14 源池 / ~129 任务**，
是路线 B 的 8×、AutoCTS++ 的 ~65%；4×A100 单次 ~14h（含 SNR 预探针 ~18h），≤1 天可行。

---

## 0. TL;DR

- **源池 14 个**（5 域，编码长度 51→1152，通道 3→64，类别 2→39），**~129 源任务**。
- **探针实测可靠（SNR>1）3 个**（SpokenArabicDigits/PhonemeSpectra/Cricket，占任务 34%）；
  其余 11 个为 **"推测可靠、未实测"**——上云 **step 0 用探针同款 SNR 预探针逐一过闸**，
  SNR<1 者踢入 reserve。这把"未实测"从风险转成一次性廉价前置检查（4×A100 ~4h）。
- **δ per-dataset**：3 个实测填死（0.03/0.05/0.08），11 个按 **N/C 阶梯**给推测初值
  （0.05–0.08，标注推测），上云 SNR 预探针 / 全量早期按实测 R' 分布微调。
- **长度分化点保住**：中序列 51–315（9 个）+ 长序列 **896/1024/1152 编码**（SCP1/
  EthanolConcentration/SCP2，其中 SCP1/SCP2 原生不裁）。
- **目标池同域支撑齐**：HAR/Epilepsy ← 5 个 HAR 源 + 2 Motion 邻域；FaceDetection(EEG/MEG)
  ← 4 个 EEG/MEG 源（非孪生范式）。**泄漏零**：Epilepsy/FaceDetection/UCI-HAR 均不在源池。

---

## 1. 最终源池清单（14 个）

分三层。`δ*` = 探针实测；无星 = N/C-anchored 推测初值（上云前须过 SNR 预探针）。
`L_enc` = 编码长度（不等长右零填充到 L_max、超长裁到 crop）；`探针` = 是否本轮实测 SNR。

### ① 探针实测可靠骨干（SNR>1，label 正确率 ≥95%）
| # | Dataset | 域 | 类 | 通道 | L_enc(处理) | N/C | PracBudget | δ | 计划任务 | 探针 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SpokenArabicDigits | Audio | 10 | 13 | 93 (pad, 原 4–93) | 880 | 24 | **0.03\*** | 16 | ✓ SNR2.24 |
| 2 | PhonemeSpectra | Audio | 39 | 11 | 217 | 171 | 24 | **0.05\*** | 18 | ✓ SNR2.03 |
| 3 | Cricket | HAR | 12 | 6 | 256 (crop, 原 1197) | 15 | 24 | **0.08\*** | 10 | ✓ SNR1.40 |

### ② 未实测切片主力（高类数，SNR 预探针后入池）
| # | Dataset | 域 | 类 | 通道 | L_enc(处理) | N/C | PracBudget | δ | 计划任务 | 探针 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | CharacterTrajectories | Motion | 20 | 3 | 182 (pad, 原 60–182) | 143 | 24 | 0.05 | 16 | ✗ 推测 |
| 5 | Handwriting | HAR | 26 | 3 | 152 | 38 | 24 | 0.07 | 12 | ✗ 推测 |
| 6 | UWaveGestureLibrary | HAR | 8 | 3 | 315 | 55 | 24 | 0.06 | 10 | ✗ 推测 |
| 7 | ArticularyWordRecognition | Motion | 25 | 9 | 144 | 23 | 24 | 0.08 | 8 | ✗ 推测 |
| 8 | NATOPS | HAR | 6 | 24 | 51 | 60 | 24 | 0.06 | 10 | ✗ 推测 |
| 9 | ERing | HAR | 6 | 4 | 65 | 50 | 24 | 0.06 | 6 | ✗ 推测 |

### ③ 长度 / 领域多样性填充（任务少但拉开分布 + 给 EEG 目标支撑）
| # | Dataset | 域 | 类 | 通道 | L_enc(处理) | N/C | PracBudget | δ | 计划任务 | 探针 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | MotorImagery | EEG/MEG | 2 | 64 | 512 (crop, 原 3000) | 189 | 4 | 0.05 | 4 | ✗ 推测 |
| 11 | SelfRegulationSCP1 | EEG/MEG | 2 | 6 | **896 (原生不裁)** | 281 | 2 | 0.05 | 2 | ✗ 推测 |
| 12 | SelfRegulationSCP2 | EEG/MEG | 2 | 7 | **1152 (原生不裁)** | 190 | 3 | 0.05 | 3 | ✗ 推测 |
| 13 | HandMovementDirection | EEG/MEG | 4 | 10 | 400 | 59 | 18 | 0.06 | 6 | ✗ 推测 |
| 14 | EthanolConcentration | Other(Spectro) | 4 | 3 | 1024 (crop, 原 1751) | 131 | 12 | 0.05 | 8 | ✗ 推测 |

**reserve（预探针后按需补位）**：Heartbeat(Audio,2类,61ch,L405,N/C205,+~4)、
FingerMovements(EEG,2类,28ch,**L50 边界**,N/C208,+~4，L=50 须强制过 SNR)。

---

## 2. 源池任务总数估算

| 口径 | 值 | 对照 |
|---|---|---|
| 全 14 源池 PracticalBudget 求和（切片容量上界） | **255** | 全档案 30 个求和 500 |
| ×0.65 realized（路线 B 实测饱和系数） | ~166 | 切片能力充裕，非瓶颈 |
| **可靠性节流后计划任务** | **~129** | **路线 B = 16（8×）；AutoCTS++ = 200（65%）** |
| 其中探针实测可靠（δ\*）占 | 44 / 129 = **34%** | — |

计划 129 < 切片上界 166：**瓶颈是可靠性不是切片容量**——低 N/C 数据集（Cricket 15、
ArticularyWord 23、Handwriting 38）按可靠性下调到远低于其 24 的切片上限，把任务量堆在
高 N/C + 实测可靠的骨干上。目标"接近 100+"达成，且没靠掺水凑数。

---

## 3. 覆盖检查表

| 维度 | 覆盖 | 说明 |
|---|---|---|
| **领域数** | **5** | Audio(2)·HAR(5)·Motion(2)·EEG/MEG(4)·Other-Spectro(1) |
| **编码长度分布** | **51 → 1152** | 见下分档 |
| **通道跨度** | **3 → 64** | 3(Handwriting/UWave/CharTraj/Ethanol)·…·24(NATOPS)·64(MotorImagery) |
| **类别跨度** | **2 → 39** | 2(EEG)·…·39(PhonemeSpectra) |

**长度分档（AutoCL temporal-contrast 分化点）**：
- **短–中（50–100）**：NATOPS 51 · ERing 65 · SpokenArabicDigits 93 —— temporal contrast 应"没用"的一端
- **中（100–320）**：ArticularyWord 144 · Handwriting 152 · CharTraj 182 · PhonemeSpectra 217 · Cricket 256 · UWave 315
- **长（≥400）**：HandMovementDirection 400 · MotorImagery 512 · **SCP1 896 · Ethanol 1024 · SCP2 1152** —— temporal contrast 应"有用"的一端
- **原生长序列锚**：SCP1(896)/SCP2(1152) 不裁，保真长度；MotorImagery(原 3000)、
  Ethanol(原 1751) 裁到 512/1024 封成本（探针证 Cricket 裁 256 不损 CLS 分辨）。
- **分化两端都有非孪生多域样本**（短端 Audio+HAR，长端 EEG+Spectro），不是单域拉长度。

---

## 4. 目标池同域支撑确认

目标池（已定，固定）：**UCI-HAR（外部）+ Epilepsy（UEA）+ FaceDetection（UEA）**。

| 目标 | 域 | 源池同域非孪生支撑 | 判定 |
|---|---|---|---|
| UCI-HAR | 加速度计 HAR | Handwriting/UWave（加速度计笔迹/手势）· NATOPS/ERing/Cricket（体感/指环/运动）· + Motion 邻域 CharTraj/ArticularyWord | **支撑强**（5 HAR + 2 Motion） |
| Epilepsy | 加速度计 HAR | 同上（不同受试者/传感器/类别，leave-dataset-out 标准协议，非有害近亲） | **支撑强** |
| FaceDetection | EEG/MEG | MotorImagery(运动想象)·SelfRegSCP1/2(慢皮层电位)·HandMovementDirection(运动方向) —— 4 个不同 EEG/MEG 范式 | **支撑充分**（HandMovementDirection 由原调研目标释放为源，正好补 EEG） |

**关键调整点**：调研原把 HandMovementDirection/JapaneseVowels 设为目标；现目标改
HAR+Epilepsy+FaceDetection，**HandMovementDirection 释放回源池**成为 EEG 支撑，
JapaneseVowels（L7–29<50）整体出局。FaceDetection 目标由 4 个非孪生 EEG/MEG 源支撑，
不滑向"太远"。

---

## 5. 泄漏确认（铁律 6）

| 目标池数据集 | 是否在源池 | 确认 |
|---|---|---|
| UCI-HAR（外部数据集） | 否（不在 UEA，非源候选） | ✓ 无泄漏 |
| Epilepsy（UEA, HAR 域） | **否**（逐条核对源池 14 个不含） | ✓ 无泄漏 |
| FaceDetection（UEA, EEG/MEG） | **否**（调研曾列为源候选，现为目标，已剔除） | ✓ 无泄漏 |

目标池 3 个数据集及其全部样本绝不进源池；TS2Vec 亦只在源池 fit（铁律 6）。✓

---

## 6. 剔除清单

| 数据集 | L | 原因 | 类别 |
|---|---|---|---|
| **PenDigits** | 8 | L<50 且探针 SNR 0.39（CLS 零分辨，纯噪声 label） | L<50 + 低 SNR |
| **LSST** | 36 | L<50（编码失配）；原为核心切片主力，L≥50 打穿之 | L<50 |
| RacketSports | 30 | L<50 | L<50 |
| Libras | 45 | L<50（45<50） | L<50 |
| JapaneseVowels | 7–29 | 最长 case 29<50，整体失配（调研原目标，现无 Audio 目标） | L<50 |
| InsectWingbeat | 2–22 | L<50 且 5 万×200 规模过重 | L<50 + 极端规模 |
| DuckDuckGeese | 270 | 1345 通道，单变量 task_embed 每任务上千次编码，爆显存 | 极端通道 |
| PEMS-SF | 144 | 963 通道，同上 | 极端通道 |
| EigenWorms | 17984 | 单 iter 成本 ≈HAR 140×，单次 (task,cls) 数小时 | 极端长度 |
| AtrialFibrillation | 640 | N=30，切不出足够任务、R' 不可靠 | 样本过小 |
| StandWalkJump | 2500 | N=27，同上 | 样本过小 |
| BasicMotions | 100 | N=80，同上 | 样本过小 |
| Epilepsy | 206 | **目标池，泄漏** | 泄漏 |
| FaceDetection | 62 | **目标池，泄漏** | 泄漏 |

**变长处理判断（约束 1）**：
- **SpokenArabicDigits（原 4–93，最短 case<50）→ 整体保留、右零填充到 L_max=93、逐通道
  z-score 仅统计真实步**。理由：① 探针实测此口径 SNR 2.24 达标（填充零放大相对增强度量
  是假象但不致命，见探针 §3.3）；② "剔除 <50 的 case"会丢弃未知大比例样本、拉低 N/C；
  ③ 决定 encoder 适配的是**编码长度 93≥50**，带 mask 的零填充不破坏 encoder（探针已证）。
- **CharacterTrajectories（原 60–182，全部 case≥50）→ 保留、pad 到 182，无 case 丢弃**。
- 结论：两个变长源都**不需丢 case**；最长<50 的（JapaneseVowels/InsectWingbeat）整体剔除。

---

## 7. 算力估算（新源池）

**成本模型**（调研 §5，探针校准）：单 (task,cls) `s ≈ 84.2 × L_eff/128 ×（弱通道因子）`；
每任务 **20 CLS**（路线 A 建议口径，见杠杆）；runs = 任务 × 20。A100 对此小模型 ~2.5× 4060。

| Dataset | 任务 | crop | s/run | runs | 4060-h |
|---|---|---|---|---|---|
| SpokenArabicDigits | 16 | 93 | 66 | 320 | 5.9 |
| PhonemeSpectra | 18 | 217 | 149 | 360 | 14.9 |
| Cricket | 10 | 256 | 168 | 200 | 9.3 |
| CharacterTrajectories | 16 | 182 | 120 | 320 | 10.7 |
| Handwriting | 12 | 152 | 100 | 240 | 6.7 |
| UWaveGestureLibrary | 10 | 315 | 207 | 200 | 11.5 |
| ArticularyWordRecognition | 8 | 144 | 95 | 160 | 4.2 |
| NATOPS | 10 | 51 | 41 | 200 | 2.3 |
| ERing | 6 | 65 | 43 | 120 | 1.4 |
| MotorImagery | 4 | 512 | 505 | 80 | 11.2 |
| SelfRegulationSCP1 | 2 | 896 | 589 | 40 | 6.5 |
| SelfRegulationSCP2 | 3 | 1152 | 758 | 60 | 12.6 |
| HandMovementDirection | 6 | 400 | 263 | 120 | 8.8 |
| EthanolConcentration | 8 | 1024 | 674 | 160 | 30.0 |
| **合计** | **129** | | | **2580** | **~136 h** |

| 硬件 | 全量 label（20 CLS） | + SNR 预探针（step 0） |
|---|---|---|
| RTX 4060 Laptop ×1 | ~136 h（5.7 天） | ~175 h |
| A100 ×1（数据并行） | ~54 h（2.3 天） | ~70 h |
| **A100 ×4** | **~13.6 h** | **~17.5 h（≤1 天 ✓）** |
| A100 ×8 | ~6.8 h | ~8.8 h |

- **单次 ≤1 天 ⟹ 4×A100 足够**（全量 ~14h；含预探针 ~18h，仍一晚跑完）。8×A100 更宽裕。
- **SNR 预探针（step 0）**：11 个未实测源 × ~2 难任务 × 8 CLS × 4 种子 ≈ 700 runs ≈
  4×A100 ~4h。其 (task,cls,seed)→R' **并入全量缓存不浪费**（铁律 3，--resume）。
- **成本双雄**：EthanolConcentration@1024（30h）与 PhonemeSpectra（任务多，14.9h）。
  Ethanol 裁 512 → 15h（总降到 ~121h / 4×A100 ~12h）；但会牺牲一个非 EEG 长锚，权衡。
- **杠杆**：① 每任务 CLS 数 20→10 直接砍半；**注意** `configs/labels.yaml` 路线 B 口径是
  40 CLS（20 shared+20 random），若路线 A 沿用 40 则算力 ×2（4×A100 ~27h>1 天）——
  **建议路线 A 用 20 CLS/任务**；② 长序列 crop；③ early=290 已固化。

---

## 8. 风险标注

| 风险 | 量化 | 应对预案 |
|---|---|---|
| **源池 34% 任务实测、66% 推测** | 3/14 数据集探针实测；11 个未实测 | **step 0 SNR 预探针逐一过闸**（探针同款、廉价）：SNR>1 入池、SNR<1 踢 reserve。把"推测"一次性转"实测"，再烧全量 |
| **推测 δ 偏差** | 11 个 δ 为 N/C 阶梯推测（0.05–0.08） | 预探针出各数据集实测 σ/SNR → 按探针 §2.2 曲线（δ≈达 95% 正确率处）微调；全量早期抽查 label 正确率复核 |
| **低 N/C 数据集 SNR 可能不达标** | ArticularyWord(23)/Cricket(15,已实测勉强)/Handwriting(38) 最可能 SNR<1 | 它们是**多样性填充非骨干**，失败即踢，用 reserve（Heartbeat+4/FingerMovements+4）或把任务加到高 N/C 骨干（SpokenArabicDigits N/C880 可吸收更多）回填 |
| **多个未实测同时失败** | 极端情形任务数掉到 <100 | 骨干 3 个（44 任务）实测稳；再放宽 SpokenArabicDigits/PhonemeSpectra/CharTraj 切片预算至 20+，单此三者即可补到 ~90；纳入 reserve 两个补齐 |
| **长序列只有 2 类 EEG 撑** | 长端 SCP1/SCP2/MotorImagery 均 2 类 | Ethanol（Spectro,4 类,1024）作非 EEG 长锚已缓解；长端分化只需"长度"信号，2 类不损分化观测 |

**"推测可靠、未实测"占比：任务 66% / 数据集 79%（11/14）**。这是 L≥50 只保下 3 个实测源的
硬约束下的必然，**核心缓解是 step 0 SNR 预探针**——它把占比从"上云后才知道靠不靠谱"
降为"烧全量前已逐一验证"。若预探针后实测源不足以撑 129，按上表回填，**骨干 44 任务
（全实测）是保底下界，已 2.75× 路线 B**。

---

## 附：与调研 §4 原 12-源池的差异

| 变化 | 原（调研 §4） | 现（本报告） | 原因 |
|---|---|---|---|
| PenDigits | 核心切片主力（N/C1099） | **剔除** | L8 SNR0.39 |
| LSST | 核心切片主力（N/C352） | **剔除** | L36<50 |
| HandMovementDirection | 目标池 | **升为源（EEG 支撑）** | 目标改 FaceDetection |
| JapaneseVowels | 目标池 | **剔除** | L7–29<50 |
| Cricket crop | 1024 | **256**（探针证不损） | 省 45h |
| δ | 全局 0.02 | **per-dataset**（0.03–0.08） | 探针标定 |
| 源池规模 | 12 / ~157 任务 | 14 / ~129 任务 | L≥50 剔 2 主力、补 EEG/Motion/HAR |

---

## 落盘产物
`results/route_a_pool/REPORT.md`（本报告）。依据数据：`results/uea_survey/`（30 档案统计、
切片估算、算力）、`results/route_a_probe/`（analysis.json / delta_yield.json：3 实测 δ 与 SNR）。
配置更新：`configs/labels.yaml`（delta → per-dataset）；`CLAUDE.md`（L≥50 encoder 适用边界）。
