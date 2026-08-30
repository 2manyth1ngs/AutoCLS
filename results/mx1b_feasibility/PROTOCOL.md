# MX-1b 协议：探针条件化可行性预测——标签感知特征的最后一次预注册尝试（判读带先于计算写死）

日期：2026-08-30 ｜ 脚本：`scripts/17_feasibility_loo_b.py` ｜ 单测：
`tests/test_feasibility_b.py` ｜ 前作：MX-1 判 not_established
（`results/mx1_feasibility/REPORT.md`，协议锁 5af6c14）。本协议与测试、实现
先 commit（协议锁），之后才跑数据。

## §1 问题、机制动机与主张边界

MX-1 的失败归因（post-hoc，本协议将其转为可检验假设）：E 是**标签盲**的
分布特征，而毒性是**标签依赖**的任务性质（SCP1/SCP2 为 E 最近邻但毒性
0.00 vs 0.45——分布几乎同、任务难度天差地别）。MX-1b 换特征不换判据：
用**零训练、标签感知**的探针特征（iters=0 探针统计 + 折 TS2Vec 表征空间的
类别可分性几何）预测同一目标（未见数据集的三态分布），E 降格为被检验的
增量假设。

主张边界：
- 特征仍属 zero-cost proxy 类（用目标标签、**零候选训练**、零 TS2Vec 训练，
  铁律 6 不变）；主张措辞为"探针条件化预测"，不再称纯分布 zero-shot。
- **勿复活条款（写死）**：本协议是"数据集级可行性之学习预测器"的**最后
  一次**预注册尝试。若判 not_established，该方向关闭，看门层改走金丝雀
  直测（预检闸 v2），不再为同一主张立第三份协议；"任务级毒性预测"是
  不同主张，不受此条款覆盖但须独立立项。
- 头条仍只到数据集级毒性水平；cell 级如实报不当头条（同 MX-1）。

## §2 探索性偷看披露（先验的诚实条件）

预注册前已看过（必须如实声明，先验在此条件下标定）：
1. MX-1 全部结果（16 折、全臂、post-hoc 1-NN/OOD 诊断）；
2. 同 16 数据集上三个**单变量** Spearman：基线均值↔破坏率 0.32（n.s.）、
   超 chance 幅度↔破坏率 0.39（n.s.）、headroom↔提升率 −0.15（n.s.）
   ——P 组特征的单变量信号已知偏弱，故 P 组单独过判据的先验低；
3. **G 组特征（折 TS2Vec 空间类别几何）从未计算过**——这是本协议真正的
   新赌注，无任何偷看。

## §3 特征（公式先于计算写死；均为零训练量）

面板/label/折结构/折 TS2Vec 全部复用 MX-1 定稿产物（320 cell、
`results/mx1_feasibility/labels.jsonl`、16 折 `folds/<ds>/shared_ts2vec.pt`
——铁律 6 按折已成立，不重训）。

**P 组（探针统计，每任务 3 维；来源 = labels.jsonl 既有字段）**：
- p1 = base_mean（iters=0 基线 5 种子均值）
- p2 = chance = 1/C（C = 任务类别数）
- p3 = band（2·max(σ, 1/n_eval)，噪声尺度）

**G 组（类别可分性几何，每任务 5 维；折 TS2Vec 前向 + 任务标签，零训练）**：
窗口采样与 MX-1 特征完全同口径（`encode_task_windows` 同 seed 同 idx——
采样索引镜像其 rng 逻辑，单测断言一致），窗口向量 = 逐时间步表征对 T 取
均值 → (m, D)，**逐行 L2 归一化**后计算：
- g1 = NCC 准确率（各类质心，最近质心分类的 in-sample 正确率）
- g2 = log1p(Fisher 比)：trace(类间散度)/trace(类内散度)
- g3 = 质心版 silhouette：mean[(b−a)/max(a,b)]，a=到本类质心距离，
  b=到最近他类质心距离
- g4 = min 类间质心距 / mean 类内到质心距（最难分的一对的裕度）
- g5 = g1 − 1/C（chance 校正 NCC lift）

冗余共线（p1/p2 与 g1/g5）容忍——MLP 不敏感，特征选择不做（防调参渗漏）。
特征缓存 `results/mx1b_feasibility/folds/<ds>/probe_features/<tid>.npz`
（含 G 统计 + (m,D) 池化矩阵 + 采样窗口标签，供审计）。

## §4 预测器与臂（超参与 MX-1 §4.2 逐字同款，禁止另调）

MLP [128,64]、Adam lr=1e-3 wd=1e-3、全批 300 epoch、类别加权 CE、CPU、
5 种子 softmax 均值、z-score 折内标准化（只喂训练折）——一切同 MX-1，
**不允许为 MX-1b 调任何超参**（防"换特征顺手调参"的渗漏）。

| 臂 | 输入 | 用途 |
|---|---|---|
| **bm（主）** | [P⊕G] ⊕ v_cls | 主判据载体 |
| **be** | [P⊕G⊕E] ⊕ v_cls | **D3b：E 增量检验**（E 取 MX-1 折特征） |
| bp | [P] ⊕ v_cls | 诊断：光靠探针统计够不够 |
| bg | [G] ⊕ v_cls | 诊断：光靠几何够不够 |
| bm_a1 | bm + 评估时特征 cross-assignment（5 assign 种子） | **D2b 消融**（4b 同语义） |
| B0/B1 | 任务盲（按折算，泄漏契约同 MX-1 §4.3） | 对照 |

## §5 量、检验与判读带（先于计算写死）

量与检验全同 MX-1 §5：err_TV 主误差、多项抽样地板参照、16 折 exact
Wilcoxon 单侧 α=0.05、cell 级 Brier 次判据、D2b 逐折功效异质性预声明同款。

- **D1b**：err_TV(bm) < err_TV(B0)，p<0.05。
- **D2b**：err_TV(bm) < err_TV(bm_a1)，p<0.05。
- **D3b**（E 增量，只记账不做门）：err_TV(be) < err_TV(bm)，p<0.05。
- S1b（次）：Brier(bm) < Brier(B1)，p<0.05。

| 条件 | 判定 |
|---|---|
| D1b ∧ D2b | **成立**：可行性层复活为"探针条件化预测"（zero-training 类）；E 增量按 D3b 如实报 |
| D1b ∧ ¬D2b | 特征携带数据集信号未证实：如实报告，按 bp/bg 归因 |
| ¬D1b 且 p_D1b < 0.15 | 边缘：交用户拍板（勿复活条款仍悬置于拍板之上） |
| 其余 | **不成立 → 勿复活条款生效**：数据集级学习预测方向永久关闭，看门层走金丝雀直测（B 路线，另立轻协议） |

## §6 稳健性与诊断（预注册，不改判读带）

- R1b：剔 3 个 k1 骨干，13 折重算 D1b（同 MX-1 R1）。
- R2b：bp / bg 两臂对 B0 的 D1 型检验各自报告（特征组归因）。
- R3b：SCP1/SCP2 展品复查——bm 对这两折的预测破坏率之差应与观测同向
  （0.00 vs 0.45）；这是机制假设"几何见得到难度差"的直接读数。
- 诊断：g1..g5 与数据集破坏率的单变量 Spearman 一并出（供 REPORT 归因，
  不判读）。

## §7 预测（先验；已条件于 §2 披露）

| # | 预测 | 先验 |
|---|---|---|
| P1b | D1b 成立 | 30% |
| P2b | D2b 成立 | 25% |
| P3b | D3b：E 在 P⊕G 之上仍有显著增量 | 15% |
| P4b | bg（纯几何）单独过 D1 型检验 | 20% |
| P5b | R3b：SCP1/SCP2 预测破坏率差与观测同向 | 55% |
| P6b | Spearman(bm 预测破坏率, 观测) ≥ 0.5 | 30% |

## §8 执行与成本

- 顺序：协议+tests+实现 → **协议锁单独 commit** → --features（轻 GPU：
  16 折 × 32 任务重编码窗口，复用既有折 TS2Vec，≈1.5–2 h @4060，锚点带
  硬件标签）→ --train/--analyze（CPU 分钟级）→ REPORT。
- GPU 规程同 MX-1 §8（清桌面负载 + nvidia-smi；Start-Process 脱离 +
  --resume 幂等；收官重验文件数）。
- 产物：`results/mx1b_feasibility/{PROTOCOL.md, folds/<ds>/probe_features/,
  predictions.jsonl（按折）, analysis.json, REPORT.md}`。
- 铁律 6 自检：特征只用既有折 TS2Vec 前向（fit 池指纹校验沿用）；目标池
  零参与。
