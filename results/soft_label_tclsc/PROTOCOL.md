# 实验 2 协议锁：噪声感知软 label 重训 T-CLSC + shuffle 消融（+ shared-only 对照 + label 翻转注入曲线）

日期：2026-09-02 ｜ 本机 RTX 4060 Laptop（桌面负载未清，GPU 非确定性 ±0.01 记 limitation）｜
来源：`docs/DESIGN_NOTES.md` §1.3-1 / §1.4 / §6 实验 2 ｜ 判据/预测**先于数据**写死。
数据：`results/labels_workpoint/`（560 R' @工作点、80 条 k=3 噪声探针、28 任务特征），零新增 R'。
脚本：`scripts/31_soft_label_tclsc.py`；共享模块 `autocls/comparator/soft_labels.py`。
**与 CLAUDE.md 冲突声明**：触碰"细粒度 task-aware 排序判死勿复活"条款；用户 2026-09-02 指示
以 DESIGN_NOTES 为准——本实验重测的是判决前提（硬 δ 是否唯一杀手），不是复活旧协议。

## 0. 问题

C2 存活 label 由 δ=P95 硬阈值产出（弃样 55%，存活 label 任务盲天花板 0.9525）。DESIGN_NOTES §1.1
指出中位噪声 ≈ 中位信号 ≈ 0.010：任务信号不是被淹没，而是 z≈0.7 的弱信号被硬阈值整段丢弃。
本实验回答三问：**Q1** 保留 0.6<P<0.97 带（软 label）后 shuffle-E 消融是否首次非零；
**Q2** "random 参与的 85% label 与 shared-shared 同分布"假设是否成立；**Q3** label 噪声→比较器精度的响应曲线
（全项目缺这条曲线）。

## 1. 噪声模型（数据前写死；per-dataset × per-制式 × 两成分）

对每个配对 (a, b) 在任务 t 上，观测差 Δ = margin(R'_a, R'_b)（分类：R'_a − R'_b，方向经 task_types，铁律 8）。
Δ = Δ_true + η，η 为两条 k=1 测量误差之差。**同 CLS 跨 weight-seed 重复差**（k=3 探针，每数据集 60 个）
与 η 同分布（噪声跨 CLS 同分布假设），故 P(a ≻ b | Δ) = F_η(Δ)（Δ_true 取平先验）。
- **分布族**：Student-t（loc=0，scale s、df ν 由对称化重复差 MLE 拟合，ν 截到 [2, 100]）。重尾由 ν 承担。
  敏感性臂：经验 CDF（对称化 120 值，线性插值）。
- **制式**：cell (t, cls) 为 strict 破坏（R' < 基线均值 − 带，`autocls.workpoint` 同式）→ 破坏制式。
  探针 cell 里破坏制式 ≥ 5 个时单独拟合 scale（df 共用），否则单制式。混合配对 s_pair = √((s_a² + s_b²)/2)。
- **计数成分**：s_task = s_ref · √(κ·n_ref/n_eval,t + (1 − κ))，κ = 计数成分方差份额 0.595
  （UM@30 两成分分解 0.0126²/(0.0126²+0.0104²)，`results/step0_um_eval_expansion/REPORT.md` §3；
  HW 无独立分解，**借用同 κ**——已知假设，写进 limitations）。n_ref = 该数据集探针任务 n_eval 均值。
- **软目标** p = F_η(Δ)；**信息权** w = |2p − 1|。p 关于 (a, b) 交换满足 p(b,a) = 1 − p(a,b)。

## 2. 零训练判定：信息加权任务盲天花板（Q3 的 (iii) 口径）

shared 10 CLS 的 45 个无序对 × 28 任务。任务盲预测器对每对只能选一个方向 d；
加权天花板 = Σ_pairs max_d Σ_t w_t·1[sign(Δ_t)=d] / Σ_pairs Σ_t w_t。
- 校验：w_t = 1[|Δ_t| ≥ δ_d]（硬 δ）必须复现 **0.9525**（pooled）/ HW 1.000 / UM 0.989；
  w_t = 1（δ=0）复现 0.7473。
- 主读数：w = |2p − 1|（Student-t 模型）的 pooled / HW 内 / UM 内天花板；以及 **中带**（0.6 ≤ max(p,1−p) < 0.97）
  单独的天花板与其权重质量占比。

## 3. 训练臂（划分/超参/评估集全部沿用 C2：HW 留出 06/07/08、UM 留出 06/07/11/15；lr 1e-4、bs 64、200 ep、curriculum 40）

| 臂 | 训练 label | 种子 | 用途 |
|---|---|---|---|
| H0 hard-δ | C2 原口径（动态配对 + δ 弃样 + 硬 y） | 0/1/2 | 同机基线（C2 数字在另一台机器状态下产生） |
| S1 soft-BCE | δ=0 不弃样，目标 p（Student-t） | 0/1/2 | **主臂** |
| S2 weighted-hard | δ=0 不弃样，硬 y=sign(Δ)，样本权 w=\|2p−1\|（Δ=0 权 0） | 0/1/2 | 软 label 的保守版 |
| S1e soft-BCE（经验 CDF） | 同 S1，噪声模型换经验 CDF | 0 | 分布族敏感性 |
| C-shared | H0 口径但只用 shared-shared 配对 | 0/1/2 | Q2：85% 同分布假设 |
| F(f) flip 曲线 | H0 口径，训练配对按 (seed, task, a, b) 哈希以概率 f 翻转 y，f ∈ {0.1, 0.2, 0.3, 0.4}（f=0 即 H0） | 0/1 | Q3 响应曲线 |
| X soft×bypass | S1 口径 + E 换 MeanPoolTaskEncoder（实验 1 的绕过臂，raw） | 0/1/2 | 实验 1×2 交叉：同时修两条因果链 |

翻转决定用 md5(seed, task, a, b) 的前 8 位十六进制 / 2³² < f，跨 epoch 对同一无序对一致；反向样本同步翻转。

## 4. 评估（每臂每跑，7 留出任务）

- **硬评估集**：C2 的 δ 过滤单向对（1320 对）+ 反向 → acc、McNemar（within / cross × 5 assignment 种子 + zero），与 C2 逐字同口径。
- **软评估集**：δ=0 全部无序对（7 任务 × 190 = 1330 对）+ 反向 → 加权准确率 acc_w = Σ w·1[round(out)=y_hard] / Σ w、
  软 log-loss（对 p）、**中带子集**（0.6 ≤ max(p,1−p) < 0.97）的 acc 与其归一化值 acc / mean(max(p,1−p))。
  消融差用 1000 次配对 bootstrap（按无序对重采样）给 95% CI。
- **C-shared** 额外报留出集上 shared-shared 子集 vs random 参与子集的 acc。

## 5. 判读表（先于数据写死）

| 问 | 判定量 | 档位 |
|---|---|---|
| **Q1** δ 是否唯一杀手 | S1/S2 的 within Δ（硬评估集 McNemar；R1–R3 中位数种子） | **零**：mean Δ < 0.005 且 n_sig ≤ 1；**可检出**：Δ ≥ 0.005 且 n_sig ≥ 3（或软评估集 acc_w 的 Δ 在 ≥3/5 种子 CI 排除 0）；**达标**：C2 线 Δ ≥ 0.02 且 n_sig ≥ 4 |
| Q1 读法 | 零 → "task-aware 判死"证据升级为**与 δ 无关**（label 侧单因不成立，锅在别处或信号确不可学）；可检出/达标 → δ 是（部分）杀手，§3.5-3 路线开放 | |
| **Q2** 85% 假设 | C-shared 留出：acc(random 参与) − acc(shared-shared) | ≥ −0.05 → 假设成立；< −0.05 → random 的 label 分布不同，天花板论证不覆盖 85% |
| **Q3(a)** 加权天花板 | Student-t 权下 pooled 天花板 | < 0.95 → 软 label 集含被 δ 弃掉的任务依赖质量；≥ 0.95 → 中带也任务盲（信号不存在于 label 层） |
| **Q3(b)** 响应曲线 | acc(f) 相对 acc(0) 的下降；曲线上下降 0.02 所在的 f* | 与硬 δ label 的等效翻转率 f_eff = mean(1 − max(p,1−p))（存活对）对比：f* > f_eff → label 噪声不是比较器瓶颈 |
| **X** 交叉 | soft×bypass 的 within Δ 档位 | 与 S1、实验 1 bypass-raw 三者对照：只有 X 非零 → 两条因果链互相遮蔽坐实（DESIGN_NOTES §3.4） |

## 6. 预测（先于数据）

1. 加权天花板（Student-t 权）pooled ∈ [0.85, 0.94]（先验 60%）；≥ 0.95（25%）；< 0.85（15%）。
2. S1 硬评估集 normal acc ∈ [0.74, 0.82]（先验 65%；软目标不应显著伤害通用排序）。
3. Q1：零（先验 50%）、可检出（30%）、达标（20%）。
4. Q2：|差| ≤ 0.05（先验 55%）。
5. Q3(b)：f* ∈ [0.15, 0.35]（先验 60%）；f_eff（硬 δ 存活对）∈ [0.02, 0.08]（先验 65%）。
6. X 交叉：与 S1 同档（先验 60%）；X 高一档（25%）；X 低一档（15%）。

## 7. 成本

GPU 4060（桌面负载在，锚点：C2 每跑 ≈ 5 min 无负载）：SetTF 臂 3+3+3+1+3+8 = 21 跑，δ=0 臂配对量 ×2 → 估 ≈ 21 × 8 min ≈ 2.8 h；
X 臂 3 跑无 SetTF ≈ 5 min；天花板 CPU 秒级。与实验 1 并行共用 GPU。全程 `--resume`（每跑 JSON 落盘即续）。
