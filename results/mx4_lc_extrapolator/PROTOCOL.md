# MX-4 WP5 协议：学习曲线外推器驱动的逐次减半淘汰（零 GPU，缓存模拟）

日期：2026-09-04 ｜ 脚本：`scripts/38_lc_sh_sim.py` ｜ 模块：`autocls/search/learning_curve.py` ｜
数据：`results/mx3_pq/{rprime_cache.jsonl, noise_probe.jsonl, delta_calibration.json}`（只读）｜
对照基线：`results/mx4_prior_sh/analysis.json`（WP1，`SH6-3-3_prior` @290 mean regret 0.0184）｜
立项依据：CLAUDE.md "路径二 / WP5"（用户 2026-09-03 记录）；分支 `wp5-lc`（自 63d26fd），与 WP3（worktree `mx4`）互不相扰。

## 0. 诚实声明（与常规预注册的区别）

本协议**不是盲预注册**。写判读线之前已看过：
- WP1 REPORT 的全部读数（SH6-3-3_prior 0.018 / SH6-3-1_prior 0.036 / all20@60→top3 0.017 / G4 新鲜种子 0.015 vs 0.018）；
- CLAUDE.md 记录的 @60→@290 目录 20 条秩相关中位 0.72；
- **本会话 2026-09-04 的头顶空间读数**（在同一缓存上用临时脚本算，未涉及任何外推器）：
  SH6-3-3_prior @290 regret 0.0184 里，**不可约部分**（目录 oracle − 先验 top-6 内 oracle）0.0118（中位 0），
  淘汰决策可改进上限 0.0067；@60 截点误杀 top-6 冠军 9/42 实例；先验 top-6 内 @60 与 @290 的 Spearman 中位 0.36
  （目录 20 条 0.735，差别是破坏性配置撑起了后者）；逐数据集淘汰损失：HMD 0.0135 / UM 0.0103 / Cricket 0.0101 /
  PS 0.0092 / UWave 0.0021 / HW 0.0013 / Ethanol 0。
**外推器本身的任何读数都未算过。**判读线、臂表、预测在正式跑之前写死。

## 1. 问题

| 问 | 内容 | 检验 |
|---|---|---|
| Q1（主） | 把 SH 淘汰规则从"按当前截点分"换成"按外推器预测的 @b_final 分"，能否降低 SH6-3-3_prior 的 regret@290 | L1 |
| Q2（次） | 外推器能否在更省的档位（SH4-2-2 / SH6-3-1）追平现行 SH6-3-3 的 0.018 | L2 |
| Q3（机制） | 在先验 top-6 内，预测分对 @290 真值的排序是否优于 @60 当前分 | L3 |
| Q4 | 增益是配置质量还是同轨迹运气（新鲜种子真值） | L4 |
| Q5（诊断） | GBDT 学到的是否只是"往先验收缩"或"回归均值"（对照 blend / nocls / slope） | D（描述性） |

## 2. 数据与口径（沿用 WP1，铁律 2/6 同口径）

- 真值：`rprime_cache.jsonl` 2835 行 = 45 切片 × 21 CLS（10 shared + 10 random + GGS）× 截点 {60,120,290}，同一轨迹，k=1 canonical。
- 折：7 折 leave-one-dataset-out（`pq_common.fold_split`）；训练侧 = 其余 6 数据集 + SCP2 = 39 切片 × 21 = **819 条轨迹**（含 GGS 行，
  它只是一条 CLS）；考侧 = 靶数据集 6 切片，候选 = 目录 20 条（GGS 不入候选，与 WP1 同）。
- 搜索实例 = (靶, 切片, b_final)：b_final=290 三级 (60,120,290) 42 实例（主）；b_final=120 两级 (60,120) 42 实例（次）。
  oracle = 目录 20 条 max R'@b_final；regret = oracle − 臂。
- 先验：WP1 唯一定义（训练侧 shared 行 within-(切片, 预算) z 的均值），warm start 全部用 prior 排名（WP1 头条臂口径）。
- **外推器**（`LearningCurveExtrapolator`，任务无关，只见向量）：
  - 输入 = 已观测截点的**原始 R'**（不是 z：部署时 @60 只训了 k0 条，算不出 20 条口径的 z）⊕ `CLSEncoder(mode="binned")` 编码
    ⊕ 切片常量 [log10 n_train, log10 n_eval, n_classes]；
  - 目标 = 增益 R'@b_final − R'@b_last_obs；排名分 = R'@b_last_obs + 预测增益；
  - 每折 3 个模型：(obs 60 → 290)、(obs 60 → 120)、(obs 60,120 → 290)；回归器复用 `FieldSurrogate`（hgb 默认超参 / ridge 默认超参）。
- 淘汰规则臂（同一 SH 档位、同一 warm start、同一成本，只换"每级留谁"）：
  - `cur`：按上一截点真 R' 留前 k（= WP1 `SH*_prior`，逐位应相同，见 C0）；
  - `lc_hgb`（主臂）：按 hgb 外推器预测的 R'@b_final 留前 k；
  - `lc_ridge`：同上，ridge；
  - `lc_hgb_nocls`：hgb 但不给 CLS 编码（只有观测分 + 切片常量），测"回归均值"能解释多少；
  - `blend`：z_k(R'@b_prev) + λ·prior(k)，z 在**当前在训的 k 条**内算（部署可得），prior 缺席记 0；
    λ ∈ {0.25, 0.5, 1, 2, 4} 在训练侧切片上按 SH6-3-3 mean regret 选（并列取小），每折一次；
  - `slope`：≥2 个观测截点时对数线性外推 R'@b_final = R'@b1 + (R'@b1 − R'@b0)·ln(b_final/b1)/ln(b1/b0)；只有 1 个截点时退化为 `cur`。
- SH 档位：(6,3,3)、(6,3,1)、(4,2,2)、(4,2,1)、(10,4,3)、(10,4,1)（b_final=120 取前两级）；纯多保真 all20@60→top1 / top3
  （20 条 @60 后按规则留 k）。并列按 cls_key 升序。
- 成本：与 WP1 完全一致（iters 逐级累加；缓存 `seconds` 增量换算 GPU 秒），外推器拟合 <1 s/折不计。
- 参考臂：`oracle`、`prior3`、`prior6_oracle`（先验 top-6 内 @b_final 最好者 = 任何淘汰规则在该 warm start 下的下界）。
- 新鲜种子（L4）：`noise_probe.jsonl` 靶切片 00/01 × shared-10 的 rep1/rep2 @290 均值当真值；选择只用 canonical 轨迹（WP1 §7 同规则）。

## 3. 判读（先于正式跑写死；配对单侧 Wilcoxon α=0.05，n 如实报，胜/负/平如实报）

| 门 | 检验 | n | 先验 |
|---|---|---:|---|
| **C0**（一致性） | `SH6-3-3_cur` @290 与 WP1 `SH6-3-3_prior` 逐实例相同（mean 0.0184） | 42 | 必须过，否则实现有误不读任何门 |
| **L1**（主） | regret[SH6-3-3_lc_hgb] < regret[SH6-3-3_cur]，@290 | 42 | 30%（上限 0.0067，最多 9 实例可动，功效低） |
| **L2**（次） | ∃ T ∈ {SH4-2-2, SH6-3-1}：mean regret[T_lc_hgb]@290 ≤ mean regret[SH6-3-3_cur]@290，且 H1 "T_lc_hgb 更差"的单侧 Wilcoxon p ≥ 0.05 | 42 | 45% |
| **L3**（机制） | 先验 top-6 内：(a) 预测分 vs R'@290 的 Spearman 逐实例中位 > cur 的（0.36）；(b) δ(数据集,290) 存活对的配对准确率 lc_hgb > cur。两者同时成立 | 42 实例 | 60% |
| **L4** | 新鲜种子真值下 mean regret[SH6-3-3_lc_hgb] ≤ mean regret[SH6-3-3_cur]（描述性，n=14 功效不足） | 14 | 50% |

读数码：**LC-WIN** = L1 ∧ L3；**LC-CHEAP** = ¬L1 ∧ L2 ∧ L3（不更准但更省）；否则 **LC-DEAD**（学习曲线在本栈等价于"看当前分"，入负翼）。
L4 独立报：WIN/CHEAP 但 L4 不过 → 与 WP1 §3.1 同法写"部署口径成立、配置质量口径只能写不差"。
D（诊断）只描述，不参与读数码。任一门不过如实入 REPORT，不 reframe。

## 4. 预测（先验写死）

- P1：先验 top-6 内 lc_hgb 预测分 vs R'@290 的 Spearman 中位 ∈ [0.40, 0.60]（55%）。
- P2：SH6-3-3_lc_hgb @290 mean regret ∈ [0.013, 0.019]（65%）。
- P3：SH6-3-1_lc_hgb @290 mean regret < 0.030（现行 0.036）（55%）。
- P4：在 SH6-3-1 上 blend 拿到 lc_hgb 增益的 ≥ 50%（(cur − blend) ≥ 0.5·(cur − lc_hgb)）：学到的大半是往先验收缩（55%）。
- P5：lc_hgb_nocls 与 cur 在 SH6-3-3 上 |Δ| < 0.003：没有 CLS 身份就学不到东西（60%）。
- P6：all20@60→top3_lc_hgb ≤ all20@60→top3_cur（0.017）（50%）。

## 5. 已知坑（写在跑之前）

- 目标噪声：k=1 单跑 R' sd 0.035–0.048，增益目标的噪声约 √2 倍；819 行拟合，hgb 300 树深 4 有过拟合风险，ridge 作对照。
- 部署真实性：@60 只有 k0 条被训，故特征只用逐候选量，不用候选集统计；blend 的 z 在 k 条内算，k=3 时噪声大。
- 平局：多数实例 cur 与 lc 会留同一批人；Wilcoxon 只在非平局对上有信息，n_tie 如实报。
- L2 的"不显著更差"在 n=42 下功效低，起约束作用的是均值条件；两个档位取"存在"有多重性，先验已折价。
- 更密截点缓存（workpoint_followup / iters_workpoint / step05）切片与协议不同，**本轮不用**；若主读数在边缘再立项。
- WP3 官方口径截点是 {25,50,100}%，与本轮 {60,120,290} 尺度不同：即便赢了，接入部署前须按官方截点重新拟合。WP3 产出可作外部复核集但绝不入拟合。
- HMD 评估集小（n_eval 24–74），其 6 实例噪声大且是淘汰损失最大的数据集；分层表如实报。

## 6. 不做的事

- 不改 `36_prior_sh_sim.py` / `37` / `39`（WP3 会话在用 36 的纯函数；本脚本以 import 复用）；不碰 `results/mx4_deploy/`。
- 不用目标数据集任何行拟合外推器或选 λ（铁律 6）；模块不含任务类型 / 数据集概念（铁律 7）。
- 不跑 GPU、不造新 label、不动 space.yaml。

## 7. 跑后修订记录（如实登记，不改判读线）

（空）
