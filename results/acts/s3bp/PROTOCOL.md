# WP17-ACTS S3-B′ PROTOCOL：field 结构直接吃 E（零 GPU）

**会话定位**：S3-B 读数码 S3B-E-NOT-BETTER（`results/acts/s3b/REPORT.md`）。它测的是"E → 回归轴坐标 a(t) → 低秩模型"，没有测 T-CLSC 线上唯一有正读数的结构：**field 头**（logit 中的任务项 = 对每个字段的加性权重，R 检验 / T 检验 / K3）。用户 2026-09-24 提问"刚才有没有用 diff / field"，随后拍板先做零 GPU 近似。

本协议把 E 直接放进 field 结构：z(t, c) ≈ wᵀφ(c) + Σ_k u_k(t)·v_kᵀφ(c)，其中 u(t) = E 的低维投影。
- 用 ridge 一次拟合，不经过 a(t)，也不受低秩 K ≤ 3 限制；
- 等价于 R 检验的 field 头（s₀ + ⟨A·B·t, x⟩），只是用逐任务 z 回归代替配对 BCE。

产物 `results/acts/s3bp/`；代码 `scripts/87_acts_s3bp.py`；单测 `tests/test_acts_s3bp.py`；输入只读：S2 缓存（同 S3-A′）+ S3-B 的 `E_pt/_means.npz` 与 `E_sh/*/_means.npz`。

## §0 已看过的读数（如实声明）
1. S3-A / S3-A′ / S3-B 全部读数，包括：
   - S3-A′ 任务 1，ridge [φ, a₁·φ]、a₁ 取 ④ 估计：random-20 中位 Spearman 0.20，μ-only 0.14，真 a₁ 0.23；
   - S3-B：E_pt + mean 的 ρ_a1 0.45 / regret@3 0.014，E_sh + mean 0.53 / 0.019；
   - 均值池化 E 的集内 / 跨集余弦：pt 0.79 / 0.61，sh 0.96 / 0.79；
   - 参照：任务盲 0.021、④ 0.010、③ 0.006。
2. **没看过**：E 进入 field 结构后的任何读数。

## §1 设计（每个 LOSO 折只在训练源拟合）
- **行**：训练任务 × (shared-30 ∪ 有效 random)；标签 Z，即 S2 钉底 z，同 S3-A′。
- **φ**：`CLSEncoder(acts_space, binned)`，同 S3-A′。
- **u(t)**：该折训练任务的 E_mean（256 维）先按列标准化，再做 PCA 取前 r 维；留出任务用同一变换。
  - E_pt：每任务一份；E_sh：用该折的共享 TS2Vec 编码（训练与留出任务同一编码器）。
  - r ∈ {4, 8, 16}，**主 r = 8**。
- **设计矩阵** [φ, u₁φ, …, u_rφ]。ridge 的 α ∈ {1, 10, 100, 10³, 10⁴}，由训练源内层 LOSO 最小化 MSE 选定。内层折沿用外层折的 PCA 与 E_sh 编码器，只影响 α 选择，不触及外层留出源。
- **臂**：{E_pt, E_sh} × r ∈ {4, 8, 16}。
- **cross-E**：留出任务改用循环下一个源同行号任务的 E（同 S3-B 的 `cross_assign`）。
- **参照**：
  - μ-only（ridge on φ）；
  - field-④（S3-A′ 主臂，u = ④ 估出的 â₁）；
  - 任务盲 / ④ / ③ 的 shared-30 regret（S3-A 口径）。

## §2 读数
- **(a) 名单外排序**：留出源每任务在有效 random（≥ 10 条）上 Spearman(预测, 原始 R0)，报中位 / 均值 [源聚类 CI]。
- **(b) 目录内选择（S3-A 口径，与 S3-B 同表）**：shared-30 按预测排前 3 → 按 end0 选 → 用 end1 评，regret@3 与 regret@1；对任务盲、④ 做源级配对差 [CI]、p。
- **(c) 名单扩展**：S3-A′ 的 30 vs 50 候选、共同分母 o_adj 的 Δ50。
- cross-E 下 (a)、(b) 的变化，[CI]、p。
- 统计同前：17 源聚类 bootstrap 10,000 次、源均值 Wilcoxon。

## §3 门（先于数据；只在主 r = 8 的两个 E 臂上判，另报 r = 4 / 16）
- **F1 内容**：满足任一条即过——
  - 该臂的 shared regret@3 在 cross-E 下变差（Δ > 0 且 p < 0.05）；
  - random-20 逐源平均 Spearman 相对 μ-only 提高且 p < 0.05。
- **F2 价值**：shared-30 regret@3 相对任务盲改善 ≥ 0.010 且 p < 0.05。
- **F3 替代 ④**：shared-30 regret@3 − ④ 的 95% CI 上界 ≤ +0.005。
- **F4 名单外**：random-20 中位 Spearman ≥ 0.50（S3-A′ SPACE 门）。
- **读数码**：`FE-` + 过的门，按 CONTENT / VALUE / REPLACE / SPACE 顺序串接（如 `FE-CONTENT+VALUE`）；两个臂任一过即算过（报告写明哪一臂）。一条都不过 → **FE-INERT**。

## §4 C0
- μ-only、field-④ 两个参照复现 S3-A′ `analysis.json` 的中位 Spearman（0.14 / 0.20），逐位；
- 任务盲 / ④ / ③ 的 shared regret@3 复现 0.0213 / 0.0098 / 0.0061。

## §5 预测（先验）
| # | 预测 | 先验 |
|---|---|---|
| P1 | 至少一臂过 F1（E 的内容被用上） | 40% |
| P2 | 至少一臂过 F2 | 25% |
| P3 | 至少一臂过 F3 | 20% |
| P4 | 至少一臂过 F4 | 5% |
| P5 | E_pt 主臂 shared regret@3 ≤ S3-B 的 E_pt + mean（0.014） | 45% |
| P6 | r = 16 的 random-20 Spearman 高于 r = 4（更多维度有用） | 35% |

## §6 执行
PROTOCOL + 单测先红 → 协议锁 → 实现 → C0 → `--analyze` → REPORT → commit → CLAUDE.md / memory。

## §7 铁律对照
3（只读缓存）；5（先红、§0 声明）；6（PCA、ridge、α 只在训练源；E_sh 用本折编码器，留出源不进其训练）；7；8。

## §8 执行台账（跑后填写）
- 协议锁 3ee4443（4 例先红）→ 实现 7a68ec0（4 例绿）→ `--analyze` 本地 CPU 约 1 h。
- **C0 修订（看过主读数后，只涉及参照比较）**：初版把 shared 参照写死成四舍五入值（0.02131 / 0.00984 / 0.00611），C0 误报 ✗；实际值与 S3-A `analysis.json` 逐位相同（0.021251 / 0.009772 / 0.006109），μ-only / field-④ 中位 Spearman 也逐位相同。改为 `c0_shared_refs` 读文件比较，只重写了 analysis.json / tables.md 的 C0 字段，主分析未重跑。
- 读数码 **FE-VALUE+REPLACE**（预测 3/6），由 E_sh r = 8 给出：regret@3 0.0096，vs 任务盲 +0.012 [0.002, 0.025] p 0.041，E − ④ −0.000 [−0.004, 0.003]；F1 ✗（cross-E +0.007 p 0.50）、F4 ✗（Spearman 0.15 ≈ μ-only）。E_sh r = 8 在 17/17 折都选中 α 上界 10⁴；r 不单调（0.010 / 0.0096 / 0.013）。REPORT.md 入库。
