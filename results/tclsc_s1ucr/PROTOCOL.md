# WP16-TCLSC 阶段 1（n 放大）PROTOCOL：diff 头 + φ / 转导指纹，UCR 84 数据集分组 7 折（零 GPU）

**会话定位**：φ 臂（PHI-DEAD）与转导指纹臂（FP-DEAD）在 UEA 8 源 / 7 折 LODO 下的共同形状 = **diff 头用了条件变量、但外推不了**
（φ：只迁移粗排序；指纹：反向消融 = 过拟合 7 个训练集的指纹分布）。oracle 身份 token 在同分布下 +0.205 说明头不是瓶颈。
最省的解释是**训练侧只有 7 个数据集**——学"任务描述 → 偏好"这个函数的样本数是 7。本协议把 n 放大 12×：WP7 留下的 UCR 84 语料
（`results/ucr_rprime/rprime_cache.jsonl`，84 数据集 × 20 CLS × 3 截点，零新 R'），84 数据集分组 7 折（每折 12 靶、72 训练集）。
WP7 在**打分器**形态（GBDT f(CLS, φ)→z）上 hgb regret@3 0.0209 vs pooled 0.0199（p 0.41，与置换同带）；本协议是**比较器 + diff 头**形态。

数据只读：`results/ucr_rprime/rprime_cache.jsonl`、`results/ucr_module/phi_ucr.jsonl`（WP7 φ14，取前 12 列 = `PHI_COLS`）、
`.claude/worktrees/wp11-mech/results/mx4_mech/ucr_noise.jsonl`（δ 标定材料）。分支 wp9-p1。产物 `results/tclsc_s1ucr/`。

## §0 已看过的读数

- φ 臂（UEA LODO）：phi_diff_hard M1 Δ +0.011 p 0.12、M0 Δ +0.035 p 3e-11、regret@3 0.031 vs 0.029；逐折 3 正 4 负。
- 指纹臂（UEA LODO）：nonprobe Δ_M1 −0.042、regret@3 0.031 vs 0.027、probe6_top3 0.015；逐折 UWave/Cricket 正、HW/PS 负。
- WP7 U1（UCR 84 LODO，打分器形态）：hgb 0.0209 vs pooled 0.0199 p 0.41；twin_knn 0.0186 p 0.083；26/84 贴顶；交互份额 9.6%。
- WP12：UCR 好半区先验信息量 0.005、自测 @25% 0.133。
- **δ 标定（本协议新看的探索读数，§0 如实声明）**：WP11 的 UCR 噪声轨迹（53 集，seed 1/2 + 主缓存 seed 0，1431 个重复对）
  |ΔR'| P50 0.013 / P90 0.045 / **P95 0.059**；逐集 P95 中位 0.042。按 mx3_pq 的标定规则（δ = P95 上取整到 0.005）取 **全局 δ = 0.060**。
- **没看过**：任何在 UCR 语料上训的比较器。

## §1 假设

H-N：在 72 个训练数据集上，diff 头能学到可外推的 φ→偏好 / 指纹→偏好函数，表现为留出数据集上 cross 消融正向且 regret@3 低于任务盲。
对立：交互本身在 UCR 语料上份额小（9.6%，含 k=1 噪声）且好半区高秩（WP12），n 放大也读不出——表现与 n=7 相同（粗排序有、好半区无）。

## §2 材料与口径

- 语料：84 数据集，每集 1 任务（`<name>_00`），目录 20 条（shared-10 与 mx3_pq 逐位相同 + random-10），截点 {60, 120, 290}；
  engine task id = `<name>_00@<b>`（33 号同式）。
- 折：数据集名排序后 `i % 7` 分组（每折 12 靶，72 训练集），逐位确定。
- **δ = 0.060 全局**（训练配对与评估 M0 同值；§0 标定）。
- CLS 编码 binned 116；头 diff / concat；标签 hard；超参 `pretrain_tclsc.yaml` 原样（lr 1e-4、batch 64、200 epoch、curriculum 40）。
- ctx：
  - φ：`PHI_COLS` 12 列（`phi_ucr.jsonl`），折内训练集标准化 ⊕ 预算 one-hot（33 号 `context_for(ctx_kind="phi_budget")` 同式）；
  - 指纹：训练折先验前 6 条 shared 在目标 @60 的实测（`fp_probes` / `fp_raw` / `fingerprint`，33 号同式），折内标准化 ⊕ 预算 one-hot。
- 排名 3 种子多数票；消融 cross（条件变量换成他训练集的，3 种子）/ zero；指纹臂两视图 nonprobe（top_k=7）/ all20（top_k=10）。
- 指标同前（M1 / M2 / M0 / regret@1 / @3 / Spearman），**252 实例**（84 靶 × 3 预算）；M1 只算有存活对的实例。
- 同成本参照 `probe6_top3`（@120/@290，168 实例）。

## §3 臂（`scripts/60_tclsc_s1ucr.py` 自带臂表，复用 33 号 `ArmSpec` / `build_model` / `save_model` / `context_for`、34 号 loader、56/57/59 号指标）

| 臂 | 任务支路 | ctx | 头 | 用途 |
|---|---|---|---|---|
| `u_blind_diff_hard` | none | — | diff | 同头任务盲对照 |
| `u_phi_concat_hard` | ctx | φ⊕b | concat | 头对照 |
| **`u_phi_diff_hard`** | ctx | φ⊕b | diff | φ 主臂 |
| **`u_fp_diff_hard`** | ctx | 指纹⊕b | diff | 指纹主臂 |

4 臂 × 7 折 × 3 种子 = 84 跑；每跑 4320 记录 ≈ 20–30 s（4060）≈ 40 min。

## §4 门（先于数据）

- **C0**：`u_blind_diff_hard` pooled M0 ∈ [0.65, 0.85]（δ=0.06 下任务盲比较器的合理带；越界说明 δ / 编码接错）；WP7 pooled 先验的 regret@3 0.0199
  作外部参照（同语料、不同形态）。
- **G-U1（φ 用了）**：`u_phi_diff_hard` Δ_M1（cross）≥ 0.03 且 p < 0.05（n ≤ 252）。
- **G-U2（φ 有价值）**：`u_phi_diff_hard` regret@3 < `u_blind_diff_hard`，配对 Wilcoxon p < 0.05。
- **G-U3（指纹用了，nonprobe）**：`u_fp_diff_hard` nonprobe Δ_M1 ≥ 0.03 且 p < 0.05。
- **G-U4（指纹有价值，nonprobe）**：`u_fp_diff_hard` nonprobe regret@3 < `u_blind_diff_hard` nonprobe regret@3，p < 0.05。
- **G-U5（同成本）**：`u_fp_diff_hard` all20 regret@3（@120/@290）< `probe6_top3`，p < 0.05。
- G-U6（头对照）：`u_phi_concat_hard` Δ_M1 零——只报。

**读数码**（φ 与指纹各判一次，再合成）：
- φ：G-U1 ∧ G-U2 → **UCR-PHI-ALIVE**；G-U1 ∧ ¬G-U2 → **UCR-PHI-USED-NO-GAIN**；¬G-U1 → **UCR-PHI-DEAD**。
- 指纹：G-U3 ∧ G-U4 ∧ G-U5 → **UCR-FP-ALIVE**；G-U3 ∧ G-U4 → **UCR-FP-EXTRAPOLATES**；G-U3 → **UCR-FP-USED-NO-GAIN**；否则 **UCR-FP-DEAD**。
- 合成读法：任一 ALIVE / EXTRAPOLATES → "n=7 是瓶颈"成立，下一步把 UCR 训的比较器带到 UEA 官方口径考试；两者都 DEAD / USED-NO-GAIN →
  "n 放大也读不出"，预测式在比较器形态下的可迁移信号止于粗排序，剩真 E（GPU）与换形式。

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 带内 | 85% |
| P2 | G-U1 过 | 60%（n=7 时 M0 已显著，n 放大后 M1 有望） |
| P3 | G-U2 过 | 35% |
| P4 | G-U3 过 | 40% |
| P5 | G-U4 过 | 25% |
| P6 | G-U5 过 | 15% |
| P7 | 指纹臂 nonprobe cross Δ_M1 由负转正（n=7 是 −0.042） | 55% |
| P8 | φ 臂 M0 cross Δ 仍显著（p<0.01） | 80% |

## §6 成本

84 跑 ≈ 40 min（4060，CPU 空闲时）；排名/分析分钟级；零新 R'。

## §7 执行顺序与协议锁

1. 本文件 + `tests/test_tclsc_s1ucr.py`（先红）→ commit（协议锁）。
2. 实现 `scripts/60_tclsc_s1ucr.py`（折 / 记录 / ctx / 训练 / 排名 / 分析）→ 单测绿 → commit。
3. `60 --train`（脱离启动 + `--resume`）→ `60 --rank --analyze` → REPORT。

## §8 铁律对照

- 铁律 3 零新 R'；铁律 5；铁律 6 折内：φ 标准化、探针选择、指纹标准化只用训练集；铁律 7 ctx 只在脚本侧装配；铁律 8 方向经 `is_better`。
- 关键决策 8 不触及。语料口径 = WP7 的 UCR 全数据集任务（CROP_LEN 512、N≤1024 分层子采样），与 UEA 切片口径不并排比较绝对值。

## §9 执行台账（跑后填写）

（空）
