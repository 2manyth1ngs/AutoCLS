# WP16-TCLSC 阶段 1（φ 臂）PROTOCOL：diff 头 + 可迁移任务变量 φ，7 折 LODO（零 GPU）

**会话定位**：重启 T-CLSC 第三步。L3 阶段 0 证明 diff 头能用 oracle 身份 token（Δ_M1 +0.205，`results/tclsc_l3/REPORT.md`）；
身份 token 对未见数据集无定义，阶段 1 问的是：**换成对新数据集可计算的任务变量后，LODO 下比较器还用不用它、用了值不值。**
阶段 1 有多个候选变量（φ 元特征 / 真 E / E⊕φ / 转导指纹）；本协议只做零 GPU 的 **φ 臂**，真 E 臂另立（需 `33 --fit-folds` GPU 请示）。

**φ**：WP6-TF 的 12 维切片特征（`results/mx4_task_feature/task_features.jsonl`，`PHI_COLS` = log10_T / log10_C / k / log10_N /
base_mean / base_band / spec_conc / hf_ratio / acf_tau / ex_kurt / chan_cv / samp_cv），**只用目标训练划分**计算（WP6-TF 已合规）。

数据 `results/mx3_pq/` 只读，零新 R'（铁律 3）。分支 wp9-p1。产物 `results/tclsc_s1phi/`。

## §0 已看过的读数

- L3 阶段 0（S2 身份 token）：diff+hard Δ_M1 +0.205 p 9e-4，regret@1 0.062 / diff+zdiff 0.027 vs 任务盲 0.080。
- L1（LODO cls_only_binned）：pair acc 0.758、top-10 acc 0.528、regret@1 0.066、regret@3 0.032（`results/tclsc_l1/analysis.json`）。
- WP6-TF：同一份 φ 进 GBDT 残差打分器（不是比较器）与 5 个 φ 置换臂同带（regret@3 0.0314 vs 0.0313）——一句话约束：
  φ 在"打分器 + 任务盲先验残差"形态下没读出信号；本协议换的是形态（diff 比较器 + 反对称头），不是特征。
- WP1：prior3 regret@3 0.032、SH6-3-3 0.018（@290，LODO）——外部参照。
- **没看过**：任何以 φ 作 ctx 的比较器；diff 头在 LODO 下的任务盲读数。

## §1 假设

H-φ：diff 头 + φ 在 LODO 下 cross 消融掉分（用了 φ），且 regret@3 低于同头任务盲臂（用了有价值）。
对立：用 7 个训练数据集学 φ→偏好的函数不可辨识（WP6-TF 的读法），表现为 Δ 零或"用了但更差"（过拟合 7 个点）。

## §2 材料与口径

- 划分：S1 = 7 折 LODO（靶 = HW / UM / PS / UWave / Ethanol / Cricket / HMD；SCP2 只训不考）；考侧 126 实例。
- ctx = **标准化 φ(12) ⊕ 预算 one-hot(3) = 15 维**。标准化 μ/σ 只用该折**训练切片**（铁律 6），存进模型 meta（`phi_mu` / `phi_sd` / `phi_cols`），
  排名时按 meta 重建，逐位一致。σ 下限 1e-6。
- CLS 编码 binned 116；头 diff（主）/ concat（对照）；标签 hard（主）/ zdiff（副臂，慢）；超参 `pretrain_tclsc.yaml` 原样。
- 排名：每臂 3 种子多数票（`rank_local_models(ctx_rows=)`）。**消融**：
  - cross：每个考侧切片的 φ 换成**另一数据集的训练切片**的 φ（`cross_slice_map`，3 个种子；预算位不动）；
  - zero：φ 部分置 0（= 训练均值），预算位不动。
- 指标同 L3：M1 好半区配对 acc（主）、M2 |Δz| 加权全对 acc、M0 δ 对 acc、regret@1 / @3、Spearman；pooled 126 实例（M1 只算有存活对的实例）；
  逐折表；配对 Wilcoxon。

## §3 臂（`scripts/33` 的独立表 `PHI_ARMS`，全部 S1、3 种子、铁律 6 合规）

| 臂 | 任务支路 | ctx | 头 | 标签 | 用途 |
|---|---|---|---|---|---|
| `blind_diff_hard` | none | — | diff | hard | 同头任务盲对照（价值门的基线） |
| `phi_concat_hard` | ctx | φ⊕b | concat | hard | 头对照：concat 是否仍零 |
| **`phi_diff_hard`** | ctx | φ⊕b | diff | hard | **主臂** |
| `phi_diff_zdiff` | ctx | φ⊕b | diff | zdiff | 副臂（每跑 ≈ 5 min，可后补） |

外部参照（不重跑）：L1 `cls_only_binned`（concat 头任务盲）、WP1 prior3 / SH6-3-3 的 regret@3。

## §4 门（先于数据）

- **C0**：`blind_diff_hard` pooled 配对 acc 与 L1 `cls_only_binned` 0.758 差 ≤ 0.02（diff 头任务盲 ≈ concat 头任务盲；超出则只报不判，
  后续门一律用 `blind_diff_hard` 作基线）。
- **G-P1（用了 φ）**：`phi_diff_hard` Δ_M1 = normal − mean_s cross_s ≥ **0.03** 且 Wilcoxon p < 0.05；zero 消融同向（只报）。
- **G-P2（用了有价值）**：`phi_diff_hard` 的 regret@3 < `blind_diff_hard` 的 regret@3，126 实例配对 Wilcoxon p < 0.05。
- **G-P3（头对照）**：`phi_concat_hard` Δ_M1 < 0.03 或 p ≥ 0.05（零）——只报，不进读数码。
- 次级：M2 / M0 的 Δ；逐折 Δ_M1 的符号数；regret@1；`phi_diff_zdiff` 若跑则同上报。

**读数码**：

| G-P1 | G-P2 | 码 | 含义 / 下一步 |
|---|---|---|---|
| ✓ | ✓ | **PHI-ALIVE** | 可迁移任务变量 + diff 头在 LODO 成立：真 E 臂与 E⊕φ 臂立项（GPU），并进官方口径 |
| ✓ | ✗ | **PHI-USED-NO-GAIN** | 比较器用 φ 但不涨（7 点过拟合）：转导指纹臂（零 GPU）先于真 E；φ 降维 / 正则作副线 |
| ✗ | — | **PHI-DEAD** | φ 这一路在比较器形态下也零：真 E / 转导指纹是剩下的两条，真 E 先做 E 诊断（F1）再决定 |

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 带内 | 80% |
| P2 | G-P1 过 | 35% |
| P3 | G-P2 过 | 25% |
| P4 | G-P3 零（concat 仍零） | 70% |
| P5 | 逐折 Δ_M1 > 0 的折 ≥ 5/7（若 G-P1 过） | 50% |
| P6 | `phi_diff_zdiff` regret@3 < `phi_diff_hard`（若跑） | 45% |

## §6 成本

- hard 臂 3 × 7 折 × 3 种子 = 63 跑 × ~12 s ≈ 15 min（4060）；zdiff 副臂 21 跑 × 5 min ≈ 105 min（可后补）；排名/分析分钟级。零新 R'。

## §7 执行顺序与协议锁

1. 本文件 + `tests/test_tclsc_s1phi.py`（先红）→ commit（协议锁）。
2. 实现：33 `PHI_ARMS` / `ctx_kind="phi_budget"` / `PQContext(phi_file=)` / 折内标准化 / meta φ 统计；`scripts/58_tclsc_s1phi.py` → 单测绿 → commit。
3. `58 --prepare --train`（hard 三臂）→ `58 --rank --analyze` → REPORT；zdiff 副臂视时间后补。

## §8 铁律对照

- 铁律 3 零新 R'；铁律 5 先测后写；铁律 6 φ 只用目标训练划分、标准化只用源折；铁律 7 φ 只在脚本侧装配成浮点向量，模型/引擎不见其语义；
  铁律 8 方向经 better_fn。

## §9 执行台账（跑后填写）

- 2026-09-14 协议锁 371dc0d（15 例先红）→ 实现 33 PHI_ARMS / phi_budget / phi_file / meta φ 统计 + 58 号 → 166 例绿（含邻近）。
- `--train` 63 跑本地 4060（12–17 s/跑，`train.log`）；58 号 `--rank` 首跑因 `load_source` 取 57 号版本（2 元组）修一行。
- `--rank --analyze` → **PHI-DEAD**（G-P1 Δ_M1 +0.011 p 0.12 ✗；G-P2 regret@3 0.031 vs 0.029 ✗；G-P3 ✓）；但 M0/M2 cross 消融 p<1e-10（92:23）——
  φ 被用于粗排序、对好半区无增益。详见 REPORT.md。zdiff 副臂未跑。
