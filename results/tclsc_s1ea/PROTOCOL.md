# WP16-TCLSC 阶段 1（A：响应画像 E）PROTOCOL：与 CLS 字段同坐标的机制特征作任务嵌入，UCR 84 分组 7 折（零 GPU）

**会话定位**：真 E 臂读数 E-INERT（`results/tclsc_s1e/REPORT.md`）坐实了 AutoCTS++ 的 task-embedding module 在跨数据集 zero-shot 下的失效形态：
SetTF E 学成"数据集身份"（集内余弦 ≈ 1），比较器依赖它的存在却不依赖内容。用户 2026-09-15 拍板自设 E 模块，第一步 **A = 响应画像 E**：
不学习，直接在目标**训练划分**上测"数据对每个 CLS 相关变换的响应"——(i) 标签保持率（变换后 1-NN 类别边际 / 变换前），(ii) 无标签失真
（变换前后余弦），(iii) 难度 / 水平-尺度信息 / 趋势——坐标与 CLS 字段对齐，让 diff 头不必自己发现"哪一维对应哪个字段"。
依据：WP13c / WP14 的正例正是这一族（highpass05 ← ret_highpass ρ +0.62–0.87、inst_norm ← level_scale_info −0.43）。

语料与折 = n 放大协议（`results/tclsc_s1ucr/`：UCR 84、`folds.json` 分组 7 折、δ 0.060、binned、diff 头、hard 标签、3 种子），
零新 R'。任务盲与 φ12 对照直接复用 s1ucr 已训模型的 `rankings_ucr.json`（同折同种子）。分支 wp9-p1。产物 `results/tclsc_s1ea/`。

## §0 已看过的读数

- s1ucr（同考场）：u_blind_diff_hard M0 0.772 / M1 0.602 / regret@3 0.0220；u_phi_diff_hard 0.782 / 0.611 / 0.0209（cross Δ_M1 +0.019 p 0.13；regret@3 vs blind p 0.045）；
  u_fp_diff_hard nonprobe Δ_M1 +0.018 p 0.067。
- s1e：SetTF E 集内余弦 0.9999 / 跨集 0.58，zero 消融 −0.028、cross +0.002。
- WP11-MECH（UCR 53 集，字段级规则 0/7 过 G1）、WP13c / WP14（删除型变换可预测）、WP15（非增强轴随基翻转）。
- **没看过**：任何响应画像特征作比较器 ctx；84 集全表的保持率（WP11 只算了合格 A 53 集且 op 集合不同）。

## §1 假设

H-A1：响应画像 E 在 LODO 下被 diff 头使用（cross 消融正向）且强于 φ12（φ12 描述"长什么样"，E_A 描述"对变换怎么反应"）。
H-A2：其价值超出粗排序——regret@3 低于任务盲，且 E_A ⊕ φ12 低于 φ12。
对立：好半区偏好高秩（WP12），响应画像只能覆盖增强字段那一层（官方口径价值小），选择价值仍零。

## §2 特征定义（`scripts/63_tclsc_s1ea.py --features` → `ea_ucr.jsonl`，每集一行，只用 `split_task_train_eval` 的 train 侧，与 42 号 φ 同划分；X 用 41 号加载器原样（注册表 crop_len ≤ 512，不再逐样本 z-score，与 WP14 `compute_phi_v2` 同口径））

**标签保持率的定义与 WP14 已过留出的正例同式**：`info(F) = clip((LOO-1NN acc − chance)/(1 − chance), −1, 1)`（F 逐维标准化，wp14 51 号 `_info`），
`ret_<op> = info_after − info_raw`。两种形式按字段语义分发：
- **确定性输入变换**（部署时训练 / 评估同一前向）→ *within* 形式：info(op(X)) 在变换后集合内 LOO；
- **随机增强 / 视图变换**（两视图之一被变换，锚不变）→ *cross* 形式：查询 = op(x)，参考 = 未变换 X（排除自身），随机 op 固定种子 (0,1,2) 均值。

| 块 | 维 | 定义 |
|---|---|---|
| **R 标签保持率**（11） | `ret_<op>` | cross：jitter@0.5、rescale@0.5、point_mask@0.5、freq_mask@0.5（`autocls.contrastive.augmentations` 原函数）、polarity（−x）、time_reverse；within：inst_norm、detrend、highpass05、lowpass25、diff1（wp14 `apply_input_transform` 同式，移植到 `mech_features`）。（input_mask@0.5 在原始空间 C=1 下与 point_mask@0.5 同一算子，不重复列） |
| **S 无标签失真**（10） | `sim_<op>` | 上表除 polarity（恒 −1）外 10 个 op：mean cosine(x, op(x))（展平，随机 op 种子 0） |
| **D 结构**（4） | `hardness_1nn`、`level_scale_info`、`trend_info`、`trend_strength` | wp11 `hardness_1nn`（逐样本 z-score 后 LOO 1-NN 难度）；wp14 51 号 `_info(concat(mean_t, std_t))`、`_info(concat(slope, mean))`、`linear_fit` R² 均值 |

E_A = 25 维；nan → 折内训练集列均值填补并记 `n_nan`。ctx = 折内标准化 E_A（μ/σ 只用训练集，σ 下限同 φ）⊕ 预算 one-hot(3)。

## §3 臂（`scripts/63` 自带臂表，S1 分组 7 折，3 种子，UCR 训练器复用 60 号 build_records / catalog_for / grouped_folds）

| 臂 | ctx | 头 | 用途 |
|---|---|---|---|
| **`u_ea_diff_hard`** | E_A(25) ⊕ b | diff | **主臂** |
| `u_ea_concat_hard` | E_A ⊕ b | concat | 头对照 |
| `u_eaphi_diff_hard` | E_A ⊕ φ12 ⊕ b | diff | 与 φ 叠加 |
| ref `u_blind_diff_hard` / `u_phi_diff_hard` | — / φ12 | diff | s1ucr 已训（同折同种子），从其 `rankings_ucr.json` 重评 |

63 跑 × ~30 s ≈ 35 min（4060）；特征计算 84 集 × (11 op × ≤3 种子) 的 LOO 1-NN（n ≤ 1024、T ≤ 512）≈ 10 min CPU。

## §4 门（先于数据；252 实例，指标同 s1ucr：M1 好半区 / M2 / M0 / regret@1/@3；cross = E_A 换成他训练集的，3 种子；zero = E_A 置 0）

- **C0**：从 s1ucr 的 `rankings_ucr.json` 重评 u_blind_diff_hard M0 = 0.772 ± 0.001、u_phi_diff_hard 0.782 ± 0.001（同函数逐位）。
- **G-A1（用了）**：`u_ea_diff_hard` cross Δ_M1 ≥ **0.03** 且 p < 0.05。
- **G-A2（有价值）**：`u_ea_diff_hard` regret@3 < `u_blind_diff_hard`，配对 Wilcoxon p < 0.05。
- **G-A3（超出 φ）**：`u_eaphi_diff_hard` regret@3 < `u_phi_diff_hard`，配对 p < 0.05。
- G-A4（机制对齐，只报）：对 AutoCL 的 4 个随机增强字段 f ∈ {jitter, rescale, point_mask, freq_mask}，`ret_f` 与该集的**字段偏好** u_f
  （@290 目录 20 条里 f ≥ 0.5 档候选的 within-task z 均值 − f = 0 档候选的 z 均值，两侧各 ≥ 2 条才定义）在 84 集上的 Spearman；
  报 |ρ| ≥ 0.3 且 p < 0.05 的字段数（预期符号 +：保持率高 → 开该增强更好）。
- G-A5（头对照，只报）：`u_ea_concat_hard` Δ_M1 零。

**读数码**：

| G-A1 | G-A2 | G-A3 | 码 | 含义 |
|---|---|---|---|---|
| ✓ | ✓ | ✓ | **EA-ALIVE** | 自设 E 成立：进 B（度量学习）与 UEA 官方口径 |
| ✓ | ✓ | ✗ | **EA-PHI-EQUIV** | 有价值但不比 φ12 多：响应画像与形状描述同源，B 用 φ ⊕ E_A 合并输入 |
| ✓ | ✗ | — | **EA-USED-NO-GAIN** | 用了但只到粗排序（与 φ 同形）：好半区上界再次坐实，B 只值得做偏好几何对齐的验证版 |
| ✗ | — | — | **EA-DEAD** | 响应画像进不了 diff 头：机制特征线到此 |

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 逐位 | 95% |
| P2 | G-A1 过 | 55% |
| P3 | G-A2 过 | 40% |
| P4 | G-A3 过 | 30% |
| P5 | G-A4 ≥ 2/4 个字段对齐（|ρ| ≥ 0.3，p < 0.05） | 45% |
| P6 | G-A5 concat 零 | 80% |
| P7 | E_A 臂 M0 ≥ φ 臂 M0（0.782） | 55% |

## §6 执行顺序与协议锁

1. 本文件 + `tests/test_tclsc_s1ea.py`（先红）→ commit。
2. 实现：`autocls/data/mech_features.py`（自 wp11-mech 移植，callable op 契约）、`scripts/63_tclsc_s1ea.py`（--features / --train / --rank / --analyze）→ 单测绿 → commit。
3. `63 --features` → `63 --train`（脱离启动）→ `63 --rank --analyze` → REPORT。

## §7 铁律对照

铁律 3 零新 R'；铁律 5；铁律 6 特征只用目标训练划分、标准化只用训练集；铁律 7 `mech_features` 是任务侧模块（按 task_type 分发），比较器不见其语义；铁律 8 方向经 `is_better`。

## §8 执行台账（跑后填写）

（空）
