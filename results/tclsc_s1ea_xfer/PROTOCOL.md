# WP16-TCLSC 阶段 1 A″ 跨语料部署检验 PROTOCOL：UCR 84 全训的响应画像比较器 zero-shot 排 UEA 8 集切片（零 GPU）

**会话定位**：A′（`results/tclsc_s1ea_confirm/REPORT.md`）读 EA-NOT-CONFIRMED+UCR-REP：E_A ⊕ φ12 的选择价值在 UCR 84 集 LODO（训练侧 72 集）
跨种子成立且内容依赖，在 UEA 8 集 LODO（训练侧 7 集）反向。读法 = 分界变量是训练侧数据集数。A″ 直接检验这个读法：**把训练侧换成 UCR 84 全部**，
zero-shot 排 UEA 的切片，与 UEA 自训（n=7）的臂在同 126 实例上配对。这同时是官方口径部署（训练侧 UCR 84 → 目标 UEA 靶）前的必经门。
用户 2026-09-15 拍板。分支 wp9-p1。产物 `results/tclsc_s1ea_xfer/`。

## §0 已看过的读数

- UEA 7 折 LODO 自训（126 实例）：blind_diff_hard regret@3 0.0294 / regret@1 0.0657 / M0 0.756；phi_diff_hard 0.0311 / 0.0771 / 0.780；eaphi_diff_hard 0.0354 / 0.0791 / 0.793。
- UCR 84 LODO（种子 0–5）：eaphi regret@3 0.0188–0.0202、cross 消融 → 0.022；u_blind 0.0197–0.0220。
- UEA 45 切片的 E_A 已算（`tclsc_s1ea_confirm/uea/ea_uea.jsonl`，集内余弦 0.83）。
- **没看过**：任何 UCR 训练的比较器在 UEA 切片上的排名。

## §1 假设

H-X1：UCR 84 全训的 `u_eaphi_diff_hard` 在 UEA 126 实例上 regret@3 低于 UEA 自训任务盲（0.0294）——训练侧数据集数是瓶颈，跨语料（单变量 → 多变量）不是。
H-X2：其价值超出同语料任务盲——低于 UCR 全训的 `u_blind_diff_hard`（区分"84 集学到的任务盲先验更好"与"响应画像给出偏好"）。
H-X3：内容依赖——把目标切片的 E_A ⊕ φ 换成其他 UEA 数据集切片的，regret@3 变差。
对立：语料位移（单变量 UCR → 多变量 UEA，φ 的 base_mean / base_band 口径不同，log10_C / chan_cv 在 UCR 恒 0）压过 n 的增益；或 UCR 学到的任务盲先验本身就比 UEA 自训强（价值不在画像）。

## §2 材料

- 训练：`results/ucr_rprime/rprime_cache.jsonl` 84 集 × 20 × 3（δ 0.06），**无折**（train = 84 集，test = ∅），3 种子（0/1/2），
  臂 `u_blind_diff_hard` / `u_phi_diff_hard`（60 号训练器，φ = `phi_ucr.jsonl`）/ `u_eaphi_diff_hard`（63 号训练器，E_A = `ea_ucr.jsonl`）；标准化统计 = 84 集。
  产物 `results/tclsc_s1ea_xfer/models/fold0/`（fold 名固定 fold0）。
- 目标：UEA 7 个折靶的 42 切片 × 3 预算 = 126 实例（与 s1phi / A′ 同实例集；SCP2 3 切片只作附录）；候选 = `results/mx3_pq` 目录（每切片 20 条）；
  δ = mx3_pq `delta_calibration.json`（M0 / M1 用）。ctx = UEA 切片自己的 E_A / φ12（`tclsc_s1ea_confirm/uea/ea_uea.jsonl`）经 **UCR 的 μ/σ** 标准化；
  **跨语料护栏（先于数据）**：训练侧 σ 在下限（≤ 10 × PHI_SD_FLOOR，即 UCR 恒定的 log10_C / chan_cv）的列在目标侧置 0；标准化值裁到 [−4, 4]。
- 参照臂（UEA 自训，n=7）：`blind_diff_hard` / `phi_diff_hard`（s1phi）与 `eaphi_diff_hard`（A′）的 normal 排名合并重评。
- 消融（UCR 训练的 ctx 臂）：cross = ctx 换成**其他 UEA 数据集**切片的（58 号 `cross_slice_map`，供体池 = 45 切片，3 种子）；zero = E_A 列置 0（u_eaphi）/ φ 列置 0（u_phi）。

## §3 臂

| 臂 | 训练侧 | ctx | 用途 |
|---|---|---|---|
| **`u_eaphi_diff_hard`** | UCR 84 | E_A ⊕ φ12 ⊕ b | **主臂** |
| `u_blind_diff_hard` | UCR 84 | — | 同语料任务盲对照（H-X2） |
| `u_phi_diff_hard` | UCR 84 | φ12 ⊕ b | φ 对照 |
| ref `blind_diff_hard` / `phi_diff_hard` / `eaphi_diff_hard` | UEA 7 折自训 | — / φ / E_A ⊕ φ | 部署基线（H-X1） |

成本：9 跑 × ≈25 s（84 集 5040 记录，4060）≈ 4 min；排名 CPU ≈ 2 min。

## §4 门（先于数据；126 实例配对 Wilcoxon）

- **C0**：三条参照臂重评 regret@3 = 0.0294 / 0.0311 / 0.0354（±0.0002）。
- **G-X1（部署价值，主）**：`u_eaphi_diff_hard` regret@3 < `blind_diff_hard`（UEA 自训），p < 0.05。
- **G-X2（画像价值，主）**：`u_eaphi_diff_hard` regret@3 < `u_blind_diff_hard`（UCR 全训），p < 0.05。
- **G-X3（内容依赖）**：`u_eaphi_diff_hard` cross 消融 regret@3 > normal，p < 0.05。
- 只报：regret@1 同三项；M0 / M1 与 cross / zero Δ；`u_phi_diff_hard` 同项；`u_blind_diff_hard` vs `blind_diff_hard`（84 集任务盲先验 vs 7 集）；逐靶 regret@3。

**读数码**：

| G-X1 | G-X2 | 码 | 含义 |
|---|---|---|---|
| ✓ | ✓ | **XFER-ALIVE** | 训练侧数据集数是瓶颈、画像给出偏好、跨语料可部署：进 B（度量学习 E，UCR 语料）与官方口径 6 靶部署 |
| ✓ | ✗ | **XFER-CORPUS** | 部署上赢，但赢的是 84 集学到的任务盲先验，画像没贡献：任务盲先验换训练语料即可，B 不做 |
| ✗ | ✓ | **XFER-RELATIVE** | 画像相对同语料任务盲有价值，但语料位移让 UCR 训练的比较器整体输给 UEA 自训：部署需要合池训练（UCR + 源池） |
| ✗ | ✗ | **XFER-DEAD** | 跨语料不成立：预测式主臂的成立域 = 同语料 LODO（n ≳ 70），写 analysis |

后缀：G-X3 ✓ → `+CONTENT`，✗ → `−CONTENT`。

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 | 95% |
| P2 | G-X1 过 | 25% |
| P3 | G-X2 过 | 40% |
| P4 | G-X3 过 | 45% |
| P5 | `u_blind_diff_hard`（UCR）regret@3 ≥ `blind_diff_hard`（UEA）（84 集先验在 UEA 上不比 7 集好） | 55% |
| P6 | `u_eaphi_diff_hard` 在 UEA 上 M0 ≥ 0.75 | 50% |
| P7 | `u_eaphi_diff_hard` regret@1 < `blind_diff_hard`（p < 0.05） | 25% |

## §6 执行顺序与协议锁

1. 本文件 + `tests/test_tclsc_s1ea_xfer.py`（先红）→ commit。
2. `scripts/65_tclsc_s1ea_xfer.py`（--train / --rank / --analyze）→ 单测绿 → commit。
3. `65 --train`（前台，4 min）→ `--rank --analyze` → REPORT → commit → memory。

## §7 铁律对照

铁律 3 零新 R'；铁律 5；铁律 6 训练只见 UCR，UEA 只考试（E_A / φ 用目标训练划分）；铁律 7 ctx 只是浮点向量；铁律 8 方向经 `is_better`。

## §8 执行台账（跑后填写）

（空）
