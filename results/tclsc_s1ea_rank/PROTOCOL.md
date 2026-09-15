# WP16-TCLSC 阶段 1 R 检验 PROTOCOL：瓶颈是"E 被要求预测高秩对象"还是 E 本身——低秩字段加性头（field 头）在 n=7 与 n=72 上对照（零 GPU）

**会话定位**：A/A′/A″ 之后的读法是"预测式主臂需要同域 ≳ 70 个训练集"。用户 2026-09-15 提出另一种解释：**复合空间让 E 要预测的对象在数据集维上高秩**
（WP12 残差 90% 要 22/86 维、8 字段 ICC≈0、多开关次可加），高秩目标才需要很多数据集；而单开关空间里同类特征 n=20 就可预测（WP14 ρ +0.62–0.87）。
本协议直接检验：**把 E 的活改成低秩的**——任务向量只进"字段级加性项"，组合效应留给任务盲主项——同一个 E_A ⊕ φ12、同一 n=7 考场，选择价值是否出现；
同时在 n=72 上看这个头是否保住了价值（低秩头是不是太弱）。分支 wp9-p1。产物 `results/tclsc_s1ea_rank/`。

## §0 已看过的读数

- UEA 7 折 LODO（126 实例）：blind_diff_hard regret@3 0.0294 / regret@1 0.0657；phi_diff 0.0311；eaphi_diff 0.0354（cross 不伤，M0 0.793）。
- UCR 84 LODO（种子 0/1/2，252 实例）：u_blind 0.0220 / 0.0611；u_phi 0.0209；u_eaphi_diff 0.0202 / 0.0458（cross → 0.0231）；种子 3/4/5 eaphi 0.0188。
- 单开关空间（WP14）：删除型开关可由原始空间特征以 n=20 预测；本轮 G-A4：同名字段保持率与字段偏好 |ρ|≈0.25 且反号。
- **没看过**：任何 field 头的数字。

## §1 假设与头的定义

**field 头**（`tclsc_model.py` `head="field"`，`field_rank` r）：
logit(a, b | t) = s₀(l_a) − s₀(l_b) + ⟨w(t), x_a − x_b⟩，其中 s₀ = Linear+ReLU+Linear（任务盲主项，与 diff 头同深同宽）、x = binned CLS 向量（116 维每字段 one-hot）、
w(t) = A·B·t（B: fz→r，A: r→116，无偏置）——任务向量对偏好的影响限定为**逐字段逐档的加性权重**，且数据集维秩 ≤ r；反对称由构造保证。
diff 头是 MLP([l, t])，允许 E 调制任意字段交互（高秩）。

H-R1：n=7 下 field 头（r=4）+ E_A ⊕ φ12 的 regret@3 低于任务盲 0.0294——低秩的活 7 个集就够学，瓶颈在目标秩不在 E。
H-R2：其价值依赖内容（cross 变差）。
H-R3：n=72 下 field 头仍优于任务盲（不因低秩失去 diff 头在 UCR 上的价值）。
对立：field 头在两个 n 上都不比任务盲好（加性项太弱，交互是必需的——瓶颈不是"秩"这一维能解释的）；或 n=7 仍不行而 n=72 行（瓶颈是 E / n）。

## §2 材料

- 考场 1（n=7）：`results/mx3_pq/` 7 折 LODO，126 实例，变体目录 `rank/uea/`，ctx 表 = `tclsc_s1ea_confirm/uea/ea_uea.jsonl`（E_A ⊕ φ12），
  33 号 `table_budget`；参照臂 blind / φ（s1phi）、eaphi_diff（A′）合并重评。
- 考场 2（n=72）：UCR 84 分组 7 折（`tclsc_s1ucr/folds.json`，δ 0.06），252 实例，种子 0/1/2，63 号训练器；参照臂 u_blind / u_phi（s1ucr）、u_eaphi_diff（s1ea）合并重评。
- 消融：cross（E_A ⊕ φ 换成他数据集训练切片的，3 种子）/ zero（E_A 列置 0，φ 不动）。

## §3 臂（binned，hard 标签，3 种子，超参不变）

| 考场 | 臂 | 头 | r | ctx |
|---|---|---|---|---|
| UEA | **`fw4_eaphi_hard`** | field | 4 | E_A ⊕ φ12 ⊕ b |
| UEA | `fw_eaphi_hard` | field | fz（不限秩） | 同上（秩是否重要的对照） |
| UCR | **`u_fw4_eaphi_hard`** | field | 4 | 同上 |
| UCR | `u_fw_eaphi_hard` | field | fz | 同上 |

成本：UEA 42 跑 × 12 s ≈ 9 min；UCR 42 跑 × 19 s ≈ 14 min（4060）。

## §4 门（先于数据；配对 Wilcoxon，regret@3）

- **C0**：参照臂重评 regret@3 逐位（UEA 0.0294 / 0.0311 / 0.0354；UCR 0.0220 / 0.0209 / 0.0202，±0.0002）。
- **G-R1（n=7 解锁，主）**：UEA `fw4_eaphi_hard` < `blind_diff_hard`，p < 0.05。
- **G-R2（内容依赖）**：UEA `fw4_eaphi_hard` cross 消融 regret@3 > normal，p < 0.05。
- **G-R3（n=72 保值）**：UCR `u_fw4_eaphi_hard` < `u_blind_diff_hard`，p < 0.05。
- 只报：fw4 vs eaphi_diff（两考场）；fw（不限秩）同三项；regret@1；M0 / M1 及 cross / zero Δ；逐靶。

**读数码**：

| G-R1 | G-R2 | G-R3 | 码 | 含义 |
|---|---|---|---|---|
| ✓ | ✓ | — | **RANK-BOTTLENECK** | 瓶颈 = 高秩目标：低秩的活 n=7 就够，E 本身没问题；后缀 G-R3 ✓ `+RETAINED` / ✗ `−TRADEOFF`（低秩头在 n=72 上是否保住价值） |
| ✓ | ✗ | — | **RANK-EXISTENCE** | n=7 上有价值但换内容不伤 = 存在效应，不算 E 的功劳 |
| ✗ | — | ✓ | **E-OR-N** | 低秩头在 n=72 保值、n=7 仍不行：瓶颈是 E 的信息量 / 同域数据集数，不是目标秩 |
| ✗ | — | ✗ | **HEAD-TOO-WEAK** | 加性项在两个 n 上都没价值：字段交互是必需的，"秩"这一维解释不了 |

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 | 95% |
| P2 | G-R1 过 | 35% |
| P3 | G-R2 过 | 40% |
| P4 | G-R3 过 | 55% |
| P5 | UEA fw4 regret@3 < eaphi_diff 0.0354（p < 0.05） | 60% |
| P6 | UEA fw（不限秩）regret@3 > fw4（秩约束有用） | 55% |
| P7 | UCR fw4 M1 ≥ 0.60 | 45% |

## §6 执行顺序与协议锁

1. 本文件 + `tests/test_tclsc_s1ea_rank.py`（先红）→ commit。
2. `tclsc_model.py` field 头（构造顺序不动其他头）、33 `ArmSpec.head_rank` + 臂、34 加载、63 臂表、64 `rank_uea(arms=)`、`scripts/66_tclsc_s1ea_rank.py` → 绿 → commit。
3. `66 --train`（脱离，两考场顺序）→ `--rank --analyze` → REPORT → commit → memory。

## §7 铁律对照

铁律 3 零新 R'；铁律 5；铁律 6 同前；铁律 7 field 头只见 binned 向量与浮点 ctx；铁律 8 方向经 `is_better`；关键决策 1（binned MLP）维持——field 头的加性项吃的是同一 binned 向量。

## §8 执行台账（跑后填写）

- 2026-09-15 协议锁 87e9d2a（9 例先红）→ 实现 field 头 / 33 / 34 / 63 / 64 / 66 → 绿 cdad42d（WP16 165 例全绿）；UCR 段目录 bug 1bff851 后重跑。
  训练 UEA 42 × 12 s + UCR 42 × 21 s ≈ 23 min（4060）；`--rank --analyze` CPU ≈ 8 min。
- 读数 **E-OR-N**（预注册主臂 fw4：G-R1 ✗ 0.0325 vs 0.0294、G-R2 ✗、G-R3 ✓ 0.0189 vs 0.0220 p 4e-4；C0 六臂逐位），预测 3/7（P6 反向）。
  **阶梯外**：对照臂 fw（不限秩）UEA regret@3 0.0269 < 任务盲 0.0294 p 0.035、cross → 0.0321 p 3e-4、zero → 0.0333 p 5e-4；UCR 0.0183 全项目最好。须 R′ 确认。详见 REPORT.md。
