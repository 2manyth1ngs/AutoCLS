# WP16-TCLSC 阶段 1 T 检验 PROTOCOL：转导型响应画像 E_T（目标上一条探针短训的表示空间）vs 原始空间 E_A，field 头，UEA n=7 / UCR n=72（本地 GPU 小量）

**会话定位**：R 检验 + §9 消融给出的账：field 头下数据集特征的贡献 = top-3 +0.009 acc（UEA）/ +0.002（UCR），来自原始空间 1-NN 的响应画像 E_A。
诊断（2026-09-15 讨论）：提取方式是次因——原始空间 1-NN 只能抓删除型变换的效应，WP14 阶段 2 显示增强类字段（rescale、polarity）要靠**表示空间**的响应
（干净基一条轨迹 @60/@290 的 rep_ret_*）才可预测。本协议把 E_A 的坐标原样搬到表示空间：在目标训练划分上用一个**固定的任务盲探针**（GGS）短训一条轨迹，
在 @60（= SH 第一级已有的训练量，部署时零额外成本）与 @290 取表示，算同一套 11 个变换的响应。用户 2026-09-15 拍板。分支 wp9-p1。产物 `results/tclsc_s1et/`。

## §0 已看过的读数

- UEA n=7（126 实例，regret@3）：blind 0.0294 / φ 0.0311 / eaphi_diff 0.0354 / **fw（E_A ⊕ φ，field 头）0.0269** / fwb（无特征）0.0304 / fw@噪声 0.0300。
- UCR n=72（252 实例）：u_blind 0.0220 / u_fw 0.0183 / u_fwb 0.0196。
- WP14 事后转导探索（UCR 53，干净基）：rep_ret_highpass05@60 → rescale、rep_ret_polarity@290 → polarity 有留出信号（事后，未预注册）。
- **没看过**：GGS 探针在任何 UEA 切片 / UCR 集上的表示空间响应；E_T 作 ctx 的任何数字。

## §1 假设

H-T1（提取方式）：E_T@60 ⊕ φ12 在 field 头下的 regret@3 低于 E_A ⊕ φ12（fw）——表示空间的响应比原始空间更贴近 CL 目标在乎的不变性。
H-T2：E_T@60 ⊕ φ12 低于任务盲，且内容依赖（cross 变差）。
对立：表示空间响应与原始空间同信息（ET-EQUIV），或探针表示引入的噪声（60 iters 的表示还没成形）反而更差。

## §2 特征定义（`scripts/69_tclsc_s1et.py --features` → `et_uea.jsonl` / `et_ucr.jsonl`，每切片 / 每集一行）

- 探针：`pq.ggs_cls()`（AutoCL GGS，任务盲、不由 R' 选出，无泄漏），`build_encoder` + `pretrain_encoder(max_iters=290, checkpoints=(60, 290))`，seed 0，batch 128，
  只用目标**训练划分**（与 φ / E_A 同划分，`split_task_train_eval` 0.3 / `run_seed_for(tid, "train_eval_split")`）。
- 表示：`encode_features`（时间维 max-pool，与 R' 评估同）。R0 = enc(X)，Rp = enc(op(X))，op = `mech_features.RESPONSE_OPS` 的 11 个（原始空间施加，随机 op 种子 (0,1,2) 均值）。
- 每个截点 b ∈ {60, 290} 的列（22 维）：`rep_<op>@b`（11：within 形式 op → info(Rp) − info(R0)；cross 形式 op → info_cross(Rp → R0) − info(R0)，同 E_A 的分发）、
  `rcos_<op>@b`（10：mean cos(R0[i], Rp[i])，polarity 除外）、`rinfo@b`（info(R0)）。两截点共 44 维；nan → 折内均值。
- 同时记 @290 的 R'（与缓存里 GGS 的 R' 对账，4060 vs pod 允许 ±1 样本）作 C0。

## §3 臂（field 头不限秩，binned，hard，3 种子）

| 考场 | 臂 | ctx |
|---|---|---|
| UEA / UCR | **`ft60_hard` / `u_ft60_hard`** | E_T@60（22）⊕ φ12 ⊕ b |
| UEA / UCR | `ft290_hard` / `u_ft290_hard` | E_T@290 ⊕ φ12 ⊕ b |
| UEA / UCR | `ftea_hard` / `u_ftea_hard` | E_T@60 ⊕ E_A（25）⊕ φ12 ⊕ b |
| ref | blind / fw / fwb（UEA：s1phi、R 检验、§9）；u_blind / u_fw / u_fwb（UCR） | 已训 |

成本：特征 = 45 + 84 条 GGS 轨迹到 290 iters + 每截点 ≈ 30 次编码（4060 估 30–60 s/集，≈ 1–2 h，脱离启动）；训练 UEA 63 跑 × 12 s + UCR 63 跑 × 21 s ≈ 35 min。

## §4 门（先于数据；配对 Wilcoxon regret@3）

- **C0**：参照臂逐位（UEA blind 0.0294 / fw 0.0269；UCR 0.0220 / 0.0183，±0.0002）；探针 @290 R' 与缓存 GGS 行对账（中位 |Δ| ≤ 0.01）。
- **G-T1（提取方式，主）**：UEA `ft60_hard` < `fw_eaphi_hard`，p < 0.05。
- **G-T2（vs 任务盲，主）**：UEA `ft60_hard` < `blind_diff_hard`，p < 0.05。
- **G-T3（内容依赖）**：UEA `ft60_hard` cross 消融 regret@3 > normal，p < 0.05。
- 只报：ft290 / ftea 同三项；UCR 三臂 vs u_blind / u_fw；regret@1；M0 / M1；逐靶；探针 @60 vs @290 的 rep 列相关。

**读数码**：

| G-T1 | G-T2 | 码 | 含义 |
|---|---|---|---|
| ✓ | ✓ | **ET-ALIVE** | 表示空间响应优于原始空间：部署形态 = SH 第一级探针的表示 → E_T → field 头名单；进官方口径 |
| ✗ | ✓ | **ET-EQUIV** | 转导型有价值但不比 E_A 多：提取方式不是瓶颈 |
| — | ✗ | **ET-DEAD** | 转导型连任务盲都不过：探针表示的响应不携带可迁移偏好 |

后缀：G-T3 ✓ `+CONTENT` / ✗ `−CONTENT`。

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 | 90% |
| P2 | G-T1 过 | 35% |
| P3 | G-T2 过 | 55% |
| P4 | G-T3 过 | 50% |
| P5 | UEA ft290 regret@3 < ft60 | 40% |
| P6 | UEA ftea regret@3 < min(ft60, fw) | 45% |
| P7 | UCR u_ft60 < u_blind（p < 0.05） | 65% |

## §6 执行顺序与协议锁

1. 本文件 + `tests/test_tclsc_s1et.py`（先红）→ commit。
2. `mech_features.rep_profile`（表示空间原语）、33 / 63 新 ctx 列表与臂、`scripts/69_tclsc_s1et.py`（--features / --train / --rank / --analyze）→ 绿 → commit。
3. `69 --features`（脱离）→ `--train`（脱离）→ `--rank --analyze` → REPORT → commit → memory。

## §7 铁律对照

铁律 2 探针 iters 显式 290、截点 60/290 同轨迹；铁律 3 零新 R'（探针 @290 的 R' 只作对账不入缓存）；铁律 5；铁律 6 探针只在目标训练划分上短训、比较器不见目标；
铁律 7 `rep_profile` 在 mech_features（任务侧），比较器只见浮点向量。

## §8 执行台账（跑后填写）

- 2026-09-15 协议锁 a32a6d5（7 例先红）→ 实现 → 绿 fa9b546；`--features --train` 脱离：特征 UEA 45 切片（均 28.5 s）+ UCR 84 集（均 37.6 s）≈ 75 min，比较器 126 跑 ≈ 35 min（4060）。
  修正：64 号 rank_uea 的 zero 消融改为置零臂的全部非 φ 列（原只置零 E_A 列，对 E_T 臂无效）后重排。
- 读数 **ET-EQUIV+CONTENT**（C0 逐位；G-T1 ✗ 0.0207 vs fw 0.0269 p 0.054；G-T2 ✓ vs 任务盲 0.0294 p 0.001；G-T3 ✓ cross 0.0338 p 2e-7），预测 4/7。
  UEA M1 0.562 首次超任务盲；UCR u_ft60 0.0197 < u_blind p 5e-4 但 ≈ u_fw 0.0183。详见 REPORT.md。
