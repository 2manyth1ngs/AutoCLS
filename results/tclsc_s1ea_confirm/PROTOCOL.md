# WP16-TCLSC 阶段 1 A′ 确认版 PROTOCOL：响应画像 E_A ⊕ φ12 的选择价值与内容依赖，UEA 8 集 7 折 LODO（主）+ UCR 84 新种子（副），零 GPU

**会话定位**：A 臂（`results/tclsc_s1ea/REPORT.md`）按预注册阶梯读 EA-DEAD（好半区 M1 门 ✗），但阶梯外两条同样预注册的读数首次在阶段 1 过线：
regret@3 0.0198 vs 任务盲 0.0220（p 0.009）且换成他集画像 regret 变差（p 5e-4）；叠加臂 eaphi M0 0.794、vs φ regret@3 p 0.078 未过。
用户 2026-09-15 拍板做确认版。**确认的对象是"E_A 有超出 φ12 的选择价值，且价值来自它的内容"**——在 A 臂**没看过的数据**上（UEA 8 集，
最初设想的考场）作主检验，在 A 臂数据上换训练种子作副检验；好半区 M1 只报不设门。分支 wp9-p1。产物 `results/tclsc_s1ea_confirm/`。

## §0 已看过的读数

- UCR 84（A 臂，种子 0/1/2）：eaphi regret@3 0.0202 vs φ 0.0209 p 0.078、vs blind 0.0220 p 3e-4；cross 消融 regret@3 → 0.0231 p 2e-7；M0 0.794；
  ea_diff regret@3 0.0198 / cross 0.0222 p 5e-4；regret@1 0.045 vs 0.061。
- UEA 7 折 LODO（s1phi，binned，diff）：blind_diff_hard M0 0.7557 / regret@3 0.0294 / regret@1 0.0657；phi_diff_hard 0.7805 / 0.0311 / 0.0771。
- **没看过**：E_A 在任何 UEA 切片上的值；E_A 臂在 UEA 上的任何数字；UCR 上种子 3/4/5 的任何臂。

## §1 假设

H-C1：E_A ⊕ φ12 在 UEA LODO 下 regret@3 低于 φ12（选择价值超出形状描述量）。
H-C2：该价值依赖 E_A 的内容——把目标切片的 E_A 换成他数据集训练切片的，regret@3 变差。
对立：A 臂的价值信号是 UCR 语料 + 三个种子的偶然（新考场 / 新种子不复现）；或价值只是"存在效应"（cross 不伤）。

## §2 材料

- **考场 1（主）**：`results/mx3_pq/` 缓存（8 集 45 切片 × 20 CLS × 3 预算，δ 按 `configs/labels.yaml` 逐 (数据集, 预算)），S1 7 折 LODO，126 实例；
  变体目录 `results/tclsc_s1ea_confirm/uea/`（56 号 `prepare_variant_dir`），binned，diff 头，hard 标签，3 种子（0/1/2），超参不变。
  E_A 逐**切片**算（`pq.load_pq_datasets` → `materialize_task` → `split_task_train_eval(eval_ratio 0.3, seed=run_seed_for(tid, "train_eval_split"))` 的 train 侧，
  与 40/42 号 φ 同划分；`response_profile` 同 A 臂 25 维、种子 (0,1,2)）→ `ea_uea.jsonl`（每切片一行，并入 φ12 列）。
  参照臂 `blind_diff_hard` / `phi_diff_hard` 不重训：取 `results/tclsc_s1phi/binned/rankings_s1phi.json` 的 normal 排名（同折同种子）合并重评。
- **考场 2（副）**：UCR 84 分组 7 折（`results/tclsc_s1ucr/folds.json`，δ 0.06），**种子 3/4/5** 重训 `u_blind_diff_hard` / `u_phi_diff_hard`（60 号）与
  `u_eaphi_diff_hard`（63 号，E_A 表 = `results/tclsc_s1ea/ea_ucr.jsonl` 原样），252 实例；输出 `results/tclsc_s1ea_confirm/ucr_s345/`。
- 33 号加通用 `ctx_kind="table_budget"`（`ArmSpec.ctx_cols` + `PQContext(ctx_file=)`，折内 nan 感知标准化只用训练切片）；臂表 `EA_ARMS`：
  `ea_diff_hard`（E_A 25）/ `eaphi_diff_hard`（E_A ⊕ φ12 = 37）。

## §3 臂

| 考场 | 臂 | ctx | 用途 |
|---|---|---|---|
| UEA | **`eaphi_diff_hard`** | E_A ⊕ φ12 ⊕ b | **主臂** |
| UEA | `ea_diff_hard` | E_A ⊕ b | 副臂（只报） |
| UEA | ref `blind_diff_hard` / `phi_diff_hard` | — / φ12 | s1phi 已训 |
| UCR s345 | `u_eaphi_diff_hard` / `u_phi_diff_hard` / `u_blind_diff_hard` | 同 A 臂 | 副检验 |

成本：UEA 42 跑 × ≈12 s（4060）≈ 9 min；UCR 63 跑 × ≈19 s ≈ 20 min；E_A 45 切片 < 1 min。

## §4 门（先于数据；指标同前：regret@1/@3、M0、M2、M1（只报）；cross = E_A 换成他数据集训练切片的（58 号 `cross_slice_map`，3 种子）；zero = E_A 置 0，φ 部分不动）

- **C0**：UEA 重评 blind_diff_hard M0 = 0.7557、phi_diff_hard M0 = 0.7805（±0.001，与 s1phi analysis.json 逐位）。
- **G-C1（价值超出 φ，主）**：UEA `eaphi_diff_hard` regret@3 < `phi_diff_hard`，126 实例配对 Wilcoxon（H1 更低）p < 0.05。
- **G-C2（内容依赖，主）**：UEA `eaphi_diff_hard` cross 消融后 regret@3（3 种子均值）> normal，配对 p < 0.05。
- **G-C3（新种子复现，副）**：UCR s345 `u_eaphi_diff_hard` regret@3 < `u_phi_diff_hard`，252 实例配对 p < 0.05。
- 只报：G-C4 UEA eaphi regret@1 vs φ；G-C5 M1 cross Δ（两考场）；G-C6 zero 消融；G-C7 `ea_diff_hard` 同四项；G-C8 UCR s345 的 cross 消融 regret@3。

**读数码**（主考场 UEA 决定，UCR 副检验作后缀）：

| G-C1 | G-C2 | 码 | 含义 |
|---|---|---|---|
| ✓ | ✓ | **EA-CONFIRMED** | 自设 E 的价值与内容依赖跨语料成立：进 B（度量学习 E）与官方口径部署 |
| ✓ | ✗ | **EA-VALUE-NO-CONTENT** | 有价值但换内容不伤 = 存在效应（与真 E 同形）：B 不做，写 analysis |
| ✗ | ✓ | **EA-CONTENT-NO-VALUE** | 内容被用但在 n=7 训练集下不超过 φ：n 的问题，B 只值得在 UCR 语料上做 |
| ✗ | ✗ | **EA-NOT-CONFIRMED** | A 臂信号不跨语料：机制特征线在比较器形态下到此 |

后缀：G-C3 ✓ → `+UCR-REP`，✗ → `−UCR-REP`。

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 逐位 | 95% |
| P2 | G-C1 过 | 30% |
| P3 | G-C2 过 | 55% |
| P4 | G-C3 过 | 45% |
| P5 | UEA eaphi M0 ≥ φ 0.7805 | 60% |
| P6 | UEA eaphi regret@1 < φ（p < 0.05） | 35% |
| P7 | UEA eaphi regret@3 < blind 0.0294（p < 0.05） | 45% |

## §6 执行顺序与协议锁

1. 本文件 + `tests/test_tclsc_s1ea_confirm.py`（先红）→ commit。
2. 实现 33 `table_budget` / `EA_ARMS`、`scripts/64_tclsc_s1ea_confirm.py`（--features / --train / --train-ucr / --rank / --analyze）→ 单测绿 → commit。
3. `64 --features` → `--train`（UEA，脱离）→ `--train-ucr`（脱离）→ `--rank --analyze` → REPORT → commit → memory。

## §7 铁律对照

铁律 3 零新 R'；铁律 5；铁律 6 E_A 只用目标训练划分、标准化只用训练折切片；铁律 7 `table_budget` 在 33 号眼里只是浮点向量；铁律 8 方向经 `is_better`。

## §8 执行台账（跑后填写）

（空）
