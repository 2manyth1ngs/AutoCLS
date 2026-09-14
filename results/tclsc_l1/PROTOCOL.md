# WP16-TCLSC-L1 PROTOCOL：T-CLSC 的 CLS 支路——编码 × 目标函数 分解（零 GPU 主体）

**会话定位**：重启 T-CLSC（CLAUDE.md 顶部"本会话"节）的第一步 L1。问题不是"E′ 为什么没用"，而是把 T-CLSC 的
**CLS 支路**与同标签下的字段代理之间 0.735 → 0.780 的差距拆成两个可单独改的因子：**编码**（32 维标量 → 按档 one-hot；
one-hot 进 `cls_mlp` 第一层 Linear 在数学上 = AutoCTS++ 的每 op 一张 `nn.Embedding` 表求和）与**目标函数**（δ 存活配对
BCE → within-(task, budget) z 回归）。MLP 结构固定不动。

分支 / 产物：主目录分支 `wp9-p1`（本 WP 不新开分支，改动只在 `scripts/33`、新 `scripts/56_tclsc_l1.py`、
`tests/test_tclsc_l1.py`、本目录）。数据 `results/mx3_pq/` **只读**，不产生任何新 R'（铁律 3 零增量）。

## §0 已看过的读数（如实声明）

- WP2 表（`results/mx4_surrogate/REPORT.md` §1，同一 126 实例、同一 rank→score 口径）：
  ridge_scalar 0.773 / ridge_binned 0.765 / hgb_scalar 0.788 / **hgb_binned 0.780** / **T-CLSC main 0.735**（regret@1 0.074）。
  逐折 T-CLSC：HW 0.720 / UM 0.816 / PS 0.648 / UWave 0.856 / Ethanol 0.459 / Cricket 0.856 / HMD 0.784。
  → 一个**线性**模型在标量编码上已 0.773，先验上"换编码就追平"不成立，故本协议必须带目标函数臂（L1c）。
- `results/mx3_pq/runs/*/cls_only_s0.json`（模型输出口径，**不是**本协议口径）：normal acc Cricket 0.854 / Ethanol 0.438 /
  HMD 0.598 / HW 0.730 / PS 0.735 / UWave 0.854 / UM 0.805；单折训练 18–23 s（pod 5090，200 epoch）。
- **没看过**：cls_only 在 rank→score 口径下的数字；任何 binned 编码的 T-CLSC；任何 MLP 回归臂。
- 材料状态：`results/mx3_pq/` 本地只有 `rprime_cache.jsonl`（2835 行 = shared 1350 + random 1350 + GGS 135，零 error）、
  `delta_calibration.json`、`cls_catalog.json`、`rankings.json`（pod 训的 main / cls_b / cls_only，scalar）、`runs/*.json`。
  **折特征 npz 与模型 .pt 不在本地**（留在 pod 卷）→ 凡吃 E 的臂都要先重跑 `33 --fit-folds`（GPU），列为阶段 2。

## §1 问题与假设

H-ENC：`cls_mlp` 在 32 维标量输入上要从一维里学非单调档位效应（8 个 ratio + temperature），改 116 维按档 one-hot 后
配对准确率上升。
H-OBJ：δ 存活配对 BCE 丢掉了大部分行（每折训练对 ≈ 800 vs 代理回归 2340 行）且只学"方向"，改成 within-task z 回归后
同一 MLP 追上 ridge_binned。
两者可同时成立、同时不成立；本协议的价值在于**分解**，不在于任一臂赢。

## §2 材料与口径（与 WP2 逐位一致）

- 实例：7 靶 × 6 切片 × 3 预算 = 126；候选 = 目录 20 条（10 shared + 10 random；GGS 剔除，同 `build_records` 默认）。
- 真值 R'：`rprime_cache.jsonl`；δ：`delta_calibration.json`（`(dataset, budget)` 查表）。
- 指标（`scripts/37_surrogate_loo.py` 的纯函数逐位复用）：δ 存活无序对配对 acc（rank→score）、top-10 内 acc、Spearman、
  regret@1 / @3；pooled = 有存活对的实例均值；逐折 = 该折实例均值。
- 排名聚合：本地模型经 `scripts/34_pq_search.py::rank_candidates`（多种子对称化得分 → 多数票胜场 → 平均分 → key 升序），
  与 pod 版 rankings.json 同一函数。**打分时的 CLS 向量按模型 meta 的 `encoder_mode` 编码**（缺键 = 旧模型 = scalar；
  `cls_dim` 与模式不符即报错）。
- LODO：训练侧 = 7 个源数据集（含 SCP2 低权）去掉靶；靶只考试（铁律 6）。L1c 的回归拟合同样按折剔靶。

## §3 臂

| 臂 | 编码 | 目标 | 种子 | 来源 | 阶段 |
|---|---|---|---|---|---|
| C0 `main_pod` | scalar | δ-BCE（E+ctx） | 3（投票） | `rankings.json["main"]` | 0 |
| C0 `cls_only_pod` | scalar | δ-BCE（无 E） | 1 | `rankings.json["cls_only"]` | 0 |
| ref `hgb_binned` / `ridge_binned` / `ridge_scalar` | — | z 回归 | 0 | WP2 `analysis.json` 复算 | 0 |
| **L1a** `cls_only_scalar` | scalar 32 | δ-BCE（无 E） | 0,1,2（投票） | 本地 `33 --train --arms cls_only --encoder-mode scalar`，产物 `results/tclsc_l1/scalar/` | 1 |
| **L1a** `cls_only_binned` | binned 116 | δ-BCE（无 E） | 0,1,2（投票） | 同上 `--encoder-mode binned`，产物 `results/tclsc_l1/binned/` | 1 |
| **L1c** `mlp_z_scalar` | scalar 32 ⊕ b/290 | z 回归 MSE | 0,1,2（预测均值） | `56 --analyze` 内拟合 | 1 |
| **L1c** `mlp_z_binned` | binned 116 ⊕ b/290 | z 回归 MSE | 0,1,2（预测均值） | 同上 | 1 |
| **L1b** `main_binned` | binned 116 | δ-BCE（E+ctx） | 0,1,2（投票） | 需 `33 --fit-folds`（GPU）后 `--train --arms main --encoder-mode binned` | 2（须用户同意开 GPU） |

固定项（先于数据写死）：
- L1a 训练超参 = `configs/pretrain_tclsc.yaml` 原样（lr 1e-4、batch 64、200 epoch、curriculum 40）；只改编码模式；
  两个编码各自独立目录（`33` 的产物路径 `models/<fold>/cls_only_s{seed}.pt` 不带模式，同目录会互相覆盖）。
- L1c 模型 `MLPZ`：主干与 `TCLSC.cls_mlp` 同形 Linear(d,128)→ReLU→Linear(128,128)，接 ReLU→Linear(128,1)；
  Adam lr 1e-3、weight_decay 1e-4、全批 300 epoch、MSE；输入标准化不做（one-hot / [0,1] 标量本就同尺度）；
  种子 0,1,2 预测取均值；CPU 确定性（`torch.manual_seed`，单线程）。**不调参**——L1c 问的是目标函数，不是最优回归器。
- 特征：`scripts/36_prior_sh_sim.py::features`（CLS 编码 ⊕ iters/290），与 WP2 代理臂逐位相同。

## §4 判读门（先于数据）

- **C0（硬门）**：用 `56 --c0` 从 `rankings.json["main"]` 复算 pooled 配对 acc = **0.735 ± 0.001**、regret@1 = 0.074 ± 0.001；
  ref 三臂与 WP2 `analysis.json` 逐位一致。不过 → 口径没对齐，不往下跑。
- **C0-local（机器带，只报不判）**：`cls_only_scalar`（本地 3 种子投票）− `cls_only_pod`（1 种子）的 pooled 配对 acc，
  |Δ| ≤ 0.02 视为同机带内；超出则 G1 只用本地–本地比较，并在 REPORT 单列。
- **G1（编码，cls_only）**：Δ_enc = `cls_only_binned` − `cls_only_scalar`（均本地 3 种子投票）
  pooled ≥ **+0.020** **且** 逐折 Δ>0 的折 ≥ **5/7**。
- **G2（编码带到主臂，阶段 2）**：`main_binned` − 0.735 ≥ +0.020 且 ≥ 5/7 折；阶段 2 未跑 → 记 n/a，不参与读数码。
- **G3（目标函数）**：`mlp_z_binned` pooled 配对 acc ≥ **0.765**（ridge_binned 水平）。
- **G3′（回归目标下编码的边际）**：`mlp_z_binned` − `mlp_z_scalar` ≥ +0.020（只报）。
- **规避读法**：所有 Δ 都必须附逐折表与 126 实例配对 Wilcoxon（`36::paired_wilcoxon_less`）p；top-10 内 acc 一并报
  （WP2 已知所有臂 0.39–0.51，此处不设门，只看是否出带）。

**读数码**（只由 G1 / G3 决定）：

| G1 | G3 | 码 | 对 L2 / L3 的含义 |
|---|---|---|---|
| ✓ | ✓ | **L1-ENC+OBJ** | 两者都改：L2 用 one-hot 节点特征，L3 的标签改连续 z |
| ✓ | ✗ | **L1-ENC-ONLY** | binned 定为 T-CLSC 默认编码；标签形式暂不动，L2 优先 |
| ✗ | ✓ | **L1-OBJ-ONLY** | 差距在 δ 配对标签而非编码：L3 先改标签（连续 z / 软差分损失），L2 后置 |
| ✗ | ✗ | **L1-FLAT** | 差距在模型族（MLP vs 树）或训练制度（curriculum / lr / epoch）：L2（图 + 消息传递）前先做一轮 MLP 训练制度对照 |

## §5 预测（先验，写在数据前）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 硬门过（0.735 / 0.074 逐位复现） | 95% |
| P2 | G1 过（编码 +0.02 且 5/7） | 40% |
| P3 | G3 过（MLP 回归 ≥ 0.765） | 65% |
| P4 | G3′ 过（回归目标下 binned 仍 +0.02） | 30%（ridge / hgb 对编码都无差别） |
| P5 | `mlp_z_binned` ≥ hgb_binned 0.780 | 25% |
| P6 | C0-local 在 0.02 带内 | 70%（4060 vs 5090 单种子非确定性，见 memory `rprime-nondeterminism-on-5090`） |
| P7 | 编码收益（若有）集中在 PS / Ethanol 两折（T-CLSC 最差折 0.648 / 0.459） | 50% |

## §6 成本

- 阶段 0（C0 / ref）与 L1c：零 GPU，本地分钟级。
- 阶段 1（L1a）：2 编码 × 7 折 × 3 种子 = 42 跑 × 20–60 s（4060 / CPU）≤ 1 h；无 SetTF，显存 < 1 GB。
- 阶段 2（L1b）：`--fit-folds` 7 折 + all8 的共享 TS2Vec 与窗口编码（GPU，估时待标定，开跑前按
  `wp-stage-reports-and-pod-consent` 请示）。

## §7 执行顺序与协议锁

1. 本文件 + `tests/test_tclsc_l1.py`（先红）→ commit（协议锁）。
2. 实现 `33 --encoder-mode` 与 `scripts/56_tclsc_l1.py` → 单测全绿 → commit。
3. `56 --c0`（硬门）→ `56 --analyze --arms mlp_z`（L1c）→ `33 --train`（两编码）→ `56 --rank --analyze`（L1a + 读数码）。
4. REPORT.md（含逐折表、Wilcoxon、预测对账）；阶段 2 单独请示。

## §8 铁律对照

- 铁律 1：binned 档位只来自 `configs/space.yaml`（`CLSEncoder(mode="binned")` 已有）；不新增取值集合。
- 铁律 3：零新 R'；`results/mx3_pq/` 只读，变体目录只复制输入文件（缓存 / δ / 目录）。
- 铁律 5：先测后写；本文件先于数据。
- 铁律 6：LODO；L1c 拟合按折剔靶（`lodo_split` 单测钉死）。
- 铁律 7：比较器 / MLPZ 不见任务类型；方向由标签侧 `is_better` 给定（沿用 33 / 36 的已有路径）。
- 关键决策 1（"CLS 编码用向量，不用图/GIN"）：本 WP 不违反——两种向量模式都出自 `space.yaml`；L2 若立项再改该条。

## §9 执行台账（跑后填写）

- 2026-09-14 协议锁 3e80e1b（27 例先红）→ 实现 `33 --encoder-mode` + `56` → 27 绿 + 邻近套件 220 + 60 绿。
- `--c0` PASS：main 0.7349 / 0.0745，ref 五臂与 WP2 逐位；cls_only_pod（rank→score 口径首次算出）0.7405。
- `--analyze`（L1c，零 GPU，< 1 min）先跑：mlp_z_scalar 0.727 / mlp_z_binned 0.764。
- `--prepare` 两目录各复制 4 个输入文件；单折计时 Handwriting/scalar/s0 8.8 s（4060），normal acc 与 pod 逐位相同。
- `--train` 42 跑（2 编码 × 7 折 × 3 种子）8.4–10.8 s/跑，`train.log`；`--rank --analyze` → 读数码 **L1-FLAT**
  （G1 Δ +0.012 ✗ / G3 0.7644 ✗ 差 0.0006 / G3′ +0.037 ✓）。详见 REPORT.md。
- 阶段 2（main_binned）未跑（折特征不在本地）。
