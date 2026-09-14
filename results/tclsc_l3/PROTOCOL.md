# WP16-TCLSC-L3 PROTOCOL：融合头 × 标签形式——E′ 是"偷懒"还是"没得学"（阶段 0 零 GPU）

**会话定位**：重启 T-CLSC 第二步（CLAUDE.md 顶部"本会话"节第 7 条）。L1 已定 CLS 支路用 binned 116 维
（`results/tclsc_l1/REPORT.md`）。本协议问的是式 17–21 的融合头：晚期 concat 让任务向量可以只作加性偏移
（实验 1 F2 测到的形状）；把头改成**反对称打分差** s(l_a, t) − s(l_b, t)（任何加性 g(t) 相减抵消，任务信息**只能**经交互起作用）
或 **FiLM 调制**（任务向量调制 CLS 编码本身，式 15 的位置）之后，给定完美任务信号，比较器能不能用它。

**为什么必须 2×2**：MX-3-PQ A8 把 one-hot 数据集身份直接喂给现有头，cross 消融仍零——在 δ 配对标签上连完美任务信号都没有被用的理由。
所以头单独换、标签不动，结果读不出原因。L3 = 头（3）× 标签（2），外加两条任务盲对照。

**阶段划分**：阶段 0（本协议）用 **oracle 身份 token** 代替真 E，S2 划分（8 数据集全参与、每数据集留 2 切片考试），
零 GPU；它回答"头能不能用任务信号"这个**机制**问题，**不是 zero-shot 读数**。阶段 1（另立协议，须 GPU 请示）把胜出的
(头, 标签) 换成真 E（SetTF，需 `33 --fit-folds` 重造折特征），7 折 LODO。

数据 `results/mx3_pq/` 只读，零新 R'（铁律 3）。分支 wp9-p1。产物目录 `results/tclsc_l3/`。

## §0 已看过的读数

- A8 `onehot_oracle`（`results/mx3_pq/runs/S2/onehot_oracle_s0.json`，scalar 编码）：normal 与 cross 档；MX-3-PQ REPORT 判 cross zero。
- L1：binned 把 cls_only 的 top-10 内配对 acc 从 0.44 抬到 0.53–0.54（事后）；δ 存活对上 binned − scalar = +0.012。
- WP6-TF 阶段 0：同数据集切片身份作分数（c1_pure_z）regret@3 0.031 → 0.014（p 5.7e-08）——**身份上界在 z 标签形式下存在**。
- C2 / 实验 2：δ 存活对任务盲天花板 0.95+，全对软标签下"任务信号"≈ 全体 label 质量的 2%。
- **没看过**：任何 diff / FiLM 头；任何 z 差分标签训练的比较器；binned 编码下的 A8。

## §1 假设

- H-HEAD：concat 头让任务向量退化为偏置；反对称 / FiLM 头下同一任务信号会被用上（cross 消融掉分）。
- H-LABEL：δ 配对硬标签把任务信号滤掉；z 差分软标签（不过 δ、全对）保留它。
- 两者可同时成立；也可能身份上界在**配对比较器形式**下根本取不出来（L3-FLAT）。

## §2 材料与口径

- 划分：S2（`pq.s2_split`）——每数据集前 4 切片训、后 2 考（SCP2 2/1）；考侧 15 切片 × 3 预算 = **45 实例**，候选 = 目录 20 条。
- 任务信号：ctx = 数据集 one-hot(8) ⊕ 预算 one-hot(3) = 11 维（A8 同款，只在脚本 33 内构造；铁律 7）。
- CLS 编码：binned 116（`--encoder-mode binned`）。
- 训练超参：`configs/pretrain_tclsc.yaml` 原样（lr 1e-4、batch 64、200 epoch、curriculum 40）。
- 标签
  - `hard`：现行 δ 配对 BCE（`dynamic_pairs`，δ 查表）。
  - `zdiff`：任务内**全部**无序对（不过 δ、不弃平局），方向经 `better_fn`（铁律 8），目标 p = σ(k·d)，
    d = |z_a − z_b| × sign（better_fn 判定），z = 任务内（engine task = 切片×预算）R' 的 z 分数；**k = 2.0 写死**
    （|Δz| = 1 → p 0.88）；样本权 w = 1；每 epoch 无序对方向随机（concat 头才需要，反对称头无关）；
    curriculum 同 hard（前 40 epoch 只用 shared 对）。走 `SoftTCLSCEngine.train_soft`。
- 头（`TCLSC(head=…)`，默认 `concat` 逐位不变，参数名集合不变）
  - `concat`：式 17–21 原样。
  - `diff`：t = set_fc(ctx)（fz 维）；s(l, t) = Linear(emb+fz→fz)+ReLU+Linear(fz→1)；logit = s(l_a,t) − s(l_b,t)。无任务支路时 s(l)。
  - `film`：t → (γ, β) = Linear(fz→2·emb)；l̃ = (1+γ)⊙l + β；s(l̃) = Linear(emb→fz)+ReLU+Linear(fz→1)；logit = s(l̃_a) − s(l̃_b)。
  - diff / film 对 (a, b) 互换严格反对称：P(a,b) + P(b,a) = 1（单测钉死）。

## §3 臂（`scripts/33` 的独立表 `L3_ARMS`，全部 S2、3 种子、`iron_law_6_compliant="s2_oracle"`）

| 臂 | 任务支路 | 头 | 标签 | 用途 |
|---|---|---|---|---|
| `l3_none_hard` | none | concat | hard | 任务盲对照（= cls_only 的 S2 版） |
| `l3_none_zdiff` | none | concat | zdiff | 任务盲对照（标签换了但没任务信号） |
| `l3_concat_hard` | ctx | concat | hard | **G0**：A8 在 binned 下的复现 |
| `l3_concat_zdiff` | ctx | concat | zdiff | 标签单独换 |
| `l3_diff_hard` | ctx | diff | hard | 头单独换 |
| `l3_diff_zdiff` | ctx | diff | zdiff | 头 + 标签 |
| `l3_film_hard` | ctx | film | hard | 头单独换（调制式） |
| `l3_film_zdiff` | ctx | film | zdiff | 头 + 标签（调制式） |

8 臂 × 3 种子 = 24 跑，每跑 ≈ 10 s（4060）。

## §4 读数与门（先于数据）

**排名口径**（`scripts/57_tclsc_l3.py --rank`）：每臂 3 种子 → `rank_candidates` 多数票（34 号同式），候选向量 binned，
ctx 按 (切片, 预算) 构造；**消融 = 排名时把 ctx 的数据集 one-hot 换成别的数据集**（`cross_dataset_map` 错位，3 个种子），
预算位不动。任务盲臂只有 normal。

**三个指标**（45 实例，rank→score）：
- **M1 好半区配对 acc**（真值前 10 内、δ 存活对；L1 里对编码最敏感的那格）——**主指标**。
- M2 |Δz| 加权全对配对 acc（不过 δ；权重 = |z_i − z_j|）。
- M0 δ 存活对配对 acc（WP2 口径；预期任务盲天花板附近、对任务信号不敏感，只作参照）。
- 附 regret@1 / @3、Spearman。

**消融 Δ**：Δ_M = M(normal) − mean_s M(cross_s)，45 实例配对 Wilcoxon（H1：normal > cross）。

**门**：
- **C0**：本地 `onehot_oracle`（scalar，pod 同臂）normal acc 与 `runs/S2/onehot_oracle_s0.json` 差 ≤ 0.005（GPU 带；L1 里 4060 与 pod 逐位）。
- **G0（A8 复现）**：`l3_concat_hard` 的 Δ_M1 < 0.03 或 p ≥ 0.05（"零"）。不过 → 读数码 **L3-ENC-SUFFICES**（binned 编码已让 concat 头用上身份，头不是问题）。
- **G-A（头单独换）**：`l3_diff_hard` 或 `l3_film_hard` 的 Δ_M1 ≥ **0.03** 且 p < 0.05。
- **G-B（头 + 标签）**：`l3_diff_zdiff` 或 `l3_film_zdiff` 同上。
- **G-C（标签单独换）**：`l3_concat_zdiff` 同上。
- 次级（只报）：Δ_M2 ≥ 0.01 且 p < 0.05；任务盲臂 vs 各 ctx 臂 normal 的 M1 差；hard vs zdiff 在 M0 上的差（zdiff 不得比 hard 低 0.02 以上，否则标签换法伤通用排序）；33 号 run json 的 cross 档（δ 对、McNemar）作 M0 侧的第二读数。

**读数码**（按顺序判）：

| 条件 | 码 | 含义 / 下一步 |
|---|---|---|
| G0 ✗ | **L3-ENC-SUFFICES** | 头不是瓶颈；阶段 1 直接用 concat + 真 E |
| G0 ✓ ∧ G-A ✓ | **L3-HEAD-ONLY** | 光换头够；阶段 1 用胜出头 + hard 标签 + 真 E |
| G0 ✓ ∧ G-A ✗ ∧ G-C ✓ | **L3-LABEL-ONLY** | 标签是唯一瓶颈；阶段 1 用 concat + zdiff + 真 E |
| G0 ✓ ∧ G-A ✗ ∧ G-C ✗ ∧ G-B ✓ | **L3-HEAD+LABEL** | 两者都要；阶段 1 用胜出头 + zdiff + 真 E |
| 全 ✗ | **L3-FLAT** | 身份上界在配对比较器形式下取不出；下一步不是换 E 而是换形式（逐 CLS 打分回归 + 任务向量，非配对） |

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 过 | 95% |
| P2 | G0 过（A8 在 binned 下仍零） | 75% |
| P3 | G-A 过 | 25% |
| P4 | G-B 过 | 45% |
| P5 | G-C 过 | 30% |
| P6 | film 的 Δ_M1 ≥ diff 的 Δ_M1（两标签都算，取 zdiff） | 50% |
| P7 | zdiff 臂的 M0 不比同头 hard 臂低 0.02 以上 | 60% |
| P8 | 任务盲 `l3_none_zdiff` 的 M1 ≥ `l3_none_hard` + 0.02（L1 的 G3′ 在 S2 上复现） | 55% |

## §6 成本与阶段 1

- 阶段 0：24 跑 ≈ 5 min（4060）+ C0 1 跑 + 排名/分析分钟级。零 GPU 时数意义上"零新 R'"，GPU 只用于比较器训练（本地）。
- 阶段 1：`33 --fit-folds`（7 折 + all8 共享 TS2Vec 与窗口编码）估时待标定，须按 `wp-stage-reports-and-pod-consent` 请示；另立 PROTOCOL 附录。

## §7 执行顺序与协议锁

1. 本文件 + `tests/test_tclsc_l3.py`（先红）→ commit（协议锁）。
2. 实现：`tclsc_model.py` head、`autocls/comparator/zdiff_labels.py`、`33` L3_ARMS / zdiff / head 透传、`34 load_pq_model` head、
   `56.rank_local_models(ctx_rows=)`、`scripts/57_tclsc_l3.py` → 单测全绿 → commit。
3. `57 --c0` → `57 --prepare --train` → `57 --rank --analyze` → REPORT.md → commit。

## §8 铁律对照

- 铁律 1：编码档位来自 space.yaml；k=2.0 是标签超参不是空间取值。
- 铁律 3：零新 R'。
- 铁律 5：先测后写、判读带先于数据。
- 铁律 6：阶段 0 是 S2 机制上界，明写"非 zero-shot"；zero-shot 读数只在阶段 1 的 LODO。
- 铁律 7：模型 / 引擎 / zdiff 标签模块不见数据集概念；one-hot 只在脚本 33 构造。
- 铁律 8：zdiff 的方向经 `better_fn` 注入。
- 关键决策 1（向量编码）：不变。式 17–21 的改动是本会话授权的重启对象。

## §9 执行台账（跑后填写）

- 2026-09-14 协议锁 1f88493（32 例先红）→ 实现 head / zdiff_labels / 33 L3_ARMS / 34 / 56 ctx_rows / 57 → 33 例绿 + 邻近 320 绿。
- `--c0` 首跑 FAIL（0.7853 vs 0.7932）：head 改动挪了 `set_fc` 构造顺序 → 同 seed 初始化变；恢复顺序 + 金标单测后 PASS（0.7922）。
- `--train` 24 跑本地 4060：hard 10–16 s、zdiff 280–370 s（全对 45k 对/epoch），`train.log`；`--rank --analyze` → **L3-HEAD-ONLY**
  （G0 ✓ / G-A ✓ diff +0.205 p 9e-4 / G-B ✓ / G-C ✓），预测 5/8。详见 REPORT.md。
- 阶段 1（真 E + LODO）未开：折特征需 `33 --fit-folds`（GPU 请示）。
