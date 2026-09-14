# WP16-TCLSC-L2 PROTOCOL：CLS 作流水线 DAG + 共享 GIN（式 13–14），零 GPU

**会话定位**：重启 T-CLSC 的 CLS 支路第二步。L1 定了 binned 编码（每字段一张 embedding 表求和），剩下 MLP 与 hgb 的 0.016 差距与好半区 0.53 的天花板
是不是**交互结构**的缺口（sim_func–temperature–loss、增强复合顺序）。AutoCTS++ 式 13–14 用 GIN 吃 (A, F)：F = 每节点嵌入，A = 架构 DAG；
论文读出取 "Hyper" 枢纽节点（官方代码用均值池化）。CLS 不需要造图——它本来就是一条流水线：`order` 字段（Table 8 的 5 种增强施加顺序，
`augmentations.py::ORDERS`）就是 6 个增强节点之间的边，其余阶段按数据流相连。

数据 `results/mx3_pq/` 只读，零新 R'。分支 wp9-p1。产物 `results/tclsc_l2/`。考场 = L1 的 7 折 LODO `cls_only`（任务盲，δ 硬标签，126 实例）。

## §0 已看过的读数

- L1：cls_only_binned pair acc 0.758 / top-10 acc 0.528 / regret@1 0.066 / @3 0.032；hgb_binned 0.780 / 0.508 / 0.056 / 0.033；mlp_z_binned 0.764。
- WP2 字段重要性：temperature 0.28 ≫ sim_func 0.17 > norm 0.09 > freq_mask 0.08；WP15：温度斜率跨基 ICC −0.75（温度效应依赖 sim/loss 基）。
- **没看过**：任何图编码的比较器。

## §1 假设

H-G：给定流水线拓扑 A，GIN 在 7 折 LODO 上的任务盲配对 acc 高于 binned MLP（+0.02），且高于完全图 / 无边对照（拓扑本身有贡献）。
对立：19 个固定节点上 MLP 已看见全部字段，关系先验不加信息（L2-FLAT）；或收益全来自"逐字段 embedding + 池化"这个结构而与拓扑无关（L2-EMB-ONLY）。

## §2 编码与模型

- **节点** = 19 个旧字段 + 1 个枢纽 `hub`（N = 20）。节点特征 F：每字段一张 `nn.Embedding(n_levels + 1, d)`（最后一档 = 无效 None）
  ⊕ 数值字段的标量序数（scalar 口径的 [0,1] 值）× 该字段可学向量 s_f；枢纽 = 可学向量。
- **边 A（对称、无自环、按每条 CLS 的 `order` 变化）**，`cls_graph.pipeline_edges(order)`：
  - 6 个增强节点按 `ORDERS[order]` 串成链；链尾 → `emb_jitter` → `emb_mask` → `norm`；
  - `norm` → `sim_func` → `temperature` → `loss_type`；
  - `instance / temporal / cross_scale / kernel / pool / adj_neighbor` 各 → `loss_type`；
  - `order` 节点、枢纽：枢纽 ↔ 全部节点。
  拓扑档：`pipe`（上）/ `full`（完全图）/ `none`（无边；读出只能用均值）。
- **GIN**（式 14）：H^(k) = MLP^(k)((1+ε_k)·H^(k−1) + A·H^(k−1))，MLP = Linear(d,d)+ReLU+Linear(d,d)，层间 ReLU，L_n = 2，ε_k 可学（初值 0）。
  读出 `hub`（论文）或 `mean`（官方代码）→ l_a（d = embedding_dim = 128）。其余（`cls_fc` / `pred_fc` / 头）不动。
- 打包：`GraphPacker.encode(cls)` → [levels(19) | scalars(19) | A(20×20)] = **438 维**浮点向量，走现有 `ClsRecord.vec` 通路（引擎零改动）；
  `TCLSC(cls_encoder="graph", graph_opts=…)` 内部解包。meta 记 `encoder_mode="graph"` + `graph_opts`（拓扑 / 读出 / 标量 / 层数），排名侧按此重建打包器。

## §3 臂（`scripts/33` 的 `cls_only` 规格 + `--encoder-mode graph --graph-*`，7 折 × 3 种子）

| 臂 | 拓扑 | 读出 | 标量 | 用途 |
|---|---|---|---|---|
| ref `mlp_binned` | — | — | — | L1 的 cls_only_binned（不重跑） |
| **`gin_pipe_hub`** | pipe | hub | ✓ | **主臂**（论文式读出） |
| `gin_pipe_mean` | pipe | mean | ✓ | 读出对照（官方代码式） |
| `gin_full_mean` | full | mean | ✓ | 拓扑对照：有消息传递、无拓扑 |
| `gin_none_mean` | none | mean | ✓ | 结构对照：逐字段 embedding + 池化，无消息传递 |
| `gin_pipe_hub_noscalar` | pipe | hub | ✗ | 标量序数的边际 |

5 臂 × 7 折 × 3 种子 = 105 跑 × ~12 s ≈ 25 min（4060）。

## §4 门（先于数据；口径 = L1：126 实例 rank→score，3 种子多数票）

- **C0**：L1 `cls_only_binned` 复算逐位 0.758（同一分析函数）；`gin_none_mean` 与它差 ≤ 0.03（结构对照落在同带，否则实现有误）。
- **G-L2a（图 vs MLP）**：`gin_pipe_hub` − `mlp_binned` 配对 acc pooled ≥ **+0.020** 且逐折 ≥ 5/7 为正。
- **G-L2b（拓扑有贡献）**：`gin_pipe_hub` − `gin_full_mean` ≥ +0.010 且 ≥ 5/7 折（取 pipe 两种读出中较高者也报）。
- 次级：top-10 内 acc（L1 主要意外的那格）、regret@1/@3、hub vs mean、scalar 边际、Wilcoxon。

**读数码**：

| G-L2a | G-L2b | 码 | 含义 |
|---|---|---|---|
| ✓ | ✓ | **L2-GRAPH-WIN** | 拓扑先验有效：后续 T-CLSC（含 diff 头）CLS 支路改 GIN；进 S2 身份 token 复验 |
| ✓ | ✗ | **L2-EMB-ONLY** | 收益来自逐字段 embedding + 池化，不来自流水线边：改用 `gin_none_mean`/`full` 即可 |
| ✗ | — | **L2-FLAT** | 19 个固定节点上关系先验无信息：CLS 支路定为 binned MLP，写进模型章 |

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 过 | 85% |
| P2 | G-L2a 过 | 30% |
| P3 | G-L2b 过（条件于 P2） | 40% |
| P4 | hub ≥ mean（pipe） | 50% |
| P5 | 标量序数边际 ≥ +0.005 | 45% |
| P6 | 任一 GIN 臂 top-10 acc ≥ 0.55（超 L1 的 0.528） | 40% |

## §6 成本

105 跑 ≈ 25 min（4060）；零新 R'。

## §7 执行顺序与协议锁

1. 本文件 + `tests/test_tclsc_l2.py`（先红）→ commit。
2. 实现 `autocls/comparator/cls_graph.py`（GraphSpec / pipeline_edges / GraphPacker / GraphCLSEncoder）、`TCLSC(cls_encoder="graph")`、
   33 `--encoder-mode graph --graph-topology/--graph-readout/--graph-scalar`、34 loader、56 `encoder_for_meta` / `rank_local_models`、`scripts/61_tclsc_l2.py` → 单测绿 → commit。
3. `61 --prepare --train`（脱离启动）→ `61 --rank --analyze` → REPORT。

## §8 铁律对照

- 铁律 1：档位 / 字段名 / 数值口径全部来自 `space.yaml`（`GraphSpec.from_space`）；拓扑只写字段**名**的连接关系并在导入时对照 `field_names` 校验；
  增强顺序取 `augmentations.py::ORDERS`（Table 8 的既有真源）。旧 scalar/binned 路径逐位不变。
- 铁律 7：图编码器不见任务概念。关键决策 1（"CLS 编码用向量，不用图/GIN"）由本 WP 显式复检；读数为 WIN 时改写该条。

## §9 执行台账（跑后填写）

- 2026-09-14 协议锁 ce996cf（17 例先红）→ 实现 cls_graph / TCLSC graph / 33 / 34 / 56 / 61 → 17 例绿 + 邻近 352 绿（commit 5ac42ab）。
- `--train` 105 跑本地 4060 脱离启动（41–75 s/跑，与其它测试并行时变慢），零事故；`--rank --analyze` → **L2-FLAT**
  （G-L2a +0.003 4/7 ✗；G-L2b +0.005 4/7 ✗；none_mean 0.769 最高但好半区更差），预测 1/6。详见 REPORT.md。
- 结论：CLS 支路定为 binned MLP；真 E 臂用 binned。
