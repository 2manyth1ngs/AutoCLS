# WP17-ACTS S3-C-D PROTOCOL：6 个 UEA 数据集的部署口径检验（T-CLSC / 任务盲比较器 / 4 探针）

**会话定位**：S3-C（`results/acts/s3c/REPORT.md`）的源侧搜索验证是在留出源的**切片**上做的（切片 E、切片 5 折 CV acc），不是部署形态。用户 2026-09-25 的初衷是"看这几个 UEA 数据集在部署口径下表现怎么样"，并拍板：
- 用整个数据集的 E（= AutoCTS++ Algorithm 2 的目标侧做法）；
- 按 WP3 旧口径：官方 train 训练、官方 test acc、按 test acc 在前 3 里选 1、seed 0；
- 三个臂：T-CLSC（A）、任务盲比较器（D）、4 探针（④）；
- 不跑目录 shared-30 全量。

产物 `results/acts/s3cd/`；代码 `scripts/90_acts_s3cd.py`；单测 `tests/test_acts_s3cd.py`；真跑走 83 的 phase W（整集官方口径执行体 `39.run_trajectory_cell`，`--profile s1pp`）。

## §0 已看过的读数（如实声明）
- S3-C 全部读数，含逐源表。6 个源的切片搜索最终 acc：A 0.608 / D 0.597 / 目录任务盲前 3 0.617；④ 目录内 0.620。
- WP3 / WP6 / WP10 在这 6 个数据集官方口径上的旧读数（旧目录 shared-10 + GGS；SH6-3-3 / SH10-4-3 冠军；TS2Vec 论文数），以及 S1″ W 步（Cricket / UWave 官方 vs cv5 排序 ρ 0.885 / 0.860）。
- **没看过**：任何 ACTS 比较器或 ④ 在整数据集官方口径上的提名与真值。

## §1 设计
- **靶**：Handwriting / PhonemeSpectra / UWaveGestureLibrary / Cricket / EthanolConcentration / SelfRegulationSCP1，UEA 官方 train / test 划分，原生长度；任务 id = `<D>_whole_official`，尺寸取自 `configs/uea30.yaml`。
- **零样本**：每个靶用 S3-C 中"该数据集留出"那一折训练好的比较器（`models/{A,D}_<D>.pt`）；④ 的低秩模型也只在该折训练源上拟合。靶数据不参与任何拟合。
- **目标 E**（A 臂）：只用官方 TRAIN（按 TRAIN 统计量标准化）fit 每任务单变量 TS2Vec，参数与 S3-B 的 E_pt 相同（seed 0、batch 16、原码 iters 规则、OOM 才切段 1000）；按 `window_idx(n_train, task_id)` 取 100 窗编码，池化到 152；SetTF 用 A 模型自带的编码器。
- **搜索**（A、D）：与 S3-C 相同——Ks 20,000 剪枝采样（种子 `run_seed_for(task_id, "s3c_ks")`，两臂共用同一份样本）→ 40 块 × 500 块内 RR 取前 10 → 400 条块赢家 RR 取前 10 → `evolutionary_search` 10 代（p1 0.8 / p2 0.2）→ RR 前 3。
- **④**（S3-A ④ 原样）：
  - 探针 = 该折 μ 前 2 + b₁ 两端，共 4 条 shared 配置；在靶上按官方口径跑 seed 0，取 25% 截点的 test acc；
  - 探针地板 = 3 次 iters = 0 的 test acc 均值；majority = test 划分多数类占比；
  - 5 维特征 → ridge（该折训练源的切片特征上拟合，α 与 S3-A 同规则内层 LOSO 选）→ â → 低秩打分 μ + âB_K（K = S3-A 该折内层选的 K）→ shared-30 前 3。
  - 已知口径错位：ridge 在切片的 CV acc 上拟合，这里吃的是官方 test acc。
- **选择与报数**（WP3 旧口径）：每臂 3 条都在官方 TRAIN 上按 `ts2vec_official` 全量训练（seed 0），按**终点 test acc** 选 1 条并报其 test acc。选择用到了 test，读数偏乐观，三个臂同等偏乐观，所以臂间可比。
- **参照**：TS2Vec 论文数（`uea30.yaml` literature）；本项目旧部署数（WP3 / WP10 报告里的 SH6-3-3 / SH10-4-3 冠军），只作背景（目录与环境不同：旧数在 RunPod pod 上跑，C 可取 inf；本轮在实验室服务器上跑，C = inf 用 1e10 替身）。
- **成本**：A 18 格 + D 18 格 + ④ 探针 24 格 + iters = 0 探针 18 次 + ④ 前 3 ≤ 18 格，共约 80 格全量训练，四卡约 1–2 h。

## §2 读数（描述性，6 个数据集）
- 每个数据集：A / D / ④ 的选中 test acc，TS2Vec 论文数，旧部署数；各臂 3 条提名的 test acc 与平均；A 与 D 提名的重合；④ 的前 3 与 S3-C 源侧 ④ 的前 3 是否一致。
- 配对（6 个数据集）：A − D、A − ④、④ − D 的均值、符号计数、Wilcoxon p。n = 6，只作描述，**不设门**。

## §3 预测（先验）
| # | 预测 | 先验 |
|---|---|---|
| P1 | 6 集均值 ④ ≥ A | 60% |
| P2 | 6 集均值 A ≥ D | 55% |
| P3 | A 在 ≥ 4/6 个数据集上 ≥ TS2Vec 论文数 | 45% |
| P4 | ④ 在 ≥ 4/6 个数据集上 ≥ TS2Vec 论文数 | 55% |
| P5 | A 与 D 的提名（前 3）在 ≥ 3 个数据集上至少重合 1 条 | 30% |

## §4 执行
PROTOCOL + 单测先红 → 协议锁 → 实现 `--prep / --embed / --search / --plan-a / --score-probe / --plan-b / --analyze` → 本地冒烟 → 服务器：embed → search → plan-a → 83 run + probe0（phase W）→ 拉回 → 本地 score-probe → 服务器 plan-b → 83 run → 拉回 → analyze → REPORT。

## §5 铁律对照
2（iters 规则显式、截点同轨迹）；3（runs.jsonl、resume）；5（先红、§0 声明）；6（比较器 / ridge / μ·B 只在训练源；靶只考，E 只用 TRAIN）；7；8。

## §6 执行台账（跑后填写）
