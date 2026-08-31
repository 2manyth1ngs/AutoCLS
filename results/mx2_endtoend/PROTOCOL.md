# MX-2 端到端零样本搜索协议：UM↔HW 双向 leave-one-dataset-out（判读带/先验先于数据写死）

日期：2026-08-31 ｜ 脚本：`scripts/25_mx2_endtoend.py` ｜ 单测：
`tests/test_mx2_endtoend.py` ｜ 前置：宽面板终判（2a4ccbe，任务级条件化关闭、
排序主张回归任务无关先验）；C2 语料（`results/labels_workpoint/`，28 任务 ×
20 CLS @工作点网格收官）。本协议为 MX-2 论文头条指标的预注册：四臂 regret
判据、外部锚对账线、全部先验在二期计算开始前锁定。

## §1 材料（全部先于本协议存在，零改动）

- **真值网格**：`results/labels_workpoint/rprime_cache.jsonl`——28 任务
  （Handwriting 12 @60 iters / UniMiB-SHAR 16 @30 iters，工作点口径）×
  每任务 20 CLS（shared 10 全局 + random 10 任务专属，
  `cls_catalog.json` 唯一真源）×k=1 canonical。**判定真值 = 该 k=1 网格**
  （GPU 非确定性 ±0.005 已知且远小于臂间差量级；k=3 确认仅供误差棒展示，
  不改判定）。
- **配对/δ**：workpoint 标定 δ（HW 0.11 / UM 0.075，`labels.yaml` 唯一真源）。
- **方向**：A = 训 UniMiB-SHAR → 考 Handwriting 12 任务（主）；
  B = 训 Handwriting → 考 UniMiB-SHAR 16 任务（副）。合计 28 搜索实例。
- **E 管线（铁律 6）**：每方向共享 TS2Vec **只 fit 训练侧数据集**
  （`get_shared_ts2vec` 指纹断言 fit_datasets=[训练侧]），考侧任务仅冻结
  前向编码；SetTF init 复用 `set_encoder_init.pt`（数据无关的确定性初始化）。
  既有 `labels_workpoint/shared_ts2vec.pt`（双数据集 fit，C2 口径）**禁用**。
- **T-CLSC**：`configs/pretrain_tclsc.yaml` 原样；每方向 3 模型种子（0/1/2）。

## §2 搜索与四臂（写死）

每个考侧任务的候选集 = 该任务的 20 CLS 目录。排名 = 3 种子成对概率矩阵 →
`symmetrize` → `pair_vote_matrix` 多数票 → 胜场降序（平票按跨种子平均对称
得分、再按 cls_key，逐位确定）。臂定义（R′ 均指真值网格 @工作点）：

| 臂 | 定义 |
|---|---|
| oracle | 网格 20 候选最大 R′（regret 基准点） |
| **T-CLSC@1** | 排名 top-1 的真值 |
| T-CLSC@3 | 排名 top-3 真评估取优（部署形态，K=3 写死） |
| GGS | AutoCL Table 5 GGS 配置固定迁移（`m1_ggs_har.GGS`；不在目录内 → 28 格 @工作点 canonical 补测） |
| random@1 | 20 候选均值（解析期望） |
| random@3 | 不放回抽 3 取最大的解析期望（次序统计量精确式） |
| bottom | 网格最小值（垫底参照） |

regret_t(臂) = oracle_t − 臂_t。**数据集级冠军**（外部锚用）：方向内各任务
胜场按 **shared-10 子阵** Borda 聚合（random 候选任务专属、跨任务不可比）。

## §3 判读带（写死）

- **D1m（唯一成败门）**：pooled 28 任务，regret@1(T-CLSC) vs regret@1(random)
  配对 Wilcoxon 单侧（T-CLSC 更小）**p<0.05**。
- 分层报告（不设门）：方向 A（n=12）、方向 B（n=16）各自同口径 p。
- 描述性附表（不判定）：@3 各臂、GGS 对比、avoid-worst（T-CLSC@1 落入
  任务内垫底四分位 = 20 候选倒数 5 的次数/28）、k=3 确认误差棒。
- **外部锚（如实报，不设成败门）**：方向 A shared-10 Borda 冠军 cls\* 与
  GGS 各 ×3 种子，全量 Handwriting 官方 split 标准协议
  （full=2320 iters@bs128，`configs/train.yaml`；SVM-RBF = AutoCL B.1）；
  对账线 = cls\* 三种子均值 ≥ TS2Vec 论文 Handwriting 公开数字 **0.515** − 0.05
  （论文数字 analyze 时对原文复核，协议先记 0.515）。UCI-HAR 锚**本轮不做**
  （目标池物理隔离优先；如需另立轻协议本地跑）。

| 条件 | 判定 |
|---|---|
| D1m 过 | MX-2 头条成立：零样本排序显著优于随机 + 成本表主张 |
| D1m 不过 | 头条降级为成本/avoid-worst 描述性结果，如实归档（不 reframe） |

成本表（bookkeeping，硬件标签必带）：在线成本 = top-3 真评估 vs 目录直搜
20 格 vs 全量口径直搜（20 × full_iters），秒数取缓存实测。

## §4 预测（先验，条件于宽面板/C2/M3 已见）

| # | 预测 | 先验 |
|---|---|---|
| P1m | D1m p<0.05 | 65% |
| P2m | 方向 A 单独 p<0.05（n=12） | 45% |
| P3m | 方向 B 单独 p<0.05（n=16） | 55% |
| P4m | pooled mean regret@3(T-CLSC) ≤ mean regret(GGS) | 40% |
| P5m | 外部锚对账线命中（cls\* ≥ 0.515−0.05） | 55% |
| P6m | avoid-worst：T-CLSC@1 零次落垫底四分位（0/28） | 60% |

## §5 执行（全计算上云 1×5090，wp 模式）

顺序：`--prepare`（方向别 TS2Vec fit ×2 + 28×2 任务特征 + T-CLSC 6 跑 +
排名落盘 + 确认工单）→ `--run-cells --resume`（GGS 28 格 + 确认格
{T-CLSC@1, GGS, oracle}×2 额外 weight seed，逐条落盘/OOM 防护/device_name）
→ `--anchor --resume`（全量 HW 6 跑）→ `--analyze`（纯聚合，云上跑完
拉回本地重跑须逐字节一致）。预算 ~1–2 GPU·h。产物：
`results/mx2_endtoend/{PROTOCOL.md, dir_A/, dir_B/, rankings.json,
rprime_extra.jsonl, confirm_cache.jsonl, anchor_cache.jsonl,
analysis.json, REPORT.md}`。协议锁 commit 先于一切云计算。
