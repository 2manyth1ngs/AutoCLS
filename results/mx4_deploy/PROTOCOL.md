# MX-4 WP3 协议草稿：全量标准协议上的"零样本名单 + 逐次减半"部署（待用户拍板，未开跑）

日期：2026-09-03 ｜ 状态：**草稿**——判读线与臂表已写死，但开跑前需用户拍板 §7 的三个决定 ｜
脚本：`scripts/39_sh_deploy.py`（待写，复用 `35_protocol_alignment.py` 的 `_cell` / 协议装载 / runs.jsonl 缓存键）｜
底座：分支 mx4 已合并 protocol-alignment（`configs/train.yaml: protocols.ts2vec_official`，`results/protocol_alignment/`）。

## 0. 诚实声明

- 探索读数已看过的部分：切片协议下 SH6-3-3 regret 0.018（WP1）；官方口径下 zs_top3 六靶 5/6 追平 TS2Vec、HW 未追平而 GGS 0.532 已达平（protocol-alignment REPORT §4）。
- 本协议把这两件事接起来：**在官方口径的轨迹上重验淘汰赛**，并回答 HW 的"名单里没有好配置"问题。切片协议的 0.018 不外推，一切按官方口径重测。

## 1. 问题

| 问 | 内容 | 检验 |
|---|---|---|
| Q1（头条） | 官方口径下，淘汰赛冠军追平 TS2Vec 论文数字的数据集数是否 ≥ zs_top3 的 5/6，且 HW 追平 | W1 |
| Q2 | 淘汰赛 vs 零样本 top-3 全跑（同成本档）在官方口径下是否仍有增益 | W2 |
| Q3 | 切片协议的零样本排名对官方口径的效度（Spearman，shared-10） | W3（描述性） |
| Q4 | 名单扩到 shared-10 + GGS + 代理提名后，HW 是否有可追平候选 | W4 |

## 2. 口径

- 协议：`ts2vec_official`（iters 200 若 N·T·C ≤ 1e5 否则 600；bs 8；drop_last；SWA；scaler 只在 TRAIN 拟合；官方 SVM 网格含 max_iter=1e7）。
  六靶官方 iters：Handwriting 200；UWave / PhonemeSpectra / Ethanol / Cricket / HMD 600。
- **截点**：按官方 iters 的 {25%, 50%, 100%}：200 → (50, 100, 200)；600 → (150, 300, 600)。截点评估用 **SWA 当前平均权重快照**
  （训练器改造：截点把 AveragedModel 的平均权重交给评估回调，训练随机流不动；同 seed 下截点 b 的快照 ≡ 单独训 b 步的 SWA 终值，单测钉死）。
- 评估：官方划分 test 集准确率（acc/F1），与 protocol-alignment §4 同口径；同时记 SVM 交叉验证分（cv 口径备用）。
- 候选目录（每靶 21 + 提名）：
  - shared-10（先验按源池 z 均值，来自 `results/mx3_pq` 缓存）；
  - **GGS**（源池缓存有其 R'，先验分照算；不在 shared-10 是 HW 未追平的直接原因）；
  - 代理提名 10 条：`hgb_binned`（WP2）按折 LODO 拟合；候选 = 训练侧 z 均值最高的 20 条已评配置各做 1–3 个字段的局部变异
    （门控字段随 space.yaml 约束联动、须过 `validate`），共 2000 次变异，实测到训练集字段汉明 ≤ 3 才保留，排除已在目录 / 已评者，
    取预测 z 前 10（seed = 0 + 靶序号，逐位确定）。**不用均匀采样**：19 字段空间里随机点离 461 条已评配置通常 ≥ 8 个字段，半径 3 筛不出东西。
- 排名臂：`prior`（shared-10 + GGS 按先验）、`hybrid`（先验 + 提名按代理）、`tclsc`（`rankings.json[D]["290"]` Borda，protocol-alignment 同式，消融对照）。
- 淘汰赛：SH6-3-3（前 6 @25% → 留 3 @50% → 3 条到 100% 取最好）；对照 SH6-3-1、all@25%→top3。选择只用截点真值，最终裁判 = 100% 截点 test acc（best-of-3，与 zs_top3(best) 同款上界）并报 cv 选择口径。
- 种子：目录带截点 1 种子（seed 0）；冠军与主要对照臂补 seed 1、2 到终点（3 种子均值与 §4 六靶表并排）。
- 成本：GPU 秒按缓存 `pretrain_seconds` + 各截点 `eval_seconds` 增量累加；iters 口径同 WP1。

## 3. 阶段与工单（`--dry-run` 计数）

| 阶段 | 工单 | 说明 |
|---|---|---|
| S0 一致性 | 6 靶 × GGS × seed 0，带截点 | 100% 截点 acc 须与 `runs.jsonl` targets 的 GGS seed 0 逐位（或 ≤ GPU 复跑噪声 ±0.005）一致——证明 SWA 截点改造不改终点口径 |
| S1 目录 | 6 靶 × (10 + 1 + 10) × seed 0，三截点 | 淘汰赛缓存模拟的全部输入；PS 的 test 3353 条评估 ≈ 20 s/截点 |
| S2 终点复核 | 每靶 SH 冠军 + prior3 冠军 × seed {1, 2} | 3 种子均值进头条表 |
| S3 分析 | 零 GPU | 淘汰赛模拟、W1–W4、成本表、与 §4 六靶表并排 |

估时（4090 实测 pretrain 5–37 s/格 + 评估）：S1 ≈ 126 格 ≈ 1.6 h @4090，本地 4060 约 3–5 h；S0+S2 ≈ 0.3 h。

## 4. 判读（先写死；n=6 靶，只报计数与配对差，不做显著性宣称）

| 门 | 检验 | 先验 |
|---|---|---|
| W1（头条） | SH6-3-3_prior 冠军 3 种子均值 ≥ TS2Vec − 0.03 的靶数 ≥ 5/6 **且** Handwriting 达标 | 55% |
| W2 | 6 靶配对：SH6-3-3 冠军 acc ≥ prior3（top-3 全跑取最好）于 ≥ 4/6，均值差 ≥ 0 | 60% |
| W3 | shared-10 上切片 R'@290 排名 vs 官方口径 100% 截点 acc 的 Spearman 中位 ≥ 0.5（描述性） | 50% |
| W4 | HW 目录（21 + 提名）里存在 100% 截点 acc ≥ 0.485 的候选 | 70%（GGS 已知 0.532） |

读数码：**DEPLOY-WIN** = W1 ∧ W2；**DEPLOY-PARITY** = W1 ∧ ¬W2（追平但淘汰赛无增益）；**DEPLOY-DEAD** = ¬W1。

## 5. 预测

- P1：官方口径截点 25% 处的排名与 100% 的 Spearman 中位 < 0.6（短轨迹早截点更吵）；SH6-3-1 比 SH6-3-3 差 ≥ 0.02（65%）。
- P2：HW 的 SH 冠军 = GGS 或提名配置，而非 shared-10 成员（60%）。
- P3：Cricket / Ethanol 所有臂同带（贴顶 / 随机 encoder 已近论文），淘汰赛与零样本无差（80%）。
- P4：S0 一致性逐位通过（75%），否则 ≤ ±0.005。

## 6. 已知坑

- 官方轨迹只有 200/600 步、bs 8：截点 25% = 50/150 步，噪声可能大到让淘汰赛退化为随机；这是本协议要回答的问题，不是要绕过的问题。
- PhonemeSpectra 评估 20 s/次，三截点 × 21+ 格 ≈ 25 min，占 S1 大头。
- input_mask 本轮不进空间（保持 19 字段）；只对 SH 冠军补一条 `+input_mask` 次读数（沿用 protocol-alignment 的 ladder 开关）。
- `rankings.json` 的 tclsc 臂只覆盖 shared-10 + 各切片 random-10，提名配置无 T-CLSC 分；tclsc 臂只在 shared-10 上比。

## 7. 拍板记录（2026-09-03）

1. 候选提名：**采用**（用户授权由执行方决定）——shared-10 + GGS + 变异提名 10（§2 规则）；HW 的教训是好配置不在名单里，提名是 WP2 打分器在部署里唯一的用武之地。
2. 截点比例：**{25%, 50%, 100%}**（执行方决定）——同一份缓存零成本可模拟"跳过 25% 的两级淘汰赛"作对照臂（§2 已列），25% 处噪声大不大由数据回答。
3. 硬件：**4090 pod**（用户拍板），卷 tudo4ntlot（EUR-NO-1，六个 UEA 缓存已在卷上）。
4. 协议锁产物：`candidates.json`（6 靶 × {shared-10, GGS, 提名 10, 先验, T-CLSC 序}），本地冒烟（Handwriting × 3 格 × iters 8，三截点）通过。
5. 状态：**草稿 → 锁定**，随本 commit 生效；开跑后判读线不改。
