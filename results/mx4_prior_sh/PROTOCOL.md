# MX-4 WP1 协议：源池先验 + 目标侧逐次减半（零 GPU，缓存模拟）

日期：2026-09-03 ｜ 脚本：`scripts/36_prior_sh_sim.py` ｜ 数据：`results/mx3_pq/`（只读）｜
计划：`~/.claude/plans/zero-shot-mx3pq-verdict-20260903-md-1-c-valiant-catmull.md`（用户拍板 2026-09-03：头条 = "零样本先验给候选名单 + 目标侧逐次减半定冠军"）。

## 0. 诚实声明（与常规预注册的区别）

本协议**不是盲预注册**。判读线写死之前，本会话已用临时脚本在同一缓存上做过一轮探索性复算
（先验@1 pooled 0.046、SH6-3-3 @290 0.018、随机起点 0.036、先验@3 0.032、新鲜种子 0.005 vs 0.011）。
本协议的作用是：把那轮复算固化成可复现脚本 + 单测，并把"过/不过"的判据、臂表、成本口径**在正式跑之前**写死，
使结果可审计、可被下一个数据集复用。先验概率栏如实标注"已看过探索读数"。

## 1. 问题

| 问 | 内容 | 检验 |
|---|---|---|
| Q1（主） | 目标侧逐次减半（SH）能否在同成本档下把 regret 压到零样本 top-3 真评估之下 | G1 / G3 |
| Q2 | 零样本先验作为 SH 的起点是否优于随机起点（零样本部分的价值） | G2 |
| Q3 | SH 的增益是配置质量还是同轨迹运气 | G4（新鲜种子真值） |
| Q4 | 零样本单选：源池先验是否优于手工强默认 GGS | G5 |

## 2. 数据与口径（全部沿用 MX-3-PQ，铁律 2 同口径）

- 真值：`results/mx3_pq/rprime_cache.jsonl`（45 切片 × 20 目录 CLS × 预算 {60,120,290}，同一轨迹三个截点，k=1 canonical）。
  搜索实例 = (折靶数据集, 切片, 最终预算 b_final)；候选 = 该切片 20 条目录 CLS；oracle = 目录内 max R'@b_final。
- 折：7 折 leave-one-dataset-out（`pq_common.fold_split`）；训练侧 = 其余 6 + SCP2（三预算全用）。
- 先验（唯一定义）：训练侧每条记录取 within-(切片, 预算) z 分数（`(R' − mean)/std`）；每个 shared CLS 的先验 = 其 z 的均值
  （跨训练侧全部切片与预算）。random CLS 不在先验中（目录外候选由 GBDT 打分）。
- GBDT：`autocls.search.surrogate.FieldSurrogate(kind="hgb")` 默认超参，特征 = `CLSEncoder(mode="scalar")` 32 维 ⊕ `b/290`，
  目标 = 训练侧 z；每折一次拟合。
- hybrid：shared 按先验（加 +10 偏置置顶）、random 按 GBDT。
- T-CLSC 对照：`results/mx3_pq/rankings.json` 的 `main` 排名（3 种子多数票，33/34 号产物），同实例。
- 成本：iters 计（每条轨迹按截点累加：k0·b0 + k1·(b1−b0) + k2·(b2−b1)），并用缓存 `seconds`（该截点为止累计墙钟）
  换算 GPU 秒；零样本 @1 成本 = b_final、@3 = 3·b_final；目录直搜 = 20·b_final。

## 3. 臂表

零样本臂（每实例，三预算）：`prior1/prior3`、`gbdt1/gbdt3`、`hybrid1/hybrid3`、`tclsc1/tclsc3`（rankings.json main）、
`clsb1`、`clsonly1`、`ggs`、`random1/random3`（不放回抽 k 的解析期望，`25_mx2_endtoend.expected_max_of_k`）、`bottom`、`oracle`。

多保真臂（b_final=290 用截点 (60,120,290)；b_final=120 用 (60,120)；b_final=60 无 SH）：
`SH{k0}-{k1}-{k2}_{warm}`，(k0,k1,k2) ∈ {(6,3,1),(6,3,3),(10,4,1),(10,4,3)}（b_final=120 时取前两级），
warm ∈ {prior, hybrid, gbdt, random（20 次抽样期望）}；纯多保真 `all20@60→top1/top3`（b_final=290）。
SH 规则：按 warm 排名取前 k0 条，在 b0 上取真 R'（同一轨迹截点）留前 k1，在 b1 上留前 k2，最终在 b_final 上取 k2 条里最好的；
并列按 cls_key 升序（逐位确定）。

## 4. 判读（先于正式跑写死；配对单侧 Wilcoxon α=0.05，n 如实报）

| 门 | 检验 | n | 先验（已看过探索读数） |
|---|---|---:|---|
| G1（主） | regret[SH6-3-3_prior] < regret[prior3]，@290 | 42 | 70% |
| G2 | regret[SH6-3-3_prior] < regret[SH6-3-3_random]，@290 | 42 | 75% |
| G3 | regret[SH6-3-3_prior] < regret[tclsc3]，@290 | 42 | 65% |
| G4 | 新鲜种子真值（噪声探针 rep1/rep2 均值）下 mean regret[SH6-3-3_prior] ≤ mean regret[prior3]（描述性，n=14 功效不足不作显著性判定） | 14 | 60% |
| G5 | regret[prior1] < regret[ggs]，pooled 三预算 + 分层描述 | 126 | 60% |

读数码：**SH-WIN** = G1 ∧ G2 ∧ G3 ∧ G4；**SH-PARTIAL** = G1 ∧ (G2 ∨ G3)；否则 **SH-DEAD**。G5 独立报，不参与读数码。
任一门不过都如实入 REPORT，不 reframe。

## 5. 预测（先验写死）

- P1：SH6-3-3_prior pooled@290 mean regret ∈ [0.010, 0.030]（75%）。
- P2：SH6-3-3_random 比 SH6-3-3_prior 差 ≥ 0.010（70%）。
- P3：all20@60→top3 与 SH6-3-3_prior 同带（|Δ| < 0.010）但成本 ≥ 1.8×（65%）。
- P4：prior1 在 @290 与 GGS 持平（|Δ| < 0.01），@60/@120 优于 GGS（60%）——探索读数 @290 0.071 vs 0.074。
- P5：SH6-3-1_prior（只把 1 条跑到 290）regret ≥ 2× SH6-3-3_prior（60%）：最后一级只留 1 条会被 @120 的噪声坑。

## 6. 已知坑（写在跑之前）

- SH 的 @60/@120 读数与 @290 真值同轨迹：这是部署真实情形（部署的就是那条轨迹），但 G4 用探针新鲜种子单独复核。
- `rankings.json` 只覆盖目录候选，GGS 在目录外，其 regret 可为负；tclsc 臂缺席的实例直接 fail-fast，不做子集检验。
- 42 靶切片里 HMD 评估集小（n_eval 24–74），单实例噪声大；分层表按数据集如实报。
- 先验只覆盖 shared-10；"prior1 输给 GBDT/hybrid"意味着 random 候选里有更好的，"赢"则说明目录里 random 基本不夺冠（探索读数：oracle − shared-10 oracle = 0.008）。
