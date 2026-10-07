# WP18 PROTOCOL：扩展空间（AutoCL ⊕ 输入视图）上的 T-CLSC 全空间搜索 + 任务感知，部署精度与消融

**用户 2026-10-07**：主张不变（全空间搜索 + 任务感知），其余设计全可改，目标 = 部署报表数字 + 消融可见；不再做编码器搜索。

## §0 已看过的东西（如实声明）
- VP 探针（`results/acts/vp`）中间读数：视图效应强烈依赖数据集（Heartbeat psd +0.04、EthanolConcentration spec 至 +0.10、PhonemeSpectra diff +0.04、UWave psd −0.4、SCP1 psd −0.07），训练量 2× / 4× 平均 −0.009、maxavg 读出平均 −0.006 → 只加视图一条新轴。
- 视图地标 E_V 在源 / dev 靶上与 VP 实测视图效应多数同号（UWave、Ethanol、SCP1）。
- UT 11 靶的 TEST 已在 UT / UT-ENS / AL 读过三次；本轮 dev 期**不读** UT 的 TEST。UT 候选池分析（零 GPU）：66 条候选 TEST 均值 0.549、CV 前 5 均值 0.571、CV 与 TEST Spearman 中位 +0.2。
- dev 靶 = 6 个旧靶（HW / PS / UWave / Cricket / Ethanol / SCP1，本身是源，比较器 / 专家 / FV 都留一源），TEST 历史上读过多次，作开发集。

## §1 设计（相对 UT 流水的改动）
1. **空间**：AutoCL 19 字段（acts_space 规范化 + 剪枝）⊕ 新轴 view ∈ {raw, diff, psd, spec}（`configs/wp18.yaml`、`autocls/search/xspace.py`）。
2. **源标签**：S2 + XS 192 任务 9.5k 条旧标签（view = raw）⊕ 新标签 3,072 条（每任务：S2 全池先验前 4 条 shared × 3 个非 raw 视图 + 4 条已标 random × 随机非 raw 视图；cv5 终点、同折）。
3. **比较器**：10 个 T-CLSC field 头成员（任务盲 s₀，LambdaRank 顶部权重 + 软配对权重，验证加权配对准确率早停），扩展编码 120 维。
4. **任务感知（E 只做相似度 / 低秩交互）**：E = E_pt ⊕ E_A ⊕ E_V（块内标准化、块等权）。
   - ME：源专家（每源残差岭回归）× π = softmax(E 余弦 / τ)；(α, τ, γ) 内层留一源。
   - FV：视图 one-hot ⊗ [1, E_V] 的低秩交互，岭回归在源残差上拟合；(α, γ) 内层留一源。
   - MEFV = ME + FV（完整方法候选）。
5. **搜索**：Ks 2 万扩展候选，搜索者 = 成员集成 + 前 5 个成员，进化 kp 10 / 10 代，各取前 2 → 并集 U（≤ 12）。
6. **选择 + 最终**：官方 TRAIN 分层 3 折 CV（tcv 折与全量同官方迭代数、bs = min(8, N)，只评终点）→ 前 3 → 完整 TRAIN × 种子 {0,1,2} → 三个编码器终点读出逐样本 L2 后拼接 → 官方 SVM → TEST，3 种子均值；同时报 CV 第 1 单模型。
   - **小 TRAIN 规则**：任一验证折 < 10 个样本时 CV 无分辨率，改按比较器集成分数取前 3（AF、SWJ 在 UT 里命中；dev 靶都不命中）。
7. **臂（消融）**：MEFV（完整）/ ME（去 FV）/ FV（去源专家）/ UNI（E → 均匀）/ BLIND（无任务感知）/ LEG（MEFV 但视图钉 raw = 旧空间）。

## §2 读数与判读（dev，6 靶，集成 3 种子均值）
- 主读数：各臂 dev 均值；完整方法候选 = MEFV。
- **DEV-GO（上 UT）**：MEFV ≥ BLIND + 0.010 且 MEFV ≥ LEG，且 MEFV > TS2Vec 论文 dev 均值 0.624。
- **DEV-ALT**：若 MEFV 不过而 ME 或 FV 过 → 以过者为完整方法上 UT。
- **DEV-STOP**：都不过 → 不上 UT，回到设计（记 REPORT）。
- UT 阶段（另写 §3 判读带后再跑）：完整方法 vs TS2Vec 0.580 / InfoTS_s 0.631；消融 vs BLIND / UNI / LEG。

## §3 预测（dev）
| # | 预测 | 先验 |
|---|---|---|
| P1 | 搜索出的 U 里非 raw 视图占比在 Ethanol / PS 高于 UWave / Cricket | 60% |
| P2 | FV 内层 γ > 0（E_V 对视图偏好有可迁移信息） | 55% |
| P3 | MEFV dev 均值 > BLIND | 45% |
| P4 | MEFV dev 均值 > LEG | 50% |
| P5 | DEV-GO | 30% |
