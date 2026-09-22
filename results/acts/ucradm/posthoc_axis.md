# UCR-ADM 事后（零 GPU，2026-09-22）：源级偏好偏差的结构——一条种子稳定的"反向轴"

用户提问"反向源少、要不要给正向源少切片"后的核查。数据 = S1″ 13 源（含被拒的 HMD / SCP2）+ UCR 10 源，shared-30 两种子均值终点 acc，任务内 z 化后按源取均值画像（23 × 30）。

## 1. 每个源的偏差都是真的（不是只有反向源）
每源画像减去"其余源均值"得残差；残差按 seed 0 / seed 1 分别算再相关：**23 源全部 0.55–0.97**（Handwriting 0.94、UniMiB 0.95、PAMAP2 0.97、Ethanol 0.93、Semg 0.97；最低 EOGH 0.55、SCP2 0.68）。正向源的偏差范数小（0.30–0.59），反向源大（Ethanol 0.84 / SCP2 0.85 / HMD 0.90 / Semg 0.93），但都可复现。

## 2. 偏差集中在一条轴上，源在轴上的位置种子稳定
残差矩阵 SVD：PC1 37%、PC2 19%、PC3 10%。**源在 PC1 上的得分按 seed 0 / seed 1 分别算的 Spearman = 0.96**。
负端 → 正端：Semg −3.8、HMD −3.6、Ethanol −3.4、SCP2 −2.9、CinC −1.4、SCP1 −1.2、UWave −0.6、NonInvasive1 −0.2 ｜ FiftyWords +0.3、ShapesAll +0.4、MixedShapes +0.8、Handwriting +0.8、EOGH +0.9、Cricket +0.9、LargeKitchen +0.9、Opportunity +1.2、UniMiB +1.3、Adiac +1.4、SleepEDF +1.4、PTB-XL +1.5、PAMAP2 +1.7、PS +1.7、FordA +1.8。
反向源之间残差同向：Ethanol–SCP2 0.54、Ethanol–Semg 0.48、Ethanol–SCP1 0.44、SCP2–HMD 0.48；正向源残差与 Ethanol 反向（UniMiB −0.67、HW −0.57、SleepEDF −0.47）。
轴的配置载荷：负端偏好 8c482976（info_nce / dot / layer_norm / τ 0.1）、fc7fd37f / 11ba2bc4（info_nce / cosine / none / τ 1.0）、重 crop（0.8–0.95）、point_mask 0.3 / 0.5；正端偏好 2a667428（triplet / euclidean / layer_norm / τ 0.01）、cb08a823、a04011c2（dot / l2 / τ 0.1）、轻 crop（0.1–0.4）。按字段粗看：τ 1.0 −0.13 vs 0.01 +0.07、info_nce −0.06 vs triplet +0.06、norm none −0.06 vs l2 +0.07——温度 / 损失几何 / 增强强度混在一条轴上；与 WP15 整集"基偏好"的对应未核（Semg 在本轴负端却是 WP15 里 B_I − B_T 最负的集，−0.20），不要直接等同。

## 3. 轴的位置不能由便宜描述量读出（n = 23）
轴得分 vs 多数类占比 −0.06、oracle +0.08、oracle − 多数类 +0.03、中位 acc +0.13、探针地板 +0.17、oracle − 探针 −0.28、k +0.19、T −0.39、n +0.36、C +0.08。难度 / chance 程度不解释轴；只有窗长（长 → 负端）和样本数（多 → 正端）有弱相关。与 WP15 A3 / GEO-E / WP12 的"数据集级偏好真实但描述量读不出"同形。

## 4. 读法
- "反向源"不是噪声：它们是同一条可复现轴的远端，且互相同向——这正是任务感知比较器**可学、可迁移**的对象（从 Ethanol / SCP2 / HMD 学，迁到 Semg）。
- 用切片数给正向源降权改变不了这件事的两个硬处：(a) 16 片 Ethanol 仍是轴上同一个点（集内共同序 0.78），不增加独立的反向样本；(b) 轴上远端四源恰是可靠性最差的（HMD / SCP2 ρ_seed 0.25–0.61、Ethanol h_t 0.06–0.25），加权只放大噪声。同一意图的干净做法 = 全源 16 片不变、训练目标改成去掉池主效应的残差（或按 |残差| 加权配对），正向源自动少贡献、反向源自动多贡献、不丢数据；门用 regret@3 + 轴符号命中率。
- 真正的关口是 E 能否把目标放到轴上（§3 说便宜量不行）；S2 前应先做零 GPU 检验：E_A / φ / E_T 对 23 源轴得分的 LODO 预测 ρ ≥ 0.5 才值得铺，否则 E 支路会再次惰性。测量式替身：轴两端由两条具体配置张成（8c482976 vs 2a667428），目标上 @25% 各跑一次即可读出方向（0.5 全跑当量）。
- 补源方向：按轴找负端源（不是按域）。候选来自 S1″ 已拒的 HMD / SCP2（可靠性不过，只能作靶）与零 GPU 筛表里画像 ρ 低的 Haptics / Mallat / SmallKitchen；三源 ≈ 15 GPU-h 可验它们是否落在负端。
