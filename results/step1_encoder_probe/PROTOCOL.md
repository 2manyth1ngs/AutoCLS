# Step 1 协议：encoder 换族探针（transformer 四问，判据先于数据 commit）

日期：2026-08-29 ｜ 脚本：`scripts/13_step1_encoder_probe.py` ｜ 单测：
`tests/test_step1_encoder_probe.py` ｜ 局部配置：本目录 `encoder_transformer.yaml`
（全局 configs/encoder.yaml **不动**，探针经 build_encoder(config_path=…) 注入）｜
本协议与代码先 commit，数据后跑。

## §1 目的与来源

三个病灶（贴顶 / 学不动 / 轨迹地板）的定罪证据都指向"膨胀 CNN 家族 ×
评估栈"（卷积深度 1→10 基线不动 = 家族级；Step0/0.5 = 扩容与 iters 两根
杠杆拉满，地板谷底 0.030 仍高于信号 2.3×）。本探针检验关键决策 2 预留的
transformer 分支能否同时松动三个病灶——**单变量实验：只换 encoder，切片/
评估栈/CLS 空间/种子机器全部冻结**。

已知风险（先行如实记录）：随机 attention 特征可能同样强（Q1 会否决）；
transformer 小数据短训练可能更不稳（Q4 会否决）或学不动（Q3 会否决）；
训练配方沿用 CNN 的（lr=1e-3、bs128、无 warmup）——失败可能部分归因配方
不匹配，记 limitation，不在本探针内调参（避免自由度膨胀）。

## §2 设计（先于数据写死）

四问四数据集，任务定义与既有缓存逐字段同源（同注册表同切片种子）：

| 问 | 数据集（病灶代表） | 任务 | 臂 | 内容 |
|---|---|---|---|---|
| Q1 贴顶开了吗 | SpokenArabicDigits（CNN 基线 0.912） | 前 2 | {60,240} | 基线 5 种子 + 10 shared × k=1 |
| Q2 学得动了吗 | MotorImagery（CNN inert，基线≈0.511） | 前 2 | {60,240} | 同上 |
| Q3 对照没坏吧 | Handwriting（唯一健康源） | 前 2 | {60,240} | 同上 |
| Q4 地板多低 | UniMiB（唯一可扩评估） | 前 2 | {60,240} | 基线 3 种子 + 10 shared × **k=3**，expanded n=2000（step0 同 extras） |

- 规模：基线 36 cell + 信号 120 cell（3 数据集 × 2 任务 × 10 CLS × 2 臂）+
  地板 120 cell（2×10×3 种子 × 2 臂）= **276 个 cell**，全部带
  `encoder: transformer` 字段、独立缓存（严禁与 CNN 缓存混键）。
- 种子：与 2c/step0 同派生（canonical / noise1,2 / `__baseline__:rep{r}`）。
- CNN 参照：Q1–Q3 的对照基线从 `results/baseline_probe/baseline_cache.jsonl`
  按 task_id + classes 校验后读取（同任务同切分的 CNN iters=0 值）。

## §3 判据（四门，先于数据写死）

零假设带 = 各任务 transformer 基线的 2·max(σ, 1/n_eval)（summarize_task 同式）。
"越上带" = 训练后 R' > 基线均值 + 带。

- **Q1**（两条都要）：(a) transformer 基线均值 ≤ 同任务 CNN 基线均值 − 0.05
  （headroom 真的开了）；(b) 某臂 ≥6/20 cell 越上带（开出来的空间用得上）。
- **Q2**：某臂 ≥6/20 越上带 **且** best gain ≥ +0.05（CNN 参照：4↑/20、
  最好 +0.049——门槛设在 CNN 之上）。
- **Q3**：某臂 ≥12/20 越上带 **且** best gain ≥ +0.15（CNN 参照 15/20、
  +0.397——允许弱于 CNN 但必须是真学习者）。
- **Q4**：某臂 δ_nd ≤ **0.015** 且该臂 nd_share ≥ 0.6（δ 公式/regime 判定与
  step0 逐字同式；0.015 = Step0.5 给出的量化门槛：现栈谷底 0.030 的一半，
  信号 p50 0.013 之下）。

**判读表**：

| 条件 | 判定 |
|---|---|
| Q3 ✗ 或 Q4 ✗ | **transformer_closed**：对照崩或地板不达标，任一都使重建无意义；预留分支关闭，出路 X 证据链收口（扩容/iters/换族三重排除） |
| Q3 ✓ Q4 ✓ 且 (Q1 ✓ 或 Q2 ✓) | **GO_full**：全量重建（工作点重扫→δ 重标→27 候选池复审→label v2） |
| Q3 ✓ Q4 ✓ 且 Q1 ✗ Q2 ✗ | **GO_narrow**：现池重建可行、池扩仍死——交拍板 |

## §4 有效性闸（不过则暂缓解读，用户拍板）

1. 276 cell 满格、零 OOM 记 NaN、零重复键、finite ≥95%（transformer 短训
   发散风险高于 CNN，NaN 予以容忍但超 5% 判闸失败并列明分布）；
2. UM extras 与 step0 逐位同集断言、n_eval_exp=2000；
3. CNN 参照匹配：Q1–Q3 六个任务的 baseline_probe 记录 classes 逐字段吻合；
4. 跑前清 GPU 桌面负载并记录 nvidia-smi（既定规程）。

## §5 预测对账（先验，先于数据写死）

| # | 预测 | 先验 |
|---|---|---|
| P1 | Q1 过（随机 attention 基线显著低于随机卷积） | 35% |
| P2 | Q2 过（MI 学得动） | 20% |
| P3 | Q3 过（HW 对照健康） | 40% |
| P4 | Q4 过（地板 ≤0.015） | 25% |
| P5 | 有效性闸全过 | 75% |

联合 GO_full 先验 ≈ 15%——诚实地低；但本探针无论落哪格都终结一个悬案：
过 → 重建有据；不过 → "扩容 + iters + 换族"三重排除，出路 X 的负结果
完备性在文献里罕见。

## §6 执行约束与成本（计时烟囱实测锚，随机数据、零真实读数）

- 实测：SAD 0.53 s/iter、HW/UM ≈0.65 s/iter、MI 1.79 s/iter（L=512 注意力
  平方项）；固定开销 ~2s/cell。transformer 比 CNN 慢 2–8×，臂网格因此收缩为
  {60,240}、每数据集 2 任务。
- 预算：基线 ~3 min；信号 HW≈66min + SAD≈54min + MI≈180min；地板 UM≈201min
  → **点估 ~8.4 h / 保守 ~12 h**（过夜）。
- 4060、OMP=1、threading 后端、串行、`--resume` 幂等、Start-Process 脱离、
  .ps1 纯 ASCII；worklist 按"基线 → 60 臂 → 240 臂"排序（便宜先跑，
  中断也有全臂覆盖）。
- 如实报告：任何门失败都注明"CNN 训练配方未调"的归因限制（§1）。
