# 代理条件化协议：逐格零训练代理读交互项（选拔场，判读带先于计算写死）

日期：2026-08-31 ｜ 脚本：`scripts/29_proxy_conditioning.py`（无对比预训练，本地 4060）｜
单测：`tests/test_proxy_conditioning.py` ｜ 用户 2026-08-31 拍板："开始思路一（代理条件化）"。

## §0 地位声明（选拔场）

本目录是**选拔场**。任何读数（含正读数）不作论文确证证据；过闸只解锁"代理条件化确认协议"
的预注册（新格子、判据见数前锁 commit）。选拔/确认分离（教训：`results/gate2_pilot_b`
选拔 strong → `results/wide_panel` 确认判负）。

## §1 新主张家族声明 + 不翻旧案条款

**主张家族 = 逐格零训练代理（per-cell proxy）**：对每个 (任务 t, CLS c) 直接把该 CLS 的
视图生成管线（输入增强 → 随机初始化编码器 → 嵌入变换）施加到该任务数据上，**测量**类别
结构的存活程度。**不学任何跨数据集映射、不用任何数据集固有特征、不做任何锚点插值、
不做对比预训练**。它是一次测量而非一次预测。

**决定性靶心 = 交互项**。破坏矩阵 SS 分解（本协议 §2 已见）：CLS 主效应 0.242 /
数据集 0.310 / **交互 0.448**。既有全部判负路线读的都不是交互：MX-1/1b 从主效应学映射、
闸 2/behavior_recon 从任务间相似度插值。本协议**只认扣除两个主效应后的残差相关**，
pooled 相关（含主效应）仅作陷阱对照报告——它显著而残差不显著，等于只是重新发现了
"某些 CLS 全局有毒"，全局先验早已做到，**不构成任何新主张**。

本协议**不翻**下列已关闭判定：任务级条件化（P/E 特征核插值，`results/wide_panel` D1w
p=0.9944）、数据集级学习预测器（`results/mx1b_feasibility` norevive）、测量条件化/转导
补全（`results/behavior_recon` 闸门 fail）、毒哨、E 主链、细粒度 task-aware 排序
（`results/source_pool_v2/PROTOCOL.md` 勿复活名单）。本协议无特征输入、无学习映射、
无锚点插值，不属上述任一家族。

## §2 偷看披露（判据锁定前已见的全部读数）

1. 共享 10 CLS 的字段分布（本会话计算）：代理**可见 10 字段** = resize/rescale/jitter/
   point_mask/freq_mask/crop/order/emb_jitter/emb_mask/norm（`norm` 属嵌入变换组，
   无训练前向下生效）；**盲区 9 字段** = instance/temporal/cross_scale/kernel/pool/
   adj_neighbor/loss_type/sim_func/temperature。
2. |log₁₀(temperature)| 与 CLS 全局破坏率 Spearman **ρ=+0.508**（n=10，mx1 labels）；
   破坏率前三的 CLS 温度为 100/0.01/10。→ 盲区字段的效应主要落在被扣除的 CLS 主效应。
   同批算出的增强字段相关（n=10，未校正，纯描述）：crop −0.776、emb_mask +0.458、
   rescale −0.389、resize −0.182、freq_mask −0.107、jitter +0.080、point_mask +0.041。
3. SS 分解复核（mx1 labels，16 数据集×2 任务×10 CLS）：cls 0.242 / ds 0.310 / 交互 0.448。
4. 工程标定（2 格，单次抽样，非结论）：NATOPS_00 g1 clean 0.849 → aug 0.837；
   SelfRegulationSCP2_00 g1 clean 0.568 → aug 0.805（**增强后反升**）；v1 长度
   1152→936。→ 长度改变可能直接改变随机编码器几何，见 §6 长度归因诊断。
5. 既有全部判负读数（wide_panel / mx1* / behavior_recon / canary / anchor_sim）。

**声明**：本协议的代理值（fid/var）、其 SS 分解、以及任何代理-结果关联，在判据锁定前
**从未计算**。§7 的 `--audit` 计数（格数/满格性/拒收）不属指标。

## §3 数据装配（写死；@290 口径的既有 R' 只读，不产新 R'）

**格子宇宙**：复用 `results/behavior_recon` 已对账的装配（25 数据集；PenDigits 排除、
PTB-XL/NIFECG 钉源 crop256、route_a_probe 排除、7 份基线副本去重）。任务粒度：
5 个宽面板数据集（CT/UWave/PS/HW/UM）取 `_00.._07` 共 8 任务，其余 20 个取 `_00/_01`。
→ **80 任务 × 10 共享 CLS = 800 格**，两向残差自由度 ≈ 711。

**结果变量 y(t,c)** = 该格既有 k=1 canonical R' @290（`dm.collect_trained` 同口径）。
三态标签（`fl.tri_state`，band=2·max(σ,1/n_eval)）仅用于 §6 的 D2x/D3x，不用于主判据。

**代理计算口径（写死）**：
1. `Xt, yt = materialize_task(task, X, y)`；`tr_idx, _ = split_task_train_eval(yt,
   task_type, eval_ratio=0.3, seed=run_seed_for(tid, "train_eval_split"))` —— **只用训练
   划分**（增强在预训练期作用于训练数据），eval_ratio 与 canonical split 与生产逐字一致。
2. 每任务 **R=5 个重复**，第 s 个重复的编码器 = `torch.manual_seed(run_seed_for(tid,
   f"proxy_encoder:{s}")); build_encoder(input_dims=Xt.shape[-1])`（随机初始化，
   **不预训练**；iters=0 ≡ 随机初始化前向是既有口径，`autocls/baseline_probe.py`）。
   **同一任务的 5 个编码器跨全部 10 条 CLS 共用**（配对），编码器种子效应因此在任务内
   对齐、在残差化中抵消。
3. 干净基线：`Z0 = encode_features(enc_s, Xt[tr_idx])`（时间维 max-pool，(n,320)）。
4. 增强臂：`v1, v2, _, _ = augment_two_views(torch.from_numpy(Xt[tr_idx]).float(),
   params={6 增强字段}, order=cls.order, generator=g)`，`g.manual_seed(run_seed_for(
   f"proxy:{tid}", f"{cls_key}:{s}"))`（增强在 CPU，照抄 `cl_trainer` 口径）；
   `z = enc_s(v1.to(device)); z = apply_emb_transform(z, cls, g); Z1 = z.max(dim=1).values`。
   → 代理测量的是**对比目标实际看到的表征**（增强后 + 嵌入变换后）。
5. 几何：`class_geometry(l2_normalize_rows(Z), yt[tr_idx])`（`scripts/17` 逐字复用，
   在单位球面上算，5 维 [g1 NCC, g2 Fisher, g3 silhouette, g4 margin, g5 lift]）。

**主代理（写死，单一）**：`fid(t,c) = mean_s [g1(Z1_{t,c,s}) − g1(Z0_{t,s})]`
——增强后 NCC 类别可分性相对干净表征的**变化量**（负 = 类结构被破坏）。
**次代理（不入主判）**：`var(t,c) = mean_s [1 − cos(pool(z(v1)), pool(z(v2)))]`（InfoTS
variety 语义）；`Δg2/Δg3/Δg4` 三维备查。

## §4 第 0 步：代理自身交互份额杀门（先行，**不看结果变量**）

对 fid 的 80×10 矩阵做两因素 SS 分解（`dm.ss_decomposition` 同式）。若代理值主要随 CLS
变、在同一 CLS 内几乎不随任务变，则其残差方差趋零，**结构上不可能解释结果的交互**。
本判读只用 x，不涉 y，故不是偷看。

| 档 | 条件（fid 的交互份额 `share_inter`） | 动作 |
|---|---|---|
| **杀** | < 0.10 | 停：代理是准 per-CLS 常数，REPORT 记杀因，不跑第 1 步 |
| **琥珀** | [0.10, 0.25) | 继续，过闸加严：主判据另需 ρ≥0.15（不接受边缘档） |
| **绿** | ≥ 0.25 | 继续，按 §5 原判读带 |

## §5 第 1 步：决定性检验 D1x（交互残差相关）

两因素加性残差化（对 x=fid 与 y=R' 各做一次）：
`r_ij = v_ij − v_i· − v_·j + v··`（i=任务，j=CLS；80×10 满格，无缺格）。

**D1x** = `Spearman(r_x, r_y)`，**单侧**（H1: ρ>0，即类结构保得越好 R' 越高）；
零分布 = **在任务内置换 r_y**（保任务块结构、破坏交互配对），`n_perm=10000`，
固定种子 20260831，`p = (1 + #{ρ_perm ≥ ρ_obs}) / (n_perm + 1)`。

| 判读带（先于计算写死） | 结论 |
|---|---|
| p<0.05 ∧ ρ ≥ 0.15 | **成立**：代理读到交互 → 立确认协议 |
| p<0.05 ∧ ρ ∈ [0.05, 0.15) | **边缘**：信号真但弱，如实报，确认与否交用户（琥珀档不接受本行） |
| 其余 | **不成立**：逐格代理家族在本口径下关闭 |

**陷阱对照（必报）**：pooled `Spearman(x, y)`（不残差化）与仅扣 CLS 主效应的半残差版。
若 pooled 显著而 D1x 不显著 → REPORT 首句必须写明"只是重新发现 CLS 全局毒性，非条件化"。

**衰减天花板注记**：y 的 k=1 三态重测信度 κ=0.471（`results/behavior_recon`），R' 的
复测噪声同源，故可观测 ρ 上限约为真值的 0.7 倍；判读带已按此下调，不因此事后放宽。

## §6 诊断与敏感性（预注册，不改判读带）

- **S-std**：按任务标准化残差（除以任务内残差 SD）后重算 D1x——防贴顶任务残差被压缩稀释。
- **S-var**：次代理 var 单独跑 D1x；**S-2d**：fid+var 二维（InfoTS 双准则）对三态的
  多项逻辑回归折内 AUC（探索性）。
- **S-geom**：Δg2/Δg3/Δg4 各自跑 D1x（探索性，多重比较不校正，只报不判）。
- **长度归因诊断（因 §2.4 预注册）**：`ΔT(t,c) = (T_v1 − T)/T` 与 fid 的相关；以及把
  ΔT 作为协变量偏出后 D1x 的偏残差 Spearman。若代理主要经长度起作用，如实写入 REPORT
  与 limitation（**不改判读带**：能预测即有用，但机制归因必须诚实）。
- **D2x（数据集级，看门层安慰奖）**：每数据集 fid 均值 vs 该数据集破坏率，n=25 Spearman，
  达标线 ρ≥0.5 ∧ p<0.05（沿用 canary V1 同款带）。过则代理可把看门层从"21 格实测
  ≈7 min @5090"降到零预训练前向，**独立于 D1x 成立与否**。
- **D3x（格级判别，描述性）**：fid 对 destroy vs 非 destroy 的 AUC（pooled 与任务内）。
- **盲区注记**：代理对 9 个训练配置字段结构性不可见（§2.1）。假设"这些字段的效应主要是
  数据集无关的主效应"由 §2.2 的 ρ=+0.508 支持，但这是 n=10 的弱证据，写入 limitation。

## §7 预测表（先验主动下调；教训 `results/anchor_sim/REPORT.md` §5"小预算测量功效
先验一贯过乐观"）

| # | 预测 | 先验 |
|---|---|---|
| PX1 | 第 0 步绿档（fid 交互份额 ≥0.25） | 45% |
| PX2 | D1x 成立（p<0.05 ∧ ρ≥0.15） | 25% |
| PX3 | D1x 至少边缘（p<0.05） | 45% |
| PX4 | D2x 过（数据集级 ρ≥0.5） | 50% |
| PX5 | 陷阱情形：pooled 显著而 D1x 不显著 | 35% |
| PX6 | 长度诊断显示 |ρ(ΔT, fid)| ≥ 0.3（代理相当程度经长度起作用） | 40% |

## §8 执行

顺序：协议锁 commit → tests（红）→ 实现 → tests（绿）→ `--audit` → commit →
`--proxy`（算 fid/var，约 35 min @4060，落 `proxy_cache.jsonl` 支持 `--resume`）→
`--step0` 杀门 → `--analyze`（D1x + 全部诊断）→ REPORT + memory + commit。
零新 R'、既有 R' 缓存只读；本地 4060（跑前清 GPU 负载——`桌面 GPU 负载破坏确定性`
教训；代理为多实例均值，微小数值抖动不影响判读，但 `device_name` 照记）。
产物：`results/proxy_conditioning/{audit.json, proxy_cache.jsonl, step0_share.json,
analysis.json, REPORT.md}`。所有判定标注 **@290 + n 口径**。
