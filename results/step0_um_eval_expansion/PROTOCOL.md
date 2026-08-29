# Step 0 探针协议：UniMiB 评估侧扩容 + 两制式 δ（判据先于数据 commit）

日期：2026-08-29 ｜ 脚本：`scripts/11_step0_um_expansion.py` ｜ 单测：
`tests/test_step0_um_expansion.py` ｜ 本协议与代码先 commit，数据后跑。

## §1 目的与来源

v2 测量台设计（2026-08-29 会话拍板）的第一步"在家验尺子"：在**不换池、不换
encoder** 的前提下，实测"评估侧单边扩容 + 两制式 δ"这套测量方案能否把任务盲
天花板压下来。上游证据链：

- 评估集规模探针（CONFIRMED）：噪声中位 ∝ 1/√n_eval（β_med=0.502）、尾部不
  跟随（β_p95=0.272）→ 扩容必须搭配破坏性制式隔离才能落到 δ。
- outlet_y §5：任务特异票 |Δ| p50=0.023 / p75=0.039——δ 须 ≤0.033 档才放进一半。
- 切片余量实测（本次协议前只读诊断）：UM 16 任务全部可扩到 n_eval=2000
  （leftover 1836–9289；最紧 _13 恰好 2000）；HW 全集 1000 无余量，不参与。

**范围声明（防误杀，先于数据写死）**：本探针只有 UniMiB 一个数据集。UM 是
现行池里最同质的（16 任务同一份加速度计数据切片、label 破坏参与 92%、UM 内
天花板 0.9890）。跨数据集任务分歧（ρ 0.240 vs 数据集内 0.574）本探针**够不到**
（HW 侧无法扩容）。因此本探针的任何"天花板仍高"读数**最多判"UM 内无可及
分歧"，不得外推为 task-aware 全局死刑**——判读表 §4 按此写死。

## §2 设计（先于数据写死）

- **训练侧与工作点逐字同口径**：UM 16 任务 × 10 shared CLS（route_a_probe
  探针目录前 10）× 3 种子（rep0=canonical、rep1/2=`run_seed_for(tid,
  "{key}:noise{rep}")`，split 恒 canonical）× iters=30 = **480 个训练 cell**。
  训练数据、n_train、种子与 labels_workpoint 完全一致——rep0 的 canonical 读数
  必须与既有缓存逐位复现（有效性闸 §5）。
- **每 cell 训练一次、两个读数**（同一 encoder、同一 SVM）：
  - canonical：既有 eval 集（n≈307，split canonical），逐样本判对向量落盘；
  - expanded：canonical eval ∪ extras（目标合计 n_eval=2000），extras 逐样本
    判对向量独立落盘。
- **extras 构造**（`expand_eval_indices`，纯函数+单测）：eligible = 全体
  `y ∈ task.classes` 的样本 − task.sample_indices（切片器同语义重建）；按
  leftover 类别占比最大余数法分层、`default_rng(run_seed_for(tid,
  "eval_expand"))` 确定性抽取 `2000 − n_can` 个；与任务样本零交集（运行时断言）。
  训练侧零触碰——extras 只过 encode+predict，不进任何训练。
- **iters=0 基线臂**：16 任务 × 3 权重种子（`run_seed_for(tid,
  "__baseline__:rep{r}")`）× 两个读数 = 48 行。CLS 在 iters=0 完全不参与
  （tests/test_baseline_probe.py 既有断言），记 `cls_key="__baseline__"`。
  用途：expanded 分辨率下的破坏判定带（§3 regime）。

## §3 量与口径（公式先于数据写死）

- **噪声对**：同 (task, cls) 跨 3 rep 的两两 |ΔR'|（480 对/读数口径）。
- **δ 公式**（与 2c calibrate 逐字同式）：δ = max(P95(噪声对), 2×max 量子)，
  上取整到 0.005。分别算 canonical / expanded × 全体 / 非破坏制式四个值；
  **δ_nd(exp)（expanded × 非破坏）是 E1 的主量**。
- **制式判定（regime，per (task, cls)）**：strict 破坏 ⇔ mean_k3(R'_exp) <
  base_mean_exp − band_exp，band = 2·max(σ_base, 1/n_eval_exp)
  （summarize_task/destruction_participation 同带子；k=3 均值参与判定，比 k=1
  稳）。非 strict 即非破坏制式。
- **两制式配对规则（E3 生产模拟）**：任务内 shared 无序对 (a,b)——任一方
  strict 破坏 → 用 δ_all(exp) 过滤（破坏制式，本来就是大差距对）；双方非破坏
  → 用 δ_nd(exp)。方向取 rep0（k=1，生产口径）expanded 值经 is_better；
  严格平局弃。
- **任务盲天花板**（noise_structure_analysis.ceiling 同语义，重实现+单测）：
  对每个存活 shared 无序对聚跨任务票，ceiling = Σ max(方向票)/Σ 票。
- **E2 信号存在性上界**：k=3 均值 expanded R'、δ=0（仅弃严格平局）的 UM 内
  天花板——"若测量无噪声，UM 内部有多少真分歧"的上界估计。
- **守卫（防误放，outlet_y 同款诊断）**：新增存活 ND 票 = 双方非破坏、
  |Δ_exp|≥δ_nd(exp) 且 |Δ_can|<0.075（今日生产口径会弃）的票；其三种子方向
  全同率（rep0/1/2 的 Δ_exp 同号）须 ≥0.95。

## §4 判读表（先于数据写死；四格，单一 kill-gate 被拆分以防误杀）

| # | 条件 | 判定与去向 |
|---|---|---|
| A | E1 失败（δ_nd(exp) > 0.045 或噪声中位比 ∉ [0.30, 0.55]） | **尺子未兑现**：测量方案本身失败，先诊断（尾部结构/环境噪声）再谈信号；不产生任何关于 task-aware 的结论 |
| B | E1 过 且 E2 天花板 ≥ 0.98 | **UM 同质**：UM 内部无可测分歧——Step 0 无法裁决 task-aware，结论是 Step 2 池必须异质化；不是 kill |
| C | E1 过 且 E2 < 0.98 且 E3 过（两制式天花板 < 0.95 且守卫 ≥ 0.95） | **GO**：扩容+两制式在真实池上兑现，v2 测量台成立，进 Step 1（encoder 探针）与建池 |
| D | E1 过 且 E2 < 0.98 且 E3 失败 | **真 kill**：信号被 E2 证实存在，但修好的尺子仍够不着或只够得着噪声票——"扩容+两制式"这条路对 task-aware 关闭，出路 X 为唯一主张 |

守卫失败（天花板 <0.95 但守卫 <0.95）计入 D 的"够得着的是噪声票"分支，
如实分开报告。

## §5 有效性闸（不过则暂缓解读，用户拍板）

1. 480 训练 cell + 48 基线行满格、零 OOM 记 NaN、零重复键、每 cell 3 rep 齐全；
2. extras 与 task.sample_indices 零交集（运行时断言）；n_eval_exp = 2000（16/16）；
3. **逐位复现闸**：rep0 canonical R' vs labels_workpoint `rprime_cache`（160 参
   考）+ rep1/2 canonical vs `noise_probe`（任务 00/01，40 参考）——匹配数
   ≥80% 逐位相同且 max|Δ|≤0.02（新鲜进程豁免带；**跑前必须清空 GPU 桌面负载
   并记录 nvidia-smi 状态**，评估集规模探针 §7 操作规程）；
4. canonical δ_all 在任务 00/01 子集上复算应落 0.075±0.02（标定锚）。

## §6 派生诊断（不判定）

- 旧生产口径连续性：rep0 canonical、δ=0.075 的 UM 内天花板（应≈发布 0.9890）；
- 16 任务版 canonical δ_all（原标定仅 2 任务，首次全池噪声面）；
- E2 的 k=1 对照（k3 均值 vs rep0 单种子在 δ=0 下的天花板差 = 噪声翻转票占比）；
- 存活量表：两制式下 ND 票数 vs 今日口径存活数（label 产量变化的直接预估）；
- per-任务 regime 分布（strict 破坏 CLS 数/任务）。

## §7 预测对账（先验，先于数据写死）

| # | 预测 | 先验 |
|---|---|---|
| P1 | 噪声中位比 med(exp)/med(can) ∈ [0.30, 0.55]（√n 纯值 0.39） | 60% |
| P2 | δ_nd(exp) ≤ 0.045 | 55% |
| P3 | E2 天花板 ∈ [0.80, 0.97]（低于同质线） | 65% |
| P4 | E3 全过（两制式天花板 <0.95 且守卫 ≥0.95） | 35% |
| P5 | 有效性闸全过（含逐位复现） | 70% |

P4 先验诚实偏低：outlet_y 变体 B 近似（混合口径）UM 内 0.9472，本探针的
扩容+两制式比该近似更强，但 UM 同质性（§1 范围声明）压先验。

## §8 执行约束与成本

- 4060、OMP=1、threading 后端（勿删）、串行、`--resume` 幂等、device_name 落盘、
  Start-Process 脱离启动、.ps1 纯 ASCII；缓存 `probe_cache.jsonl` +
  `baseline_cache.jsonl`（独立于 labels_workpoint，零写入既有缓存）。
- **跑前清空 GPU 桌面负载**（Wallpaper Engine 等）并核对 nvidia-smi 待机
  <0.5GB/温度正常，启动日志记录状态——违反则逐位闸预期失守。
- 成本：480 训练 cell（UM@30 锚 ~5–7s/条 + expanded 编码 ~+1s）+ 48 基线行
  （~2s/条）≈ **点估 55 min / 保守 2.5 h**（桌面负载残留按 2.5× 计）。
- 如实报告：判读落 A/B 格时明说"本探针不构成对 task-aware 的裁决"。
