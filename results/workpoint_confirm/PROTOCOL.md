# workpoint confirm —— Handwriting / UWaveGestureLibrary 60 条确认（协议锁）

日期：2026-08-11 ｜ 本机 RTX 4060 Laptop ｜ 判据/网格/选点规则/预测先于数据写死，
commit 后开跑。动机：Insect 刚证明 n=20（2 任务）判定可以是局部性质（60 条 C 层
撤销），而 Handwriting / UWave 的 usable 判定同样是 n=20——**在确认之前源池 38
这个数字不可信**。用户两条裁决（2026-08-11）：① 开跑含 @20ep 加测；② UWave
选点用 0.02 让分窗。

## §1 范围与臂

- **Handwriting 确认**：@120 iters（40ep，realized 峰）与 @60 iters（20ep）各扩到
  60 条（10 shared CLS × 前 6 任务）。既有 20 条/臂（任务 00–01，4060）从
  `results/iters_workpoint/uea/trained_fill.jsonl` 幂等复制入本 outdir 作 resume 种子；
  新增 40 条/臂 = 任务 02–05。
- **UWave 工作点扫描**：网格 {1,2,5,10,20} ep × 3 steps/ep（任务 00–01 实测
  n_train=312/273）= iters **{3,6,15,30,60}**，20 条/点（前 2 任务）。@290 历史臂
  （96.7ep，20 条，5090，`stage1_rerun_20260804/.../rprime_cache.jsonl`）作为参照臂
  并列入判定表，不新跑。
- **UWave 确认**：在 §2 选出的工作点扩到 60 条（前 6 任务）。
- epoch 错位预注册：Handwriting_02（246→2 spe）、UWave_03/04（156/195→2 spe）
  使 6 任务口径 steps/ep 不唯一——**判定一律按 iters 臂**（`allow_mixed_spe`，
  Insect 任务 02 先例）。
- 目标池不碰（铁律 6）。iters 入缓存键（resume 键 = task_id, cls_key, iters）。
  新记录全部带 `device_name`。

## §2 UWave 工作点选择规则（纯函数 `select_workpoint`，先测后写）

1. 候选臂 = 过 **T1+T2+T3(20)** 的臂（扫描臂 ∪ @290 参照臂）。
2. 取 best_gain 最大者；**与最大者差 ≤0.02 视为并列，并列取更少 iters**
   （让分不传递）。理由：@290 的 +0.077 带 5090 ±1 样本抖动；破坏参与随 iters
   单调涨（followup §4 规律 2）；Handwriting 弃 290 选 20–40ep 先例。
3. 候选集空 → 回落 @290 确认（唯一既有 usable 点）。此分支下 5090 的 20 条作
   种子 + 4060 补 40 条，60 条内硬件混合记 limitation。

## §3 确认判据（n=60，逐字沿用 followup 协议；实现 `confirm_verdict` 原样）

- **A 确认（可入池）**：↑≥37/60（Wilson 90% 下界 >0.5）且 T3(60)（↑≥30 且
  ↑≥2×↓）且 T2（best_gain ≥0.05）且 T1（基线 ≤0.90）。
- **B 维持但未坐实**：T3(60) 过、↑∈[30,36] → judgment 维持但不作入池依据。
- **C 反转**：T3(60) 不过 → 60 条优先于 20 条，usable 判定撤销。
- **Handwriting 主判定臂 = @120（realized 峰）**；@60 同判据独立分层；
  **入池判定 = 两臂最高层级**（工作点是段不是点，followup §6.4-2；段内任一
  预注册臂 A 即坐实）。UWave 判定臂 = §2 选出的单臂。
- 同时报告（不立门槛）：各臂破坏参与率 strict/loose（δ=0.05 主、0.03 敏感），
  δ 存活率；对照 Handwriting@20ep 既有 71%/0%。
- 基线与 ±2σ 带：`results/baseline_probe/baseline_cache.jsonl` 5 种子（幂等复制入
  outdir），↑/↓ 与 strict 带子同源（summarize_task null_band）。

## §4 成本（锚点带硬件标签；上轮"~15 min 合计"系用 5090 秒数误估，已纠正）

| 项 | 新增 | 锚点 | 点估 |
|---|---:|---|---:|
| Handwriting @120 | 40 | 15.1 s/条（4060 同臂实测） | 10.1 min |
| Handwriting @60 | 40 | 7.1 s/条（4060 同臂实测） | 4.7 min |
| UWave 扫描 5 臂 | 100 | 0.23 s/iter（⚠ 5090→4060 按 Handwriting 双机比 2.36× 换算） | ~11 min |
| UWave 确认（≤60 iters 臂） | 40 | 同上 | 3–10 min |
| （回落）UWave 确认 @290 | 40 | ⚠ 67 s/条（换算） | 45 min |
| 启动开销 ×8 | — | ~15 s/次 | 2 min |

主路径点估 ~35 min（预算 70）；回落分支 ~73 min（预算 90）。

## §5 预测（对账用；全部涉及 2→6 任务外推 = Insect 翻车同型，带宽已放大，
且声明：followup P4/P5 曾落在二项带外，本带同样可能不覆盖）

- **P1** Handwriting@120：↑∈[38,52]（点 45）、↓≤6，T3(60) 过；A 层概率 ~0.6。⚠
- **P2** Handwriting@60：↑ 与 @120 差 ≤6 条，↓ ≤ @120 的 ↓（破坏单调内插）。
- **P3** Handwriting 6 任务 strict 破坏参与：@60 ≤20%、@120 ≤30%（2 任务时
  0%/13%）。⚠ 纯先验。
- **P4** UWave 扫描：≥1 臂过 T2；存在过 T1+T2+T3(20) 臂的概率 ~0.5；若存在，
  落在 {15,30,60} 而非 {3,6}。⚠ 机制推理（@290 strict 65% + 10↑ 暗示信号需一定
  训练量），无直接数据。
- **P5** UWave 确认：↑∈[22,38]（点 30，恰在 T3(60) 门槛）；C 撤销概率 ~0.45、
  A 层概率 ≤0.25。依据：n=20 的 10↑/3↓ 弱于 Insect 撤销前的 12↑/0↓。⚠
- **P6** UWave 确认臂 strict 破坏参与 ∈[40,80]%。
- 情景先验（主观）：双 A ~20%；HW A + UW 撤销/B ~40%；HW 仅 B ~20%；
  HW C（终点信号）~10–15%。

## §6 结果解读（先写死，防事后编故事；用户原文）

- **双 A** → 源池 38 任务可信，进入下一阶段讨论（"干净 label 95% 来自一家"仍在）。
- **Handwriting 坐实、UWave 撤销** → 源池 22 任务，一家独大更严重。
- **Handwriting 撤销** → 明确的终点信号：全项目唯一"高产+干净"源不成立，源池
  实质为空。**该认真考虑调整项目主张（task-aware CLS 排序 → 跨任务 CLS 可行性
  预测），而不是继续找数据集。如实报告，不试图补救。**

## §7 执行防护（命令重放事故教训，2026-08-10）

- 种子写入幂等：`seed_fill.py` 按键（task_id, cls_key, iters）增补，只 append
  缺失键，禁止截断式 `"w"`；重放无害。baseline_cache 存在即跳过。
- 队列脱离启动（Start-Process）一次；启动前查 inflight/既有进程防重复拉起。
- 收官三重校验：满格（220 新增无 null）/ 零重复键 / 配对审计（classes,
  split_seed）零拒收，然后才出 REPORT。
- OMP_NUM_THREADS=1 + threading 后端（route-a 吞吐结论）。
