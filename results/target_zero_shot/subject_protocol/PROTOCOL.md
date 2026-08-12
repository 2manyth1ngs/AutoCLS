# 目标池 zero-shot 评估·出路 A：跨受试者协议（判据先于数据）

日期：2026-08-12 ｜ 拍板：用户选出路 A（首轮停机报告 §7）｜ 执行：
`scripts/5_target_zero_shot.py --subject`（缓存进本目录；判据/网格/候选池与首轮
**逐字不变**，唯一改动 = 切分口径）

## 1. 协议改动（唯一变量：切分）

1. **任务内 train/eval**：`split_train_eval_by_group`（受试者整组进一侧，组随机
   排列首次跨越 eval_ratio×n 即停，类覆盖校验 fail-fast）。canonical seed =
   `run_seed_for(task_id, "subject_eval_split")`，任务内恒定（5 基线种子与全部
   trained 记录同切分——与源池"split 恒 canonical、weight 逐种子"口径一致）。
2. **sweep 真值 val**：受试者留出 **{8, 17, 21, 25}**（首轮诊断 rng seed=0 抽出，
   停机报告 §9-1 要求固定，本协议预注册）。fit 5886 / val 1466。
3. 数据加载、任务切片（8 任务、RandomClassSubsetSlicer）、候选池 31、比较器
   排名（已收官，复用 `../ranking.json`）、encoder 预训练（全量 train 自监督）
   均不变；task_features / tclsc checkpoint / encoders 与首轮共享（切分无关）。

## 2. 判据（逐字沿用，不因首轮停机而放松）

- 臂判定 T1（基线均值 ≤0.90）/ T2（best_gain ≥0.05）/ T3（↑ 过半且 ≥2×↓）+
  测满 20 条；选点 `select_workpoint`（0.02 让分窗、并列取最少 iters）。
- **停机规则不变**：跨受试者口径下仍无 usable 臂 → 停下报告（此时"UCI-HAR
  任务级标定不可用"就是真结论，不再有切分伪影的辩护）。
- 过闸 → 直通步骤 4-5（用户已拍板 A 路线整链）：31+2 sweep（val 真值）→
  top-3/GGS×3/winner 补种子 官方 test 评估 → winner @2320 桥接。
- 测试集纪律：test 仅 --final 触碰（预算 ≤9 次 SVM 评估）；标定/sweep 只用 train。

## 3. 预测（先于数据写死；沿用"宽带 + 情景先验"配方）

- P1 探针任务（00/01）跨受试者基线 ∈ [0.83, 0.94]，两任务均值过 T1（≤0.90）
  概率 ~0.5（全量 9 通道 6 类口径 0.891；任务通道子集 7/6 通道、类子集 6/5 类，
  方向不定）。
- P2 若 T1 过：trained 正增益在 ≥2ep 臂出现，best_gain ∈ [0.02, 0.10]，
  usable 臂存在概率 ~0.5；工作点若有，落 {2, 5, 10} ep 的先验最高。
- P3 若走到步骤 5：GGS@工作点 test ∈ [0.89, 0.945]；winner test ∈ [0.90, 0.955]；
  winner − GGS ∈ [−0.02, +0.03]（比较器是任务盲质量先验，赢 GGS 不是默认预期）。
- P4 真值对账：val 真最优落比较器 top-10 概率 ~0.6；τ(比较器, val) ∈ [0.1, 0.5]
  （顶端排序未经真值检验，垫底段拉高 τ）。
- 情景先验：{T1 再挂停机 ~35%，过闸且完整走完 ~50%，过闸但 sweep 出幺蛾子
  （OOM/翻车）~15%}。

## 4. 成本预算（4060）

标定基线 40 + fill 100 ≈ 20 min；sweep 33 预训练 @W ep（W=2/5/10/20 →
~25/30/45/75 min）+ final ~20 min + 桥接 3 min。点估 1.2-2 h / 保守 3 h。
