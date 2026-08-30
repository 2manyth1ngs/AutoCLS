# 破坏矩阵协议：可行性边界的任务依赖性存在性检验（判读带先于计算写死）

日期：2026-08-30 ｜ 脚本：`scripts/14_destruction_matrix.py` ｜ 单测：
`tests/test_destruction_matrix.py` ｜ **零 GPU**：全部来自既有缓存，无新测量。
本协议与代码先 commit，计算后出报告。

## §1 问题与来源

抓手三（任务感知可行性预测）的存在性前提：**"哪些 CLS 是破坏性的"这条边界
是否随任务/数据集移动**。若破坏性几乎全是 CLS 固有属性（全局 65% 那一份），
E 无信号可吃、抓手三关闭；若数据集×CLS 交互份额可观，则可行性 label
（幅度 0.1–0.4、天然过 δ）能承载可测的 task-aware，且贴顶数据集在此 label
类型下复活（它们恰恰最擅长产破坏分离）。

## §2 数据源与配对（先于计算写死）

- **Panel A（主，@290 统一口径，16 UEA 数据集 × 2 任务）**：
  trained = `stage1_rerun_20260804/.../rprime_cache.jsonl`（260）+
  `baseline_probe/trained_fill.jsonl`（60）；baseline =
  `baseline_probe/baseline_cache.jsonl`（默认档，剔 task_id 含 "+c"）。
- **Panel B（次，工作点口径，HW@60 12 任务 + UM@30 16 任务）**：
  trained = `labels_workpoint/rprime_cache.jsonl`（shared 10）；baseline =
  `iters_workpoint/{uea,candidates}/baseline_cache.jsonl`（iters=0 行）。
- 配对校验（逐行）：同 task_id、trained.classes == baseline.classes、
  split_seed == canonical（`run_seed_for(tid,"train_eval_split")`）、
  cls_key ∈ shared 10（探针目录）、iters 匹配面板口径。拒收计数如实报告。
- 两面板 iters 口径不同，**不跨面板合并**；各自出矩阵与分解。

## §3 量与判定（公式先于计算写死）

- **破坏判定**（summarize_task 同带）：任务基线 mean/σ 取自 ≥3 个种子，
  band = 2·max(σ, 1/n_eval)；D[cls, task] = 1 ⇔ trained R' < mean − band。
  trained 为 k=1 canonical（局限：边界附近单种子误判存在，破坏差距大故有限）。
- **速率表**：p[cls, dataset] = 该数据集内任务的 D 均值。
- **两因素 SS 分解**（速率表上）：SS_cls（CLS 主效应 = 任务盲部分）、
  SS_ds（数据集主效应）、SS_inter（交互）。
  **任务依赖份额 TDS = (SS_ds + SS_inter) / SS_total**——E 理论上能吃的部分。
- **mixed CLS**：在 ≥3 个数据集上破坏（p≥0.5）**且**在 ≥3 个数据集上安全
  （p=0）的 CLS 条数（/10）——边界移动的直接携带者。
- 诊断（不判定）：数据集破坏率与 L 的 Spearman；per-CLS 破坏与 log L 的对齐。

## §4 判读带（先于计算写死；以 Panel A 为主判）

| 条件 | 判定 |
|---|---|
| TDS ≥ 0.25 **且** mixed ≥ 3/10 | **成立**：抓手三立项（可行性 label 管线 + E 条件化分类器 + E 消融预注册） |
| TDS ∈ [0.10, 0.25) 或 mixed = 2 | **边缘**：交用户拍板 |
| TDS < 0.10 且 mixed ≤ 1 | **不成立**：破坏性以 CLS 固有为主，抓手三关闭 |

Panel B 作跨口径佐证（仅 2 数据集，SS_ds/inter 自由度小，不参与主判）。

## §5 有效性

Panel A 配对命中应 ≈320（32 任务 × 10 CLS，容许个别 null）；每数据集 2 任务
齐全 ≥14/16；基线每任务 ≥3 种子。方向锚（软）：baseline_probe §2.2 的
per-dataset ↓ 计数（20 CLS 口径）与本矩阵 shared-10 破坏率排序应大体同向
（NATOPS/Ethanol 高、HW/Heartbeat 低）。

## §6 预测（先验，先于计算写死）

| # | 预测 | 先验 |
|---|---|---|
| P1 | TDS ∈ [0.15, 0.45]（既非纯固有也非纯任务） | 50% |
| P2 | mixed ≥ 2/10 | 55% |
| P3 | 数据集破坏率与 L 正相关（ρ>0.3；resize/crop 类在长序列上更毁） | 45% |
| P4 | Panel B 的 HW vs UM 破坏模式差异与已知 strict 份额差（5% vs 66%）同向 | 70% |
