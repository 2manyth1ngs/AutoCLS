# SWA 噪声探针协议（先于数据写死，2026-08-13）

## 背景与假设

排查二（commit 26d02ee）发现唯一未对齐的训练期差异：TS2Vec 官方 `encode` 用
`AveragedModel` 等权平均权重评估，本项目用原始权重。decisive/outlet_y 已把核心
障碍定位为：任务特异信号 |Δ| 中位 0.023–0.033 vs 单种子轨迹噪声 P95=0.110（HW），
δ 按噪声标定 → 存活 label 任务盲天花板 0.9525 → E 零边际（C2）；required-k≈11。

**假设**：SWA 平均整条训练轨迹 → R' 跨 weight_seed 重复差收窄 → required-k 降至
可行区。正当性 = 向 TS2Vec 官方实现修正，非为结果调参。

官方语义（`reference/AutoCTS_plusplus/NAS_Net/ts2vec.py`，与 yuezhihan/ts2vec
逐行一致）：`__init__` 创建 AveragedModel 后立即快照 init 权重（:53-54）；每次
`optimizer.step()` 后 update（:152，逐 iter）；等权 running mean（K 次更新 =
K+1 个快照含 init 的等权平均）；`encode` 用平均权重（:177）。

## 实验矩阵（与 labels_workpoint 噪声探针逐字同口径）

- 数据集×工作点：Handwriting @60 iters（20ep）、UniMiB-SHAR @30 iters（5ep）
  （labels.yaml `workpoint.iters_per_dataset`；池加载走 2c `load_pool`，28 任务）
- 每数据集：前 2 任务（`tasks[:2]`）× shared CLS 前 10（`probe_cls_catalog.json`，
  2c 同源）× 3 个 weight_seed × SWA {off, on} → **240 条**
- 种子派生（与 2c 完全一致）：
  - rep0（canonical）：`weight=run_seed_for(tid, cls_key)`，
    `split=run_seed_for(tid, "train_eval_split")`
  - rep1/2：`weight=run_seed_for(tid, f"{cls_key}:noise{rep}")`，split 仍 canonical
- 环境：本地 4060、threading 后端、OMP_NUM_THREADS=1、串行；swa 字段入行与
  done-key，独立落盘 `results/swa_probe/probe_cache.jsonl`，不触碰历史缓存
- 预估 30–35 min GPU wall（锚点 HW 7.1s/条、UniMiB 5.6s/条 + SWA <2% 开销）

## 对照基准（既有实测，逐字引用）

- 噪声（labels_workpoint k=3 探针，全部产自本机 4060）：
  HW 中位 0.029 / P95 0.110 / max 0.160；UniMiB 中位 0.020 / P95 0.075 / max 0.098
- HW best_gain +0.391@20ep（workpoint confirm n=60）；HW 56↑/1↓ @n=60
- 任务盲天花板：k=1 δ_wp **0.9525**；δ=0 **0.7473**；探针 cell k=3 δ=0 **0.8427**(n=178)
- required-k：HW (0.110/0.033)²≈11；UM (0.075/0.023)²≈11

## 主判据（SWA-on 的 HW 噪声 P95；P95 取法与 2c calibrate 同式：
sorted diffs 的 `round(0.95*(n-1))` 索引，diffs = 同 (task,cls) cell 内 3 种子两两 |ΔR'| 池化）

| P95_on | 判定 |
|---|---|
| < 0.05 | 噪声减半以上，required-k 降到 ~3 内 → **task-aware 有救**，值得重做 label + 重训 T-CLSC + 重测 E 消融 |
| 0.05–0.08 | 部分改善 → 报 required-k 与天花板实际变化，用户拍板 |
| > 0.08 | 改善不足（原 0.110），**SWA 不是出路** |

阈值理由（独立于结果）：required-k ≤3 需噪声降到 0.110/√(11/3)≈0.058 以下，取 0.05 略严。

## 否决条件（任一命中即判 SWA 无效，无论噪声降多少）

1. **信号被压**：SWA-on 的 HW best_gain 相对 SWA-off **同 20 cell** 的 best_gain
   下降超过 30%（分母用 off 臂同 cell 值——探针只覆盖 confirm n=60 的子集，
   off 臂 best_gain 未必恰为 +0.391；off 臂值为既有常数，此选择先于 on 臂数据）。
   用户原始阈值（on best_gain < 0.7×0.391 = 0.274）同表并报。
   best_gain 口径：rep0（canonical）R' − 基线均值（`build_baseline_bands`，
   iters=0 基线缓存；SWA 不影响 iters=0，单测已固定），对 20 cell 取 max。
2. **实现有 bug**：off 臂 120 条无法逐位复现 labels_workpoint 既有值
   （rep0 ↔ rprime_cache.jsonl，rep1/2 ↔ noise_probe.jsonl；全部同机 4060 产出，
   预期逐位相同）。

## 派生诊断（同批数据，零额外 GPU）

1. **δ 重标定**（用户拍板：2c calibrate 口径）：
   `δ = max(P95(同 CLS 重复对 |ΔR'|), 2×量子)` 上取整到 0.005，per-dataset。
2. **required-k**：`k ≥ (P95_on / 容p50)²`，容 p50 = HW 0.033 / UM 0.023
   （信号侧沿 outlet_y 实测不变），对照原 11。
3. **任务盲天花板**（探针 cell 口径，outlet_y 同法）：4 探针任务、shared-shared
   任务内配对、rep-mean R'、按 per-dataset δ 过滤后，任务盲多数方向可达 acc。
   报 on 臂 @δ_new 与 @δ=0；off 臂 @δ=0 须复现 0.8427 作内部校验。
   口径注记：探针 4 任务 ≠ 全池 28 任务，对照 0.9525/0.7473 时如实标注。
4. **副诊断（信号侧）**：两臂各报 best_gain 与 ↑/↓ 条数（显著判据 = 现行工作点
   δ：HW 0.110 / UM 0.075，rep0 口径），对照 56↑/1↓@n=60 的量级折算。

## 落盘

`results/swa_probe/{probe_cache.jsonl, analysis.json, REPORT.md}`；
脚本 `scripts/7_swa_noise_probe.py`（--run/--analyze，--resume 幂等 append-only，
种子写入无截断式 open；device_name 逐行落盘）。

## 预注册排除声明

- 本实验只动评估权重（训练轨迹逐位不变，单测固定）；不调 lr/iters/batch 等任何超参。
- 若 SWA 压不动噪声、或压噪声同时压信号（否决 1），如实判"此路不通"并收官——
  这是排除最后一个便宜选项，不是失败。
- 目标池不碰（铁律 6）：本实验只用源池 Handwriting / 候选池 UniMiB-SHAR。
