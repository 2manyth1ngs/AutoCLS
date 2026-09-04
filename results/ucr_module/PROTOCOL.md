# WP7-UCR 协议锁：在不相交的 UCR 语料上预训练"CLS × 任务特征"打分模块，冻结后插入主管线（零样本 task-aware 的最后一次检验）

日期：2026-09-04 ｜ 分支 `wp7-ucr`（自 wp6-tf 7ea08e8）｜ 脚本 `results/ucr_survey/scripts/ucr_survey.py`（调查/预取）、
`scripts/41_ucr_rprime.py`（语料 R'）、`scripts/42_ucr_module.py`（模块 / U1 / U2 / U3）、`scripts/ucr_pod/{make_bundle,setup,run,finish}.sh` ｜
单测 `tests/test_ucr_rprime.py`（9）、`tests/test_ucr_module.py`（7）｜ 计划 `~/.claude/plans/ts2vec-e-quirky-bubble.md`（用户拍板：
shared-10 + random-10；φ = 12 + 趋势/季节 2；回归 z；纯 UCR 冻结 + UCR⊕源池两臂；5090 pod；拉回即停 pod）。

## 0. 诚实声明

- 已看过：WP6-TF 全部读数（`results/mx4_task_feature/REPORT.md`）；判负后的只读诊断（数据集两两排序块结构、分半自洽、
  字段效应 × 12 维统计量 1380 个相关在随机水平、源侧重加权先验反向、多样化名单更差、探针方差）；文献检索（InfoMin / Iwana & Uchida 2021 /
  InfoTS / AutoTCL / TS-ARM 2024 / FFORMA / landmarking / catch22）。
- **UCR 语料上任何 R' 读数、任何 φ、任何模块读数都未算过**（本地只跑过 ChlorineConcentration 的 4/8/12 截点 smoke 与 4060 计时 pilot，
  pilot 数据不入语料）。U1/U2/U3 的臂表、门、预测在此写死。

## 1. 主张与不翻旧案

主张 = **数据集级零样本 task-aware 在 n≈84 个数据集的语料上是否可辨识，并能否迁移到主管线的多变量靶**。
不翻：切片级 task-aware、配对 δ 比较器、E 主链、可行性预测器、核插值、行为锚插值、逐格代理、学习曲线外推（关键决策 8）、
WP6-TF 的 n=8 判负。差异：语料 n 从 8 → 84、模块与源池分离、输出直接是分数。

## 2. 语料（`results/ucr_survey/ucr_survey.csv`，119 个调查完成）

- 来源 `aeon.datasets.tsc_datasets.univariate` 128，Zenodo 镜像下载到 `data/candidates/raw/ucr_cache`（`scripts/candidates/_ucr._ensure_local`，`@problemName` 校验）。
- **硬排除 9**（与源池 / 靶同源，铁律 6）：CricketX/Y/Z、UWaveGestureLibraryX/Y/Z/All、Phoneme、EthanolLevel。
  名字巧合已核非同源、不排除：SemgHandMovementCh2（sEMG）、FaceAll/FaceFour/FacesUCR（轮廓图像）、HandOutlines（X 光）。Semg 三选一（丢 Gender/Subject）。
- 可用判据（先写死）：等长 ∧ 无缺失 ∧ 单变量 ∧ L ≥ 50 ∧ 每类 ≥ 2 ∧ n_eval = 0.3·min(N, 1024) ≥ 50。
  结果：**84 可用**；剔除 变长 12 / n_eval<50 14 / L<50 4 / 缺失 3 / Semg 重复 2。
- 与换池调研重叠的 14 个 UCR 档案（`data/candidates/*.npz`）本次一律从 .ts 重新装载（同口径），它们从未进过源池，无泄漏。
- 语料规模：T 分位 [60, 128, 272, 720, 1024]，任务 N 分位 [166, 343, 750, 1024, 1024]；T ≤ 200 有 36 个、200–600 有 22 个、>600 有 26 个。

## 3. 语料 R' 口径（与 `results/mx3_pq` 逐位同，只换数据集）

整数据集 = 一个任务 `<name>_00`：全部类、channels=(0,)、N ≤ 1024 全取 / N > 1024 按类分层 canonical 子采样（`run_seed_for(name, "ucr_subsample:0")`）；
T > CROP_LEN 前缀截断（**CROP_LEN = 512**，用户 2026-09-04 拍板；26 个 T>600 的数据集受影响，φ 的 log10_T 取截后长度）。两道闸：每类 ≥ 2、n_eval ≥ 50（`41.task_gates`）。
每数据集 20 条 CLS = shared-10（`probe_cls_catalog.json` 前 10）+ random-10（`sample_random_cls(space, "<name>_00", n=10, seed=42)`）；
一条轨迹三截点 {60,120,290}；bs 128、lr 1e-3、eval_ratio 0.3、canonical seeds、swa 关、crop_mode fixed；`guarded_run_multi` 原子落盘；
resume 键 (task_id, cls_key, iters)。iters=0 基线 5 种子。不跑噪声探针 / δ / 复现校验。执行：5090 pod，`ucr_pod/run.sh` 按 l_enc 分档并发，
`finish.sh` 串行重扫 + 基线 + `--verify`（满格 84 × 20 × 3、零重复键、基线 5 种子齐）。**拉回本地后第一件事 stop pod + confirm-stopped 3/3，然后才分析。**

## 4. φ（14 维，三组）

G1 40 号 12 维（meta 4：log10 T / log10 C / k / log10 N；landmark 2：iters=0 基线均值、band；信号 6）；
G2 趋势强度、季节强度：经典分解（周期 = 周期图峰值、限 [2, T/4]；趋势 = 移动平均；季节 = 相位均值；强度 = max(0, 1 − Var(残差)/Var(成分+残差))，
tsfeatures / TS-ARM 同式，不依赖 statsmodels），逐 (样本, 通道) 后取中位。UCR 在 canonical 训练划分上算；UEA 45 切片 = 40 号 12 维 ⊕ 分解 2 维；
官方 6 靶在官方 TRAIN 上算（base 取 `protocol_alignment` audit_iters0）。

## 5. 模块与臂

f(CLS_binned 116 ⊕ b/290 ⊕ φ 14) → within-(数据集, 预算) z。
- `hgb`（主，`FieldSurrogate("hgb")` 默认超参）；`ridge_int`（CLS 字段 {cross_scale, pool, norm, emb_mask} one-hot × 标准化 φ 显式交互，alpha 3）；
  `twin_knn`（φ 标准化后 k=3 最近语料数据集的 shared 偏好均值；random 用任务盲 HGB）；`hgb_shuffle_0..4`（测试时把靶 φ 换成 derangement 供体数据集的 φ）；
  `pooled`（任务盲：语料 shared 均值 + 任务盲 HGB）。
- 冻结：`module.pkl`（全部 UCR 拟合）。进主管线：`ucr_direct`（分 = f）、`ucr_dev`（hybrid + [f − f̄]）、`ucr_dev_z`（s0z + [f − f̄]）、
  `ucr_direct_shuffle_0..4`、`ucr_plus_source_direct` / `ucr_plus_source_dev_z`（UCR ⊕ 该折 7 个训练 UEA 数据集切片行逐折重拟合）。

## 6. 三级验收（判读带先写死；配对单侧 Wilcoxon α=0.05）

| 级 | 场地 | 主判据 | 门 |
|---|---|---|---|
| **U1** | UCR 留一数据集（84 折 × 3 预算 = 252 实例，候选 = 该数据集 20 条） | regret@3（主）、regret@1、入前 6、配对 acc（δ=0 全对）、ρ | `hgb` regret@3 < `pooled`（p<0.05）∧ < 五个 shuffle 臂的逐实例均值（p<0.05）。不过 → **UCR-U1-DEAD**：预测型 task-aware 在 n=84 判死，不做 U2/U3 |
| **U2** | UEA 7 折（126 实例；40 号 C0 同款：hybrid 臂须与 WP1 逐位） | 入前 6 @290（主）、regret@3（主）、SH6-3-3 @290（描述） | A1：`ucr_direct` 入前 6 @290 ≥ hybrid + 3 ∧ regret@3 < hybrid3（p<0.05）；A2：< s0z（p<0.05）∧ 真增益 ≥ 2 × shuffle 均值增益。A1 ∧ A2 → **UCR-TF-WIN**；否则 **UCR-U1-ONLY** |
| **U3** | 官方口径 6 靶（`mx4_deploy`） | SH6-3-3 冠军 acc；HW 上 GGS / e490b8 名次；HAR 不做 | 描述性 |

覆盖诊断：每个 UEA 靶到最近 UCR 数据集的标准化 φ 距离（孪生距离）；U2 失败时按它区分"无信号"与"靶超出覆盖"。

## 7. 预测（先验写死）

| # | 预测 | 先验 |
|---|---|---|
| P1 | U1 过 | 40% |
| P2 | U1 里 `twin_knn` 也显著优于 pooled | 30% |
| P3 | A1 ∧ A2（UCR-TF-WIN） | 25% |
| P4 | `ucr_plus_source_direct` 的 regret@3 优于 `ucr_direct` | 55% |
| P5 | 官方 HW 上 `ucr_direct` 把 GGS 排进前 6 | 20% |
| P6 | φ 分组置换重要性（训练侧）stl 组 > meta 组 | 40% |
| P7 | 孪生距离最大的两个靶 ⊂ {PhonemeSpectra, HandMovementDirection}（多通道 / 特殊模态） | 50% |

## 8. 成本与执行记录

5090 拟合（`t ≈ 14.7 + 0.0139·T + 5.18e-5·N·T` 秒/轨迹中位，均值 ×1.25）：84 × 20 = 1680 条轨迹。
CROP_LEN=1024：≈ 20.6 GPU-h（档 A 5.1 / B 4.9 / C 10.6），分档并发 6/3/2 → 墙钟 ≈ 8.6 h ≈ $9；
CROP_LEN=512：档 C 成本约减半、并发可到 4 → 墙钟 ≈ 4.4 h ≈ $4–5。本地 4060 pilot（ChlorineConcentration N=1024 T=166）：
单条 50.8 s（无争用）、中位 88 s（与单测同跑）。（跑后修订记录写在下方。）

### 跑后修订记录（如实登记，不改判读线）
- 2026-09-04 pod `7r0xkwmqbeoxjm`（RTX 5090 32 GB，卷 hgj73ti51t EU-RO-1，$0.99/h；deploy 循环误开的第二个 pod `28w37wrhpqmmms` 22 s 内已 stop，confirm 3/3）。
  torch 2.8.0+cu128 sm_120（与 MX-3-PQ 同指纹）；md5 ALL MATCH；pod 重算 tasks.json 与本地 84 任务逐项一致。
- pilot（3 数据集并发，各 20 条）：ECG5000（T 140, N 1024）中位 16.8 s / p90 26.6；Ham（431, 214）23.1 / 49.8；StarLightCurves（512 截, 1024）29.6 / 65.6；
  零 error 零 OOM。实测 = 模型的 0.6–0.85×（模型按 mx3_pq 切片拟合、含 ×1.25 均值修正）→ 全量 ≈ 11 GPU-h。
- 全量分档：TIERS="0:200:7 201:512:4"（峰值模型 7×3.3 ≈ 23 GiB / 4×7.4 ≈ 30 GiB；并发档 --on-oom skip，finish.sh 串行重扫兜底）。
