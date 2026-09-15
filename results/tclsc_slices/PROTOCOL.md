# WP16 SX 检验 PROTOCOL：每集加切片——比较器切片轴 S ∈ {6, 9, 12} 嵌套读数（源侧新 R'，pod 5090）

**会话定位**：LC 检验（`results/tclsc_lc/REPORT.md`，LC-SLICE）读出三条标签轴里只有"每集切片数"在满量仍涨（E_T regret@3 S4 0.0355 → S6 0.0207，
换种子 0.0354 → 0.0277；6 种子均值 0.0354 → 0.0242 vs 任务盲 0.0323 → 0.0311），用户 2026-09-15 拍板"每集加切片立项"，并要求先核两件事：
**数据集能不能这么切、我们的切法对不对（参照 AutoCTS++）**。本协议把每个折数据集的切片从 6 加到 12，读 S ∈ {6, 9, 12} 的嵌套曲线。
产物 `results/tclsc_slices/`；脚本 `scripts/74_tclsc_slices.py`；pod 三件套 `scripts/pa_pod/sx_{bundle,setup,run}.sh`。
用户授权（memory `wp16-slice-expansion-pod-authorization`）：协议锁 + 单测 + 冒烟后直接开 pod，跑完 stop + confirm 3/3。

## §0 已看过的读数

- LC 检验切片轴（S ∈ {2, 4, 6}，UEA 7 折）：seeds 0/1/2 E_T 0.0330 / 0.0355 / 0.0207、任务盲 0.0351 / 0.0306 / 0.0294；seeds 3/4/5 E_T 0.0407 / 0.0354 / 0.0277、任务盲 0.0362 / 0.0341 / 0.0328；
  6 种子均值 S6：E_T **0.0242**、任务盲 **0.0311**（本协议 C0 参照）。S6 satu M1 0.562（s012）/ 0.507（s345）。
- 满量点种子方差 0.007（E_T 0.0207 vs 0.0277）→ 本协议每点 **6 种子**。
- **没看过**：任何 S > 6 的读数；新切片的 R' / E_T 一条都没有。

## §1 切法核对（用户要求；先于数据）

### 1.1 我们的切片是什么

一个切片 = (随机类别子集, 随机通道子集, 随机样本子集 ≤ 1024)，`autocls/data/task_slicer.py::slice_pool_tasks` + `slice_strategies.RandomClassSubsetSlicer`，
参数全取注册表 `configs/source_pool.yaml`（类别档 `n_classes_range`、通道档 (⌈C/2⌉, C)、`n_samples 1024` / `min_samples 32`、`seed 0`）；同一数据集内**类别子集互不重复**（多样性主力）。
现有 6 切片 = 注册表 `planned_tasks` 流的前 6 个（`pq.load_pq_datasets`），**RNG 前缀不能动**（改 n_tasks 会改前 6 个切片的通道 / 样本抽样，既有 R' 缓存全失效，铁律 3）。

### 1.2 AutoCTS++ 怎么切（`exps/dataset_slice.py`，已读源码）

每个源数据集出 `sample_num`（默认 20）个子集：**变量（传感器）子集 × 连续时间窗**——变量数在 [3, min(300, N)]、窗长在 [7000, min(25000, T)]，
两维各按 25 / 50 / 75 / 100% 四档分层（16 格，每格 ⌈20/16⌉ = 2 个），变量 `random.sample` 不放回、窗起点随机；**子集之间不去重、允许重叠**（同一数据集的 100% 档窗几乎就是整条序列）。
论文口径：11 个源数据集 → 100 个子集 × 2 个预测设置 = 200 个任务，每任务 20 条 shared + 20 条 random 架构-超参对。
对应关系：他们的"变量子集"↔ 我们的通道子集（分类任务再加类别子集 = 标签总体的子集）；"时间窗"↔ 我们的样本子集（分类实例无时序，取无放回子样本）；
"尺寸分层"↔ 我们的类别档 [lo, hi] + 通道档；"允许重叠"↔ 我们的切片同样重叠（样本 / 类别 / 通道均可交叠，`slicing-leakage-real-but-t2t3-immune`）。
**差异**：① 我们强制类别子集不重复（他们不去重）——更保守；② 我们每集 6 个、他们每集 ≈ 9（100 / 11）到 20 个；③ 他们同一子集再乘 2 个预测设置（P/Q），我们同一切片乘 3 个预算截点（60/120/290）。
结论：切法同义、我们只是每集数量少一半——加到 12 正好对齐他们的每集量级。

### 1.3 数据集能不能切到 12（注册表 + `results/mx3_pq/tasks.json`，零数据加载）

| 数据集 | 类数 | 类别档 | 可用类别组合 | 现有 6 切片 | 12 切片可行性 |
|---|---:|---|---:|---|---|
| Handwriting | 26 | 8–14 | 4.7e7 | k 9–13、351–515 样本、通道 2–3 | ✓ 组合充裕；样本 = 所选类全部（无样本子集多样性，靠类别 / 通道） |
| UniMiB-SHAR | 17 | 7–14 | 1.1e5 | k 9–14、1024 样本（封顶） | ✓ 组合充裕 + 样本子集真实（≥ 1024 可选） |
| PhonemeSpectra | 39 | 4–8 | 8.1e7 | k 6–8、1024（封顶）、通道 6–10 | ✓ 同上 |
| UWaveGestureLibrary | 8 | 4–8 | 163 | k 4–8、220–440 | ✓ 163 组合 ≫ 12 |
| Cricket | 12 | 6–10 | 2497 | k 6–10、90–150 样本 | ✓ 组合充裕；样本少（n_train 66–110）与现有切片同量级 |
| EthanolConcentration | 4 | 2–4 | 11 | k 2–4、261–524 | △ 11 组合 − 已用 6 = 5 个不重复 + **1 个重复类别子集**（只靠通道 / 样本子集区分；AutoCTS++ 本来就允许） |
| HandMovementDirection | 4 | 2–4 | 11 | k 2–4、109–234 | △ 同上（5 + 1 重复） |
| SelfRegulationSCP2 | 2 | 2–2 | 1 | 3 切片、380 样本 | ✗ **不扩**：唯一类别组合、只有通道子集可变；本来就是训练用低权、不作折；保持 3 |

### 1.4 扩展切片的构造（新代码，单测钉死）

- `slice_strategies.ExcludingClassSubsetSlicer(exclude=现有 6 切片的类别子集)`：同一 RNG 流内先出未用过的类别组合，组合耗尽后才允许重复（Ethanol / HMD 的第 6 个）。
- `source_pool.slice_extension_tasks(entry, registry, y, n_new=6, seed_offset=1000)`：与注册表**同参**（类别档 / 通道档 / n_samples / min_samples），独立 RNG 流 `slice_seed + 1000`，
  id = `{name}_x00 … _x05`（`pq.dataset_of` 仍剥到数据集名；与注册表流的 `_06…` 不撞 id，不碰 canonical 缓存），tag = `ext`。**注册表 planned_tasks 不改**，原 6 切片逐字段不变（单测断言）。
- 嵌套：S6 = 原 6；S9 = 原 6 + x00–x02；S12 = 原 6 + x00–x05。**考侧永远只用原 6 切片**（126 实例，与 T / LC 逐位可比）；新切片只进训练侧。
- 跑完 `--slice` 后把实测重叠表（类别子集 Jaccard、样本 Jaccard、通道 Jaccard，新 vs 原、新 vs 新）填入 §1.5（协议修订，先于任何 R'）。

### 1.5 实测重叠（`--slice` 后填写）

（待填）

## §2 标签与特征（源侧新 R'，铁律 3 缓存 + --resume）

- 每个新切片 20 条 CLS = shared-10（探针目录前 10，与 mx3_pq 同）+ random-10（按 task_id 派生，seed 42）+ 1 条 GGS 锚（与 mx3_pq 网格同构，比较器训练不吃 GGS）；
  每条一条轨迹 290 iters、@60/@120/@290 三截点各评一次 R'（`evaluate_rprime_trajectory`，canonical 种子 / eval_ratio 与 32 号同），缓存 `rprime_cache.jsonl`。
- δ：沿用 `results/mx3_pq/delta_calibration.json` 的逐 (数据集, 预算) 值（δ 是数据集级标定，新切片同数据集）。
- 基线探针（φ 的 base_mean / base_band）：iters=0 随机初始化 R' 5 种子（`baseline_probe` 同式），缓存 `baseline_cache.jsonl`。
- 特征表 `ctx_ext.jsonl`：φ12（40 号公式）⊕ E_A 25（`response_profile`）⊕ E_T（GGS 探针 290 iters、@60/@290 各 22 维，69 号 `probe_profiles`）——列集与 `results/tclsc_s1et/uea/ctx_uea.jsonl` 相同；ft60_hard 只用 rep@60 ⊕ φ12。
- 42 切片 × 21 = 882 条轨迹，按 mx3_pq 缓存 `trajectory_seconds`（5090）估 ≈ 7.3 h；基线 42 × 5 条 ≈ 5 min；E_T 探针 42 × ≈ 1 min；比较器 S9/S12 训练 168 跑 ≈ 20 min。合计 ≈ 8.5–9 h 5090 ≈ $9。

## §3 比较器

臂 `ft60_hard`（E_T@60 ⊕ φ12 ⊕ b，field 头）vs `blind_diff_hard`；7 折 LODO；**6 种子 0–5**；S ∈ {6, 9, 12}。
S6 复用：seeds 0–2 = T 检验 / s1phi 模型，seeds 3–5 = `results/tclsc_lc/rep_s345/models/S6`（LC §2 已证与 33 号训练路径逐位同一）。
S9 / S12：73 号 `train_fold_uea`（rows = mx3_pq 行 ∪ 扩展行，ctx 表 = ctx_uea ∪ ctx_ext，标准化只用当前训练切片，铁律 6）。
考侧 = 目标数据集原 6 切片 × 3 预算 = 126 实例；指标 regret@3（主）、regret@1、M1、M0、E 增益 = 任务盲 − E_T；逐折。

## §4 判读（先于数据）

- **C0**：S6 六种子 pooled 与 §0 逐位（E_T 0.0242 / 任务盲 0.0311，容差 0.0005）。
- **G-S1**（主）：E_T regret@3 S12 vs S6 配对 Wilcoxon（126 实例 × 6 种子 pooled，H1 S12 更低）p < 0.05 **且** 均值差 ≤ −0.002。
- **G-S2**：E 增益(S12) > E 增益(S6)（涨的是 E 的部分，不是任务盲部分）。
- **曲线**：73 号 `axis_verdict` 规则在 (6, 9, 12) 上（满量 12 比 9 低 ≥ 0.001 且增益仍涨 → RISING）。

**读数码**：G-S1 ✓ ∧ G-S2 ✓ → **SLICE-WIN**（加切片是对的杠杆，继续到 S=18/24 可立项）；G-S1 ✓ ∧ G-S2 ✗ → **SLICE-BLIND**（切片抬的是任务盲部分）；
G-S1 ✗ ∧ 曲线 RISING → **SLICE-RISING**（量级不足 6 种子分辨、继续加才知道）；其余 → **SLICE-FLAT**（切片轴在 6 → 12 已饱和，标签轴全部关闭）。
**副读数（零 GPU，不进读数码）**：S12 折模型（seeds 0–2）代入 71 号 D′ 口径（官方 6 靶信任域 SH6-3-3 seed-0 缓存模拟），六靶均值 / 追平数 / 冠军是否变，对照 D′ S6（0.564 seed 0）。冠军变了的靶列 picks（是否补 finals 另议）。

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 过 | 90% |
| P2 | S12 E_T regret@3 比 S6 低 ≥ 0.002 | 55% |
| P3 | G-S1 显著（p < 0.05） | 45% |
| P4 | E 增益(S12) > +0.0069（S6 六种子值） | 50% |
| P5 | 任务盲 regret@3 S12 < S6（任务盲部分也涨） | 55% |
| P6 | 副读数：S12 官方六靶 seed-0 均值 ≥ D′ S6 0.564 | 50% |
| P7 | S9 落在 S6 与 S12 之间（单调） | 60% |

## §6 铁律对照

铁律 3（缓存 + --resume + --verify 行数 / 重复键）；铁律 5（本文件 + `tests/test_tclsc_slices.py` 先红）；铁律 6（新切片只做训练侧；ctx 标准化只用训练切片；目标折的新切片**不进**该折训练——LODO 按数据集剔除，新切片同数据集自然剔除）；
铁律 7（比较器只见浮点向量）；铁律 8（`is_better`）。pod 规矩：`--device cuda` + CUDA 闸、nohup + --resume、跑完 stop + confirm 3/3。

## §7 执行台账（跑后填写）

（空）
