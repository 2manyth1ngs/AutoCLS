# WP17-ACTS S1′ PROTOCOL：S1 修订版——CV 终点标签 + n_lo = max(80, 7k) + 无偏噪声样本 + 源准入实测 + Cricket 预算消融

**会话定位**：S1 读数 ACTS-S1-NOISE（`results/acts/s1/REPORT.md`）后，用户 2026-09-19 拍板按两轮外部分析修订并重跑（`results/acts/s1/REPORT.md §5`、`diag_step1.md`）。目标不变：在 AutoCTS++ 式三轴 OA 构造下读 G-A（同集任务是否分化、分化是否超过种子噪声）；本轮把测量地板的三个来源（评估集太小、best-of-3 挑噪声、锚坐地板边缘）各修一处，并把"源池能不能进"改成实测。执行机器仍是实验室 8k-4090（`/home/b311/data/wangwenjie/kejunyu/AutoCLS`，口径声明沿用 S1 PROTOCOL §1.10）。
产物 `results/acts/s1p/`；真源 `configs/acts.yaml` 的 `profiles.s1p`（S1 基线值不动，S1 可复现）；代码 `scripts/acts_common.py` + `scripts/83_acts_corpus.py --profile s1p`；单测 `tests/test_acts_s1p.py`；pod 脚本 `scripts/pa_pod/acts_lab_run_s1p.sh`。

## §0 已看过的读数（如实声明）

- S1（1,008 格 holdout best-of-3）：G-A1 ✓ 0.177（Cricket 0.341 / PS 0.228 / HMD −0.038），G-A2 ✗ 0.240（Cricket 0.917 / PS 0.060 / HMD −0.225）；failed 份额 0.54 且与 aug_max 无关，随温度单调（τ 0.01→10：0.46→0.68）；存活区内 25% → 100% 保真度中位 **−0.13**；CV 分数与 holdout 在存活区一致性 0.47；两条锚 48 对里两边地板 22 / 翻转 9 / 两边存活 17，总体 MS_noise 0.708 vs 存活条件 0.266。
- 零 GPU 重算（2026-09-19）：**终点标签**（同一地板）failed 0.69、对半分 Cri 0.55 / HMD −0.06 / PS 0.37、真实份额 Cri 0.79 / HMD 0.54 / PS −0.22、合并 0.31；末两点均值：0.66 / 0.55 / −0.22 / 0.39 / 0.77 / −0.29 / 0.29 / 0.31——逐集数字在 2 锚下不稳，合并份额 0.24 → 0.31。
- **τ=10 全剪不过 G-A0**：官方 29 靶冠军按温度 τ=1 十八 / τ=10 七 / τ=0.01 三 / τ=100 一；"τ ≥ 10 全剪"保留 21/29 = 0.72（丢 ERing / SAD / Libras / SCP1 / RacketSports / PEMS-SF / FaceDetection），"τ ≥ 1 全剪" 3/29。剪枝规则**维持** S1 两条。
- 代码事实：`linear_eval._fit_svm_ts2vec` 在 `n // k < 5 or n < 50` 走硬间隔回退（无网格）；`cl_trainer` 为 AdamW 常数 lr、无 schedule，SWA 自 init 快照逐 iter 更新、同 generator 逐 epoch 重洗 → 3× 轨迹前 1/3 逐位等于 1× 独立跑（GPU 非确定性除外）。
- 发表数字（TS2Vec 论文 acc − 多数类）：HMD 0.338 − 0.405 = −0.07；Ethanol 0.308 − 0.251 = +0.06；SCP2 0.578 − 0.50 = +0.08；SCP1 0.881（WP10）− 0.5 = +0.38。
- **没看过**：任何 CV 终点口径的标签；任何 n_lo = max(80, 7k) 切片的 R′；Ethanol / SCP2 / SCP1 / UWave / HW / UM 在 OA 切片上的任何标签；Cricket 3× 预算下的任何数字。

## §1 设计（相对 S1 的改动，其余同 S1 PROTOCOL §1）

### 1.1 切片
- 三轴 OA(16, 4³)、通道 / 窗长线性四档不变；**样本数轴 n_lo = max(80, 7k)**（`slicing.n_min 80`、`n_lo_per_class 7`），cap 1024，对数四档；分层抽样**每类 ≥ 5**（`n_min_per_class 5`，5 折分层每折训练部分都含全部类）。机制理由：外层 5 折后折内训练集 0.8n，官方网格条件 `0.8n // k ≥ 5 ∧ 0.8n ≥ 50` ⇔ n ≥ 6.25k ∧ n ≥ 62.5，取 7k / 80 留余量，避免探针协议在 n/k 阈值两侧不连续切换（有网格 / 硬间隔）。实测下界：Cricket 84、UWave 80、HMD 80、SCP1/SCP2 80、Ethanol 80、HW 182、UM 119、PS 273，均 ≤ 各源 N。
- 覆盖代价：靶里 n_train ∈ [50, 80) 只有 DuckDuckGeese（C 本已越界）；ERing 30 / BasicMotions 40 的最近子集 log 距仍 < 0.5。

### 1.2 评估 = 分层 5 折 CV、终点标签
- 预训练在**整个切片**上（无监督；评估样本以无标签形式被看过——与部署侧"官方 train 预训练 / test 评估"的差异如实声明，训练配方 batch 8 / iters 规则 / SWA / scaler / 网格一字不改，只改评估聚合）。标准化按通道在整个切片上做。
- 每任务固定 5 折分层划分（seed = `run_seed_for(task_id, "cv_folds")`），每折内在训练折特征上跑官方网格（`fit_svm(grid="ts2vec")`，折内 0.8n ≥ 7k·0.8 ⇒ 永远走网格）、测试折出 acc；截点分 = 5 折 acc 均值。
- **R′ = 终点（100% iters）截点分**，不再 best-of-3（口径：官方协议训固定 iters 后评估终态、无 early stopping；best-of-3 = 在三个噪声里挑最大）。与 AutoCTS++ best-of-5 的偏离如实记录（口径一致优先）。25% / 50% 截点仍评估，只作保真度旁读数。
- 地板探针：同口径 iters = 0 × 3 种子（随机初始化 encoder 编码 + 同一 5 折网格），floor = max(多数类, 探针均值)；`error` / NaN / R′ ≤ floor → R′ = floor、failed（不剔除）。地板定义不改（S1 §3.6-4）。

### 1.3 标签与噪声
- shared-20 在**全部任务**跑 seed 0 与 seed 1，标签 = 两种子终点分的均值后再套地板；random-20 单种子。within-task z 在 40 条上算（shared 为两种子均值、random 单种子，两者噪声不同，如实声明）。
- 噪声估计 = 无偏样本：20 条 shared 全部（不按结果选）。G-A2 用**两种子均值的噪声** MS_noise,mean = mean over (task, shared) 的 (z0 − z1)² / 4；同时报单种子总体噪声 (z0 − z1)² / 2 与"两边存活"条件噪声（只供好半区排序保真度解读）。锚臂取消。

### 1.4 源准入（实测，同一把尺子量全部候选）
- 候选 9 个：现 8 源（HW / UM / PS / UWave / Ethanol / Cricket / HMD / SCP2）+ SelfRegulationSCP1（若入源则从 WP10 靶集移出，靶 29 → 28；只在源池不足 7 时启用）。发表数字（§0）只作先验记录，不作门。
- 准入块：每候选取 OA 前 8 行里 n 档 = 3 的两行（行 1 `(1,2,3)` → `_a01`，行 7 `(3,0,3)` → `_a07`）× shared-20 × seed {0, 1} = 80 格（CV 终点口径），与主批同 key、全部复用。
- **准入规则 ADM**：两任务上"两种子都存活"的 shared 配置数 ≥ 8，且其 Spearman(seed 0, seed 1) 的两任务均值 ≥ 0.30 → 准入；不满足 → 不入 S2 源池、不作试点。源池底线 7：不足时按序启用 SCP1；仍不足 → 记 limitation 不再补。
- 试点集（G-A 门）：PhonemeSpectra、Cricket、UWaveGestureLibrary 三集固定；HMD 若过 ADM 则作第 4 集一并进 G-A（门的"≥ 半数数据集"随集数走）。

### 1.5 Cricket 预算消融
- Cricket 8 任务 × shared-20 × seed 0，一条 3× 轨迹（iters = 3 × 官方规则值，200 → 600 / 600 → 1800），截点分位 (1/12, 1/6, 1/3, 2/3, 1)：前三个截点逐位 = 官方 1× 轨迹的 25 / 50 / 100%（§0 代码事实）。arm `ablation3x`，160 格。
- 读数：存活区内 ρ(1/3 截点, 1 截点) 与 ρ(2/3, 1)（"1× 终点的排序在 3× 下稳不稳"）；failed 份额 1/3 vs 1（地板同一探针）；两者按任务报并取中位。
- **预算决策 BUD（预注册）**：failed 份额 (1× → 3×) 下降 ≥ 0.15 **且** 存活区 ρ(1/3, 1) 中位 < 0.5 → 主批与 S4 部署两侧同改 `ts2vec_official_3x`（须先修订 train.yaml 协议块并重锁；部署侧同时报官方 1× 以保 TS2Vec 对表）；否则维持官方 1×。

### 1.6 不变项
剪枝两条（G-A0 28/29 沿用 S1 precheck）；kernel=0 规范化；OA 设计与 tier 行序；截点分位 25/50/100%（1× 格）；iters 规则；地板定义；`--check` 部署自检（改为核 5 折划分 md5）。

### 1.7 执行顺序与规模
1. 零 GPU：profile / 切片 / CV 执行体 / 计划 / 分析 + 单测 → 本地冒烟（HMD 2 任务 × 3 配置 × 8 iters，CV 执行体 + CV 探针）。
2. **阶段 A**（服务器，2 卡并行）：准入块 9 × 80 = 720 格 + Cricket 消融 160 格（3×）→ 拉回 → ADM / BUD 判定（零 GPU）→ 定源池、定预算 → 若 BUD = 3× 先修订协议再进阶段 B。
3. **阶段 B**：试点集 × 8 任务 × (shared 20 × 2 种子 + random 20) − 阶段 A 已跑格 + CV 探针 3 × 任务数；3 集 ≈ 1,200 格 + 72 探针，4 集 ≈ 1,600 + 96。
4. 拉回 → `--analyze` → REPORT → commit → CLAUDE.md / memory。

## §2 判读（先于数据）

| 门 | 定义 | 不过的含义 |
|---|---|---|
| C0 | 阶段 A / B 计划格 key 唯一、每格截点齐（1× 3 个、3× 5 个）、每格 5 折分齐；探针齐；`--check` 5 折划分 md5 与本地一致 | 先修 |
| G-A0 | 沿用 S1：剪后空间冠军保留 28/29 ≥ 0.90 | 已过 |
| **ADM** | 每候选：两任务两种子都存活的 shared ≥ 8 且 ρ(seed0, seed1) 均值 ≥ 0.30 | 该源不入 S2 / 试点 |
| **BUD** | failed 降 ≥ 0.15 ∧ 存活区 ρ(1/3, 1) 中位 < 0.5 → 3×；否则 1× | — |
| **G-A1** | 试点集对半分 ρ（half 3）均值 ≤ 0.70（shared 两种子均值画像） | 同集任务不分化 |
| **G-A2** | 真实交互份额 = (MS_int − MS_noise,mean) / MS_int：合并 ≥ 0.50 且 ≥ 半数试点集单独 ≥ 0.50 | 分化是噪声 |

**读数码**：G-A1 ✓ ∧ G-A2 ✓ → **ACTS-S1P-GO**（铺 S2，源池 = ADM 通过者）；G-A1 ✗ → **ACTS-S1P-SAME**；G-A1 ✓ ∧ G-A2 ✗ → **ACTS-S1P-NOISE**（两轮修订后仍是噪声：三轴 OA 构造在本口径下的测量地板判定为硬地板，S2 不铺，回到方案 §8 讨论 k 放大或换口径）。
只报不设门：failed 份额（总体 / 按温度）、存活区保真度（25% → 终点）、单种子总体噪声与存活条件噪声、任务相似度（同集 / 异集）、发表数字先验 vs ADM 实测的一致性、成本。

## §3 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | HMD 不过 ADM | 75% |
| P2 | Ethanol 过 ADM | 50% |
| P3 | SCP2 过 ADM | 40% |
| P4 | SCP1 过 ADM | 65% |
| P5 | BUD = 维持 1×（存活区 ρ(1/3, 1) 中位 ≥ 0.5） | 55% |
| P6 | 主批 failed 份额 ≤ 0.45（S1 终点口径 0.69） | 60% |
| P7 | G-A1 ✓ | 70% |
| P8 | G-A2 ✓ | 45% |
| P9 | 存活区 25% → 终点保真度中位 ≥ 0.30（S1 −0.13） | 50% |

## §4 执行

1. 本文件 + `configs/acts.yaml profiles.s1p` + `tests/test_acts_s1p.py`（先红）→ 协议锁 commit。
2. 实现 → 绿 → 本地冒烟 → commit → 导出 6 个新源的 `*_merged.npz`（UniMiB 用现成 candidates npz）→ bundle → 服务器 setup（`--check`）→ 阶段 A → 拉回判 ADM / BUD → commit → 阶段 B → 拉回 → `--analyze` → REPORT。
3. 服务器礼仪同 S1：nice 10、LOKY_MAX_CPU_COUNT=8、每进程一张空闲卡；跑完拉回即删 upload 大包。

## §5 铁律对照

1（档位只经 space.yaml；acts.yaml 只加 profile）；2（iters 规则显式落盘；3× 臂 iters = 3 × 规则值显式写进格；截点同一轨迹）；3（runs.jsonl + probe0.jsonl + resume + verify；阶段 A 格全部复用）；5（先红）；6（SCP1 若入源即从靶集移出；试点集只作源；靶数据只在 sanity 读尺寸）；7；8。

## §6 执行台账（跑后填写）

- 协议锁 009bf86 → 实现 ed477f8（16 单测绿、本地冒烟 12/12 + 2/2）→ 阶段 A 准备 585ed15（72 任务、plan_A 880 格、服务器 `--check` 0 mismatch）。
- 阶段 A（8k-4090 GPU 0 / 1，服务器时钟 2026-09-19 17:17 起）880 格 + 72 探针，15.85 GPU-h，verify OK（error 2 = PS_a07 一条配置两种子 SVM 480 s 超时 → failed）；`--analyze-a` → ADM 准入 HW / UniMiB / UWave + fallback SCP1（源池 4 < 7，limitation）、BUD = 1×；入库 edda5e3；`--plan-b` → plan_B 1,200 格 + 24 探针任务。
- 阶段 B（GPU 0 / 1 / 6 三进程，服务器时钟 2026-09-20 00:13:38 起、03:35:38 退出，墙钟 3.37 h，10.74 GPU-h）1,200 格 + 36 新探针，verify 1200/1200 error 0、probe0 72/72 OK；拉回前缀核对逐位一致；`--analyze` → **ACTS-S1P-SAME**（G-A1 ✗ 0.781、G-A2 ✓ 0.553）；REPORT.md、diag_s1p.md 入库（本 commit）。
- 预测 5/9（P1 ✓ P2 ✗ P3 ✗ P4 ✓ P5 ✓ P6 ✗ P7 ✗ P8 ✓ P9 ✓）。事后发现的协议盲点：G-A1 的对半分量在"强共同序 + 真实交互并存"时必 > 0.7，测不出分化（REPORT §5.1）。
