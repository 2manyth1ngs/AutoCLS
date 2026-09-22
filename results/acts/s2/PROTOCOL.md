# WP17-ACTS S2-CORPUS PROTOCOL：全语料标签（源池 19 = 正向 13 × 8 片 + 反向 4 × 16 片 + 试用 2 × 8 片）

**会话定位**：UCR-ADM（`results/acts/ucradm/REPORT.md`，UCR-ADM-GO+SAME）与两个零 GPU 事后（`headroom_H.md`：扩池不改余量量级；`posthoc_axis.md`：源级偏差集中在一条种子稳定的轴上）之后，用户 2026-09-22 拍板：**正向源 8 片（砍掉 4 个冗余 / 昂贵源：MixedShapesSmallTrain / FiftyWords / Opportunity / EOGHorizontalSignal，UniMiB 留）、反向源 16 片、HMD / SCP2 以 8 片进池试用（不行再剔除）、random-20 保留**。本协议只定义 S2 的**标签语料**（方案 `docs/AUTOCTS_STRICT_REPRO_PLAN.md` §5 的 S2）；t_i / E（S3）与比较器训练（S4）另立协议。执行机器 = 实验室 8k-4090，输出 symlink 到根盘（UCR-ADM 教训）。
产物 `results/acts/s2/`；真源 `configs/acts.yaml profiles.s2`（`extends: s1pp`）；代码 `scripts/83_acts_corpus.py --profile s2`（`--slice-all / --plan-s2 / --phase S / --analyze-s2`）；单测 `tests/test_acts_s2.py`；三件套 `scripts/pa_pod/acts_lab_{bundle,setup,run}_s2.sh`。

## §0 已看过的读数（如实声明）

- S1″ 与 UCR-ADM 的全部标签（4,856 + 2,425 行）及读数；`headroom_H.md`（21 池先验@1 0.031 名义 GO 不稳、前 3 / SH 全 B、池大小曲线平）；`posthoc_axis.md`（PC1 37%、源得分种子一致 0.96；负端 Semg −3.8 / HMD −3.6 / Ethanol −3.4 / SCP2 −2.9 / CinC −1.4 / SCP1 −1.2；轴不能由难度 / chance / 探针读出）；倾斜模拟（正向 16 → 8 = 反向权重 ×2：反向源 regret@1 0.072 → 0.038、正向不伤、前 3 不动）；砍源表（ρ_seed / 池 agree / 最像的源 / 单格耗时：MixedShapes–ShapesAll 0.93、FiftyWords–NonInvasive1 0.92、Opportunity–PAMAP2 0.89 且 ρ_seed 0.75、EOGH 80 s/格）。
- **反向源 = 轴得分 < −1（锁前看过）**：Semg / Ethanol / CinC / SCP1（池内）+ HMD / SCP2（v3′ 曾拒，本轮试用）。数据定的分组，对本协议是既定输入。
- 没看过：任何源第 9–16 行的标签；任何源第 5–8 行里非试点行（a00 / a02 / a04 / a06）的标签；任何新任务的 random-20 标签；HMD / SCP2 在 8 行上的可靠性。
- **修订 1 前看过（2026-09-22，S2 标签未 analyze）**：S1″ 试点 24 任务（PS / Cricket / UWave 各 8 行）random-20 相对 shared-30 的 oracle 扩展（seed 0 同口径）：best_R − best_S 均值 −0.002 / 中位 −0.001，超 2× 种子 sd 的任务 0/24，每任务赢过 shared 最好的 random 条数均值 0.6/20，random 低于 shared 中位的份额 0.46。即试点上 H_rand ≈ H_cat。S2 其余 160 任务的 random-20 未看。

## §1 设计

### 1.1 源池与切片数
| 组 | 源 | 每源任务 | OA 行 |
|---|---|---:|---|
| 正向 13 | Handwriting / UniMiB-SHAR / PhonemeSpectra / UWaveGestureLibrary / Cricket / PTB-XL / SleepEDF / PAMAP2 / NonInvasiveFetalECGThorax1 / ShapesAll / LargeKitchenAppliances / Adiac / FordA | 8 | 前 8 行 a00–a07（三轴各档等频） |
| 反向 4 | SemgHandMovementCh2 / EthanolConcentration / CinCECGTorso / SelfRegulationSCP1 | 16 | 全 16 行 |
| 试用 2 | HandMovementDirection / SelfRegulationSCP2 | 8 | 前 8 行 |
合计 19 源、**184 任务**（104 + 64 + 16）；反向 + 试用任务占 43%。切片器参数逐位 = s1p（n_lo = max(80, 7k)、每类 ≥ 5、cv5）；已有任务（S1″ 104、UCR-ADM 80）的 fold_md5 逐位不变（前缀稳定，单测 + `--check`）。砍掉的 4 源标签保留在 UCR-ADM / S1″ 缓存里，可作靶或消融，不进训练集。

### 1.2 候选与标签（= S1″ (a′)）
- shared-30 = S1″ 同一批 × 种子 {0, 1}；random-20 × 种子 0：S1″ 试点 24 任务（PS / Cricket / UWave 各 8 行）复用 S1″ 目录（已剪，10–20 条，标签复用），其余 160 任务按任务 rng 采（剪枝四条，与 shared 及任务内去重）。
- R′ = 终点 100% 截点 5 折均值原始 acc；探针 iters=0 × 3 只作诊断。
- 配对权重规则（S4 用，本轮只报产量）= S1″ §1.1：gap < 2·sd_gap 的对权重 0、否则 min(gap / 4·sd_gap, 1)、前四分位 ×2——这是让 HMD / SCP2 的噪声对自动归零的机制。
- 缓存种子 = S1″ `runs.jsonl` / `probe0.jsonl` + UCR-ADM 同名文件（同 key 去重），已有格不重跑；新格数与估时由 `--plan-s2` 报（§5 填）。

### 1.3 门（先于数据）
| 门 | 定义 | 不过的动作 |
|---|---|---|
| C0 | key 唯一、shared-30 = S1″、184 任务 fold_md5 跨机器一致、已有任务 md5 与 S1″ / UCR-ADM tasks.json 逐位同、verify 100% | 先修 |
| **G-P（试用源）** | HMD / SCP2 各自 8 行：v3′（Spearman(seed0, seed1) ≥ 0.5 的行 ≥ 5/8）∧ 护栏（h_t ≥ 0.10 的行 ≥ 5/8） | **不过则该源从 S2 源池剔除**（标签保留作靶 / 消融；= 用户"不行再剔除"的预注册版） |
| G-R（反向源） | 4 个反向源各 16 行：v3′ ≥ 10/16 ∧ 护栏 ≥ 10/16 | 不过 → 该源降为 8 片（训练集只用前 8 行） |
| G-A2′ | 全语料（184 任务 × shared-30）真实交互份额（S1″ §1.2 尺度：中位钉底 + 前一半噪声）≥ 0.50 | 不过 → S2-CORPUS-NOISE，S3 不铺 |
读数码：G-A2′ ✓ → **S2-CORPUS-GO**；✗ → **S2-CORPUS-NOISE**；后缀 `+PROBk`（k = 过 G-P 的试用源数）。

### 1.4 只报不设门
每源全行可靠性（ρ_seed 中位 / 最低）与 h_t；轴稳定性（184 任务重算 PC1，源得分与 UCR-ADM 事后 4 行版的 Spearman）；反向源 16 行 vs 前 8 行轴得分一致性；全语料余量 H 三层（§1.5 修订 1：H_cat / H_rand raw + adj / H_search 留 S5；**主指标 H_rand_adj regret@3**，附 H_cat、regret@1、顶部平局诊断、oracle 扩展份额）；random-20 相对 shared 的 z-bad（钉底 z < −1）份额；配对产量（§1.2 规则）中位；failed（探针地板）按 τ；成本（GPU-h、单格中位、墙钟、重启次数、重复行）。

### 1.5 修订 1（2026-09-22，锁于 S2 标签 analyze 之前；用户指出 H 只在目录上读净空）
S1″ / UCR-ADM 的 H 把 oracle 与四个底都限制在 shared-30 里（`_headroom_four` 只收 `arm == shared`），读出的是**目录内**任务感知余量；它对全空间任务感知价值只是下界，不是上界，用它判 MID / B 等于把"目录里没净空"读成"空间里没净空"。shared-30 本身是固定 seed 从剪枝后空间均匀采的（S1′ 20 剪 3 + 13 新），先验只是它们在其他源上的平均秩，所以 H 不是"先验比先验"，而是"目录内任务盲 vs 目录内任务 oracle"；问题只在分母是 30 条且顶部平局。修订：
- **H_cat**：原定义不变（oracle 在 shared-30 里按 seed 0 选、seed 1 评；四个底 LODO 先验 / SH 截点淘汰按 seed 0，seed 1 评）。
- **H_rand**：oracle 扩到 shared-30 ∪ random-20；random 只有 seed 0，报两版：`raw` = max(oracle_cat@seed1, max_j R0_j)；`adj` = max(oracle_cat@seed1, max_j R0_j − κ_n·σ_i)，κ_n = n 个标准正态最大值的期望（n = 该任务有效 random 数；n=20 → 1.87，与 GEO-S 的 1.82σ 同源），σ_i = 该任务 shared 两种子差的 sd / √2。四个底不变（先验只对 shared 排序，pick 仍在 shared 里）。附 oracle 扩展：random 抬高目录 oracle 的任务份额与均值（raw / adj）。
- **H_search**：目标侧搜索真评估 vs 随机搜索（AutoCTS++ 自己的验证方式），S5 另立协议；本协议不算。
- **判读规则**：H_rand 仍是全空间净空的下界（每任务 20 条均匀样本），**只能给 GO，不能给 NO-GO**。H_rand_adj(regret@3，任务加权) ≥ 0.03 → 任务感知门 GO；< 0.03 → 只读作"均匀采样扩不出目录外净空"，落点由 S5 的 H_search 定，不写"空间无净空"。H_cat 只回答目录内选择价值；S1″ / UCR-ADM 的 MID / B 落点按此改读为目录内量（两份报告加注，不重写）。
- 实现：`_headroom_four(M, groups, rand_end0=)` + `expected_max_normal(n)` + `_random_end0_matrix`；`tests/test_acts_s2.py::TestHeadroomRand` 先红。标签、计划、服务器上在跑的格不变。

## §2 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | G-P：HMD 过 | 30% |
| P2 | G-P：SCP2 过 | 30% |
| P3 | G-R：4 反向源全过（16 行 ≥ 10） | 60% |
| P4 | G-A2′ ≥ 0.50（S1″ 试点 0.648） | 75% |
| P5 | 轴稳定：184 任务 PC1 源得分 vs 4 行版 ρ ≥ 0.9 | 65% |
| P6 | 全语料前 3 全量实测 regret@3 ≤ 0.015（21 池 0.013） | 60% |
| P7 | 新格中位耗时 ≤ 60 s（4090 争用；UCR-ADM 52 s，S1″ 多通道更重） | 45% |
| P8 | random-20 的 z-bad 份额 ≥ 0.25（SP1 语料 25%） | 55% |
| P9 | H_rand_adj − H_cat（regret@3，任务加权）< 0.005（试点 24 任务 ≈ 0） | 60% |
| P10 | random-20 原始最好抬高目录 oracle 的任务份额 ≤ 0.5（试点 0.42），扣减后 ≤ 0.2 | 55% |

## §3 执行

1. 本文件 + `tests/test_acts_s2.py`（先红）→ 协议锁 commit。
2. 实现：`acts.yaml profiles.s2`（`sources` 分组 + 每组任务数、random 复用来源、缓存种子来源、G-P / G-R / G-A2′ 参数）；`83`：`source_task_counts`、`cmd_slice_all` 按源取任务数、`merge_seed_caches`、`build_plan_s2 / cmd_plan_s2`（phase S；random 复用 / 新采；已有格不重跑）、`--phase S` 接 run / probe0 / verify / check、`gate_rows / verdict_s2 / cmd_analyze_s2`；单测绿 → 本地 4060 冒烟 → commit → `--slice-all` → `--plan-s2`（报格数 / 估时）→ `--check` → 打包 → **报数请示后**上服务器。
3. 服务器：setup（解包、npz 清单 md5、缓存种子合并去重、输出目录 symlink 到 `/tmp/acts_s2_out`、`--check`、pytest）→ `PHASE=S` 三卡起（能用几张用几张）→ 跑完 probe0 → verify → 拉回 → `--analyze-s2` → REPORT → commit → CLAUDE.md / memory。
4. 每日报进度；共享盘写满由 run 脚本守护 + 根盘输出兜底；`pkill` 只在服务器脚本文件里用。

## §4 铁律对照

1（档位只经 space.yaml；剪枝只在 acts.yaml）；2（iters 规则显式；截点同一轨迹）；3（缓存种子合并、resume、verify；旧格不重跑）；5（先红；§0 声明轴分组与砍源表已看）；6（源只作源；HMD / SCP2 若被 G-P 剔除只作靶；砍掉的 4 源不进训练集）；7；8。

## §5 执行台账（跑后填写）

- 协议锁 ff7bcfa → 实现 119ada9（test_acts_s2 8 例绿、ACTS 五个测试文件共 83 例绿）→ 零 GPU：tasks.json 184 任务（19 源，前缀自检 OK；已有 S1″ 104 / UCR-ADM 80 任务 fold_md5 逐位同）、缓存种子合并 7,256 runs / 324 probe0、plan_S **14,643 格 = shared 11,040 + random 3,603，已有 5,683、新 8,960**；探针 552（已有 324、新 228）；random 目录 reuse 24 / new 160；cost_model 4090 33.8 h（UCR-ADM 实测低估 4.9× → 预期 135–170 GPU-h，三卡共享 ≈ 2–2.5 天墙钟）；本地 --check 184 任务 0 mismatch / plan key 一致；4060 冒烟 8/8 OK；bundle `bundles/s2/`（HEAD 119ada9，9.4 MB：代码 + 输入 + 缓存种子 + 19 源数据 md5 清单）。
- 待：用户看数后拍板上服务器 → setup（数据 md5 核对 / 根盘 symlink / 合并种子 / --check / pytest）→ PHASE=S。
