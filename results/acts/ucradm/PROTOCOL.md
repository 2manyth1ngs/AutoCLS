# WP17-ACTS UCR-ADM PROTOCOL：UCR 单变量源准入实测（v3′ 可靠性门 + chance 护栏 + 池级多样性读数）

**会话定位**：S1″ 读数 ACTS-S1PP-GO + HEADROOM-NA（`results/acts/s1pp/REPORT.md`）后，用户 2026-09-21 拍板"在 UCR 里找源进池、准入实测在实验室服务器跑、准备好请示"；范围拍板 = 零 GPU 筛表的 **A 档 10 源**（同源族只带一个代表）。本轮只做准入（每源 4 任务 × shared-30 × 2 种子），不采 random、不铺 S2；准入尺子换成 S1″ REPORT §5 建议的 **v3′ 可靠性门 + acc 单位 chance 护栏**，多样性只作读数。执行机器 = 实验室 8k-4090（部署路径固定 `/home/b311/data/wangwenjie/kejunyu/AutoCLS`，py3.8 无 aeon → 全部走 npz）。
产物 `results/acts/ucradm/`；真源 `configs/acts.yaml profiles.ucradm`（`extends: s1pp`，基线 / s1p / s1pp 不动）+ 源注册表 `configs/acts_sources.yaml`；代码 `scripts/acts_common.py` + `scripts/83_acts_corpus.py --profile ucradm`（`--plan-adm / --phase M / --analyze-adm`）；单测 `tests/test_acts_ucradm.py`；服务器三件套 `scripts/pa_pod/acts_lab_{bundle,setup,run}_ucradm.sh`。

## §0 已看过的读数（如实声明）

- S1″ 全部（`results/acts/s1pp/{REPORT,posthoc_s1pp}.md`）：三门过、准入 v2 0/13、v3′（任务级全 30 条 Spearman ≥ 0.5 过半）11/13（HMD / SCP2 1/4 拒；Ethanol / SCP1 / Opportunity 3/4 进）、H 对先验@1 0.022、源间偏好差异占秩方差 0.48、每源与其余源平均序的 Spearman（Ethanol −0.07 / SCP2 −0.12 / HMD −0.22，其余 0.35–0.90）。
- **本会话零 GPU 筛表**（`scratchpad/ucr_source_scan.md`；数据 = WP7 84 集 batch128@290 k=1 缓存 + WP15 官方口径 52 集 × 34 格单种子 + mx3_pq shared-10 画像）：A 档 10 源的官方整集净空（max − 多数类）0.43–0.93、好半区 IQR/σ_ref(0.0073) 2.3–18.5、@25%→@100% Spearman 0.54–0.90、shared-10 画像 vs UEA 先验 ρ：Adiac 0.01 / LargeKitchen 0.20 / Semg 0.32 / EOGH 0.33 / MixedShapesSmall 0.45 / NonInvasive1 0.60 / ShapesAll 0.60 / FordA 0.63 / CinC 0.83 / FiftyWords 0.88；WP7 batch128 口径下相对未训地板的 failed：EOGH 1.0 / MixedShapesSmall 0.95 / LargeKitchen 0.75 / Adiac 0.70 / FordA 0.70 / ShapesAll 0.65 / CinC 0.60 / FiftyWords 0.45 / NonInvasive1 0.30 / Semg 0.30。这些都是**整集**、**另一口径或单种子**的数，只用于挑候选，不进任何门。
- 8 月换源池调研（`results/candidate_survey/REPORT_STEP3.md`）：ElectricDevices / InsectWingbeatSound 等 UCR 集在旧口径（GGS 相对未训基线增益）下判 inert；NonInvasive1 @crop256 基线 0.776；FordA / NonInvasive1 的 npz 由该调研生成（复用）。
- **chance 护栏的标定（锁前在 S1″ 数据上算过）**：h_t = median(前一半 shared 两种子均值) − 切片多数类占比，门 0.10、≥ 3/4 任务：HW / UniMiB / UWave / Cricket / SCP1 / PTB-XL / SleepEDF / PAMAP2 / Opportunity 4/4 过；PhonemeSpectra 1/4、EthanolConcentration 1/4、HMD 0/4、SCP2 0/4 不过。即护栏在 S1″ 池上把 v3′ 的 11 减到 9（去 PS / Ethanol）——对本轮 10 个 UCR 源是预注册，对 S1″ 13 源是标定集，如实记。
- **没看过**：任何 UCR 集在 `ts2vec_official` 切片口径（cv5）下的标签；shared-30 任何一条在任何 UCR 集上的标签；任何 UCR 集的种子 1 行；任何 UCR 切片的探针地板。

## §1 设计

### 1.1 源与数据
| 源 | 域 | N | L | k | min_class | n_lo = max(80, 7k) | n_hi | 同源族（只带本代表；兄弟不进池） |
|---|---|---:|---:|---:|---:|---:|---:|---|
| NonInvasiveFetalECGThorax1 | ECG | 3765 | 750 | 42 | 75 | 294 | 1024 | Thorax2 |
| CinCECGTorso | ECG | 1420 | 1639 | 4 | 355 | 80 | 1024 | — |
| SemgHandMovementCh2 | EMG | 900 | 1500 | 6 | 150 | 80 | 900 | SemgHandGender / Subject（ucr_pool 已三选一） |
| EOGHorizontalSignal | EOG | 724 | 1250 | 12 | 59 | 84 | 724 | EOGVerticalSignal |
| ShapesAll | 轮廓 | 1200 | 512 | 60 | 20 | 420 | 1024 | — |
| MixedShapesSmallTrain | 轮廓 | 2525 | 1024 | 5 | 334 | 80 | 1024 | MixedShapesRegularTrain |
| LargeKitchenAppliances | 电器 | 750 | 720 | 3 | 250 | 80 | 750 | Computers / ScreenType / RefrigerationDevices / SmallKitchenAppliances / ElectricDevices |
| Adiac | 硅藻轮廓 | 781 | 176 | 37 | 20 | 259 | 781 | — |
| FiftyWords | 手写词 | 905 | 270 | 50 | 6 | 350 | 905 | WordSynonyms |
| FordA | 发动机噪声 | 4921 | 500 | 2 | 2394 | 80 | 1024 | FordB |

- npz 口径（与 S1″ 外部源 PTB-XL 等同一条管线 `scripts/prepare_candidates.py`）：TRAIN+TEST 合并、原生长度、整集按通道 z-score、float32。**`min_class_count` 改为 SPEC 可覆盖，UCR 十源取 5**（= ACTS 切片器每类 ≥ 5 的对应物）；调研期默认 64 会把 Adiac / ShapesAll / FiftyWords 整个剔空，如实记；NonInvasive1 / FordA 复用调研期 npz（其 min_class 75 / 2394 ≥ 64，与取 5 逐位同结果，md5 不变）。
- 注册表 `configs/acts_sources.yaml`（schema 同 `source_pool.yaml`，`source: npz`，只列本轮 10 源）；`ac.load_dataset_full` 在 source_pool → candidate_pool 之后第三级回退到它。不改 `candidate_pool.yaml`（其 `--emit-registry` 的 N ≥ 1000 初筛会拒掉 Adiac / FiftyWords / Semg / EOGH / LargeKitchen，那是调研期的尺子）。
- 铁律 6：十源与 UEA 30 靶 / 8 源无同名（单测锁定）；同源排除：Worms / WormsTwoClass（← EigenWorms 靶）、CricketX/Y/Z / UWave×4 / EthanolLevel / Phoneme（ucr_pool 硬排除）、InsectWingbeatSound（← UEA InsectWingbeat，未跑靶，标注）。

### 1.2 切片（profile `ucradm` extends `s1pp`；切片参数逐位 = s1p）
- n_lo = max(80, 7k)、每类 ≥ 5、window_min 50、n_cap 1024、cv5（`fold_seed_key cv_folds`）、iters 规则 `ts2vec_size`、截点 25 / 50 / 100%。
- **C = 1 的退化**：通道档恒为 [1]，OA(16, 4³) 的行 (a, b, a⊕b) 投影到 (窗档 b, 样本档 a⊕b) 是 4 × 4 全因子（强度 2 正交表的性质；单测锁定 16 行投影两两不同）——每源仍 16 个不同 (窗, n) 任务。
- 每源切 8 行（`pilot.n_tasks` 8，与 S1″ 同，tasks.json 80 任务）；准入用 OA 行 1 / 7 / 3 / 5（`_a01 _a07 _a03 _a05`）→ (窗档, n 档) = (2,3) (0,3) (1,2) (3,2)：n 高两档、四个窗档各一次。
- `--check`：服务器本机 npz 重建 80 任务的 n_train / T / C / majority / fold_md5 与 tasks.json 逐任务一致。

### 1.3 候选与标签
- shared-30 = S1″ 同一批（同 rng 流、同剪枝四条；**C0：catalog shared 30 个 cls_key 逐位 = `results/acts/s1pp/catalog.json`**），两种子 {0, 1}；无 random、无锚。
- R′ = 终点 100% 截点的 5 折均值原始 acc（S1″ (a′)：不套地板）；探针 iters = 0 × 3 种子只作诊断。
- 格 = 10 源 × 4 任务 × 30 × 2 = 2,400 + 探针 40 × 3 = 120。

### 1.4 准入 v3′（主门，同 S1″ posthoc §1 的尺子）
- 每任务 ρ_t = Spearman(seed 0 全 30 条, seed 1 全 30 条)；pass = #{t : ρ_t ≥ 0.5}；**准入 ⇔ pass ≥ 3**（4 任务的严格过半，= S1″ "任务过半"在 n = 4 时的取整；Ethanol / SCP1 / Opportunity 3/4 进、HMD / SCP2 1/4 拒为标定锚）。
- 缺格：某配置单种子缺 → 该配置从该任务的 ρ_t 剔除；两种子都缺 → 同上；某任务 shared 两种子齐的配置 < 20 条 → 该任务不计，过半按剩余任务数算（n < 3 时该源 NA，不准入）。

### 1.5 chance 护栏（acc 单位；S1″ REPORT §5 "PS 型近 chance 源由 acc 单位余量定"的落地）
- 每任务 h_t = median(按两种子均值排前一半的 shared 的两种子均值) − 切片内多数类占比（tasks.json `majority`）；pass_h = #{t : h_t ≥ 0.10}；**护栏通过 ⇔ pass_h ≥ 3**。
- 准入 = v3′ ∧ 护栏。

### 1.6 诊断列（只报，不设门）
v2 三条（前一半 seed ρ / 净占比 / IQR/σ_mean，S1″ §1.2 尺度）、真实份额、探针地板 failed 份额（总 / 按 τ）、两种子都超地板的中位、每任务 σ_task、单格耗时（预训练 / 评估）、error 行。

### 1.7 多样性读数（只报）
- 每源画像 = 4 任务钉底 z（S1″ §1.2：v_c = max(v, median) 再 z 化，v = shared-30 两种子均值）的均值向量（30 维）。
- 参照 = S1″ v3′ 11 源在其准入 4 行上的同口径画像的池均值（从 `results/acts/s1pp/labels.jsonl` 重算；PS / UWave / Cricket 只取 4 准入行）。
- 报：ρ(源画像, 池均值画像)；UCR 十源两两 ρ 矩阵；合池 = S1″ 11 源 + 本轮准入 UCR 源，秩方差分解 SS_between / SS_within（配置的源级平均秩，配置主效应先去掉；S1″ 13 源 0.48 / 0.52 参照）与每源 agree（与其余源平均序的 Spearman）。

### 1.8 读数码
- n_adm = 准入源数：**≥ 7 → UCR-ADM-GO**（S2 合池候选）；4–6 → UCR-ADM-PARTIAL；≤ 3 → UCR-ADM-DEAD。
- 后缀：准入源中 ρ(源, 池均值) < 0.3 的 ≥ 2 → `+DIVERSE`，否则 `+SAME`。
- 读数码只描述准入层；S2 铺不铺、靶集怎么分（XFER-DEAD 的约束：UCR 进源池须同时有 UCR 靶或"UEA 靶不伤"门）另议。

### 1.9 不变项
OA 设计与行序、s1p 切片参数、s1pp 剪枝四条与 shared-30、cv5、iters 规则、截点、探针照跑、`--check` 跨机器自检、铁律 6 名单。

## §2 判读（先于数据）

| 门 / 读数 | 定义 | 不过的含义 / 动作 |
|---|---|---|
| C0 | 计划格 key 唯一、截点齐、80 任务 fold_md5 跨机器一致、探针齐、shared-30 逐位 = S1″ | 先修 |
| v3′ | 4 任务中 ρ_t ≥ 0.5 的 ≥ 3 | 该源不入池 |
| 护栏 | 4 任务中 h_t ≥ 0.10 的 ≥ 3 | 该源不入池（chance 型） |
| 读数码 | n_adm ≥ 7 GO / 4–6 PARTIAL / ≤ 3 DEAD；+DIVERSE / +SAME | — |

只报不设门：§1.6 诊断列、§1.7 多样性、成本。

## §3 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | 准入 ≥ 7/10（UCR-ADM-GO） | 55% |
| P2 | NonInvasive1 / CinCECGTorso / ShapesAll 三源各 4/4 过 v3′ | 65% |
| P3 | EOGHorizontalSignal 被拒（v3′ 或护栏） | 40% |
| P4 | FiftyWords 被拒（稀类 6 + 18× 不平衡 → cv5 折内噪声） | 45% |
| P5 | 后缀 +DIVERSE（准入源中 ρ(源, 池均值) < 0.3 的 ≥ 2） | 50% |
| P6 | 合池 SS_between ≥ 0.48（S1″ 13 源） | 50% |
| P7 | 单格中位耗时 ≤ 45 s（4090，预训练 + 3 截点 cv5） | 55% |
| P8 | 探针地板 failed（shared 均值）≤ 0.5 于 ≥ 7 源 | 55% |
| P9 | 护栏拒掉的源 = 0（十源官方整集净空均 ≥ 0.43） | 75% |

## §4 执行

1. 本文件 + `tests/test_acts_ucradm.py`（先红）→ 协议锁 commit。
2. 实现：`prepare_candidates.py` SPEC `min_class_count` 覆盖 + 8 个 `scripts/candidates/ucr_*.py`；`configs/acts_sources.yaml` + `load_dataset_full` 三级回退；`acts.yaml profiles.ucradm`；`acts_common.admission_v3_metrics / admission_v3_decision / chance_guard / profile_diversity / between_source_decomp`；`83`：`--plan-adm`、`--phase M`（run / probe0 / verify / check 读 `plan_M.json`）、`--analyze-adm`、smoke 走 ucradm；单测绿 → 本地 4060 冒烟（iters 8）→ commit。
3. 零 GPU：`prepare_candidates.py --datasets ucr_*`（8 个 npz）→ `--slice-all`（80 任务，前缀自检）→ `--plan-adm`（2,400 格 + 120 探针，估时按 cost_model 报）→ `--check`。
4. 打包 `acts_lab_bundle_ucradm.sh`（代码 HEAD + 协议 / tasks / plan_M / catalog + 10 个 npz 与 md5）→ **请示用户** → 服务器 `acts_lab_setup_ucradm.sh`（解包 → md5 → 磁盘闸 ≥ 2 GB → CUDA 闸 → `--check` → pytest）→ `PHASE=M GPUS="…"` `acts_lab_run_ucradm.sh`（分片 → probe0 → verify → 打包）+ `/tmp/acts_watchdog` 守护（共享盘写满自愈）。
5. 拉回 → `--verify --phase M` → `--analyze-adm`（v3′ / 护栏 / 诊断 / 多样性 / 读数码）→ REPORT → commit → CLAUDE.md / memory；upload 大包用后即删。
6. 服务器礼仪同 S1″：nice 10、LOKY_MAX_CPU_COUNT=8、每进程一张空闲卡、开跑 3 分钟内确认首格、10 分钟无进度必查。

## §5 铁律对照

1（档位只经 space.yaml；剪枝只在 acts.yaml；新注册表只加不改旧路径）；2（iters 规则显式；截点同一轨迹）；3（runs.jsonl + probe0.jsonl + resume + verify；npz 缓存）；5（先红；§0 标定如实声明）；6（十源只作源；与靶无同名 / 同源；靶数据不读）；7（task_type 显式）；8（不涉及）。

## §6 执行台账（跑后填写）

- 协议锁 05db9a4 → 实现 d03ada4（18 单测绿、相关 101 例绿）→ 零 GPU：8 个新 npz（K 全保留、min_class_count 5）、`--slice-all` 80 任务前缀自检 OK、`--plan-adm` 2,400 格 + 120 探针（C0 shared-30 = S1″ OK；cost_model 估 4090 9.6 h，S1″ 经验该模型对多通道低估 3.7×，单变量预期 1–3×）、本地 `--check` 80 任务 0 mismatch / plan key 一致 → 本地 4060 冒烟（Adiac 行 0/1 × shared 2 × 2 种子，iters 8）8/8 格 + 2 探针 OK → 打包 `bundles/ucradm/`（code_head.tar + acts_inputs + acts_npz 47.6 MB + md5 20 项，HEAD d03ada4）。
- 待：用户请示 → 服务器 setup / run（PHASE=M）→ 拉回 → `--verify --phase M` → `--analyze-adm` → REPORT。
