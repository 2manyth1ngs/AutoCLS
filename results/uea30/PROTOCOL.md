# WP10-UEA30 协议：UEA 30 靶外 23 靶的目标侧全量部署（可评性分类的预注册检验）

日期：2026-09-06 ｜ 分支：`wp10-uea30`（自 `mx4` 63694c0 开，含 WP6 掩码字段 `input_mask`）｜ 脚本：`scripts/45_uea30_deploy.py`
（薄驱动，importlib 复用 35 / 36 / 39 / 40）｜ 配置真源：`configs/uea30.yaml` ｜ 单测：`tests/test_uea30_deploy.py`（21 例）+
`tests/test_sh_deploy.py` 增补 3 例 ｜ 产物：`results/uea30/{PROTOCOL.md, candidates.json, runs.jsonl, oom_failures.jsonl,
analysis.json, picks.json, tables.md, table30.md, table30.json, REPORT.md}` ｜ 协议锁 commit 见 §7。

**主张**：零样本先验名单 + 目标侧逐次减半（SH）在 UEA 30 里"靶外"的 23 个数据集上按官方口径部署，**作为可评性分类的
预注册检验**——策略选择只在数据集能分辨策略的地方（E 类）移动精度；在贴顶 / 短序列 / 小测试集 / chance 型数据集（A / B /
C / D 类，19/24）上落在预测带内、与 TS2Vec 同带。**不做**：排序器重训、开放空间提名、task-aware 任何形式（CLAUDE.md 两条判死）。

## §0 偷看声明（先于判据；全部已揭晓的读数）

1. **6 个 LODO 靶的掩码结果**（`results/mx4_mask/paper_table.md`，2026-09-06）：SH10-4-3_prior 掩码关 3 种子六靶均值 0.574，
   追平 TS2Vec 6/6、超论文 5/6；SH20-3 两版都量 0.577，超论文 6/6（首个六靶全超的臂），冠军增量对掩码关臂 ≤ 0.011（噪声内）。
   HAR 外部锚 HAR-WIN（`results/mx4_deploy_har`，冠军 0.935 / TS2Vec 0.930）。
2. **上一会话的 A–E 分类与逐靶预测**（计划文件 `encoder-encoder-wp3-4-6-ts2vec-6-6-1-gg-velvet-orbit.md` §5.1，零 GPU 推断）：
   A 贴顶 7 / B 短序列 L<50 4 / C n_test≤50 3 / D chance-EEG 5 / E 有净空可评 5；"19/24 无活动空间"。该表把 SpokenArabicDigits 的
   TS2Vec 论文数误记为 0.932，**论文 Table 10 为 0.988**（本协议按论文）。
3. **成本锚**（RTX 4090，`results/mx4_deploy` + `results/mx4_mask` + `results/protocol_alignment` 的 `checkpoint_results` 段间差标定）：
   纯预训练 s/1k iters ≈ 21.5 + 0.023·L；带三截点实况 ≈ 21 + 0.039·L；单次评估 ≈ 1.3e-6·n_eff² + 4e-7·n_test·n_eff +
   (n_tr+n_te)·L·C / 2e7 + 0.1（n_eff = min(n_train, 10000)）。C ≤ 11 范围内通道数无可测效应；PhonemeSpectra 的预训练被自身
   并发 SVM 拖慢 3×。逐靶锚点写在 yaml `anchor_4090`。
4. **ragged / 类集实测**（`load_uea_pair(scaler="train")`，2026-09-06）：23 靶 TEST 类集全部 ⊆ TRAIN；只有 CharacterTrajectories
   （TRAIN ≤180 < TEST ≤182）与 JapaneseVowels（26 < 29）会触发 35 号 `load_arrays` 的双 scaler 断言——45 号自带加载器只走
   scaler=train 一路绕开。PenDigits N·L·C = 119,904 > 1e5 → 600 iters（临界，`--check` 已断言）。
5. **旧口径下触碰过的靶**：Epilepsy / FaceDetection 在 `results/target_preflight`（切片口径，判不可评，关键决策 5）；
   NATOPS / ERing / SCP1 / FingerMovements / MotorImagery / Heartbeat / AWR / CharacterTrajectories / SpokenArabicDigits 在源池筛选
   （`results/source_pool_v2`、stage1，切片协议 @290，δ 存活率 / 贴顶 / 死平判定）。**官方口径（UEA 官方划分、ts2vec_official）下
   24 个靶外数据集从未跑过。**
6. **文献数**：TS2Vec 论文（AAAI'22，arXiv 2106.10466 v4）Table 10，子代理转录并与 `35:TABLE10` 六行 36 个值逐值对账、29 靶
   均值行复核一致；InsectWingbeat DTW 论文为 "–"。AutoTCL / InfoTS 逐靶数字未录（用户截图待补，只作附录列，不入门）。
7. **本地 CPU 冒烟**（BasicMotions / JapaneseVowels / CharacterTrajectories × 1 legacy + 1 孪生 × iters 8）只验证管线，
   6 行三截点齐全、孪生 3 行；`--check` 23 靶尺寸 / iters / chance / EigenWorms 分段 (640, 3597, 6) 全过，RSS 峰值 0.98 GB。
   这些格与正式格 iters 不同、键不同，不进任何判读。

## §1 固定项（协议锁时写死）

- 代码基线 `mx4` 63694c0；分支 `wp10-uea30`；协议锁 commit 见 §7。
- `configs/uea30.yaml` md5 `73bf2ec1d05efb648d15d6effbdd8b84`（LF；`45.cfg_md5` 去 CR 后同值）；`results/uea30/candidates.json`
  md5 `5eea62afa3055640512715017426b0e6`（去 CR 口径，与 pod 端 `md5_data_local.txt` 同法；本机 CRLF 原始字节为
  `06de7163575754b248c8c86239f56ac4`）——23 靶 × shared-10 + GGS；22 靶 `fit_on = all 8 source datasets`，SelfRegulationSCP2
  `LODO excluding SelfRegulationSCP2`；`nominated = []`。
- **排序器不重训**：先验 = `results/mx3_pq/rprime_cache.jsonl` 上 `39.prior_with_ggs` 同码；GBDT 代理不参与（无提名）。
- **不跑**：InsectWingbeat（用户拍板：L≤22 低于 encoder 适用边界 50，评估 25000/25000 占总成本 28%）；表内标注"未跑"。
- **复用**：6 个 LODO 靶（Handwriting / UWave / PhonemeSpectra / Ethanol / Cricket / HMD）用 `results/mx4_deploy` + `results/mx4_mask`
  缓存在同一套臂函数上复算（`45.reused_six_targets`，与 `paper_table.md` 逐值对账为金标单测）；HAR 外部锚复用
  `results/mx4_deploy_har/analysis.json`（legacy_full 口径，只报不入门）。
- 靶顺序 = yaml `order`（便宜靶先；评估重的 FaceDetection / SpokenArabicDigits / PenDigits 最后；EigenWorms / MotorImagery 在 pod
  冒烟通过后）。

## §2 口径

- **协议** `ts2vec_official`（`configs/train.yaml`）：iters = 200 若 N·L·C ≤ 1e5 否则 600（官方规则）；bs 8、lr 1e-3、drop_last、
  SWA、scaler=train（TRAIN 统计量套两切分）、官方 SVM 网格（C ∈ {1e-4…1e4, ∞}，cv=5，n>10000 分层子采样）。
- **截点** {25, 50, 100}% × iters（200 → 50/100/200；600 → 150/300/600），同一条轨迹的 SWA 当前均值权重各评一次；seed 0。
- **目录 22 格/靶**：11 legacy（shared-10 + GGS，`input_mask=none`）+ 11 孪生（`twin(cls, input_mask="binomial")`，
  `cls_key` 含扩展字段、`options.input_mask` 同步，`check_extension_consistency` 训练前 fail-fast）。同一次 `--catalog` 出，
  legacy 在前孪生在后；`--resume` 键 = `35.cache_key`（dataset / protocol / cls_key / seed / iters / arm / options）。
- **EigenWorms**（L=17984）：训练侧 `uea.segment_train(Xtr, 3000)`——官方 `max_train_length` 语义（sections = L//3000 = 5，
  `np.array_split` 成 5 段 × 3597，末段补 1 步零）；截点评估编码**全长** train / test（官方 SVM 用未切分 train 编码）。
  论文未给全、按 X 语义补全：官方补 NaN 后 encoder 置零，本仓库增强链不耐 NaN，直接补零（= z-score 后通道均值）。
  MotorImagery L=3000 → sections=1 不切。
- **ragged**：右侧零填充到 max(TRAIN∪TEST) 真实长度（SAD 93 / CT 182 / JV 29）；CT 的 39 个 case 内通道不等长走
  `_parse_ts_tolerant`（截到 case 内最短通道）。
- **臂**（`45.uea30_arms`，名单只在 11 条 legacy 上按先验序排，GGS 在内）：
  - 主 **SH10-4-3_prior**：先验前 10 @25% → 4 @50% → 3 @100% → 取最好；部署当量 5.0 全跑；
  - 副 **SH20-3_both**：先验前 10 各带孪生 = 20 条从 50% 起 → 3 @100%；部署当量 11.5 全跑；
  - 对照 SH6-3-3_prior（3.75）/ SH20-4-3_both（7.5）/ prior1 / prior3 / random1 / oracle22 / oracle_off / oracle_on。
  成本按 iters 逐级累加（`36.path_cost_iters`），秒按缓存 `seconds` 增量（铁律 2）。
- **3 种子** = seed 0（目录格）+ finals seeds {1, 2}（两臂冠军，同冠军合并一格）；mean ± 总体 std，与 `paper_table.md` 同口径。
- **追平** = 3 种子均值 ≥ 论文 TS2Vec − 0.03（`parity_margin`，与 WP3 同）；**噪声 sd** = sqrt(p(1−p)/n_test)。
- **评估超时** 480 s / 截点（Linux SIGALRM）；超时格写 error 行，`--resume` 不重试；OOM 写 `oom_failures.jsonl`，
  同 key ≥ 2 次 `drop_repeated_oom` 剔出工单。
- **完整性**：11 legacy + 11 孪生三截点齐全且无 error 才 `complete`；不完整靶照常出 picks / finals，但**不入 U1–U3 分母**，
  单列 INCOMPLETE（名单静默变短的可见性）。

## §3 判读带（先于数据写死）

- **U1（头条，E 类 5：PEMS-SF / SelfRegulationSCP1 / NATOPS / ERing / EigenWorms）**：主臂冠军 3 种子 ≥ TS2Vec − 0.03 于
  **≥ 3/5**（`u1_min = 3`）。
- **U2（A 7 / B 3 / D 5 = 15）**：主臂 3 种子落在 yaml `band` 内 **≥ 12/15**（`u2_min_rate = 0.8`）。带 = 上一会话 §5.1 区间按
  1.96·sqrt(p(1−p)/n_test)（p = 区间中点）放宽后裁到 [0, 1]（§4 表）。
- **U3（副臂）**：完整靶上配对 Δ = SH20-3_both − SH10-4-3_prior（3 种子均值，缺则 seed 0）：**均值 ≥ 0 且 Δ ≥ −0.01 于 ≥ 60%**。
- **C 类（AtrialFibrillation / StandWalkJump / DuckDuckGeese）只报**：值 ± 1.96·sd（n_test 15 / 15 / 50 → ±0.20 / 0.25 / 0.13）。
- 读数码：`UEA30-WIN` = U1 ∧ U2；`UEA30-PARTIAL` = U1 ∧ ¬U2；`UEA30-DEAD` = ¬U1；U3 过加后缀 `+TWIN`。
- **主表** = 6 LODO 靶 + HAR + E 类 5（可评集）；**附录** = 全 30 按类标注，均值分"全 30（已跑 29）/ 剔 C / 可评集"三列。
- 不论读数如何，都如实报：INCOMPLETE 清单、error 行、孪生冠军数、25%→100% 截点排名 Spearman（描述性）。

## §4 逐数据集预测（先于数据写死；带见 yaml）

| # | 靶 | 类 | L | n_test | iters | 论文 TS2Vec | chance | 1.96·sd | 预测带 / E 类预测区间 | 预测 SH20-3 选孪生 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | RacketSports | B | 30 | 152 | 200 | 0.855 | 0.283 | 0.056 | [0.773, 0.927] | 否 |
| 2 | Libras | B | 45 | 180 | 200 | 0.867 | 0.067 | 0.050 | [0.800, 0.930] | 否 |
| 3 | ERing | E | 65 | 270 | 200 | 0.874 | 0.167 | 0.040 | 0.93–0.95（赢） | 否 |
| 4 | JapaneseVowels | A | 29 | 370 | 200 | 0.984 | 0.238 | 0.013 | [0.946, 1.000] | 否 |
| 5 | BasicMotions | A | 100 | 40 | 200 | 0.975 | 0.250 | 0.048 | [0.941, 1.000] | 否 |
| 6 | Epilepsy | A | 206 | 138 | 200 | 0.964 | 0.268 | 0.031 | [0.917, 1.000] | 否 |
| 7 | AtrialFibrillation | C | 640 | 15 | 200 | 0.200 | 0.333 | 0.202 | 只报 ±1.96·sd | 否 |
| 8 | NATOPS | E | 51 | 180 | 600 | 0.928 | 0.167 | 0.038 | 0.93–0.96（平/赢） | 否 |
| 9 | FingerMovements | D | 50 | 100 | 600 | 0.480 | 0.510 | 0.098 | [0.402, 0.678] | 是 |
| 10 | ArticularyWordRecognition | A | 144 | 300 | 600 | 0.987 | 0.040 | 0.013 | [0.944, 1.000] | 否 |
| 11 | Heartbeat | D | 405 | 205 | 600 | 0.683 | 0.722 | 0.064 | [0.639, 0.801] | 是 |
| 12 | DuckDuckGeese | C | 270 | 50 | 600 | 0.680 | 0.200 | 0.129 | 只报 ±1.96·sd | 否 |
| 13 | PEMS-SF | E | 144 | 173 | 600 | 0.682 | 0.173 | 0.069 | 0.70–0.78（可能赢） | 是 |
| 14 | CharacterTrajectories | A | 182 | 1436 | 600 | 0.995 | 0.065 | 0.004 | [0.964, 1.000] | 否 |
| 15 | SelfRegulationSCP1 | E | 896 | 293 | 600 | 0.812 | 0.502 | 0.045 | 0.82–0.87（可能赢） | 是 |
| 16 | SelfRegulationSCP2 | D | 1152 | 180 | 600 | 0.578 | 0.500 | 0.072 | [0.427, 0.653] | 是 |
| 17 | LSST | B | 36 | 2466 | 600 | 0.537 | 0.315 | 0.020 | [0.510, 0.580] | 否 |
| 18 | StandWalkJump | C | 2500 | 15 | 600 | 0.467 | 0.333 | 0.252 | 只报 ±1.96·sd | 是 |
| 19 | EigenWorms | E | 17984 | 131 | 600 | 0.847 | 0.420 | 0.062 | 0.85–0.90（可能赢） | 是 |
| 20 | MotorImagery | D | 3000 | 100 | 600 | 0.510 | 0.500 | 0.098 | [0.402, 0.648] | 是 |
| 21 | FaceDetection | D | 62 | 3524 | 600 | 0.501 | 0.500 | 0.017 | [0.483, 0.547] | 否 |
| 22 | SpokenArabicDigits | A | 93 | 2199 | 600 | 0.988 | 0.100 | 0.005 | [0.954, 1.000] | 否 |
| 23 | PenDigits | A | 8 | 3498 | 600 | 0.989 | 0.104 | 0.003 | [0.955, 1.000] | 否 |

"预测选孪生"依据 WP6 机制探针（掩码收益与 lag1 自相关 / 插值 R² 正相关、与类数负相关）：长序列 / 少类别 / EEG 类预测孪生胜出。

**总预测（先验概率）**：

| # | 预测 | 先验 |
|---|---|---|
| P1 | U1 过（E 类追平 ≥ 3/5） | 55% |
| P2 | U2 过（A/B/D 带内 ≥ 12/15） | 60% |
| P3 | U3 过（副臂 Δ 均值 ≥ 0 且 ≥ 60% 在 −0.01 内） | 65% |
| P4 | SH20-3 选中孪生的靶数 ≥ 6/23，且 §4"是"的 8 个里命中 ≥ 5 | 50% |
| P5 | 主臂冠军 ∈ {GGS} ∪ 先验前 3 的靶数 ≥ 14/23 | 60% |
| P6 | 25%→100% 截点排名 Spearman 中位（22 目录）≥ 0.5（描述性） | 60% |
| P7 | 实测 catalog 总墙钟落在 [8.8, 14.3] h（4090，含 InsectWingbeat 剔除） | 60% |
| P8 | 全 UEA 30（已跑 29）主臂均值落在 TS2Vec 论文 29 靶均值 0.712 ± 0.02 内 | 60% |

## §5 预算（锚点带硬件标签：RTX 4090，ts2vec_official）

`45 --dry-run`（2026-09-06）：S1 506 格（23 × 22）；catalog 8.84 h + finals ≤ 1.35 h（每靶 ≤ 2 冠军 × 2 种子）；
保守 ×1.4 = **14.3 h**；pod 费用按 $0.6–0.7/h ≈ **$9–10**。最贵五靶：PenDigits 1.67 h / SpokenArabicDigits 1.31 / FaceDetection
1.21 / MotorImagery 0.64 / EigenWorms 0.59（前三个是评估主导：n_train 5890–7494 的 RBF-OvO 网格）。数据：23 个 UEA 目录未压缩
≈ 2.3 GB（FaceDetection 753 MB / MotorImagery 513 / PEMS-SF 401 / DuckDuckGeese 292 / EigenWorms 251）；pod 侧优先 Zenodo 直下 +
md5 对账，失败再上传分包。每靶跑完从 `checkpoint_results` 回算实测 s/格写 REPORT。

## §6 执行顺序

1. 本地（Windows，wp10-uea30 工作树，`data` junction 到主目录）：单测 → `45 --build-candidates` → `45 --check` → `45 --dry-run` →
   CPU 冒烟（§0.7）→ 本协议完稿 → **协议锁 commit**。
2. **请示用户开 pod**（4090，卷 tudo4ntlot）→ `pod_ctl.py resume/deploy` → `uea30_bundle.sh` → scp → `uea30_setup.sh`
   （依赖 pin、CRLF→LF、`df` ≥ 4 GB、数据 fetch + md5、pytest 子集、`--check`、pod 冒烟 7 靶看 `nvidia-smi` 峰值与 PenDigits
   单次评估 < 480 s、`--dry-run`）。
3. `nohup uea30_run.sh`：`--catalog --resume` → `--analyze` → `--finals --resume` → `--analyze` → `--table30`；监控
   `inflight.txt` / `wc -l runs.jsonl`（目标 506 + finals ≤ 92）/ `oom_failures.jsonl`（期望 0）。中途可随时 scp 拉回 runs.jsonl；
   pod 死掉 → 原样重跑（`pending()` 跳过已跑，工单逐位同）。
4. 拉回 + **收官重验**：行数、`key` 唯一、非 error 行无 `acc: null`、`inflight.txt` 空、error 行清单、本地 `45 --analyze` 与 pod
   `analysis.json` 逐字节相同。
5. `45 --table30` → `REPORT.md`（门读数、预测对账 P1–P8、逐靶实测成本、孪生冠军数、INCOMPLETE 清单）→ memory → CLAUDE.md 里程碑
   → **停机** `pod_ctl.py stop` + `confirm-stopped 3`，记入 §7。

## §7 修订记录

- 2026-09-06：协议锁 commit **a5b168f**（yaml / 45 / 单测 / pod 脚本 / candidates.json / 本协议；全量单测 2062 通过 / 3 个既有
  失败：`test_cl_trainer` 与 `test_swa` 的 bitwise 金标受 torch 版本浮点影响、`test_pq_generate::TestApplyDelta`，WP6 时已在）。
  本条与 §1 md5 口径说明为锁后回填（不改判读带）。
- 2026-09-06 ~17:20（本地）：请示开 pod，用户同意"开 4090 pod（卷 tudo4ntlot）"。旧 4090 pod `5q5avhnjqmgvw9` resume 报
  "not enough free GPUs on the host"，按 pod_ctl 备选 `deploy` 在同卷同数据中心（EUR-NO-1）新开 **pod `mvqetmtt1t4azg`**
  （uea30-4090，RTX 4090 23 GB，120 vCPU，$0.74/h，镜像 runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404，torch 2.8+cu128，
  与 WP6 的 4090 pod 镜像 torch 2.4 不同——GGS seed 0 跨 2.4/2.8 五靶逐位相同，见 WP3 里程碑）。bundle HEAD 8b4c647。
- 待记：`uea30_setup.sh` 冒烟的 `nvidia-smi` 峰值与 PenDigits 单次评估秒数；主流程开跑 / 拉回 / 停机时间与费用。
