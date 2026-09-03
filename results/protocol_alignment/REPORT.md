# 全量标准协议对齐（Protocol Alignment）——本栈在 UEA 官方划分上追平已发表模型

脚本 `scripts/35_protocol_alignment.py`；缓存 `runs.jsonl`（key = sha1[:16] of {dataset, protocol,
cls_key, seed, iters, arm, options}，`--resume` 按 key 跳过，OOM 不入缓存 → `oom_failures.jsonl`）；
预注册 picks `picks.json`；聚合 `analysis.json` / `tables.md`（`--analyze` 生成）。

## §0 预注册（2026-09-03，写于任何 GPU 跑之前；分支 `protocol-alignment`，基于 mx3pq@3ea9e0a）

### 0.1 目标与背景
用户目标：在 UEA 官方 train/test 划分（SVM-RBF 下游）上，本栈的绝对精度追平多数已发表模型。
现状：唯一全量锚 Handwriting（`results/mx2_endtoend/anchor_cache.jsonl`，5090，2320 iters@bs128）
零样本冠军 0.339 / GGS 0.390，TS2Vec 论文 0.515；UCI-HAR 同链路 0.935 ≈ AutoCL GGS 0.937。
审计（2026-09-03，三路只读）把差距定位在**训练口径**而非 encoder / 零样本选择：

| # | 项 | 本栈 legacy_full | TS2Vec 官方 |
|---|---|---|---|
| M1 | 训练量 | 2320 iters@128 = 150 样本 ≈1160 遍 | 200 if N·T·C≤1e5 else 600 iters，bs 8 ≈ 11 遍 |
| M2/3 | batch | 128，每遍 2 步含 22 样本残批，无 drop_last | 8，drop_last |
| M4 | SWA | 关 | AveragedModel（init 快照 + 逐 iter） |
| M5 | 目标函数 | 冠军/GGS 均单尺度 instance-only | 层级 instance+temporal（空间内可表示，从未跑过） |
| M6 | crop | 共享长度由 p 固定、整批共用偏移 | 每迭代 crop_l~U[2,T]、逐行偏移 |
| M7 | encoder 训练期 | 无 binomial 输入掩码 / repr_dropout | 有（p=0.5 / 0.1） |
| M8 | 归一化 | TRAIN/TEST 各自 z-score | scaler 只在 TRAIN 拟合 |
| M9/10 | SVM | C∈{1,10,100}，cv=3 | C∈{1e-4..1e4, inf}，cv=5，小样本回退 C=inf |

### 0.2 协议块（`configs/train.yaml: protocols`，现有键原样不动）
```yaml
legacy_full:     {iters: 2320, batch_size: 128, lr: 1e-3, drop_last: false, swa: false, crop_mode: fixed, scaler: split, svm_grid: legacy}
ts2vec_official: {iters_rule: ts2vec_size, batch_size: 8, lr: 1e-3, drop_last: true, swa: true, crop_mode: fixed, scaler: train, svm_grid: ts2vec}
```
`iters_rule: ts2vec_size` → 只有 Handwriting（150×152×3 = 68,400）得 200，其余五靶 600
（UWave 113,400 / PS 7,912,905 / Ethanol 1,371,033 / Cricket 775,656 / HMD 640,000）。
`legacy_full` 与同文件 `full_iters/batch_size/lr` 三键漂移即 fail-fast（铁律 2）。
所有新开关（drop_last / crop_mode / input_mask / repr_dropout / svm grid / scaler）默认关，
默认路径金标逐位不变（tests: test_cl_trainer::TestDropLastAndCropMode、test_swa 金标、
test_uea::test_pad_and_zscore_refactor_bitwise_golden、test_encoders::TestLadderSwitches、
test_linear_eval::TestSvmGrid、test_augmentations::TestCropTs2vec）。

### 0.3 臂（CLS）定义
| 臂 | cls_key | 定义 | 空间合法 |
|---|---|---|---|
| `ts2vec_equiv` | `3c231edcf2a5` | crop 0.5（仅开关）、instance+temporal、kernel 2 max、cross_scale 关、dot、τ=1、InfoNCE、norm none | ✓ |
| `ggs` | `c92058f26c1d` | AutoCL Table 5 GGS 本空间合法化版（kernel/pool 置 None） | ✓ |
| `ggs_table5` | `9dd8c689d53a` | Table 5 忠实版 kernel 5 / avg（本实现会加多尺度 instance 项） | ✗ 白名单，仅文献对照，永不作候选 |
| `zs_top1` / `zs_top3` / `zs_clsonly_top1` | 见下表 | `results/mx3_pq/rankings.json[D]["290"]` 六切片 `main`（/`cls_only`）排名在 10 个 shared 键上的 Borda（铁律 6：只读以 D 为靶的折） | ✓ |

预注册 picks（`picks.json`，零 GPU 推导，先于数据写死）：

| 目标 | zs_top1 | zs_top3 | cls_only_top1 | 唯一 CLS 数 |
|---|---|---|---|---:|
| Handwriting | a1f67db45e64 | a1f67db45e64, 4f7d413581f6, 9092e36eefe1 | 9092e36eefe1 | 6 |
| UWaveGestureLibrary | 9092e36eefe1 | 9092e36eefe1, cffb55035c3a, 8b87d66faad3 | 9092e36eefe1 | 6 |
| PhonemeSpectra | 9092e36eefe1 | 9092e36eefe1, cffb55035c3a, 8b87d66faad3 | 9092e36eefe1 | 6 |
| EthanolConcentration | a1f67db45e64 | a1f67db45e64, 9092e36eefe1, 4f7d413581f6 | a1f67db45e64 | 6 |
| Cricket | 9092e36eefe1 | 9092e36eefe1, a1f67db45e64, 4f7d413581f6 | 9092e36eefe1 | 6 |
| HandMovementDirection | 9092e36eefe1 | 9092e36eefe1, 4f7d413581f6, a1f67db45e64 | 9092e36eefe1 | 6 |

同一 CLS 的多个臂合并成一格（按 CLS 内容的规范键去重），六靶合计 36 个唯一 (D, CLS) × 3 种子 = 108 格。

### 0.4 阶段、工单与判线（执行顺序固定；`--dry-run` 计数）
| 阶段 | 格数 | 读数 / 判线 |
|---|---:|---|
| A1 `--eval-audit` iters=0 | 18 | 随机 encoder 基线；同一特征上 scaler × svm_grid 的 2×2 差分（M8、M9/10 的纯评估侧效应） |
| A2 `--eval-audit` legacy 重跑 | 2 | HW × {冠军 f48e5da1d899, GGS} × seed 0 @legacy_full；对 5090 锚 0.341/0.392 的跨机核对（±0.02 内），并作秒/iter 估时锚 |
| `--reproduce` | 12 | HW × TS2VEC_EQUIV × {official, official−swa, legacy, legacy+swa} × 3 种子。**P1：official 三种子均值 ≥ 0.485 且每种子 ≥ 0.465** |
| `--bs-scan` | 9 | HW × TS2VEC_EQUIV @ official，bs∈{8,32,128} 等遍（drop_last 口径 200/44/11 iters）× 3 种子；只作归因不作判线；bs=8 格与 repro_official 计算相同 = 同机复现对照 |
| `--official-ts2vec` | 18 | vendored 官方 fit 循环（`autocls/task_embed/ts2vec_wrapper.TS2VecWrapper`，output_dims 320、bs 8）六靶 × 3 种子。**P1′：HW 同 P1 线**；P1′ 不过 = 数据/评估侧问题，先修再上阶梯 |
| `--ladder`（仅 P1 不过时） | 12 | official 上依次 +crop_mode=ts2vec、+input_mask=binomial、+repr_dropout=0.1、三者全开；3 种子；成分均值 ≥ +0.03 才采纳 |
| `--targets` | 108 | 六靶 × 唯一 CLS × 3 种子 @ official。**P2 头条：zs_top3（best-of-3）三种子均值 ≥ TS2Vec − 0.03 的数据集数**；次读数 ≥ 该行文献六值中位数；并报 cv 选择口径（SVM 交叉验证分选 top-3 内之一，无测试集泄漏） |
| `--catalog`（可选，用户 go 后） | 120 | 六靶 × (10 shared + 切片 _00 的 10 random) × seed 0；只报与切片 R'@290 的 Spearman，shared-10 ρ<0.5 标"需重标"（另立任务，不改标签协议） |

判线阈值（TS2Vec Table 10）：

| 目标 | TS2Vec | line = TS2Vec−0.03 | 六值中位数 |
|---|---:|---:|---:|
| Handwriting | 0.515 | 0.485 | 0.3685 |
| UWaveGestureLibrary | 0.906 | 0.876 | 0.817 |
| PhonemeSpectra | 0.233 | 0.203 | 0.2145 |
| EthanolConcentration | 0.308 | 0.278 | 0.291 |
| Cricket | 0.972 | 0.942 | 0.972 |
| HandMovementDirection | 0.338 | 0.308 | 0.2835 |

先验（写死）：P1 过 0.55；P1′ 过 0.70；P2 ≥ 3/6 数据集 parity 0.40，≥ 4/6 中位数 0.55。

### 0.5 时间预算与硬件
用户拍板（2026-09-03）：GPU 阶段跑 RunPod **RTX 4090** pod（EUR-NO-1 卷 `tudo4ntlot`，仓库
`/workspace/wp/CLS`，镜像 runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404，torch 2.8.0+cu128），
估 3–4 h ≈ $3；bs 小扫描加入。跑完必须 podStop 并多次确认（pod id / 时长 / 费用入 §6）。
A2 的跨机核对因此是 **4090 vs 5090**（非计划原文的 4060）。`--dry-run` 粗估合计 2.75 h（不含
catalog 1.2 h）；A2 后用实测秒/iter 重估。

### 0.6 已知残差与口径声明（不加开关、写进 limitations）
1. `total_contrastive_loss` 对各尺度各维度取 mean（除以 2d−1）vs 官方 `/d` 且 α=0.5 → 有效学习率 ×≈0.94；
   `hierarchical_pool` ceil_mode vs 官方 max_pool floor（末尺度多一层）。
2. TS2VEC_EQUIV 的 crop 在 `crop_mode=fixed` 下共享长度固定为 0.5T、整批共用偏移（M6 是阶梯项）。
3. 官方 wrapper 与官方仓库差异：output_dims 覆盖为 320（wrapper 默认 256）、batch_size 8、无
   max_train_length（六靶 T ≤ 1751 < 3000，不触发）、种子经 np/torch 全局设定；binomial 掩码 /
   repr_dropout / SWA / 逐行裁剪 / 层级损失 与官方逐字相同。
4. 六靶全用**原生长度**（Cricket 1197、Ethanol 1751），而 zero-shot picks 来自源池切片口径
   （Cricket crop 256、Ethanol crop 1024、按类别子集切片、@290 iters bs128）——pick 与靶评估的
   口径差本身是 zero-shot 迁移的一部分，如实报。
5. `zs_top3` 主口径 = best-of-3 on test（MX-2 `tclsc3` 同款上界）；同时报 cv 选择口径。
6. scaler=train 下标签在两切分并集上编码（官方在 TRAIN 唯一值上）；六靶 TEST 类集 ⊆ TRAIN 类集，
   两者一致（脚本 `load_arrays` 断言两 scaler 的标签相同）。
7. GGS_T5 出空间，`space_legal=false`，仅文献对照；铁律 1 不破。
8. R' 标签管线、比较器、搜索、space.yaml、所有缓存、MX-3-PQ 产物一律未动；encoder 的两个阶梯开关
   造 label 永不触发（`build_encoder()` 默认关，tests 断言）。
9. 单种子目录（catalog）噪声地板 δ_nd≈0.03（memory: iters 地板）——Spearman 只作方向性读数。
10. bs 扫描的"等遍"按 drop_last 口径 N//bs 步/遍换算（200/44/11），bs 128 时每遍只 1 步、丢 22 样本。

### 0.7 补充（2026-09-03 09:10 UTC，写于 `--ladder` 开跑之前；`--reproduce` 已出 P1=不过）
P1 不过（TS2VEC_EQUIV@official 0.437±0.019）→ 阶梯将跑。若阶梯**采纳**了成分（单 rung Δ ≥ +0.03），
六靶再跑第二遍 `--targets --ladder-overrides <rung>`（stage `targets_ladder`，臂名加 `+ladder`），
rung = 采纳者中 Δ 最大的**单一实测配置**（不拼未实测的组合）。头条 P2 仍按 §0.4 的主口径
（plain `ts2vec_official`）判读；+ladder 第二遍是次读数，报在 §4b，用于回答"补上 encoder 训练期
成分后能否追平"。判线、阈值、picks 不变。

## §1 评估审计（A1 iters=0 基线 / A2 旧口径跨机核对）

执行：RTX 4090 pod `m9l7o0rgj2a67b`（EUR-NO-1，torch 2.8.0+cu128，cgroup 12.75 CPU），
`--eval-audit`，20 格；本地 4060 另跑了 seed 0 的功能冒烟（scratch，非判线）。

**A1 随机 encoder（iters=0，3 种子均值）的 2×2 评估变体**（`tables.md §1`）：

| 数据集 | split/legacy | train/legacy | split/ts2vec | train/ts2vec | Δscaler | Δgrid |
|---|---:|---:|---:|---:|---:|---:|
| Handwriting | 0.251 | 0.251 | 0.251 | 0.251 | +0.000 | +0.000 |
| UWaveGestureLibrary | 0.668 | 0.668 | 0.668 | 0.668 | +0.000 | +0.000 |
| PhonemeSpectra | 0.202 | 0.203 | 0.201 | 0.203 | +0.002 | −0.000 |
| EthanolConcentration | 0.293 | 0.285 | 0.283 | 0.283 | −0.000 | −0.003 |
| Cricket | 0.968 | 0.968 | 0.968 | 0.968 | +0.000 | +0.000 |
| HandMovementDirection | 0.234 | 0.248 | 0.239 | 0.239 | +0.000 | −0.009 |

读法：① **M8（scaler）与 M9/M10（SVM 网格）在同一特征上的效应 ≤ 0.01**——评估侧不是差距来源；
② 随机 encoder 在 Cricket（0.968 vs TS2Vec 0.972）、Ethanol（0.28–0.29 vs 0.308）、PhonemeSpectra
（0.20 vs 0.233）上已贴近文献值——这三靶对"协议对齐"几乎没有可显示的净空（memory：贴顶是源池主流）；
③ Handwriting 随机 0.251 → 差距 0.26 全在训练侧。

**A2 旧口径重跑**（legacy_full = 2320 iters@128，seed 0）vs MX-2 `anchor_cache.jsonl`（5090）：

| 臂 | 4090 | 4060（冒烟） | 5090 seed 0 | 5090 三种子均值 | Δ(4090−5090 s0) |
|---|---:|---:|---:|---:|---:|
| 冠军 f48e5da1 | 0.3412 | 0.3412 | 0.3412 | 0.3388 | +0.0000 |
| GGS c92058f2 | 0.4082 | 0.3976 | 0.3918 | 0.3902 | +0.0165 |

三机（4060/4090/5090）冠军逐位同值、GGS 在 ±0.02 内：旧口径可复现，跨机漂移不是问题。
估时锚（4090）：2320@128 冠军 117 s、GGS 49 s；200@8 TS2VEC_EQUIV 约 5 s；官方 wrapper 200@8 HW
约 10 s、600@8 PhonemeSpectra 17 s + SVM 20 s、Ethanol（T=1751）37 s。

**执行台账（如实）**：第一次开跑用 `threading` joblib 后端，PhonemeSpectra 的官方网格（cv=5 × 10 C，
39 类）单格 >25 min 未完；根因是转录时漏了官方 param_grid 里显式的 `max_iter=10000000`
——C=inf（硬间隔）在不可分多类数据上 SMO 无界不收敛（本地实测：C=1/100/1e4 各 0.4/0.5/1.3 s，
C=inf 无界 >15 min 未完；加 1e7 上限 7.4 s，val_acc 与 C=1e4 同为 0.20）。补上 max_iter（提交
10bcba3，**更忠实**而非偏离）并改 `loky` 后端后，PhonemeSpectra 整格 SVM 约 20 s。中途两次
`--resume` 重启，已跑格零丢失（6 行）。

## §2 复现阶梯：Handwriting × TS2VEC_EQUIV（P1）、bs 扫描、成分阶梯

| 格 | 协议 | SWA | acc（3 种子） | f1 | 各种子 |
|---|---|---|---:|---:|---|
| repro_official | ts2vec_official（200@8, drop_last） | 开 | **0.437±0.019** | 0.400 | 0.424, 0.424, 0.465 |
| repro_official_noswa | 同上 | 关 | 0.474±0.010 | 0.434 | 0.481, 0.460, 0.480 |
| repro_legacy | legacy_full（2320@128） | 关 | 0.344±0.006 | 0.315 | 0.342, 0.338, 0.352 |
| repro_legacy_swa | 同上 | 开 | 0.412±0.005 | 0.382 | 0.405, 0.415, 0.416 |

- **P1 不过**：0.437 < 0.485（三种子 0.424/0.424/0.465，均 < 0.485）。
- 归因（均值差）：训练口径本身（无 SWA）**+0.130**（0.344 → 0.474）；SWA 在长口径 **+0.068**、
  在短口径 **−0.037**（200 步等权平均把 init 与前期权重也计入，官方在有掩码/dropout 的轨迹上
  才是净增益，见阶梯）；合计 official − legacy = +0.093。
- 同机复现对照：`bs_scan_8`（与 repro_official 完全同配置）三种子 0.419/0.426/0.462 vs
  0.424/0.424/0.465——GPU 非确定性 ≤ 0.005。

**bs 扫描**（等遍 ≈11 遍，drop_last 口径 200/44/11 iters）：

| bs | iters | acc | 各种子 |
|---:|---:|---:|---|
| 8 | 200 | 0.436±0.019 | 0.419, 0.426, 0.462 |
| 32 | 44 | 0.368±0.021 | 0.351, 0.356, 0.398 |
| 128 | 11 | 0.312±0.011 | 0.299, 0.313, 0.325 |

同遍数下 bs 8 → 128 掉 0.12：小 batch（更多更新步、InfoNCE 负样本更少）本身就是 M1/M2 差距的
主体，与"训练越多越差"的既有四个实验相容（大 bs + 长训练 = 双重不利）。

**成分阶梯**（在 ts2vec_official 上逐成分，3 种子；采纳线 Δ ≥ +0.03）：

| 成分 | acc | 各种子 | Δ vs official | 采纳 |
|---|---:|---|---:|---|
| +crop_mode=ts2vec（官方逐行随机长度裁剪） | 0.411±0.007 | 0.414, 0.402, 0.418 | −0.026 | ✗ |
| **+input_mask=binomial（官方 p=0.5 输入掩码）** | **0.519±0.009** | 0.532, 0.512, 0.513 | **+0.082** | ✓ |
| +repr_dropout=0.1 | 0.485±0.017 | 0.468, 0.479, 0.508 | +0.048 | ✓ |
| 三者全开 | 0.498±0.004 | 0.502, 0.493, 0.499 | +0.061 | ✓ |

- **残差归因定案**：本栈 encoder 删掉的 TS2Vec 训练期 binomial 输入掩码是最后一块（+0.082，
  单独就把 TS2VEC_EQUIV 送到 0.519 ≥ 0.485 且每种子 ≥ 0.465，即 P1 线在 +input_mask 下成立）；
  repr_dropout 次之；官方式随机长度裁剪在本框架里反而 −0.026（本空间 crop 字段的固定共享长度
  0.5T 更好）。按 §0.7 规则采纳 rung = `ladder_input_mask`（Δ 最大的单一实测配置），六靶第二遍用它。
- 本地 4060 冒烟（seed 0）同向：official 0.421、+input_mask 0.531、+repr_dropout 0.465、全开 0.475。

## §3 官方地面真值（vendored TS2Vec fit 循环于本机，P1′）

| 数据集 | 本机 official（3 种子） | 论文 TS2Vec | Δ | 各种子 |
|---|---:|---:|---:|---|
| Handwriting | 0.529±0.006 | 0.515 | +0.014 | 0.531, 0.535, 0.520 |
| UWaveGestureLibrary | 0.903±0.009 | 0.906 | −0.003 | 0.912, 0.891, 0.906 |
| PhonemeSpectra | 0.235±0.001 | 0.233 | +0.002 | 0.233, 0.235, 0.236 |
| EthanolConcentration | 0.288±0.016 | 0.308 | −0.020 | 0.304, 0.266, 0.293 |
| Cricket | 0.977±0.007 | 0.972 | +0.005 | 0.972, 0.986, 0.972 |
| HandMovementDirection | 0.266±0.039 | 0.338 | −0.072 | 0.311, 0.216, 0.270 |

- **P1′ 过**（HW 0.529，每种子 ≥ 0.52）：数据加载（scaler=train、标签并集编码）、官方 SVM 网格、
  max-pool 表征、iters 规则在本机全部对齐——论文数字在 5/6 靶 ±0.02 内可复现；HMD（测试集 74 条，
  种子间 0.216–0.311）是 Table 10 数字的噪声，不是本机问题。
- 与 §2 合读：官方代码 0.529 vs 本栈 TS2VEC_EQUIV@official 0.437 的 0.09 差距，阶梯里 +input_mask
  一项即补到 0.519；剩余 ≈0.01 在 §0.6 列出的残差范围内（损失归一化 2d−1 vs d、ceil_mode）。

## §4 六靶表（主口径 = plain `ts2vec_official`，3 种子均值；P2）

| 数据集 | TS2Vec 论文 | 官方代码本机 | TS2VEC_EQUIV | GGS | GGS_T5* | zs_top1 | zs_top3(best) | zs_top3(cv) | 判定(top3) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Handwriting | 0.515 | 0.529±0.006 | 0.442±0.022 | **0.532±0.015** | 0.484±0.018 | 0.389±0.007 | 0.433±0.019 | 0.433 | median |
| UWaveGestureLibrary | 0.906 | 0.903±0.009 | 0.750±0.018 | 0.869±0.003 | 0.881±0.007 | **0.907±0.002** | 0.907 | 0.907 | parity |
| PhonemeSpectra | 0.233 | 0.235±0.001 | **0.246±0.008** | 0.203±0.002 | 0.206±0.003 | 0.190±0.006 | 0.231±0.011 | 0.229±0.003 | parity |
| EthanolConcentration | 0.308 | 0.288±0.016 | 0.280±0.010 | 0.241±0.007 | 0.245±0.005 | 0.267±0.031 | 0.294±0.021 | 0.294 | parity |
| Cricket | 0.972 | 0.977±0.007 | 0.958±0.000 | 0.986±0.011 | 0.986±0.011 | 0.968±0.013 | **0.986±0.000** | 0.986 | parity |
| HandMovementDirection | 0.338 | 0.266±0.039 | 0.239±0.026 | 0.270±0.019 | 0.315±0.046 | **0.446±0.058** | 0.446 | 0.446 | parity |

\*GGS_T5 出空间（`space_legal=false`），仅文献对照。zs_top3(best) = top-3 内测试集均值最大者（MX-2 `tclsc3`
同款上界）；zs_top3(cv) = 用 SVM 5 折交叉验证分在 top-3 内选（无测试集泄漏），六靶里 5 靶与 best 同一 CLS，
PhonemeSpectra 选到 8b87d66f（0.229，仍 parity）。各臂逐种子值见 `analysis.json: targets.per_dataset.*.per_cls`。

**P2 头条（预注册 §0.4）：zs_top3 三种子均值 ≥ TS2Vec − 0.03 的数据集数 = 5/6**（UWave / PhonemeSpectra /
Ethanol / Cricket / HMD）；≥ 该行文献六值中位数 **6/6**；cv 选择口径 parity 同为 5/6。zs_top1 口径：parity 3/6
（UWave、Cricket、HMD），median 1（HW），below 2（PhonemeSpectra 0.190、Ethanol 0.267）。

**是否追平（一句话）**：在 UEA 官方划分的六靶上，把训练口径对齐到 TS2Vec 官方（200/600 iters @ bs 8、
drop_last、SWA、train-fit scaler、官方 SVM 网格）之后，本栈的零样本 top-3 选择在 **5/6 靶追平或超过 TS2Vec
（−0.03 内），6/6 靶超过文献中位数**；唯一未追平的 Handwriting（0.433 vs 0.515）差在**零样本选择**而非空间或
encoder——同口径下 GGS 0.532 已达平、encoder 补上官方 binomial 掩码后 TS2VEC_EQUIV 0.519 亦达平（§2/§4b）。

逐靶读法：
- **Handwriting**：这是 MX-2 的锚数据集。旧口径 0.34–0.39 → 官方口径下所有臂整体抬升（GGS 0.39 → 0.53，
  TS2VEC_EQUIV 0.44）。zero-shot pick a1f67db4（resize 0.5 + point_mask 0.8 + kernel 2 cross-scale）只有 0.389，
  top-3 最好 9092e36e 0.433——切片 R'@290（bs 128、类别子集、split z-score）上的排名没迁移到 200@8 官方口径
  （§0.6 第 4 条口径差的实例）；同口径下 GGS 0.532 说明空间里有达平配置，pick 没选中它（GGS 不在 shared-10
  候选目录里）。
- **UWaveGestureLibrary**：zs_top1 = 9092e36e 0.907 = 论文 0.906；GGS 0.869、TS2VEC_EQUIV 0.750 都低——
  这一靶零样本选择明显优于两个固定基线。
- **PhonemeSpectra**：随机 encoder 0.20，TS2VEC_EQUIV 0.246 > 论文 0.233 > 官方本机 0.235；zs_top1 0.190
  低于随机 encoder（9092e36e 的 resize 0.6 + triplet τ=10 在 39 类上失效），top-3 里 cffb5503 0.231 补回。
- **EthanolConcentration**：随机 encoder 0.283 已近论文 0.308，官方本机 0.288；zs_top3 0.294 parity，
  GGS 0.241 反而低于随机——贴顶靶（§1 ②）。
- **Cricket**：全部臂 ≥ 0.958，zs_top3 0.986 与 GGS 同值；随机 encoder 0.968——贴顶靶，无信息。
- **HandMovementDirection**：zs_top1 0.446 远超论文 0.338 与官方本机 0.266，但三种子 0.365/0.473/0.500、
  测试集 74 条（±0.06 ≈ 4–5 个样本）——方向可信、幅度不可信。

预测对账（§0.4 先验）：P1 过（先验 0.55）→ **不过**；P1′ 过（0.70）→ **过**；P2 ≥ 3/6 parity（0.40）→ **过**（5/6）；
P2 ≥ 4/6 中位数（0.55）→ **过**（6/6）。

成本（4090，`runs.jsonl` 逐行 pretrain/eval 秒）：eval_audit 20 格 4.1 min、reproduce 12 格 7.1 min、bs_scan 9 格
0.5 min、official 18 格 7.8 min、ladder 12 格 1.2 min、targets 108 格 53.9 min（预训练 47.6 + SVM 6.3）；
179 行零 error、零 OOM（`oom_failures.jsonl` 空）。sklearn 在 PhonemeSpectra 的 GGS_T5 格上 4 次报
`FitFailedWarning`（50 个网格拟合里 1 个 C=inf 折系数非有限 → 该候选记 NaN，网格从其余候选选 C），格本身完成。

## §4b 六靶 +ladder 第二遍（ts2vec_official + 采纳成分 ladder_input_mask；REPORT §0.7；次读数，非头条）

| 数据集 | TS2Vec 论文 | TS2VEC_EQUIV+L | GGS+L | zs_top1+L | zs_top3(best)+L | zs_top3(cv)+L | 判定(top3) |
|---|---:|---:|---:|---:|---:|---:|---|
| Handwriting | 0.515 | 0.519 | 0.520 | 0.375 | 0.408 | 0.408 | median |
| UWaveGestureLibrary | 0.906 | 0.864 | 0.901 | 0.896 | 0.896 | 0.896 | parity |
| PhonemeSpectra | 0.233 | 0.234 | 0.191 | 0.190 | 0.235 | 0.235 | parity |
| EthanolConcentration | 0.308 | 0.272 | 0.247 | 0.297 | 0.323 | 0.297 | parity |
| Cricket | 0.972 | 0.963 | 0.995 | 0.968 | 0.986 | 0.986 | parity |
| HandMovementDirection | 0.338 | 0.338 | 0.266 | 0.495 | 0.495 | 0.495 | parity |

+ladder：zs_top3 parity 5/6，≥中位数 6/6

读法（§4b，`analysis.json: targets_ladder`；逐 CLS 的 plain → +mask 差分）：
- **+mask 不是普适增益**：对"干净"配置（增强少/无）稳定有效——TS2VEC_EQUIV 在 HW +0.077、UWave +0.114、
  HMD +0.099（PhonemeSpectra −0.012、Ethanol −0.008、Cricket +0.005），GGS 在 UWave +0.032、HW −0.013；
  对**重增强**的零样本 pick 反而有害——UWave 的 8b87d66f −0.125、HW 的三个 pick −0.014～−0.024
  （它们本身已带 point_mask 0.2–0.8 + freq_mask 0.6–0.95，再叠 Bernoulli(0.5) 输入掩码 = 双重掩码）。
- P2 计数在 +ladder 口径下不变（parity 5/6、≥中位数 6/6）；HW 仍是唯一未达平靶（zs_top3 0.408）。
- 因此**采纳 input_mask 只应作为"TS2Vec 目标/干净配置的默认"**，不应作为全空间的 encoder 默认——
  它与 CLS 的掩码类字段（point_mask/emb_mask）重叠，正是 encoder 当初去掉它的理由（铁律 6 注释、
  `dilated_cnn.py` 模块 docstring）。若要把它纳入空间，正确做法是作为 CLS 的第 20 个字段进 `space.yaml`
  而非 encoder 常开（本轮不改，见 §7）。
- HMD 的 +0.05（0.446 → 0.495）与 TS2VEC_EQUIV +0.099 一起读：本靶测试集 74 条、种子间 ±0.05，方向可信、
  幅度不可信（§4）。

## §5 目录 oracle 与 Spearman（可选末段）——**未完成，本轮不报数**

用户 12:00 UTC 批准 `--catalog`（六靶 × 20 目录 CLS × seed 0 @ ts2vec_official），11:09 UTC 开跑；
12:24 UTC 时 83/120（Handwriting / UWave / PhonemeSpectra / Ethanol 完整、Cricket 进行中），用户 12:2x UTC
拍板"现在停止、提交主结果"，pod 随即 EXITED。已跑的约 85 格留在网络卷 `tudo4ntlot` 的
`/workspace/wp/CLS/results/protocol_alignment/runs.jsonl` 里（stage=`catalog`），**未拉回本地**；本地提交的
`runs.jsonl` 剔除了 11:11 UTC 那次拉取里混入的 1 行 catalog 格（n=1 的 Spearman 无意义），287 行全部属于
主口径七个阶段。补跑/拉回：重开同卷 pod，`bash scripts/pa_pod/setup.sh`（依赖重装）后
`python scripts/35_protocol_alignment.py --catalog --resume --joblib-backend loky` 再 `--analyze`，约 20 min + $0.3。

执行观察（写进 limitations）：PhonemeSpectra 目录里的退化 CLS（acc≈0.09）让官方 SVM 网格跑到 max_iter=1e7
上限才停（单格 SVM 750 s）——目录重扫的成本被"破坏性配置 × 39 类 × C=inf"主导，与 memory
"随机空间 65% 破坏性"一致。

已知的一条目录读数（主口径 §4 已含）：Handwriting 上 shared-10 中 GGS 不在候选目录里，而目录内 top-3
最好者 0.433 < 同口径 GGS 0.532 ——R'@290 切片排名对官方口径的效度问题**已由 §4 定性**（pick 选错），
量化的 ρ 留待补跑。

## §6 铁律台账

| 铁律 | 本任务的处理 | 证据 |
|---|---|---|
| 1 空间唯一真源 | GGS_T5（kernel 5/avg，两开关全关）`space.validate` 判非法 → `space_legal=false` 白名单，只进对照列、永不入候选；脚本启动断言 `not validate(GGS_T5)` | `tests/test_protocol_alignment.py::TestArms`、`target_arms` 对白名单臂若合法即 raise |
| 2 iters 显式 + 口径一致 | `train.yaml` 只新增 `protocols:{legacy_full, ts2vec_official}`；`ts2vec_size` 规则算出的 200/600 仍是显式 int 传入 `pretrain_encoder` 并逐行落盘（`iters` 字段）；`legacy_full` 与 `full_iters/batch_size/lr` 漂移 fail-fast；early/full/budgets 三条既有口径与所有旧缓存原样不动 | `TestProtocols::test_load_protocols_drift_fail_fast`、`test_existing_keys_untouched` |
| 3 结果落盘 + resume | 单一 `runs.jsonl`，key = sha1[:16]{dataset, protocol, cls_key, seed, iters, arm, options}；`--resume` 按 key 跳过；OOM 抛 `RprimeOOM` 不入缓存 → `oom_failures.jsonl`；非 OOM 异常记 `error` 行 | `TestCacheKey`；`execute()` |
| 4 task_embed 照官方源码 | 官方地面真值走 `autocls/task_embed/ts2vec_wrapper.TS2VecWrapper`（fit 循环 = ts2vec.py 逐字副本），差异逐条写 §0.6/§7 | — |
| 5 先测后写 + 论文未给全标注 | 六个开关各有"默认路径逐位不变"金标（loss hex + 参数 md5 / pad_and_zscore md5 / eval 前向 equal / legacy 网格参数）先于实现；官方 SVM 网格未 vendored → 转录并在 docstring 标来源 | `tests/test_{cl_trainer,augmentations,encoders,linear_eval,uea}.py` 新增类 |
| 6 零样本无泄漏 | picks 只读 `rankings[<target>]`（该折的 TS2Vec/比较器/δ 均不含目标）；TS2VEC_EQUIV/GGS/GGS_T5 与数据集无关；六靶 TEST 只在评估时前向 | `TestZeroShotPicks::test_only_target_fold_is_read` |
| 7 任务类型显式分发 | 脚本顶部 `validate_task_type(TASK_CLASSIFICATION)` 硬断言；contrastive/encoders 新开关不引入任务类型假设 | — |
| 8 R' 优劣方向 | 本任务不造配对 label，不触及 | — |
| encoder 确定性（关键决策 2 注释） | `input_mask`/`repr_dropout` 只在 `--ladder` 经 `build_encoder(overrides=)` 打开；`build_encoder()` 默认关，造 label 永不触发；eval 前向与 state_dict 键不变 | `TestLadderSwitches::test_build_encoder_default_never_enables` |
| pod 台账 | 4090 pod `m9l7o0rgj2a67b`（EUR-NO-1 卷 tudo4ntlot，$0.74/h）：07:59:45 UTC podResume → 12:2x UTC EXITED（用户拍板停止），`confirm-stopped` 3/3 EXITED @12:30 UTC；时长 ≈4.45 h ≈ $3.3；catalog 中止于 ~85/120（数据留卷上） | `scripts/pa_pod/pod_ctl.py` |
| 不动清单 | R' 标签管线、比较器、搜索、`space.yaml`、所有缓存、MX-3-PQ 产物：零改动（`git diff` 只含上表文件） | 提交 109b166 / 1f6a45a |

## §7 Limitations

1. **GGS_T5 出空间**：AutoCL Table 5 的 kernel 5/avg 在本实现里两开关全关时会给 instance 对比加多尺度项，
   与原文语义不同；只作文献对照，不能当"忠实复现"。
2. **单种子目录噪声地板**：`--catalog` 若跑，只有 seed 0；轨迹噪声地板 δ_nd≈0.030（memory iters 地板），
   ρ 只作方向性读数。
3. **官方 wrapper 与官方仓库的差异**：output_dims 320（wrapper 默认 256）、batch_size 8、无
   max_train_length（六靶 T≤1751<3000 不触发）、种子经 np/torch 全局设定；其余（binomial 掩码、
   repr_dropout、SWA、逐行裁剪、层级损失）逐字相同。
4. **TS2VEC_EQUIV 的已知残差**（不加开关）：损失取 mean 除以 2d−1 vs 官方 /d·α=0.5（有效 lr ×≈0.94）；
   `hierarchical_pool` ceil_mode vs 官方 floor；crop 在 fixed 模式下共享长度固定 0.5T、整批共用偏移。
5. **口径差**：zero-shot picks 来自源池切片口径（类别子集、Cricket crop 256 / Ethanol crop 1024、@290 bs128、
   split z-score、legacy SVM），靶评估是官方口径（原生长度、train-fit scaler、官方网格、200/600@8+SWA）。
   pick→靶 的口径迁移是 zero-shot 声明的一部分，但也意味着 R'@290 排名对官方口径的效度未被本任务证明
   （`--catalog` 的 Spearman 才回答）。
6. **zs_top3 主口径是 best-of-3 on test**（MX-2 `tclsc3` 同款上界）；cv 选择口径同表并列，两者之差就是
   "top-K 真评估选优"的乐观量。
7. **标签编码**：scaler=train 下在两切分并集上编码（官方在 TRAIN 唯一值上）；六靶两者一致（脚本断言）。
8. **跨机**：A2 是 4090/4060 vs 5090 锚；GPU 逐位不确定，±0.01 内视为一致（memory: 5090 run-to-run ±1 样本）。
9. **机器相关金标**：`test_swa` 与本任务的 loss-hex/参数-md5 金标在 pod（Linux BLAS）上失败——它们标定的是
   本机 CPU 的浮点结果；pod 上改用"默认路径 vs 显式默认参数"的差分自检（IDENTICAL）。
