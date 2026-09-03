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

（§1–§7 由 `--analyze` 结果填写；§0 之后不改判线。）
