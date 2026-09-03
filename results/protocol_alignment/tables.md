# protocol_alignment tables（--analyze 自动生成）

## §1 评估审计：iters=0 随机 encoder 基线（各数据集 3 种子均值；2×2 = scaler × svm_grid）

| 数据集 | split/legacy | train/legacy | split/ts2vec | train/ts2vec | Δscaler(ts2vec) | Δgrid(train) |
|---|---:|---:|---:|---:|---:|---:|
| Handwriting | 0.251 | 0.251 | 0.251 | 0.251 | +0.000 | +0.000 |
| UWaveGestureLibrary | 0.668 | 0.668 | 0.668 | 0.668 | +0.000 | +0.000 |
| PhonemeSpectra | 0.202 | 0.203 | 0.201 | 0.203 | +0.002 | -0.000 |
| EthanolConcentration | 0.293 | 0.285 | 0.283 | 0.283 | -0.000 | -0.003 |
| Cricket | 0.968 | 0.968 | 0.968 | 0.968 | +0.000 | +0.000 |
| HandMovementDirection | 0.234 | 0.248 | 0.239 | 0.239 | +0.000 | -0.009 |

### A2 旧口径重跑（legacy_full，seed 0）vs MX-2 anchor_cache（5090）

| 臂 | 本机 acc | 本机 device | 5090 seed0 | 5090 3 种子均值 | Δ(本机−5090 s0) |
|---|---:|---|---:|---:|---:|
| champion | 0.3412 | NVIDIA GeForce RTX 4090 | 0.3412 | 0.3388 | +0.0000 |
| ggs | 0.4082 | NVIDIA GeForce RTX 4090 | 0.3918 | 0.3902 | +0.0165 |

## §2 复现：Handwriting × TS2VEC_EQUIV（3 种子 mean±std [各种子]）

| 格 | 协议 | swa | acc | f1 | 各种子 |
|---|---|---|---:|---:|---|
| repro_official | ts2vec_official | True | 0.4373±0.0194 | 0.3996 | 0.424, 0.424, 0.465 |
| repro_official_noswa | ts2vec_official | False | 0.4737±0.0097 | 0.4336 | 0.481, 0.460, 0.480 |
| repro_legacy | legacy_full | False | 0.3439±0.0059 | 0.3153 | 0.342, 0.338, 0.352 |
| repro_legacy_swa | legacy_full | True | 0.4122±0.0053 | 0.3818 | 0.405, 0.415, 0.416 |

归因：SWA@official -0.0365，SWA@legacy +0.0682，训练口径（无 SWA）+0.1298，合计 +0.0933

**P1**（official 三种子均值 ≥ 0.485 且每种子 ≥ 0.465）：不过

### bs 扫描：Handwriting × TS2VEC_EQUIV × 等遍（≈11 遍）× 3 种子

| bs | iters | acc | 各种子 |
|---:|---:|---:|---|
| 8 | 200 | 0.4357±0.0191 | 0.419, 0.426, 0.462 |
| 32 | 44 | 0.3682±0.0209 | 0.351, 0.356, 0.398 |
| 128 | 11 | 0.3122±0.0106 | 0.299, 0.313, 0.325 |

## §3 官方代码（vendored TS2Vec fit 循环）于本机

| 数据集 | 本机 official | 论文 TS2Vec | Δ | 各种子 |
|---|---:|---:|---:|---|
| Handwriting | 0.5286±0.0064 | 0.515 | +0.0136 | 0.531, 0.535, 0.520 |
| UWaveGestureLibrary | 0.9031±0.0092 | 0.906 | -0.0029 | 0.912, 0.891, 0.906 |
| PhonemeSpectra | 0.2348±0.0012 | 0.233 | +0.0018 | 0.233, 0.235, 0.236 |
| EthanolConcentration | 0.2877±0.0159 | 0.308 | -0.0203 | 0.304, 0.266, 0.293 |
| Cricket | 0.9769±0.0065 | 0.972 | +0.0049 | 0.972, 0.986, 0.972 |
| HandMovementDirection | 0.2658±0.0387 | 0.338 | -0.0722 | 0.311, 0.216, 0.270 |

**P1′**（官方 wrapper 于 Handwriting 同线）：过

## §2b 阶梯（在 ts2vec_official 上逐成分；采纳线 ≥ +0.03）

| 成分 | acc | Δ vs official | 采纳 |
|---|---:|---:|---|
| ladder_crop_ts2vec | 0.4114±0.0065 | -0.0259 | ✗ |
| ladder_input_mask | 0.5188±0.0092 | +0.0816 | ✓ |
| ladder_repr_dropout | 0.4851±0.0169 | +0.0478 | ✓ |
| ladder_all | 0.4980±0.0039 | +0.0608 | ✓ |

## §4 六靶（ts2vec_official 口径，3 种子均值；列 = 论文 / 官方代码本机 / 本栈各臂）

| 数据集 | TS2Vec 论文 | 官方本机 | TS2VEC_EQUIV | GGS | GGS_T5* | zs_top1 | zs_top3(best) | zs_top3(cv) | 判定(top3) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Handwriting | 0.515 | 0.529 | 0.442 | 0.532 | 0.484 | 0.389 | 0.433 | 0.433 | median |
| UWaveGestureLibrary | 0.906 | 0.903 | 0.750 | 0.869 | 0.881 | 0.907 | 0.907 | 0.907 | parity |
| PhonemeSpectra | 0.233 | 0.235 | 0.246 | 0.203 | 0.206 | 0.190 | 0.231 | 0.229 | parity |
| EthanolConcentration | 0.308 | 0.288 | 0.280 | 0.241 | 0.245 | 0.267 | 0.294 | 0.294 | parity |
| Cricket | 0.972 | 0.977 | 0.958 | 0.986 | 0.986 | 0.968 | 0.986 | 0.986 | parity |
| HandMovementDirection | 0.338 | 0.266 | 0.239 | 0.270 | 0.315 | 0.446 | 0.446 | 0.446 | parity |

*GGS_T5 出空间（space_legal=false），仅文献对照。
**P2**：zs_top3(best-of-3) ≥ TS2Vec−0.03 的数据集数 = 5/6（≥ 文献中位数：6/6；cv 选择口径 parity：5/6）

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
