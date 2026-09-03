## LODO pooled（126 实例）

| 臂 | 配对 acc | top-10 内 acc | Spearman | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|
| ridge_scalar | 0.773 | 0.461 | 0.411 | 0.085 | 0.035 |
| ridge_binned | 0.765 | 0.504 | 0.421 | 0.081 | 0.032 |
| hgb_scalar | 0.788 | 0.500 | 0.463 | 0.063 | 0.031 |
| hgb_binned | 0.780 | 0.508 | 0.448 | 0.056 | 0.033 |
| tclsc | 0.735 | 0.385 | 0.351 | 0.074 | 0.038 |
| random | — | — | — | 0.134 | 0.052 |

## 逐折配对 acc

| 折 | ridge_scalar | ridge_binned | hgb_scalar | hgb_binned | tclsc |
|---|---:|---:|---:|---:|---:|
| Cricket | 0.859 | 0.843 | 0.850 | 0.834 | 0.856 |
| EthanolConcentration | 0.504 | 0.541 | 0.503 | 0.539 | 0.459 |
| HandMovementDirection | 0.764 | 0.652 | 0.709 | 0.670 | 0.784 |
| Handwriting | 0.761 | 0.782 | 0.827 | 0.821 | 0.720 |
| PhonemeSpectra | 0.767 | 0.796 | 0.807 | 0.826 | 0.648 |
| UWaveGestureLibrary | 0.878 | 0.838 | 0.919 | 0.871 | 0.856 |
| UniMiB-SHAR | 0.861 | 0.867 | 0.865 | 0.861 | 0.816 |

## 逐预算（配对 acc / regret@1）

| 预算 | ridge_scalar | ridge_binned | hgb_scalar | hgb_binned | tclsc |
|---|---:|---:|---:|---:|---:|
| @60 | 0.742 / 0.075 | 0.722 / 0.070 | 0.737 / 0.045 | 0.729 / 0.045 | 0.699 / 0.071 |
| @120 | 0.805 / 0.080 | 0.806 / 0.079 | 0.835 / 0.057 | 0.833 / 0.053 | 0.770 / 0.073 |
| @290 | 0.773 / 0.098 | 0.768 / 0.095 | 0.792 / 0.086 | 0.779 / 0.071 | 0.736 / 0.079 |

## 字段主效应（ridge_binned 档内去均值系数的极差；hgb_binned 分组置换 ΔMSE）

| 字段 | ridge range | hgb ΔMSE | 最好档 → 最差档 |
|---|---:|---:|---|
| temperature | 0.908 | 0.2826 | 0.1 (+0.39) → 100.0 (-0.52) |
| freq_mask | 0.766 | 0.0833 | 0.95 (+0.35) → 0.8 (-0.41) |
| emb_jitter | 0.677 | 0.0387 | 0.0 (+0.31) → 0.8 (-0.37) |
| resize | 0.654 | 0.0436 | 0.8 (+0.39) → 0.0 (-0.26) |
| jitter | 0.596 | 0.0485 | 0.6 (+0.34) → 0.4 (-0.25) |
| point_mask | 0.534 | 0.0376 | 0.7 (+0.28) → 0.6 (-0.26) |
| sim_func | 0.478 | 0.1713 | dot (+0.19) → cosine (-0.28) |
| emb_mask | 0.433 | 0.0287 | 0.9 (+0.19) → 0.2 (-0.24) |
| norm | 0.414 | 0.0938 | none (+0.22) → l2 (-0.19) |
| rescale | 0.382 | 0.0318 | 0.0 (+0.19) → 0.95 (-0.20) |
| crop | 0.354 | 0.0204 | 0.3 (+0.19) → 0.95 (-0.17) |
| kernel | 0.342 | 0.0304 | 5 (+0.13) → 3 (-0.21) |
| order | 0.254 | 0.0339 | 3 (+0.10) → 1 (-0.15) |
| cross_scale | 0.133 | 0.0140 | True (-0.13) → True (-0.13) |
| adj_neighbor | 0.114 | 0.0069 | True (+0.11) → True (+0.11) |
| pool | 0.065 | 0.0146 | avg (+0.03) → max (-0.03) |
| loss_type | 0.018 | 0.0133 | triplet (+0.01) → info_nce (-0.01) |
| temporal | 0.015 | 0.0104 | True (+0.01) → True (+0.01) |
| instance | 0.000 | 0.0000 | True (+0.00) → True (+0.00) |

## 目录外（进化 CLS，21 实例均值）

| 量 | 值 |
|---|---:|
| bias_plain | 0.523 |
| bias_lcb | 0.295 |
| bias_cat_plain | -0.035 |
| spearman_plain | 0.234 |
| spearman_lcb | 0.252 |
| spearman_hull | 0.175 |
| spearman_cat_only | 0.448 |
| hamming_evo_median | 8.143 |
| hamming_cat_median | 3.214 |
| std_evo_mean | 0.228 |
| std_cat_mean | 0.165 |
| frac_evo_in_hull | 0.000 |
| frac_cat_in_hull | 0.500 |
| evo_true_z_mean | 0.047 |
| evo_pred_z_mean | 0.570 |

## 门

- S1：hgb_binned 配对 acc 赢 T-CLSC 5/7 折 → ✓
- S2：regret@1 0.056 vs T-CLSC 0.074，p=7.33e-03（44 胜 / 34 负）→ ✓
- S3：Spearman 0.234 → LCB 0.252；高估 0.523 → 0.295 → ✗

读数码：**SUR-WIN**

## 预测对账

| 预测 | 内容 | 读数 | 命中 |
|---|---|---|---|
| P1 | hgb_binned pooled pair acc ∈ [0.76, 0.81] | {'value': 0.7802} | ✓ |
| P2 | 主效应 range top-3 ⊂ ACTIVE；惰性 5 字段分组重要性都在末 8 位 | {'top3': ['temperature', 'freq_mask', 'emb_jitter'], 'inert_positions': {'temporal': 16, 'cross_scale': 14, 'pool': 13, 'adj_neighbor': 17, 'loss_type': 15}} | ✗ |
| P3 | 进化行高估量 ≥ +0.5 且到训练集字段汉明距离中位 ≥ 4 | {'bias': 0.5228, 'hamming_median': 8.1429} | ✓ |
| P4 | LCB 修正后合并 Spearman 提升 < 0.10 | {'gain': 0.0174} | ✓ |
