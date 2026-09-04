# 阶段 0 简报（C0 复现 / oracle 漏网拆账 / C1 数据集身份上界 / 0b 交互定位）

- **C0** ✓ 逐位复现：n=126，max|Δ| prior3 0.0e+00, gbdt3 0.0e+00, hybrid3 0.0e+00, SH6-3-3_prior 0.0e+00；SH6-3-3_prior @290 均值 ours 0.018443 vs WP1 0.018443

## oracle ∉ hybrid 前 6（漏网拆账）

| 预算 | n | 漏网 | 来源 | 数据集 | 前 6 内最好 mean regret |
|---|---:|---:|---|---|---:|
| 60 | 42 | 24 | {'random': 18, 'shared': 6} | {'Cricket': 3, 'EthanolConcentration': 6, 'HandMovementDirection': 4, 'Handwriting': 4, 'PhonemeSpectra': 2, 'UWaveGestureLibrary': 2, 'UniMiB-SHAR': 3} | 0.0160 |
| pooled | 126 | 57 | {'random': 48, 'shared': 9} | {'Cricket': 7, 'EthanolConcentration': 15, 'HandMovementDirection': 9, 'Handwriting': 4, 'PhonemeSpectra': 8, 'UWaveGestureLibrary': 4, 'UniMiB-SHAR': 10} | 0.0121 |
| 120 | 42 | 17 | {'random': 15, 'shared': 2} | {'Cricket': 2, 'EthanolConcentration': 4, 'HandMovementDirection': 2, 'PhonemeSpectra': 3, 'UWaveGestureLibrary': 1, 'UniMiB-SHAR': 5} | 0.0086 |
| 290 | 42 | 16 | {'random': 15, 'shared': 1} | {'Cricket': 2, 'EthanolConcentration': 5, 'HandMovementDirection': 3, 'PhonemeSpectra': 3, 'UWaveGestureLibrary': 1, 'UniMiB-SHAR': 2} | 0.0118 |

## 臂表（pooled 126 / @290 42）

| 臂 | 入前6率 pooled | 入前6 @290 | regret@1 | regret@3 | 前6最好 | SH @290 mean / median |
|---|---:|---:|---:|---:|---:|---:|
| prior | 0.548 | 26/42 | 0.0465 | 0.0313 | 0.0121 | 0.0184 / 0.0000 |
| gbdt | 0.413 | 20/42 | 0.0631 | 0.0312 | 0.0195 | 0.0281 / 0.0127 |
| hybrid | 0.548 | 26/42 | 0.0465 | 0.0313 | 0.0121 | 0.0184 / 0.0000 |
| s0z | 0.413 | 19/42 | 0.0562 | 0.0300 | 0.0204 | 0.0325 / 0.0158 |
| c1_pure_z | 0.643 | 29/42 | 0.0311 | 0.0141 | 0.0081 | 0.0133 / 0.0000 |
| c1_pure_hyb | 0.611 | 26/42 | 0.0252 | 0.0165 | 0.0096 | 0.0181 / 0.0000 |
| c1_blend_z | 0.619 | 27/42 | 0.0471 | 0.0145 | 0.0084 | 0.0143 / 0.0000 |
| c1_blend_hyb | 0.587 | 26/42 | 0.0361 | 0.0167 | 0.0105 | 0.0191 / 0.0000 |

## 配对检验（H1: a < b）

| 对 | n | regret3 a/b | p | 胜/负 | 入前6 @290 a/b | SH@290 a/b | SH p |
|---|---:|---|---:|---|---|---|---:|
| s0z_vs_hybrid | 126 | 0.0300/0.0313 | 2.98e-01 | 18/16 | 19/26 | 0.0325/0.0184 | 9.99e-01 |
| c1_pure_z_vs_hybrid | 126 | 0.0141/0.0313 | 5.71e-08 | 59/21 | 29/26 | 0.0133/0.0184 | 2.32e-02 |
| c1_pure_hyb_vs_hybrid | 126 | 0.0165/0.0313 | 4.28e-07 | 51/19 | 26/26 | 0.0181/0.0184 | 5.54e-01 |
| c1_blend_z_vs_hybrid | 126 | 0.0145/0.0313 | 2.21e-09 | 50/10 | 27/26 | 0.0143/0.0184 | 1.04e-01 |
| c1_blend_hyb_vs_hybrid | 126 | 0.0167/0.0313 | 1.51e-08 | 46/8 | 26/26 | 0.0191/0.0184 | 7.04e-01 |
| c1_pure_z_vs_s0z | 126 | 0.0141/0.0300 | 1.23e-07 | 59/19 | 29/19 | 0.0133/0.0325 | 2.71e-04 |

## C1 参考线（事后核对）

`{"c1_pure_z": {"delta_in_top_290": 3, "delta_regret3": 0.017204806968670446, "regret3_p": 5.710377102526618e-08, "pass": true}, "c1_pure_hyb": {"delta_in_top_290": 0, "delta_regret3": 0.014777802604473193, "regret3_p": 4.284789275401827e-07, "pass": true}, "c1_blend_z": {"delta_in_top_290": 1, "delta_regret3": 0.01682009448490515, "regret3_p": 2.2147846439760666e-09, "pass": true}, "c1_blend_hyb": {"delta_in_top_290": 0, "delta_regret3": 0.01460114691549071, "regret3_p": 1.505874869000753e-08, "pass": true}, "any_pass": true}`

## 0b 交互定位

字段 ratio（跨数据集方差 / 切片 bootstrap 方差）排名：
cross_scale 6.04, pool 5.57, norm 4.02, emb_mask 3.82, sim_func 3.77, freq_mask 3.72, temperature 3.38, resize 3.36, point_mask 3.29, emb_jitter 3.22, rescale 2.98, adj_neighbor 2.73, temporal 2.55, crop 2.36, jitter 2.24, kernel 2.05, loss_type 1.77, order 1.73, instance 0.00

逐数据集 shared-10 z 均值 vs LODO 先验 Spearman：{"Cricket": 0.83, "EthanolConcentration": -0.055, "HandMovementDirection": 0.297, "Handwriting": 0.37, "PhonemeSpectra": 0.2, "SelfRegulationSCP2": 0.224, "UWaveGestureLibrary": 0.842, "UniMiB-SHAR": 0.867}

- @60：任务 0.821 / CLS 0.078 / 交互 0.101（交互中数据集级 0.677 / 切片级 0.323）；MS 数据集级交互 / σ² = 15.6，切片级 / σ² = 1.4，σ = 0.0355（160 格）
- @120：任务 0.813 / CLS 0.085 / 交互 0.102（交互中数据集级 0.655 / 切片级 0.345）；MS 数据集级交互 / σ² = 9.5，切片级 / σ² = 0.9，σ = 0.0464（160 格）
- @290：任务 0.726 / CLS 0.132 / 交互 0.143（交互中数据集级 0.786 / 切片级 0.214）；MS 数据集级交互 / σ² = 25.8，切片级 / σ² = 1.3，σ = 0.0370（160 格）
