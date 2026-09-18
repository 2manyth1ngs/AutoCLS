# EV1 检验 tables：源侧进化搜索验证（14 切片）

读数码 **EV-DEAD**（门 {'ev_et': {'G-V1': False, 'G-V2': True, 'G-V3': False}, 'ev_blind': {'G-V1': False, 'G-V2': False}, 'rand_u': {'G-V1': False}, 'rand_nb': {'G-V1': False}}）

## @60（参照 = max(shared-10 k3, GGS k1)）

| 臂 | n | 正占比 | 超 δ 占比 | 中位 gap | mean best3 | mean mean3 | 预测超信任域占比 | 真超占比 | 超 δ 切片 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ev_et | 14 | 0.29 | 0.00 | -0.0076 | 0.7190 | 0.6915 | 1.00 | 0.12 | [] |
| ev_blind | 14 | 0.36 | 0.00 | -0.0050 | 0.7197 | 0.6881 | 1.00 | 0.17 | [] |
| rand_u | 14 | 0.36 | 0.00 | -0.0120 | 0.7193 | 0.6367 | — | 0.12 | [] |
| rand_nb | 14 | 0.29 | 0.00 | -0.0039 | 0.7331 | 0.6812 | — | — | [] |

## @120（参照 = max(shared-10 k3, GGS k1)）

| 臂 | n | 正占比 | 超 δ 占比 | 中位 gap | mean best3 | mean mean3 | 预测超信任域占比 | 真超占比 | 超 δ 切片 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ev_et | 14 | 0.43 | 0.00 | -0.0050 | 0.7311 | 0.7023 | 1.00 | 0.21 | [] |
| ev_blind | 14 | 0.29 | 0.00 | -0.0390 | 0.7118 | 0.6742 | 1.00 | 0.14 | [] |
| rand_u | 14 | 0.29 | 0.00 | 0.0000 | 0.7320 | 0.6472 | — | 0.12 | [] |
| rand_nb | 14 | 0.57 | 0.00 | 0.0044 | 0.7438 | 0.6925 | — | — | [] |

## @290（参照 = max(shared-10 k3, GGS k1)）

| 臂 | n | 正占比 | 超 δ 占比 | 中位 gap | mean best3 | mean mean3 | 预测超信任域占比 | 真超占比 | 超 δ 切片 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ev_et | 14 | 0.43 | 0.00 | -0.0058 | 0.7279 | 0.7004 | 1.00 | 0.19 | [] |
| ev_blind | 14 | 0.36 | 0.07 | -0.0052 | 0.7387 | 0.6910 | 1.00 | 0.14 | ['HandMovementDirection_00'] |
| rand_u | 14 | 0.36 | 0.00 | -0.0111 | 0.7308 | 0.6410 | — | 0.12 | [] |
| rand_nb | 14 | 0.64 | 0.00 | 0.0057 | 0.7425 | 0.6910 | — | — | [] |

## 配对（@290，H1 = 前者更高）

| 对比 | n | 前者 | 后者 | Δ | p | 胜:负 |
|---|---:|---:|---:|---:|---:|---|
| ev_et_vs_rand_u_mean3 | 14 | 0.7004 | 0.6410 | 0.0594 | 0.018 | 11:3 |
| ev_et_vs_rand_u_best3 | 14 | 0.7279 | 0.7308 | -0.0029 | 0.692 | 6:8 |
| ev_et_vs_rand_nb_mean3 | 14 | 0.7004 | 0.6910 | 0.0094 | 0.195 | 8:6 |
| ev_blind_vs_rand_u_mean3 | 14 | 0.6910 | 0.6410 | 0.0500 | 0.068 | 9:5 |
| ev_blind_vs_rand_u_best3 | 14 | 0.7387 | 0.7308 | 0.0079 | 0.350 | 8:5 |
| ev_blind_vs_rand_nb_mean3 | 14 | 0.6910 | 0.6910 | 0.0000 | 0.452 | 9:5 |
| ev_et_vs_blind_mean3 | 14 | 0.7004 | 0.6910 | 0.0094 | 0.271 | 7:7 |
| ev_et_vs_blind_best3 | 14 | 0.7279 | 0.7387 | -0.0108 | 0.593 | 5:7 |

高估（预测超信任域占比 − 真超占比）：{'ev_et': {'pred_above_share': 1.0, 'realized_above_share': 0.19047619047619047, 'over': 0.8095238095238095, 'G-V4': False}, 'ev_blind': {'pred_above_share': 1.0, 'realized_above_share': 0.14285714285714285, 'over': 0.8571428571428572}}

诊断：{'reused_rows': 0, 'n_rows': 378, 'proposals_in_nb_labeled': {'ev_et': 0, 'ev_blind': 0}, 'trust_in_top3_share': {'ev_et': 0.0, 'ev_blind': 0.0}, 'hamming_hist': {'ev_et': {'1': 0, '2': 3, '3': 39}, 'ev_blind': {'1': 0, '2': 1, '3': 41}}, 'overlap_between_guides': {'ev_et&ev_blind': 0}}

预测：{'P1': {'desc': 'G-V1 ev_et', 'hit': False}, 'P2': {'desc': 'G-V2 ev_et', 'hit': True}, 'P3': {'desc': 'G-V3 ev_et > blind', 'hit': False}, 'P4': {'desc': 'ev_es mean3 ≤ ev_et mean3', 'hit': None}, 'P5': {'desc': '高估 blind − et ≥ 0.10', 'hit': False}, 'P6': {'desc': '超 δ 只在 HMD / Ethanol', 'value': ['HandMovementDirection'], 'hit': True}, 'P7': {'desc': '≥ 50% 切片最终种群前 3 含信任域（ev_et）', 'value': 0.0, 'hit': False}, 'es_gates': {}}

