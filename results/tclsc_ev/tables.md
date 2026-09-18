# EV1 检验 tables：源侧进化搜索验证（14 切片）

读数码 **EV-DEAD**（门 {'ev_et': {'G-V1': False, 'G-V2': True, 'G-V3': False}, 'ev_es': {'G-V1': False, 'G-V2': False, 'G-V3': False}, 'ev_blind': {'G-V1': False, 'G-V2': False}, 'rand_u': {'G-V1': False}, 'rand_nb': {'G-V1': False}}）

## @60（参照 = max(shared-10 k3, GGS k1)）

| 臂 | n | 正占比 | 超 δ 占比 | 中位 gap | mean best3 | mean mean3 | 预测超信任域占比 | 真超占比 | 超 δ 切片 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ev_et | 14 | 0.29 | 0.00 | -0.0076 | 0.7190 | 0.6915 | 1.00 | 0.12 | [] |
| ev_es | 4 | 0.25 | 0.00 | -0.0491 | 0.7140 | 0.6832 | 1.00 | 0.08 | [] |
| ev_blind | 14 | 0.36 | 0.00 | -0.0050 | 0.7197 | 0.6881 | 1.00 | 0.17 | [] |
| rand_u | 14 | 0.36 | 0.00 | -0.0120 | 0.7193 | 0.6367 | — | 0.12 | [] |
| rand_nb | 14 | 0.29 | 0.00 | -0.0039 | 0.7331 | 0.6812 | — | — | [] |

## @120（参照 = max(shared-10 k3, GGS k1)）

| 臂 | n | 正占比 | 超 δ 占比 | 中位 gap | mean best3 | mean mean3 | 预测超信任域占比 | 真超占比 | 超 δ 切片 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ev_et | 14 | 0.43 | 0.00 | -0.0050 | 0.7311 | 0.7023 | 1.00 | 0.21 | [] |
| ev_es | 4 | 0.00 | 0.00 | -0.0565 | 0.7207 | 0.6823 | 1.00 | 0.00 | [] |
| ev_blind | 14 | 0.29 | 0.00 | -0.0390 | 0.7118 | 0.6742 | 1.00 | 0.14 | [] |
| rand_u | 14 | 0.29 | 0.00 | 0.0000 | 0.7320 | 0.6472 | — | 0.12 | [] |
| rand_nb | 14 | 0.57 | 0.00 | 0.0044 | 0.7438 | 0.6925 | — | — | [] |

## @290（参照 = max(shared-10 k3, GGS k1)）

| 臂 | n | 正占比 | 超 δ 占比 | 中位 gap | mean best3 | mean mean3 | 预测超信任域占比 | 真超占比 | 超 δ 切片 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ev_et | 14 | 0.43 | 0.00 | -0.0058 | 0.7279 | 0.7004 | 1.00 | 0.19 | [] |
| ev_es | 4 | 0.00 | 0.00 | -0.1071 | 0.6717 | 0.6422 | 1.00 | 0.00 | [] |
| ev_blind | 14 | 0.36 | 0.07 | -0.0052 | 0.7387 | 0.6910 | 1.00 | 0.14 | ['HandMovementDirection_00'] |
| rand_u | 14 | 0.36 | 0.00 | -0.0111 | 0.7308 | 0.6410 | — | 0.12 | [] |
| rand_nb | 14 | 0.64 | 0.00 | 0.0057 | 0.7425 | 0.6910 | — | — | [] |

## 配对（@290，H1 = 前者更高）

| 对比 | n | 前者 | 后者 | Δ | p | 胜:负 |
|---|---:|---:|---:|---:|---:|---|
| ev_et_vs_rand_u_mean3 | 14 | 0.7004 | 0.6410 | 0.0594 | 0.018 | 11:3 |
| ev_et_vs_rand_u_best3 | 14 | 0.7279 | 0.7308 | -0.0029 | 0.692 | 6:8 |
| ev_et_vs_rand_nb_mean3 | 14 | 0.7004 | 0.6910 | 0.0094 | 0.195 | 8:6 |
| ev_es_vs_rand_u_mean3 | 4 | 0.6422 | 0.6960 | -0.0538 | 0.938 | 1:3 |
| ev_es_vs_rand_u_best3 | 4 | 0.6717 | 0.7856 | -0.1139 | 1.000 | 0:4 |
| ev_es_vs_rand_nb_mean3 | 4 | 0.6422 | 0.7032 | -0.0610 | 1.000 | 0:4 |
| ev_blind_vs_rand_u_mean3 | 14 | 0.6910 | 0.6410 | 0.0500 | 0.068 | 9:5 |
| ev_blind_vs_rand_u_best3 | 14 | 0.7387 | 0.7308 | 0.0079 | 0.350 | 8:5 |
| ev_blind_vs_rand_nb_mean3 | 14 | 0.6910 | 0.6910 | 0.0000 | 0.452 | 9:5 |
| ev_et_vs_blind_mean3 | 14 | 0.7004 | 0.6910 | 0.0094 | 0.271 | 7:7 |
| ev_et_vs_blind_best3 | 14 | 0.7279 | 0.7387 | -0.0108 | 0.593 | 5:7 |
| ev_es_vs_blind_mean3 | 4 | 0.6422 | 0.7126 | -0.0704 | 1.000 | 0:3 |
| ev_es_vs_blind_best3 | 4 | 0.6717 | 0.7922 | -0.1205 | 1.000 | 0:3 |
| ev_es_vs_ev_et_mean3 | 4 | 0.6422 | 0.7247 | -0.0825 | 1.000 | 0:4 |

高估（预测超信任域占比 − 真超占比）：{'ev_et': {'pred_above_share': 1.0, 'realized_above_share': 0.19047619047619047, 'over': 0.8095238095238095, 'G-V4': False}, 'ev_es': {'pred_above_share': 1.0, 'realized_above_share': 0.0, 'over': 1.0, 'G-V4': False}, 'ev_blind': {'pred_above_share': 1.0, 'realized_above_share': 0.14285714285714285, 'over': 0.8571428571428572}}

诊断：{'reused_rows': 0, 'n_rows': 411, 'proposals_in_nb_labeled': {'ev_et': 0, 'ev_es': 0, 'ev_blind': 0}, 'trust_in_top3_share': {'ev_et': 0.0, 'ev_es': 0.0, 'ev_blind': 0.0}, 'hamming_hist': {'ev_et': {'1': 0, '2': 3, '3': 39}, 'ev_es': {'1': 0, '2': 2, '3': 16}, 'ev_blind': {'1': 0, '2': 1, '3': 41}}, 'overlap_between_guides': {'ev_et&ev_es': 1, 'ev_et&ev_blind': 0, 'ev_es&ev_blind': 0}}

## NB1 考卷上 es_field_hard（同组种子）

es：{'regret3': 0.04973842980933641, 'regret1': 0.09902686563052854, 'M1': 0.5527647912035264, 'M0': 0.7429050846985366, 'n': 108, 'regret3_by_fold': {'Cricket': 0.031, 'EthanolConcentration': 0.0455, 'HandMovementDirection': 0.0728}, 'hit3': 0.24074074074074073}
NB1 同组：{'ft60_hard': {'regret3': 0.042359745661768824, 'regret1': 0.0834022525110755, 'M1': 0.4782197600379419, 'M0': 0.7479145358614786, 'n': 108, 'regret3_by_fold': {'Cricket': 0.0297, 'EthanolConcentration': 0.0268, 'HandMovementDirection': 0.0706}}, 'blind_diff_hard': {'regret3': 0.038278981782499825, 'regret1': 0.09193658349111813, 'M1': 0.5696906080305291, 'M0': 0.7433365289317568, 'n': 108, 'regret3_by_fold': {'Cricket': 0.0278, 'EthanolConcentration': 0.0368, 'HandMovementDirection': 0.0503}}}
配对：{'es_vs_ft60_hard': {'a': 'ft60_hard', 'b': 'es_field_hard', 'metric': 'regret3', 'n': 108, 'mean_a': 0.042359745661768824, 'mean_b': 0.04973842980933641, 'p': 0.9592681433518981, 'wins_b': 23, 'losses_b': 41, 'ties': 44}, 'es_vs_blind_diff_hard': {'a': 'blind_diff_hard', 'b': 'es_field_hard', 'metric': 'regret3', 'n': 108, 'mean_a': 0.038278981782499825, 'mean_b': 0.04973842980933641, 'p': 0.9990578962197746, 'wins_b': 17, 'losses_b': 34, 'ties': 57}}
消融：{'cross': {'n': 108, 'normal': 0.04973842980933641, 'cross': 0.0425076020594719, 'mean_delta': -0.007230827749864523, 'p': 0.9898344199919765, 'wins': 34, 'losses': 51}, 'zero': {'n': 108, 'normal': 0.04973842980933641, 'cross': 0.03790528093508895, 'mean_delta': -0.011833148874247464, 'p': 0.9769721897968533, 'wins': 24, 'losses': 38}}
E 诊断：{'EthanolConcentration': {'n_slices': 87, 'cos_within': 0.9514276534386735, 'cos_cross': 0.19315426205760794, 'norm_median': 39.0592402274294, 'norm_min': 21.695185319797115, 'norm_max': 77.14604182124525}, 'Cricket': {'n_slices': 87, 'cos_within': 0.9523348158907916, 'cos_cross': 0.11624510953001434, 'norm_median': 46.82534050508369, 'norm_min': 13.952464474525996, 'norm_max': 65.72141319714584}, 'HandMovementDirection': {'n_slices': 87, 'cos_within': 0.9477547909262106, 'cos_cross': 0.1744807632528746, 'norm_median': 33.307802280428476, 'norm_min': 8.760753917832847, 'norm_max': 55.74253264613634}}

预测：{'P1': {'desc': 'G-V1 ev_et', 'hit': False}, 'P2': {'desc': 'G-V2 ev_et', 'hit': True}, 'P3': {'desc': 'G-V3 ev_et > blind', 'hit': False}, 'P4': {'desc': 'ev_es mean3 ≤ ev_et mean3', 'hit': True}, 'P5': {'desc': '高估 blind − et ≥ 0.10', 'hit': False}, 'P6': {'desc': '超 δ 只在 HMD / Ethanol', 'value': ['HandMovementDirection'], 'hit': True}, 'P7': {'desc': '≥ 50% 切片最终种群前 3 含信任域（ev_et）', 'value': 0.0, 'hit': False}, 'es_gates': {'G-V1': False, 'G-V2': False, 'G-V3': False}}

