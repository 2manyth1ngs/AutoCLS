# WP16-TCLSC 阶段 1 n 放大 tables（UCR 84 集，7 折，δ=0.06）

φ **UCR-PHI-DEAD** | 指纹 **UCR-FP-DEAD**

## 视图 all20（252 实例）— normal

| 臂 | M1 | M2 | M0 | Spearman | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|---:|
| u_blind_diff_hard | 0.602 | 0.752 | 0.772 | nan | 0.061 | 0.022 |
| u_phi_concat_hard | 0.580 | 0.748 | 0.769 | nan | 0.052 | 0.022 |
| u_phi_diff_hard | 0.611 | 0.760 | 0.782 | nan | 0.061 | 0.021 |
| u_fp_diff_hard | 0.670 | 0.773 | 0.795 | nan | 0.041 | 0.017 |

### all20 消融 Δ

| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|---|
| u_phi_concat_hard | cross | acc_top10 | 0.580 | 0.576 | +0.003 | 0.211 | 16/14/56 |
| u_phi_concat_hard | cross | wacc | 0.748 | 0.748 | -0.000 | 0.831 | 74/81/91 |
| u_phi_concat_hard | cross | pair_acc | 0.769 | 0.769 | +0.000 | 0.158 | 59/54/125 |
| u_phi_concat_hard | zero | acc_top10 | 0.580 | 0.580 | -0.000 | 0.567 | 3/4/79 |
| u_phi_concat_hard | zero | wacc | 0.748 | 0.748 | -0.000 | 0.923 | 18/36/192 |
| u_phi_concat_hard | zero | pair_acc | 0.769 | 0.769 | +0.000 | 0.541 | 14/14/210 |
| u_phi_diff_hard | cross | acc_top10 | 0.611 | 0.592 | +0.019 | 0.125 | 42/26/18 |
| u_phi_diff_hard | cross | wacc | 0.760 | 0.748 | +0.011 | 8.97e-07 | 153/90/3 |
| u_phi_diff_hard | cross | pair_acc | 0.782 | 0.769 | +0.013 | 8.94e-07 | 131/73/34 |
| u_phi_diff_hard | zero | acc_top10 | 0.611 | 0.601 | +0.010 | 0.228 | 30/17/39 |
| u_phi_diff_hard | zero | wacc | 0.760 | 0.749 | +0.011 | 3.37e-05 | 146/94/6 |
| u_phi_diff_hard | zero | pair_acc | 0.782 | 0.772 | +0.010 | 0.000284 | 109/66/63 |
| u_fp_diff_hard | cross | acc_top10 | 0.670 | 0.582 | +0.088 | 1.11e-06 | 57/21/8 |
| u_fp_diff_hard | cross | wacc | 0.773 | 0.745 | +0.028 | 3e-20 | 186/57/3 |
| u_fp_diff_hard | cross | pair_acc | 0.795 | 0.764 | +0.031 | 3.38e-20 | 165/57/16 |
| u_fp_diff_hard | zero | acc_top10 | 0.670 | 0.607 | +0.063 | 0.000154 | 42/11/33 |
| u_fp_diff_hard | zero | wacc | 0.773 | 0.750 | +0.022 | 7.76e-16 | 170/71/5 |
| u_fp_diff_hard | zero | pair_acc | 0.795 | 0.771 | +0.024 | 1.81e-13 | 132/54/52 |

### all20 逐折 Δ_M1（cross）

- u_phi_concat_hard: fold0 -0.005, fold1 +0.008, fold2 +0.016, fold3 +0.000, fold4 -0.008, fold5 +0.003, fold6 +0.012
- u_phi_diff_hard: fold0 +0.004, fold1 +0.037, fold2 +0.059, fold3 -0.059, fold4 -0.012, fold5 +0.067, fold6 +0.019
- u_fp_diff_hard: fold0 +0.103, fold1 +0.060, fold2 +0.086, fold3 +0.157, fold4 +0.027, fold5 +0.130, fold6 +0.083

### all20 价值（regret@3 配对，H1 条件臂更低）

- u_blind_diff_hard → u_phi_concat_hard: n=252 0.02202857906502077 → 0.02246349441794923 p=0.5618317122077624 胜/负/平 18/17/217
- u_blind_diff_hard → u_phi_diff_hard: n=252 0.02202857906502077 → 0.020926898926290304 p=0.04493205758140825 胜/负/平 44/27/181
- u_blind_diff_hard → u_fp_diff_hard: n=252 0.02202857906502077 → 0.016890424791902104 p=2.323253567530076e-08 胜/负/平 76/22/154

## 视图 nonprobe（252 实例）— normal

| 臂 | M1 | M2 | M0 | Spearman | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|---:|
| u_blind_diff_hard | 0.601 | 0.725 | 0.740 | nan | 0.075 | 0.021 |
| u_fp_diff_hard | 0.599 | 0.718 | 0.733 | nan | 0.066 | 0.021 |

### nonprobe 消融 Δ

| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|---|
| u_fp_diff_hard | cross | acc_top10 | 0.599 | 0.582 | +0.018 | 0.0666 | 35/31/32 |
| u_fp_diff_hard | cross | wacc | 0.718 | 0.716 | +0.003 | 0.109 | 126/116/4 |
| u_fp_diff_hard | cross | pair_acc | 0.733 | 0.725 | +0.008 | 0.00817 | 115/94/27 |
| u_fp_diff_hard | zero | acc_top10 | 0.599 | 0.571 | +0.028 | 0.105 | 25/16/57 |
| u_fp_diff_hard | zero | wacc | 0.718 | 0.717 | +0.001 | 0.359 | 118/112/16 |
| u_fp_diff_hard | zero | pair_acc | 0.733 | 0.730 | +0.003 | 0.173 | 78/76/82 |

### nonprobe 逐折 Δ_M1（cross）

- u_fp_diff_hard: fold0 +0.065, fold1 +0.006, fold2 +0.080, fold3 -0.026, fold4 -0.103, fold5 +0.042, fold6 +0.035

### nonprobe 价值（regret@3 配对，H1 条件臂更低）

- u_blind_diff_hard → u_fp_diff_hard: n=252 0.021040688947061542 → 0.020805502295224446 p=0.5407875590956586 胜/负/平 34/36/182

## 同成本参照（all20，@120/@290）

- probe6_top3 0.012519800078360207 vs u_fp_diff_hard 0.016522100437906313（blind 0.02155150283822468）n=168 p=0.9961899063403082

## 门

| 门 | 值 | 过 |
|---|---|---|
| GU1 | {"arm": "u_phi_diff_hard", "delta": {"arm": "u_phi_diff_hard", "metric": "acc_top10", "n": 86, "normal": 0.611414281066148, "cross": 0.592341696124621, "mean_delta": 0.019072584941527106, "p": 0.12531257993303085, "wins": 42, "losses": 26, "ties": 18}} | False |
| GU2 | {"metric": "regret3", "blind": 0.02202857906502077, "phi": 0.020926898926290304, "p": 0.04493205758140825} | True |
| GU6 | {"arm": "u_phi_concat_hard", "delta": {"arm": "u_phi_concat_hard", "metric": "acc_top10", "n": 86, "normal": 0.5795545023612426, "cross": 0.5764578141477354, "mean_delta": 0.003096688213507276, "p": 0.21079223114664108, "wins": 16, "losses": 14, "ties": 56}} | True |
| GU3 | {"arm": "u_fp_diff_hard", "view": "nonprobe", "delta": {"arm": "u_fp_diff_hard", "metric": "acc_top10", "n": 98, "normal": 0.599190633629409, "cross": 0.5816409308756247, "mean_delta": 0.017549702753784386, "p": 0.06663817675860685, "wins": 35, "losses": 31, "ties": 32}} | False |
| GU4 | {"view": "nonprobe", "metric": "regret3", "blind": 0.021040688947061542, "fp": 0.020805502295224446, "p": 0.5407875590956586} | False |
| GU5 | {"view": "all20@120/290", "probe6_top3": 0.012519800078360207, "fp": 0.016522100437906313, "p": 0.9961899063403082} | False |

## 预测对账

- P1 C0 带内: True
- P2 G-U1 过: False
- P3 G-U2 过: True
- P4 G-U3 过: False
- P5 G-U4 过: False
- P6 G-U5 过: False
- P7 指纹 nonprobe cross Δ_M1 由负转正: True（0.017549702753784386）
- P8 φ 臂 M0 cross Δ 仍显著（p<0.01）: True（8.936443114770513e-07）
