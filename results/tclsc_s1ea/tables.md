# WP16-TCLSC 阶段 1 响应画像 E tables（UCR 84 集，7 折，δ=0.06，252 实例）

读数码 **EA-DEAD**

## normal

| 臂 | M1 | M2 | M0 | Spearman | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|---:|
| u_ea_diff_hard | 0.592 | 0.760 | 0.780 | nan | 0.0451 | 0.0198 |
| u_ea_concat_hard | 0.604 | 0.754 | 0.774 | nan | 0.0602 | 0.0214 |
| u_eaphi_diff_hard | 0.607 | 0.771 | 0.794 | nan | 0.0458 | 0.0202 |
| u_blind_diff_hard | 0.602 | 0.752 | 0.772 | nan | 0.0611 | 0.0220 |
| u_phi_diff_hard | 0.611 | 0.760 | 0.782 | nan | 0.0609 | 0.0209 |

## 消融 Δ

| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|---|
| u_ea_diff_hard | cross | acc_top10 | 0.592 | 0.573 | +0.018 | 0.0976 | 44/34/8 |
| u_ea_diff_hard | cross | wacc | 0.760 | 0.740 | +0.020 | 1.17e-12 | 172/72/2 |
| u_ea_diff_hard | cross | pair_acc | 0.780 | 0.757 | +0.023 | 7.56e-10 | 142/82/14 |
| u_ea_concat_hard | cross | acc_top10 | 0.604 | 0.604 | +0.000 | 0.449 | 12/12/62 |
| u_ea_concat_hard | cross | wacc | 0.754 | 0.753 | +0.000 | 0.00405 | 107/73/66 |
| u_ea_concat_hard | cross | pair_acc | 0.774 | 0.773 | +0.001 | 0.00378 | 68/43/127 |
| u_eaphi_diff_hard | cross | acc_top10 | 0.607 | 0.567 | +0.041 | 0.00213 | 50/26/10 |
| u_eaphi_diff_hard | cross | wacc | 0.771 | 0.747 | +0.024 | 3.7e-18 | 176/65/5 |
| u_eaphi_diff_hard | cross | pair_acc | 0.794 | 0.765 | +0.028 | 8.84e-15 | 154/73/11 |
| u_ea_diff_hard | zero | acc_top10 | 0.592 | 0.571 | +0.021 | 0.192 | 33/25/28 |
| u_ea_diff_hard | zero | wacc | 0.760 | 0.743 | +0.018 | 3.6e-09 | 159/83/4 |
| u_ea_diff_hard | zero | pair_acc | 0.780 | 0.760 | +0.020 | 3.52e-08 | 125/61/52 |
| u_ea_concat_hard | zero | acc_top10 | 0.604 | 0.607 | -0.003 | 0.946 | 0/3/83 |
| u_ea_concat_hard | zero | wacc | 0.754 | 0.754 | +0.000 | 0.159 | 37/30/179 |
| u_ea_concat_hard | zero | pair_acc | 0.774 | 0.774 | +0.000 | 0.251 | 16/11/211 |
| u_eaphi_diff_hard | zero | acc_top10 | 0.607 | 0.583 | +0.024 | 0.0865 | 38/20/28 |
| u_eaphi_diff_hard | zero | wacc | 0.771 | 0.751 | +0.021 | 1.36e-09 | 156/86/4 |
| u_eaphi_diff_hard | zero | pair_acc | 0.794 | 0.769 | +0.025 | 1.36e-10 | 131/60/47 |

## 逐折 Δ_M1（cross）

- u_ea_diff_hard: fold0 -0.056, fold1 +0.002, fold2 +0.085, fold3 +0.125, fold4 -0.058, fold5 +0.110, fold6 -0.037
- u_ea_concat_hard: fold0 +0.001, fold1 +0.002, fold2 +0.006, fold3 +0.000, fold4 -0.009, fold5 +0.003, fold6 +0.000
- u_eaphi_diff_hard: fold0 -0.057, fold1 +0.003, fold2 +0.169, fold3 +0.063, fold4 -0.030, fold5 +0.132, fold6 +0.056

## 价值（regret@3 配对，H1 b < a）

- u_blind_diff_hard → u_ea_diff_hard: n=252 0.0220 → 0.0198 p=0.00881 胜/负/平 48/27/177
- u_phi_diff_hard → u_eaphi_diff_hard: n=252 0.0209 → 0.0202 p=0.0776 胜/负/平 40/25/187
- u_phi_diff_hard → u_ea_diff_hard: n=252 0.0209 → 0.0198 p=0.304 胜/负/平 38/29/185
- u_blind_diff_hard → u_eaphi_diff_hard: n=252 0.0220 → 0.0202 p=0.000322 胜/负/平 49/22/181
- u_blind_diff_hard → u_ea_concat_hard: n=252 0.0220 → 0.0214 p=0.0526 胜/负/平 27/13/212

## 逐预算 M0

- @60: u_ea_diff_hard 0.778, u_ea_concat_hard 0.773, u_eaphi_diff_hard 0.791, u_blind_diff_hard 0.773, u_phi_diff_hard 0.780
- @120: u_ea_diff_hard 0.776, u_ea_concat_hard 0.774, u_eaphi_diff_hard 0.789, u_blind_diff_hard 0.769, u_phi_diff_hard 0.778
- @290: u_ea_diff_hard 0.786, u_ea_concat_hard 0.775, u_eaphi_diff_hard 0.800, u_blind_diff_hard 0.774, u_phi_diff_hard 0.788

## G-A4 机制对齐（ret_f ↔ 字段偏好 u_f @290）

| 字段 | 列 | n | ρ | p | 对齐 |
|---|---|---:|---:|---:|---|
| jitter | ret_jitter | 80 | +0.073 | 0.52 | False |
| rescale | ret_rescale | 84 | -0.242 | 0.0266 | False |
| point_mask | ret_point_mask | 84 | -0.252 | 0.0208 | False |
| freq_mask | ret_freq_mask | 79 | -0.151 | 0.185 | False |

E_A nan 数据集：{}

## 门

| 门 | 值 | 过 |
|---|---|---|
| GA1 | {"arm": "u_ea_diff_hard", "delta": {"arm": "u_ea_diff_hard", "metric": "acc_top10", "n": 86, "normal": 0.5918270160986824, "cross": 0.5734224242575103, "mean_delta": 0.018404591841171946, "p": 0.09764771119552934, "wins": 44, "losses": 34, "ties": 8}} | False |
| GA2 | {"metric": "regret3", "blind": 0.02202857906502077, "ea": 0.019810246743025065, "p": 0.008805691313312035} | True |
| GA3 | {"metric": "regret3", "phi": 0.020926898926290304, "eaphi": 0.02017956430652682, "p": 0.07759560304087772} | False |
| GA5 | {"arm": "u_ea_concat_hard", "delta": {"arm": "u_ea_concat_hard", "metric": "acc_top10", "n": 86, "normal": 0.6044063469254776, "cross": 0.6040746441527869, "mean_delta": 0.000331702772690633, "p": 0.44878337825548653, "wins": 12, "losses": 12, "ties": 62}} | True |
| C0 u_blind_diff_hard | ref 0.7722080531433542 got 0.7722080531433542 | True |
| C0 u_phi_diff_hard | ref 0.7820906335497033 got 0.7820906335497033 | True |

## 预测对账

- P1 C0 逐位（参照臂 M0 ±0.001）: True
- P2 G-A1 过: False
- P3 G-A2 过: True
- P4 G-A3 过: False
- P5 G-A4 ≥ 2/4 字段对齐: False（0）
- P6 G-A5 concat 零: True
- P7 E_A 臂 M0 ≥ φ 臂 M0: False（[0.780207625822232, 0.7820906335497033]）
