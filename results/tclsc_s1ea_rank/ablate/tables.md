# R 检验 §9 消融：数据集特征对 field 头的贡献

## UEA n=7（126 实例）

| 臂 / 消融 | M1 | M0 | regret@1 | regret@3 |
|---|---:|---:|---:|---:|
| blind_diff_hard | 0.499 | 0.756 | 0.0657 | 0.0294 |
| phi_diff_hard | 0.499 | 0.780 | 0.0771 | 0.0311 |
| fw_eaphi_hard | 0.492 | 0.788 | 0.0758 | 0.0269 |
| fwb_hard | 0.516 | 0.754 | 0.0669 | 0.0304 |
| fw_eaphi_hard@zero_all | 0.522 | 0.775 | 0.0656 | 0.0301 |
| fw_eaphi_hard@noise0 | 0.546 | 0.775 | 0.0623 | 0.0289 |
| fw_eaphi_hard@noise1 | 0.487 | 0.768 | 0.0592 | 0.0318 |
| fw_eaphi_hard@noise2 | 0.493 | 0.762 | 0.0658 | 0.0294 |
| fw_eaphi_hard@noise | — | — | 0.0624 | 0.0300 |

### 配对检验

- structure: blind→fwb: n=126 0.0294 → 0.0304 p=0.61（H1 更低）胜/负/平 19/18/89
- features: fwb→fw: n=126 0.0304 → 0.0269 p=0.037（H1 更低）胜/负/平 40/23/63
- blind→fw: n=126 0.0294 → 0.0269 p=0.035（H1 更低）胜/负/平 36/21/69
- fw vs zero_all regret@3: 0.0269 → 0.0301 p_worse=0.0644 更差/更好/平 26/20/80
- fw vs noise regret@3: 0.0269 → 0.0300 p_worse=0.0357 更差/更好/平 43/27/56
- fw vs zero_all regret@1: 0.0758 → 0.0656 p_worse=0.964 更差/更好/平 26/37/63
- fw vs noise regret@1: 0.0758 → 0.0624 p_worse=0.993 更差/更好/平 33/50/43
- features: fwb→fw regret@1: n=126 0.0669 → 0.0758 p=0.901（H1 更低）胜/负/平 35/42/49
- structure: blind→fwb regret@1: n=126 0.0657 → 0.0669 p=0.692（H1 更低）胜/负/平 19/23/84

## UCR n=72（252 实例）

| 臂 / 消融 | M1 | M0 | regret@1 | regret@3 |
|---|---:|---:|---:|---:|
| u_blind_diff_hard | 0.602 | 0.772 | 0.0611 | 0.0220 |
| u_phi_diff_hard | 0.611 | 0.782 | 0.0609 | 0.0209 |
| u_fw_eaphi_hard | 0.646 | 0.794 | 0.0385 | 0.0183 |
| u_fwb_hard | 0.617 | 0.775 | 0.0508 | 0.0196 |
| u_fw_eaphi_hard@zero_all | 0.637 | 0.783 | 0.0407 | 0.0199 |
| u_fw_eaphi_hard@noise0 | 0.587 | 0.771 | 0.0398 | 0.0213 |
| u_fw_eaphi_hard@noise1 | 0.572 | 0.767 | 0.0510 | 0.0222 |
| u_fw_eaphi_hard@noise2 | 0.556 | 0.772 | 0.0436 | 0.0213 |
| u_fw_eaphi_hard@noise | — | — | 0.0448 | 0.0216 |

### 配对检验

- structure: blind→fwb: n=252 0.0220 → 0.0196 p=0.0039（H1 更低）胜/负/平 36/17/199
- features: fwb→fw: n=252 0.0196 → 0.0183 p=0.00809（H1 更低）胜/负/平 75/39/138
- blind→fw: n=252 0.0220 → 0.0183 p=5.72e-05（H1 更低）胜/负/平 85/38/129
- fw vs zero_all regret@3: 0.0183 → 0.0199 p_worse=0.0618 更差/更好/平 51/35/166
- fw vs noise regret@3: 0.0183 → 0.0216 p_worse=1e-05 更差/更好/平 110/41/101
- fw vs zero_all regret@1: 0.0385 → 0.0407 p_worse=0.108 更差/更好/平 76/64/112
- fw vs noise regret@1: 0.0385 → 0.0448 p_worse=0.000381 更差/更好/平 127/77/48
- features: fwb→fw regret@1: n=252 0.0508 → 0.0385 p=0.00368（H1 更低）胜/负/平 89/65/98
- structure: blind→fwb regret@1: n=252 0.0611 → 0.0508 p=0.0407（H1 更低）胜/负/平 54/38/160

