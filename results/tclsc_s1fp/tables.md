# WP16-TCLSC 阶段 1 转导指纹臂 tables（7 折 LODO）

读数码 **FP-DEAD**

## 视图 nonprobe（126 实例）— normal

| 臂 | M1 好半区 acc | M2 | M0 δ 对 acc | Spearman | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|---:|
| blind_diff_hard | 0.575 | 0.687 | 0.713 | 0.339 | 0.062 | 0.027 |
| fp_concat_hard | 0.588 | 0.682 | 0.705 | 0.333 | 0.073 | 0.034 |
| fp_diff_hard | 0.536 | 0.673 | 0.692 | 0.312 | 0.062 | 0.031 |

### nonprobe 消融 Δ

| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|---|
| fp_concat_hard | cross | acc_top10 | 0.588 | 0.580 | +0.008 | 0.0881 | 5/2/55 |
| fp_concat_hard | cross | wacc | 0.682 | 0.682 | -0.000 | 0.491 | 23/21/82 |
| fp_concat_hard | cross | pair_acc | 0.705 | 0.706 | -0.000 | 0.788 | 17/20/79 |
| fp_concat_hard | zero | acc_top10 | 0.588 | 0.591 | -0.003 | 0.706 | 2/3/57 |
| fp_concat_hard | zero | wacc | 0.682 | 0.683 | -0.001 | 0.98 | 11/22/93 |
| fp_concat_hard | zero | pair_acc | 0.705 | 0.707 | -0.001 | 0.961 | 2/10/104 |
| fp_diff_hard | cross | acc_top10 | 0.536 | 0.578 | -0.042 | 0.941 | 13/22/27 |
| fp_diff_hard | cross | wacc | 0.673 | 0.676 | -0.003 | 0.309 | 67/58/1 |
| fp_diff_hard | cross | pair_acc | 0.692 | 0.698 | -0.006 | 0.513 | 54/47/15 |
| fp_diff_hard | zero | acc_top10 | 0.536 | 0.572 | -0.036 | 0.889 | 9/17/36 |
| fp_diff_hard | zero | wacc | 0.673 | 0.681 | -0.007 | 0.5 | 56/59/11 |
| fp_diff_hard | zero | pair_acc | 0.692 | 0.703 | -0.010 | 0.747 | 34/33/49 |

### nonprobe 逐折 Δ_M1（cross）

- fp_concat_hard: Cricke +0.000, Ethano -0.000, HandMo -0.044, Handwr +0.026, Phonem +0.015, UWaveG +0.000, UniMiB +0.000
- fp_diff_hard: Cricke +0.023, Ethano -0.018, HandMo +0.000, Handwr -0.162, Phonem -0.068, UWaveG +0.167, UniMiB -0.025

### nonprobe 价值（regret@3 配对，H1 fp 臂更低）

- blind_diff_hard → fp_concat_hard: n=126 0.027142972253342546 → 0.03436284992167588 p=0.936311165015817 胜/负/平 14/15/97
- blind_diff_hard → fp_diff_hard: n=126 0.027142972253342546 → 0.0312568160650237 p=0.5102304102467433 胜/负/平 21/13/92

## 视图 all20（126 实例）— normal

| 臂 | M1 好半区 acc | M2 | M0 δ 对 acc | Spearman | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|---:|
| blind_diff_hard | 0.499 | 0.731 | 0.756 | 0.406 | 0.066 | 0.029 |
| fp_concat_hard | 0.507 | 0.730 | 0.758 | 0.406 | 0.074 | 0.029 |
| fp_diff_hard | 0.508 | 0.717 | 0.745 | 0.380 | 0.059 | 0.028 |

### all20 消融 Δ

| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|---|
| fp_concat_hard | cross | acc_top10 | 0.507 | 0.507 | +0.000 | 0.748 | 3/8/40 |
| fp_concat_hard | cross | wacc | 0.730 | 0.731 | -0.001 | 0.686 | 37/40/49 |
| fp_concat_hard | cross | pair_acc | 0.758 | 0.759 | -0.001 | 0.773 | 19/23/79 |
| fp_concat_hard | zero | acc_top10 | 0.507 | 0.507 | +0.000 | 0.5 | 3/3/45 |
| fp_concat_hard | zero | wacc | 0.730 | 0.732 | -0.002 | 0.933 | 26/32/68 |
| fp_concat_hard | zero | pair_acc | 0.758 | 0.759 | -0.001 | 0.937 | 7/12/102 |
| fp_diff_hard | cross | acc_top10 | 0.508 | 0.502 | +0.006 | 0.754 | 13/20/18 |
| fp_diff_hard | cross | wacc | 0.717 | 0.710 | +0.007 | 0.00469 | 81/44/1 |
| fp_diff_hard | cross | pair_acc | 0.745 | 0.728 | +0.017 | 0.011 | 68/42/11 |
| fp_diff_hard | zero | acc_top10 | 0.508 | 0.506 | +0.002 | 0.626 | 8/11/32 |
| fp_diff_hard | zero | wacc | 0.717 | 0.724 | -0.007 | 0.273 | 69/56/1 |
| fp_diff_hard | zero | pair_acc | 0.745 | 0.755 | -0.010 | 0.311 | 50/43/28 |

### all20 逐折 Δ_M1（cross）

- fp_concat_hard: Cricke +0.000, Ethano +0.000, HandMo -0.032, Handwr +0.000, Phonem +0.006, UWaveG +0.000, UniMiB +0.000
- fp_diff_hard: Cricke +0.048, Ethano -0.057, HandMo +0.091, Handwr -0.054, Phonem -0.043, UWaveG +0.444, UniMiB +0.000

### all20 价值（regret@3 配对，H1 fp 臂更低）

- blind_diff_hard → fp_concat_hard: n=126 0.029427115528154295 → 0.029043929173845205 p=0.291579966624073 胜/负/平 21/14/91
- blind_diff_hard → fp_diff_hard: n=126 0.029427115528154295 → 0.028214358851959068 p=0.23749866893868687 胜/负/平 31/19/76

## 同成本参照（all20，@120/@290）

- probe6_top3 regret@3 0.014790920288452601 vs fp_diff_hard 0.030089782837472486（blind 0.028483977687699037）n=84 p=0.9999469098956201 胜/负/平 14/36/34

## 门

| 门 | 值 | 过 |
|---|---|---|
| GF1 | {"arm": "fp_diff_hard", "view": "nonprobe", "delta": {"arm": "fp_diff_hard", "metric": "acc_top10", "n": 62, "normal": 0.536405529953917, "cross": 0.5782450076804916, "mean_delta": -0.0418394777265745, "p": 0.9411710741680978, "wins": 13, "losses": 22, "ties": 27}} | False |
| GF2 | {"view": "nonprobe", "metric": "regret3", "blind": 0.027142972253342546, "fp": 0.0312568160650237, "p": 0.5102304102467433} | False |
| GF3 | {"view": "all20@120/290", "metric": "regret3", "probe6_top3": 0.014790920288452601, "fp": 0.030089782837472486, "p": 0.9999469098956201} | False |
| GF4 | {"arm": "fp_concat_hard", "delta": {"arm": "fp_concat_hard", "metric": "acc_top10", "n": 62, "normal": 0.5877688172043011, "cross": 0.5802590032428742, "mean_delta": 0.007509813961426869, "p": 0.08814818722025558, "wins": 5, "losses": 2, "ties": 55}} | True |

## 探针（逐折）

- Handwriting: ['9092e36eefe1', 'a1f67db45e64', 'cffb55035c3a', '4f7d413581f6', 'f48e5da1d899', '8c4aa28336d6']
- UniMiB-SHAR: ['9092e36eefe1', 'a1f67db45e64', 'cffb55035c3a', '4f7d413581f6', '8c4aa28336d6', 'e490b88ee785']
- PhonemeSpectra: ['9092e36eefe1', 'a1f67db45e64', 'cffb55035c3a', '8c4aa28336d6', 'e490b88ee785', '4f7d413581f6']
- UWaveGestureLibrary: ['9092e36eefe1', 'a1f67db45e64', 'cffb55035c3a', '4f7d413581f6', '8c4aa28336d6', 'e490b88ee785']
- EthanolConcentration: ['9092e36eefe1', 'a1f67db45e64', '8c4aa28336d6', '4f7d413581f6', 'e490b88ee785', 'cffb55035c3a']
- Cricket: ['9092e36eefe1', 'cffb55035c3a', 'a1f67db45e64', '4f7d413581f6', '8c4aa28336d6', 'e490b88ee785']
- HandMovementDirection: ['9092e36eefe1', 'cffb55035c3a', 'a1f67db45e64', '8c4aa28336d6', '4f7d413581f6', 'e490b88ee785']

## 预测对账

- P1 C0 带内: True
- P2 G-F1 过: False
- P3 G-F2 过: False
- P4 G-F3 过: False
- P5 all20 Δ_M1 > nonprobe Δ_M1: True（[0.005631100049238824, -0.0418394777265745]）
- P6 fp 臂 M0（nonprobe）≥ φ 臂 M0 0.780: False（0.6923857093046869）
- P7 concat 对照 nonprobe Δ_M1 零: True
