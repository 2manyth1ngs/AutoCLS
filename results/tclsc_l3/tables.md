# WP16-TCLSC-L3 阶段 0 tables（45 实例，S2 考侧）

读数码 **L3-HEAD-ONLY**

## normal（各切片自己的数据集 token）

| 臂 | M1 好半区 acc | M2 |Δz| 加权 acc | M0 δ 对 acc | Spearman | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|---:|
| l3_none_hard | 0.766 | 0.725 | 0.770 | 0.393 | 0.080 | 0.035 |
| l3_none_zdiff | 0.737 | 0.711 | 0.739 | 0.372 | 0.064 | 0.029 |
| l3_concat_hard | 0.780 | 0.721 | 0.769 | 0.388 | 0.084 | 0.034 |
| l3_concat_zdiff | 0.771 | 0.731 | 0.779 | 0.426 | 0.046 | 0.019 |
| l3_diff_hard | 0.821 | 0.747 | 0.788 | 0.438 | 0.062 | 0.027 |
| l3_diff_zdiff | 0.816 | 0.751 | 0.803 | 0.457 | 0.027 | 0.016 |
| l3_film_hard | 0.806 | 0.749 | 0.808 | 0.447 | 0.056 | 0.020 |
| l3_film_zdiff | 0.797 | 0.753 | 0.809 | 0.455 | 0.036 | 0.019 |

## 消融 Δ = normal − cross（数据集 token 错位，3 种子均值；Wilcoxon H1 normal > cross）

| 臂 | 指标 | normal | cross | Δ | p | 胜/负/平 |
|---|---|---:|---:|---:|---:|---|
| l3_concat_hard | acc_top10 | 0.780 | 0.760 | +0.020 | 0.5 | 1/2/14 |
| l3_concat_hard | wacc | 0.721 | 0.720 | +0.002 | 0.00878 | 23/12/10 |
| l3_concat_hard | pair_acc | 0.769 | 0.767 | +0.002 | 0.0156 | 8/3/31 |
| l3_concat_zdiff | acc_top10 | 0.771 | 0.583 | +0.187 | 0.0295 | 9/4/4 |
| l3_concat_zdiff | wacc | 0.731 | 0.604 | +0.127 | 5.01e-07 | 39/6/0 |
| l3_concat_zdiff | pair_acc | 0.779 | 0.606 | +0.173 | 1.01e-06 | 39/3/0 |
| l3_diff_hard | acc_top10 | 0.821 | 0.616 | +0.205 | 0.000931 | 12/2/3 |
| l3_diff_hard | wacc | 0.747 | 0.712 | +0.035 | 1.23e-06 | 38/7/0 |
| l3_diff_hard | pair_acc | 0.788 | 0.748 | +0.039 | 1.7e-05 | 35/4/3 |
| l3_diff_zdiff | acc_top10 | 0.816 | 0.627 | +0.189 | 0.0231 | 10/3/4 |
| l3_diff_zdiff | wacc | 0.751 | 0.630 | +0.122 | 1.76e-06 | 37/8/0 |
| l3_diff_zdiff | pair_acc | 0.803 | 0.633 | +0.170 | 7.39e-08 | 38/4/0 |
| l3_film_hard | acc_top10 | 0.806 | 0.705 | +0.101 | 0.0183 | 8/2/7 |
| l3_film_hard | wacc | 0.749 | 0.682 | +0.068 | 1.88e-08 | 39/6/0 |
| l3_film_hard | pair_acc | 0.808 | 0.709 | +0.098 | 1.3e-07 | 39/2/1 |
| l3_film_zdiff | acc_top10 | 0.797 | 0.666 | +0.130 | 0.0373 | 9/4/4 |
| l3_film_zdiff | wacc | 0.753 | 0.638 | +0.114 | 2.67e-07 | 40/5/0 |
| l3_film_zdiff | pair_acc | 0.809 | 0.645 | +0.163 | 4.38e-07 | 39/2/1 |

## 门

| 门 | 值 | 过 |
|---|---|---|
| G0 | {"arm": "l3_concat_hard", "delta": {"arm": "l3_concat_hard", "metric": "acc_top10", "n": 17, "normal": 0.7796651013346515, "cross": 0.7600572581973967, "mean_delta": 0.019607843137254898, "p": 0.5, "wins": 1, "losses": 2, "ties": 14}} | True |
| GA | {"arms": ["l3_diff_hard", "l3_film_hard"], "deltas": {"l3_diff_hard": {"arm": "l3_diff_hard", "metric": "acc_top10", "n": 17, "normal": 0.820853929807217, "cross": 0.6158242708848246, "mean_delta": 0.2050296589223925, "p": 0.000931423148990945, "wins": 12, "losses": 2, "ties": 3}, "l3_film_hard": {"arm": "l3_film_hard", "metric": "acc_top10", "n": 17, "normal": 0.8063457736035592, "cross": 0.7051855055747789, "mean_delta": 0.10116026802878014, "p": 0.01826799180276182, "wins": 8, "losses": 2, "ties": 7}}} | True |
| GB | {"arms": ["l3_diff_zdiff", "l3_film_zdiff"], "deltas": {"l3_diff_zdiff": {"arm": "l3_diff_zdiff", "metric": "acc_top10", "n": 17, "normal": 0.8161620530565168, "cross": 0.6272793431097929, "mean_delta": 0.1888827099467238, "p": 0.023133008736423395, "wins": 10, "losses": 3, "ties": 4}, "l3_film_zdiff": {"arm": "l3_film_zdiff", "metric": "acc_top10", "n": 17, "normal": 0.7967231010051079, "cross": 0.6662334541659801, "mean_delta": 0.1304896468391278, "p": 0.037323402459789684, "wins": 9, "losses": 4, "ties": 4}}} | True |
| GC | {"arm": "l3_concat_zdiff", "delta": {"arm": "l3_concat_zdiff", "metric": "acc_top10", "n": 17, "normal": 0.770590294941506, "cross": 0.5833532432580876, "mean_delta": 0.18723705168341848, "p": 0.02954729556003378, "wins": 9, "losses": 4, "ties": 4}} | True |

## 33 号 run json（δ 对，模型输出口径）：normal acc / cross 档 Δ̄

- l3_none_hard: normal 0.7732, cross Δ̄ None tier None, zero_ctx Δ̄ None
- l3_none_zdiff: normal 0.7347, cross Δ̄ None tier None, zero_ctx Δ̄ None
- l3_concat_hard: normal 0.7746, cross Δ̄ 0.0001 tier ['zero'], zero_ctx Δ̄ -0.0004
- l3_concat_zdiff: normal 0.7755, cross Δ̄ 0.108 tier ['pass'], zero_ctx Δ̄ 0.0671
- l3_diff_hard: normal 0.8066, cross Δ̄ 0.0396 tier ['pass'], zero_ctx Δ̄ 0.0437
- l3_diff_zdiff: normal 0.813, cross Δ̄ 0.1125 tier ['pass'], zero_ctx Δ̄ 0.0773
- l3_film_hard: normal 0.8056, cross Δ̄ 0.0682 tier ['pass'], zero_ctx Δ̄ 0.0519
- l3_film_zdiff: normal 0.8086, cross Δ̄ 0.106 tier ['pass'], zero_ctx Δ̄ 0.0661

## 预测对账

- P1 C0 过: True
- P2 G0 过（A8 在 binned 下仍零）: True
- P3 G-A 过: True
- P4 G-B 过: True
- P5 G-C 过: True
- P6 film Δ_M1 ≥ diff Δ_M1（zdiff）: False（[0.1304896468391278, 0.1888827099467238]）
- P7 zdiff 臂 M0 不比同头 hard 臂低 0.02 以上: False（{'none': -0.0311, 'concat': 0.0103, 'diff': 0.0148, 'film': 0.0012}）
- P8 任务盲 zdiff M1 ≥ hard + 0.02: False（-0.02878769154720706）
