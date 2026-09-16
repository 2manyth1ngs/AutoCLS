# 切法核对（--slice 实测，零 GPU）

| 切片 | k | 通道 | 样本 | n_train | 类别 Jaccard vs 原(max) | 样本 Jaccard vs 原(max) | 通道 Jaccard vs 原(max) | 类别子集重复原切片 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Handwriting_x00 | 9 | 2 | 337 | 236 | 0.31 | 0.33 | 1.00 | 否 |
| Handwriting_x01 | 8 | 3 | 310 | 217 | 0.33 | 0.34 | 1.00 | 否 |
| Handwriting_x02 | 14 | 3 | 528 | 369 | 0.35 | 0.34 | 1.00 | 否 |
| Handwriting_x03 | 9 | 3 | 351 | 246 | 0.38 | 0.37 | 1.00 | 否 |
| Handwriting_x04 | 11 | 3 | 428 | 300 | 0.35 | 0.35 | 1.00 | 否 |
| Handwriting_x05 | 14 | 3 | 562 | 393 | 0.44 | 0.45 | 1.00 | 否 |
| UniMiB-SHAR_x00 | 8 | 2 | 1024 | 717 | 0.40 | 0.05 | 1.00 | 否 |
| UniMiB-SHAR_x01 | 8 | 3 | 1024 | 716 | 0.62 | 0.07 | 1.00 | 否 |
| UniMiB-SHAR_x02 | 10 | 3 | 1024 | 717 | 0.62 | 0.07 | 1.00 | 否 |
| UniMiB-SHAR_x03 | 14 | 2 | 1024 | 717 | 0.75 | 0.06 | 1.00 | 否 |
| UniMiB-SHAR_x04 | 11 | 2 | 1024 | 716 | 0.71 | 0.06 | 1.00 | 否 |
| UniMiB-SHAR_x05 | 14 | 2 | 1024 | 715 | 0.69 | 0.06 | 1.00 | 否 |
| PhonemeSpectra_x00 | 5 | 7 | 854 | 599 | 0.10 | 0.10 | 0.60 | 否 |
| PhonemeSpectra_x01 | 5 | 7 | 855 | 600 | 0.30 | 0.26 | 0.56 | 否 |
| PhonemeSpectra_x02 | 8 | 8 | 1024 | 716 | 0.33 | 0.23 | 0.78 | 否 |
| PhonemeSpectra_x03 | 8 | 7 | 1024 | 718 | 0.23 | 0.16 | 0.55 | 否 |
| PhonemeSpectra_x04 | 8 | 10 | 1024 | 717 | 0.14 | 0.10 | 0.82 | 否 |
| PhonemeSpectra_x05 | 8 | 11 | 1024 | 717 | 0.23 | 0.17 | 0.91 | 否 |
| UWaveGestureLibrary_x00 | 5 | 2 | 275 | 195 | 0.71 | 0.71 | 1.00 | 否 |
| UWaveGestureLibrary_x01 | 5 | 2 | 275 | 195 | 0.71 | 0.71 | 1.00 | 否 |
| UWaveGestureLibrary_x02 | 4 | 2 | 220 | 156 | 0.80 | 0.80 | 1.00 | 否 |
| UWaveGestureLibrary_x03 | 6 | 2 | 330 | 234 | 0.75 | 0.75 | 1.00 | 否 |
| UWaveGestureLibrary_x04 | 5 | 2 | 275 | 195 | 0.71 | 0.71 | 1.00 | 否 |
| UWaveGestureLibrary_x05 | 6 | 2 | 330 | 234 | 0.75 | 0.75 | 1.00 | 否 |
| EthanolConcentration_x00 | 2 | 2 | 262 | 183 | 0.67 | 0.67 | 1.00 | 否 |
| EthanolConcentration_x01 | 3 | 2 | 393 | 275 | 0.75 | 0.75 | 1.00 | 否 |
| EthanolConcentration_x02 | 2 | 2 | 262 | 184 | 0.67 | 0.67 | 1.00 | 否 |
| EthanolConcentration_x03 | 2 | 2 | 263 | 184 | 0.67 | 0.67 | 1.00 | 否 |
| EthanolConcentration_x04 | 2 | 2 | 263 | 184 | 0.67 | 0.67 | 1.00 | 否 |
| EthanolConcentration_x05 | 4 | 3 | 524 | 367 | 1.00 | 1.00 | 1.00 | 是 |
| Cricket_x00 | 7 | 5 | 105 | 77 | 0.55 | 0.55 | 0.83 | 否 |
| Cricket_x01 | 7 | 4 | 105 | 77 | 0.70 | 0.70 | 0.67 | 否 |
| Cricket_x02 | 6 | 6 | 90 | 66 | 0.71 | 0.71 | 1.00 | 否 |
| Cricket_x03 | 8 | 6 | 120 | 88 | 0.70 | 0.70 | 1.00 | 否 |
| Cricket_x04 | 6 | 6 | 90 | 66 | 0.50 | 0.50 | 1.00 | 否 |
| Cricket_x05 | 10 | 6 | 150 | 110 | 0.73 | 0.73 | 1.00 | 否 |
| HandMovementDirection_x00 | 2 | 7 | 109 | 77 | 0.67 | 0.61 | 0.70 | 否 |
| HandMovementDirection_x01 | 3 | 9 | 164 | 116 | 0.75 | 0.70 | 0.90 | 否 |
| HandMovementDirection_x02 | 2 | 5 | 125 | 88 | 0.67 | 0.70 | 0.67 | 否 |
| HandMovementDirection_x03 | 2 | 8 | 110 | 78 | 0.67 | 0.61 | 0.80 | 否 |
| HandMovementDirection_x04 | 2 | 8 | 125 | 88 | 0.67 | 0.70 | 0.80 | 否 |
| HandMovementDirection_x05 | 4 | 6 | 234 | 165 | 1.00 | 1.00 | 0.60 | 是 |

原切片逐字段不变：True

| 数据集 | 新 vs 新 类别 Jaccard 均值 | 新 vs 新 样本 Jaccard 均值 | 类别子集重复原切片数 |
|---|---:|---:|---:|
| Handwriting | 0.29 | 0.29 | 0 |
| UniMiB-SHAR | 0.43 | 0.04 | 0 |
| PhonemeSpectra | 0.11 | 0.09 | 0 |
| UWaveGestureLibrary | 0.47 | 0.47 | 0 |
| EthanolConcentration | 0.42 | 0.42 | 1 |
| Cricket | 0.39 | 0.39 | 0 |
| HandMovementDirection | 0.42 | 0.41 | 1 |
