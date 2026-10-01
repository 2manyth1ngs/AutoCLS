# S3-MD 部署口径读数（官方 TEST，3 折 CV 选 1，3 种子均值）

| 数据集 | ME（E 挑源） | UNI（均匀） | BLIND（S3-Z NEW_B） | S3-Z NEW（field E） | S3-Y NEW | ME 纯比较器 | ME 的 U 最好（种子 0） | TS2Vec | InfoTS_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Handwriting | 0.555 | 0.455 | 0.577 | 0.536 | 0.536 | 0.456 | 0.572 | 0.515 | 0.554 |
| PhonemeSpectra | 0.247 | 0.231 | 0.223 | 0.241 | 0.235 | 0.189 | 0.257 | 0.233 | 0.233 |
| UWaveGestureLibrary | 0.910 | 0.914 | 0.902 | 0.915 | 0.915 | 0.900 | 0.925 | 0.906 | 0.884 |
| Cricket | 0.977 | 0.977 | 0.977 | 0.931 | 0.968 | 0.977 | 1.000 | 0.972 | 1.000 |
| EthanolConcentration | 0.274 | 0.269 | 0.265 | 0.271 | 0.271 | 0.257 | 0.319 | 0.308 | 0.323 |
| SelfRegulationSCP1 | 0.837 | 0.846 | 0.882 | 0.874 | 0.874 | 0.736 | 0.836 | 0.812 | 0.887 |
| **均值** | **0.6334** | 0.6154 | 0.6377 | 0.6279 | 0.6330 | 0.5860 | 0.6516 | 0.6243 | 0.6468 |

读数码：**S3MD-BEATS-TS2VEC+E**；预测 2/6

- CV 选择损失（种子 0）：ME +0.0186、UNI +0.0302；U 平均：ME 0.5835、UNI 0.5705
- ME 各折选中超参与 π 前 3：Handwr γ 0.75 τ 0.5 → PTB-XL 0.18/FordA 0.17/SleepEDF 0.13；Phonem γ 0.75 τ 1 → PTB-XL 0.10/FordA 0.09/SleepEDF 0.09；UWaveG γ 0.75 τ 1 → ShapesAl 0.13/NonInvas 0.10/SelfRegu 0.08；Cricke γ 0.75 τ 0.5 → SleepEDF 0.13/FordA 0.11/UniMiB-S 0.09；Ethano γ 0.75 τ 0.5 → ShapesAl 0.13/UWaveGes 0.11/NonInvas 0.10；SelfRe γ 0.75 τ 1 → Cricket 0.09/UWaveGes 0.07/EthanolC 0.07
