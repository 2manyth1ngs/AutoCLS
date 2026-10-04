# S3-MD 部署口径读数（官方 TEST，3 折 CV 选 1，3 种子均值）

| 数据集 | ME（E 挑源） | UNI（均匀） | BLIND（S3-Z NEW_B） | S3-Z NEW（field E） | S3-Y NEW | ME 纯比较器 | ME 的 U 最好（种子 0） | TS2Vec | InfoTS_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Handwriting | 0.477 | 0.496 | 0.577 | 0.536 | 0.536 | 0.407 | 0.520 | 0.515 | 0.554 |
| PhonemeSpectra | 0.252 | 0.229 | 0.223 | 0.241 | 0.235 | 0.189 | 0.246 | 0.233 | 0.233 |
| UWaveGestureLibrary | 0.909 | 0.891 | 0.902 | 0.915 | 0.915 | 0.899 | 0.919 | 0.906 | 0.884 |
| Cricket | 0.991 | 0.977 | 0.977 | 0.931 | 0.968 | 0.940 | 0.986 | 0.972 | 1.000 |
| EthanolConcentration | 0.237 | 0.274 | 0.265 | 0.271 | 0.271 | 0.251 | 0.342 | 0.308 | 0.323 |
| SelfRegulationSCP1 | 0.887 | 0.846 | 0.882 | 0.874 | 0.874 | 0.643 | 0.891 | 0.812 | 0.887 |
| **均值** | **0.6257** | 0.6189 | 0.6377 | 0.6279 | 0.6330 | 0.5547 | 0.6507 | 0.6243 | 0.6468 |

读数码：**S3MD-BEATS-TS2VEC+E**；预测 1/6

- CV 选择损失（种子 0）：ME +0.0292、UNI +0.0195；U 平均：ME 0.5821、UNI 0.5768
- ME 各折选中超参与 π 前 3：Handwr γ 0.75 τ 0.5 → FordA 0.15/PTB-XL 0.13/UniMiB-S 0.10；Phonem γ 0.75 τ 1 → EthanolC 0.09/PTB-XL 0.09/SleepEDF 0.08；UWaveG γ 0.75 τ 1 → ShapesAl 0.09/Articula 0.09/ERing 0.08；Cricke γ 0.75 τ 1 → Articula 0.08/ShapesAl 0.08/PEMS-SF 0.08；Ethano γ 0.75 τ 0.5 → PhonemeS 0.15/SleepEDF 0.11/PTB-XL 0.08；SelfRe γ 0.75 τ 0.5 → PAMAP2 0.10/UWaveGes 0.10/Cricket 0.08
