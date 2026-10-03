# S3-MD 部署口径读数（官方 TEST，3 折 CV 选 1，3 种子均值）

| 数据集 | ME（E 挑源） | UNI（均匀） | BLIND（S3-Z NEW_B） | S3-Z NEW（field E） | S3-Y NEW | ME 纯比较器 | ME 的 U 最好（种子 0） | TS2Vec | InfoTS_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Handwriting | 0.555 | 0.496 | 0.577 | 0.536 | 0.536 | 0.429 | 0.556 | 0.515 | 0.554 |
| PhonemeSpectra | 0.238 | 0.229 | 0.223 | 0.241 | 0.235 | 0.194 | 0.239 | 0.233 | 0.233 |
| UWaveGestureLibrary | 0.899 | 0.891 | 0.902 | 0.915 | 0.915 | 0.889 | 0.934 | 0.906 | 0.884 |
| Cricket | 0.958 | 0.977 | 0.977 | 0.931 | 0.968 | 0.949 | 1.000 | 0.972 | 1.000 |
| EthanolConcentration | 0.279 | 0.274 | 0.265 | 0.271 | 0.271 | 0.279 | 0.304 | 0.308 | 0.323 |
| SelfRegulationSCP1 | 0.852 | 0.846 | 0.882 | 0.874 | 0.874 | 0.857 | 0.901 | 0.812 | 0.887 |
| **均值** | **0.6302** | 0.6189 | 0.6377 | 0.6279 | 0.6330 | 0.5994 | 0.6559 | 0.6243 | 0.6468 |

读数码：**S3MD-BEATS-TS2VEC+E**；预测 2/6

- CV 选择损失（种子 0）：ME +0.0324、UNI +0.0195；U 平均：ME 0.5820、UNI 0.5768
- ME 各折选中超参与 π 前 3：Handwr γ 0.75 τ 0.5 → PTB-XL 0.16/FordA 0.15/SleepEDF 0.12；Phonem γ 0.75 τ 0.5 → PTB-XL 0.13/FordA 0.11/SleepEDF 0.10；UWaveG γ 0.75 τ 0.5 → ShapesAl 0.17/Articula 0.13/NonInvas 0.11；Cricke γ 0.75 τ 0.5 → SleepEDF 0.12/FordA 0.10/UniMiB-S 0.08；Ethano γ 0.75 τ 0.5 → ShapesAl 0.10/PEMS-SF 0.09/UWaveGes 0.08；SelfRe γ 0.75 τ 0.5 → Cricket 0.11/SleepEDF 0.07/UWaveGes 0.07
