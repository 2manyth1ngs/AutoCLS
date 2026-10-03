# XS 部署读数（官方 TEST，3 折 CV 选 1，3 种子均值；主干 = S3-Z NEW_B，源 17 → 17 + 4）

| 数据集 | **ME_new** | UNI_new | ME_old（S3-MD） | UNI_old | BLIND | TS2Vec | InfoTS_s | ME π 新源份额 | ME π 前 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Handwriting | 0.555 | 0.496 | 0.555 | 0.455 | 0.577 | 0.515 | 0.554 | 0.10 | PTB-XL 0.16 / FordA 0.15 / SleepEDF 0.12 |
| PhonemeSpectra | 0.238 | 0.229 | 0.247 | 0.231 | 0.223 | 0.233 | 0.233 | 0.11 | PTB-XL 0.13 / FordA 0.11 / SleepEDF 0.10 |
| UWaveGestureLibrary | 0.899 | 0.891 | 0.910 | 0.914 | 0.902 | 0.906 | 0.884 | 0.27 | ShapesAll 0.17 / Articulary 0.13 / NonInvasiv 0.11 |
| Cricket | 0.958 | 0.977 | 0.977 | 0.977 | 0.977 | 0.972 | 1.000 | 0.14 | SleepEDF 0.12 / FordA 0.10 / UniMiB-SHA 0.08 |
| EthanolConcentration | 0.279 | 0.274 | 0.274 | 0.269 | 0.265 | 0.308 | 0.323 | 0.25 | ShapesAll 0.10 / PEMS-SF 0.09 / UWaveGestu 0.08 |
| SelfRegulationSCP1 | 0.852 | 0.846 | 0.837 | 0.846 | 0.882 | 0.812 | 0.887 | 0.15 | Cricket 0.11 / SleepEDF 0.07 / UWaveGestu 0.07 |
| **均值** | **0.6302** | 0.6189 | 0.6334 | 0.6154 | 0.6377 | 0.6243 | 0.6468 | 0.17 | |

读数码：**XS-BEATS-TS2VEC+E**；预测 2/6

- 准入 4/6：['ArticularyWordRecognition', 'ERing', 'NATOPS', 'PEMS-SF']；选中 ['ArticularyWordRecognition', 'ERing', 'NATOPS', 'PEMS-SF']；8 行门后保留 ['ArticularyWordRecognition', 'ERing', 'NATOPS', 'PEMS-SF']
- 残差幅度（任务内 z − 主干，std 均值）：旧源 1.152、新源 1.299
- U 最好 / 平均（种子 0）：ME 0.6559 / 0.5820、UNI 0.6403 / 0.5768；CV 选择损失 ME +0.0324、UNI +0.0195
- ME 各折超参：Handwr γ 0.75 τ 0.5 α_w 1；Phonem γ 0.75 τ 0.5 α_w 1；UWaveG γ 0.75 τ 0.5 α_w 1；Cricke γ 0.75 τ 0.5 α_w 1；Ethano γ 0.75 τ 0.5 α_w 1；SelfRe γ 0.75 τ 0.5 α_w 1
