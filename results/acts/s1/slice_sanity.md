# 切片分布 sanity（PROTOCOL §1.7-5）：子集云 vs 29 官方靶的 (C, L, n_train, 类别数)

子集 = 试点 24 + 全设计 128（8 集 × 16，零 GPU）；靶 = 29；云外靶（任一轴）23，逐轴 {'C': 9, 'L': 9, 'n': 14, 'k': 0}；只看试点：云外 25，逐轴 {'C': 10, 'L': 12, 'n': 14, 'k': 8}。

| 轴 | 子集云 [min, max]（试点+全设计） | 试点 [min, max] |
|---|---|---|
| C | 1 – 11 | 1 – 10 |
| L | 50 – 1676 | 51 – 1151 |
| n | 132 – 720 | 132 – 717 |
| k | 2 – 39 | 4 – 39 |

| 靶 | 类 | C | L | n_train | k | 云内(C/L/n/k) | 最近子集 log 距 |
|---|---|---:|---:|---:|---:|---|---:|
| DuckDuckGeese | C | 1345 | 270 | 50 | 5 | ✗✓✗✓ | 2.19 |
| PEMS-SF | E | 963 | 144 | 267 | 7 | ✗✓✓✓ | 2.02 |
| FaceDetection | D | 144 | 62 | 5890 | 2 | ✗✓✗✓ | 1.93 |
| PenDigits | A | 2 | 8 | 7494 | 10 | ✓✗✗✓ | 1.31 |
| StandWalkJump | C | 4 | 2500 | 12 | 3 | ✓✗✗✓ | 1.26 |
| EigenWorms | E | 6 | 17984 | 128 | 5 | ✓✗✗✓ | 1.23 |
| SpokenArabicDigits | A | 13 | 93 | 6599 | 10 | ✗✓✗✓ | 1.15 |
| AtrialFibrillation | C | 2 | 640 | 15 | 3 | ✓✓✗✓ | 1.12 |
| MotorImagery | D | 64 | 3000 | 278 | 2 | ✗✗✓✓ | 1.05 |
| Heartbeat | D | 61 | 405 | 204 | 2 | ✗✓✓✓ | 0.87 |
| ERing | E | 4 | 65 | 30 | 6 | ✓✓✗✓ | 0.77 |
| PhonemeSpectra | LODO | 11 | 217 | 3315 | 39 | ✓✓✗✓ | 0.67 |
| LSST | B | 6 | 36 | 2459 | 14 | ✓✗✗✓ | 0.67 |
| BasicMotions | A | 6 | 100 | 40 | 4 | ✓✓✗✓ | 0.62 |
| FingerMovements | D | 28 | 50 | 316 | 2 | ✗✓✓✓ | 0.61 |
| JapaneseVowels | A | 12 | 29 | 270 | 9 | ✗✗✓✓ | 0.49 |
| ArticularyWordRecognition | A | 9 | 144 | 275 | 25 | ✓✓✓✓ | 0.46 |
| NATOPS | E | 24 | 51 | 180 | 6 | ✗✓✓✓ | 0.42 |
| Libras | B | 2 | 45 | 180 | 15 | ✓✗✓✓ | 0.37 |
| Handwriting | LODO | 3 | 152 | 150 | 26 | ✓✓✓✓ | 0.37 |
| CharacterTrajectories | A | 3 | 182 | 1422 | 20 | ✓✓✗✓ | 0.34 |
| RacketSports | B | 6 | 30 | 151 | 4 | ✓✗✓✓ | 0.32 |
| UWaveGestureLibrary | LODO | 3 | 315 | 120 | 8 | ✓✓✗✓ | 0.19 |
| EthanolConcentration | LODO | 3 | 1751 | 261 | 4 | ✓✗✓✓ | 0.18 |
| Epilepsy | A | 3 | 206 | 137 | 4 | ✓✓✓✓ | 0.14 |
| SelfRegulationSCP1 | E | 6 | 896 | 268 | 2 | ✓✓✓✓ | 0.13 |
| SelfRegulationSCP2 | D | 7 | 1152 | 200 | 2 | ✓✓✓✓ | 0.12 |
| HandMovementDirection | LODO | 10 | 400 | 160 | 4 | ✓✓✓✓ | 0.10 |
| Cricket | LODO | 6 | 1197 | 108 | 12 | ✓✓✗✓ | 0.09 |
