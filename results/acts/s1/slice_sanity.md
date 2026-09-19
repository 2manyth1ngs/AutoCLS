# 切片分布 sanity（PROTOCOL §1.7-5）：子集云 vs 29 官方靶的 (C, L, n_train, 类别数)

子集 = 试点 24 + 全设计 128（8 集 × 16，零 GPU）；靶 = 29；云外靶（任一轴）19，逐轴 {'C': 9, 'L': 9, 'n': 8, 'k': 0}；只看试点：云外 22，逐轴 {'C': 9, 'L': 13, 'n': 8, 'k': 8}。

| 轴 | 子集云 [min, max]（试点+全设计） | 试点 [min, max] |
|---|---|---|
| C | 1 – 11 | 1 – 11 |
| L | 50 – 1714 | 61 – 1133 |
| n | 21 – 696 | 27 – 677 |
| k | 2 – 39 | 4 – 39 |

| 靶 | 类 | C | L | n_train | k | 云内(C/L/n/k) | 最近子集 log 距 |
|---|---|---:|---:|---:|---:|---|---:|
| DuckDuckGeese | C | 1345 | 270 | 50 | 5 | ✗✓✓✓ | 2.15 |
| PEMS-SF | E | 963 | 144 | 267 | 7 | ✗✓✓✓ | 2.12 |
| FaceDetection | D | 144 | 62 | 5890 | 2 | ✗✓✗✓ | 1.97 |
| PenDigits | A | 2 | 8 | 7494 | 10 | ✓✗✗✓ | 1.51 |
| MotorImagery | D | 64 | 3000 | 278 | 2 | ✗✗✓✓ | 1.25 |
| SpokenArabicDigits | A | 13 | 93 | 6599 | 10 | ✗✓✗✓ | 1.18 |
| EigenWorms | E | 6 | 17984 | 128 | 5 | ✓✗✓✓ | 1.16 |
| Heartbeat | D | 61 | 405 | 204 | 2 | ✗✓✓✓ | 0.98 |
| PhonemeSpectra | LODO | 11 | 217 | 3315 | 39 | ✓✓✗✓ | 0.82 |
| LSST | B | 6 | 36 | 2459 | 14 | ✓✗✗✓ | 0.79 |
| FingerMovements | D | 28 | 50 | 316 | 2 | ✗✓✓✓ | 0.72 |
| JapaneseVowels | A | 12 | 29 | 270 | 9 | ✗✗✓✓ | 0.63 |
| NATOPS | E | 24 | 51 | 180 | 6 | ✗✓✓✓ | 0.55 |
| StandWalkJump | C | 4 | 2500 | 12 | 3 | ✓✗✗✓ | 0.46 |
| RacketSports | B | 6 | 30 | 151 | 4 | ✓✗✓✓ | 0.44 |
| CharacterTrajectories | A | 3 | 182 | 1422 | 20 | ✓✓✗✓ | 0.41 |
| HandMovementDirection | LODO | 10 | 400 | 160 | 4 | ✓✓✓✓ | 0.37 |
| Cricket | LODO | 6 | 1197 | 108 | 12 | ✓✓✓✓ | 0.36 |
| SelfRegulationSCP2 | D | 7 | 1152 | 200 | 2 | ✓✓✓✓ | 0.35 |
| SelfRegulationSCP1 | E | 6 | 896 | 268 | 2 | ✓✓✓✓ | 0.32 |
| AtrialFibrillation | C | 2 | 640 | 15 | 3 | ✓✓✗✓ | 0.28 |
| EthanolConcentration | LODO | 3 | 1751 | 261 | 4 | ✓✗✓✓ | 0.26 |
| ArticularyWordRecognition | A | 9 | 144 | 275 | 25 | ✓✓✓✓ | 0.25 |
| ERing | E | 4 | 65 | 30 | 6 | ✓✓✓✓ | 0.23 |
| UWaveGestureLibrary | LODO | 3 | 315 | 120 | 8 | ✓✓✓✓ | 0.23 |
| BasicMotions | A | 6 | 100 | 40 | 4 | ✓✓✓✓ | 0.21 |
| Handwriting | LODO | 3 | 152 | 150 | 26 | ✓✓✓✓ | 0.19 |
| Epilepsy | A | 3 | 206 | 137 | 4 | ✓✓✓✓ | 0.15 |
| Libras | B | 2 | 45 | 180 | 15 | ✓✗✓✓ | 0.07 |
