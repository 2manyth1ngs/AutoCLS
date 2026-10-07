# WP18-VP REPORT：输入视图（对比在哪个域上做）上限探针

**读数码：VP-GO（增益 ≥ +0.05 的数据集 6 个：ERing diff +0.080、Ethanol spec +0.061、DuckDuckGeese spec +0.060、Epilepsy spec +0.058、UWave diff +0.053、AtrialFibrillation psd +0.067〔CV 每折 5 样本，噪声级〕；两 CLS 同号比例 0.74 ≥ 0.7），预测 2/5。**
协议锁 283a87a；实验室 8k-4090，2026-10-07 12:04 → 约 13:45（他人约 100 个满 CPU 进程，格子 CPU 受限；中途去掉 nice、12 → 24 进程，均按 key 续跑，内容不变）。344 格，单种子 0。

## 1 主表（1× 官方迭代、max 读出，两 CLS 均值；源集 = 官方 TEST，未见靶 = TRAIN 内 3 折 CV）

完整表见 `tables.md`。要点：

| 数据集 | raw | diff | psd | spec | 最好非 raw − raw |
|---|---:|---:|---:|---:|---:|
| Handwriting | 0.542 | 0.398 | 0.149 | 0.124 | diff −0.144 |
| PhonemeSpectra | 0.222 | 0.261 | 0.203 | 0.221 | diff +0.038 |
| UWaveGestureLibrary | 0.872 | **0.925** | 0.527 | 0.514 | diff +0.053 |
| EthanolConcentration | 0.264 | 0.289 | 0.278 | **0.325** | spec +0.061 |
| SelfRegulationSCP1 | 0.787 | 0.756 | 0.701 | 0.677 | diff −0.031 |
| Heartbeat（CV） | 0.708 | 0.706 | **0.735** | 0.716 | psd +0.027 |
| MotorImagery（CV） | 0.545 | 0.520 | 0.522 | 0.473 | psd −0.023 |
| SelfRegulationSCP2（CV） | 0.570 | 0.532 | 0.508 | 0.482 | diff −0.038 |
| ERing | 0.804 | **0.883** | 0.422 | 0.680 | diff +0.080 |
| Epilepsy（CV） | 0.916 | 0.927 | 0.912 | **0.974** | spec +0.058 |
| DuckDuckGeese（CV） | 0.470 | 0.500 | 0.490 | **0.530** | spec +0.060 |

- 视图效应**强烈依赖数据集、两个方向都有**：diff 在 UWave / PS / AWR 上升、在 HW（−0.14）/ SCP1 / SCP2 上降；spec 只在 Ethanol 明显升；psd 在 Heartbeat 小升、在形状类（UWave、HW、AWR）大降。
- EEG 类靶（MotorImagery、FingerMovements、SCP2、HMD）上频域视图**没有**带来增益（原假设 P1 不成立）。
- 副读数：2× / 4× 训练量平均 −0.009（更长训练整体有害，仅 ERing / PS 小升）；maxavg 读出平均 −0.006（SCP1 +0.03 为孤例）→ 训练量 / 读出不升级为轴。

## 2 预测
| # | 预测 | 先验 | 结果 |
|---|---|---|---|
| P1 | psd / spec 在 ≥ 2 个 EEG 类上 ≥ +0.05 | 45% | ✗ |
| P2 | psd / spec 在形状类上 ≤ −0.05 | 70% | ✓（UWave / HW / AWR / Cricket） |
| P3 | VP-GO | 35% | ✓ |
| P4 | 4× 对 1× 均值 > 0 | 40% | ✗（−0.009） |
| P5 | maxavg 对 max 均值 > 0 | 50% | ✗（−0.006） |

## 3 读法与偏离
- 视图轴给出了 AutoCL 空间缺的东西：**数据集级、方向相反、量级 0.04–0.14** 的偏好（AutoCL 字段的数据集级偏好历来 ≤ 0.03）——这正是 E 能显形的对象；能抬精度的数据集约三分之一（UWave、PS、ERing、Ethanol、Epilepsy、DDG、Heartbeat）；其中 Epilepsy / DDG / Heartbeat 是未见靶（CV 读数，TEST 未读），E_V 地标在 Epilepsy（psd / spec +0.6）、Heartbeat（+0.35 / +0.42）上同向。
- 按协议 VP-GO → 视图升级为扩展轴（`configs/wp18.yaml`），补源标签、重训比较器、全流程部署（`results/acts/wp18/PROTOCOL.md`）。如实记：源侧新标签在 334/344 格时（当时的中间读数码是 VP-WEAK）已开跑；最后 10 格（ERing / NATOPS / BasicMotions / Epilepsy / DDG 的部分格）把读数码推到 GO，决定未因此改变。
- 视图地标 E_V（TRAIN 上 1-NN 类别保持）对 psd / spec 的方向多数对（UWave、Ethanol、SCP1），对 diff 不可靠（UWave 地标 −0.11、实测 +0.05）。
