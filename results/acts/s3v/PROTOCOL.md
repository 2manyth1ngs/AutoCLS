# WP17-ACTS S3-V PROTOCOL：部署口径对表——AutoCTS++ 式消融 + 手工模型门

**会话定位**：
- 此前 T-CLSC 与 E 的过门都是"带 E vs 不带 E"的对比消融，没有和手工设计的方法比过精度。
- 用户 2026-09-28 拍板：改用 AutoCTS++ 的评估方式，在部署侧做一次简单的实验：
  - 完整框架 vs 它的三个消融变体（去掉 TS2Vec / 去掉 Set-Transformer / 去掉 shared 样本，= AutoCTS++ §4.2.3、Table 9–12）；
  - 与手工模型比精度；
  - 数据集 = 之前部署用过的 6 个（S3-C-D：HW / PS / UWave / Cricket / Ethanol / SCP1）；
  - **门 = 超过 TS2Vec 和 InfoTS_s**。

产物 `results/acts/s3v/`；代码 `scripts/108_acts_s3v.py`（部署：准备 / 原始窗口 / 搜索 / 选择 / 读数）+ `scripts/105_acts_s3r.py` 增 `--e-mode` 与 `--no-shared`（消融臂训练）+ `acts_common.whole_arrays_val` 与 `83` phase W 的 `val` 评估模式（只增不改）；单测 `tests/test_acts_s3v.py`；服务器 `scripts/pa_pod/acts_lab_run_s3v.sh`。

## §0 已看过的读数（如实声明）
- S3-C-D（同 6 集、官方口径，单种子、**按 test 在前 3 里选**，只作描述）：T-CLSC 搜索 0.605、任务盲比较器搜索 0.596、4 探针 0.650。
- S3-R′：带 E 模型样本内用上 E（+0.079），留出切片 regret@3 0.0120 vs 任务盲 0.0144。
- 文献数字（`results/acts/target_scan/`，PDF 原文解析）：

| 数据集 | TS2Vec（其原文） | InfoTS_s（其原文） |
|---|---|---|
| Handwriting | 0.515 | 0.554 |
| PhonemeSpectra | 0.233 | 0.233 |
| UWaveGestureLibrary | 0.906 | 0.884 |
| Cricket | 0.972 | 1.000 |
| EthanolConcentration | 0.308 | 0.323 |
| SelfRegulationSCP1 | 0.812 | 0.887 |
| **均值** | **0.6243** | **0.6468** |

  注：AutoTCL 论文 Table 12 在 UWave 上把 TS2Vec / InfoTS 记为 0.884 / 0.906，与两者原文相反，本协议一律取各方法自己论文的数字。
- **没看过**：任何按本协议（验证集选择、3 种子）得到的部署读数；三个消融臂的任何读数。

## §1 部署口径（对每个数据集 D）
- 官方 TRAIN / TEST；训练与评估沿用 `ts2vec_official`（TS2Vec 官方训练 + SVM，与 WP3 / S3-C-D 相同）。
- **比较器 = 留出 D 的那一折**（训练只见其余 16 个源），D 的标签从未参与训练。
- **E 只用 D 的官方 TRAIN**：TS2Vec 窗口表示沿用 S3-C-D 的 `E/<D>.npz`；原始窗口（去掉 TS2Vec 臂用）取同一批 100 个窗口。
- **搜索**（各臂相同）：3 个种子模型的概率取平均 → Ks 20,000（种子同 S3-C）→ 分块 500 取 10 → 进化 10 代（kp 10、p1 0.8、p2 0.2）→ 循环赛取前 3；结构性剪枝同 acts.yaml。
- **选择（= AutoCTS++ Algorithm 2）**：官方 TRAIN 按固定种子分层切出 20% 作 D_val。前 3 个配置各在其余 80% 上训练（种子 0），SVM 在 D_val 上出准确率，取最高者（平局取比较器排名靠前者）。
- **最终精度**：选中的配置在完整官方 TRAIN 上训练，种子 0 / 1 / 2 各一次，TEST 准确率取均值。

## §2 臂
| 臂 | 设计 | 模型 |
|---|---|---|
| **FULL** | E_pt（TS2Vec）+ Set-Transformer + concat 头；S3-R′ 训练制度（清洗标签、全配对、同任务留出配置验证、按验证准确率早停） | S3-R′ E 臂现成模型（每折 3 种子） |
| **w/o Set-Transformer** | E = TS2Vec 窗口表示的均值池化（按训练任务标准化），其余同 FULL | 新训 6 折 × 3 种子 |
| **w/o TS2Vec** | E = 原始窗口逐时间步 MLP（每通道共享 1 → 64 → 256，通道均值）再接同样的 Set-Transformer，端到端训练；其余同 FULL | 新训 6 折 × 3 种子 |
| **w/o shared** | 训练配对只用 random 配置之间的（shared-30 不进训练），其余同 FULL | 新训 6 折 × 3 种子 |
| 任务盲（描述） | 不输入 E（我们此前的消融） | S3-R′ B 臂现成模型 |

## §3 门与读数码
- **主门（FULL）**：6 个数据集的平均 TEST 准确率（3 种子均值）
  - 同时 > InfoTS_s 均值 0.6468 且 > TS2Vec 均值 0.6243 → **S3V-BEATS-BOTH**；
  - 只 > TS2Vec → **S3V-BEATS-TS2VEC**；
  - 否则 → **S3V-BELOW**。
- **后缀 +ABL**：FULL 的均值高于三个消融臂各自的均值（= AutoCTS++ "完整框架优于所有变体"的主张）。
- 描述项：
  - 逐数据集对 TS2Vec / InfoTS_s 的胜负；
  - 消融表（各臂逐数据集准确率与均值）；
  - 任务盲臂；
  - 选择损失（前 3 里 TEST 最好者 − 验证集选中者，种子 0）。
- 6 个数据集、3 个种子，不做显著性声明；臂间差 < 0.01 视为持平。

## §4 预测（先验）
| # | 预测 | 先验 |
|---|---|---|
| P1 | FULL 均值 > TS2Vec 0.6243 | 55% |
| P2 | FULL 均值 > InfoTS_s 0.6468（主门过） | 30% |
| P3 | +ABL | 30% |
| P4 | FULL 均值 > 任务盲臂 | 45% |
| P5 | 三个消融里 w/o shared 最差 | 35% |
| P6 | FULL 逐数据集胜 TS2Vec ≥ 4/6 | 45% |

## §5 成本与执行
- 服务器：原始窗口准备（CPU）→ 3 个消融臂 54 次训练 → 5 臂 × 6 集搜索 → 验证选择 ≤ 90 格 → 最终 ≤ 90 格（已有格复用）。服务器被占时预计 4–6 h。
- 本地 `--analyze` → REPORT。不追加新臂。

## §6 铁律对照
2（iters 按官方口径规则、截点同一轨迹）；3（逐格落盘 + resume）；5（先红、§0 声明）；6（比较器只见其余 16 源；E 与 D_val 都只来自官方 TRAIN；TEST 只用于最终报告）；7；8。

## §7 执行台账（跑后填写）
- 2026-09-28 实验室 8k-4090（GPU 0 / 4 / 6 / 1 / 5，10 槽）：
  - UTC 09:02 → 11:35，墙钟 2.55 h；
  - 54 次训练 + 30 次搜索 + 240 格（选择 90 + 最终新跑 150），verify OK、error 0、重试 0；
  - 服务器流水 a3fee92。
- 读数码 **S3V-BELOW**（无 +ABL），预测 1/6（P4 ✓）。
  - FULL 0.5987 vs TS2Vec 0.6243 / InfoTS_s 0.6468；
  - w/o Set-Transformer 0.5902 / w/o TS2Vec 0.5757 / w/o shared 0.6052 / 任务盲 0.5750。
- 事后描述（`posthoc_s3v.py`，不改门）：前 3 提名 TEST 上限与平均、排序层留出 regret@3。报告 §4 标为事后。
- 偏离：无。本地 `--analyze` 控制台编码报错，改用 `PYTHONIOENCODING=utf-8` 重跑，不影响读数。
