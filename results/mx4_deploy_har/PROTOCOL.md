# MX-4 WP3b 协议：UCI-HAR 外部锚上的"零样本名单 + 逐次减半"（本地 4060，零新标签）

日期：2026-09-04 ｜ 脚本：`scripts/39_sh_deploy.py --protocol legacy_full --datasets HAR --har-npz <main>/data/HAR/har.npz` ｜
产物目录 `results/mx4_deploy_har/` ｜ 用户原话："将 8 个所有的 R' 进入 CLS ranking 模块进行训练，用 HAR 作为目标集，看看它的 acc 和 F1"。

## 0. 定位与诚实声明
- HAR 不在源池（8 个数据集）里，是 CLAUDE.md 关键决策 5 保留的外部锚：`legacy_full` 口径（2320 iters @ bs 128，无 SWA，legacy SVM 网格）下
  GGS 复现 AutoCL Table 2（本栈 0.935 / 0.935 vs 论文 0.937 / 0.937，`results/m1_rerun/runA_bs128_iters2320.log`）。
- 先验 / 代理用**全部 8 个源数据集**的 R'（2835 行）拟合，不再做 LODO（目标不在池内，铁律 6 自动满足）。
- 探索读数：无（HAR 从未跑过 shared-10 目录）。已知：HAR 随机初始化基线高（切片协议下曾判贴顶），所有合理配置可能挤在 0.92–0.94。

## 1. 问题
| 问 | 内容 | 检验 |
|---|---|---|
| Q1 | 零样本名单 + 淘汰赛在 HAR 上选出的配置，acc / macro-F1 是否追平 GGS 锚（0.935）与 TS2Vec（0.930） | H1 / H2 |
| Q2 | 淘汰赛是否优于零样本 top-3 全跑 | H3 |
| Q3 | 目录 21 条里最好的能到哪（对 AutoCL 0.963 的距离，描述性） | 目录 oracle |

## 2. 口径
- 协议：`legacy_full`（configs/train.yaml：iters 2320、bs 128、lr 1e-3、无 drop_last、无 SWA、crop fixed、legacy SVM 网格 C∈{1,10,100} cv 3 于 ≤3000 分层子样本后全量拟合）；
  数据 `data/HAR/har.npz`（train 7352 / test 2947 × 128 × 9，z-score 只用 train 统计量，即 scaler=train）。
- 截点：2320 × {25%, 50%, 100%} = (580, 1160, 2320)；截点评估原始权重（无 SWA）。
- 候选 21：shared-10 + GGS + 变异提名 10（WP3 §2 同规则，父本 = 全 8 数据集 z 均值前 20，seed 0）。
- 排名臂：prior（shared + GGS 按全池先验）、hybrid（同 WP3）；无 T-CLSC 臂。
- 淘汰赛：SH6-3-3 主臂；对照 SH6-3-1、SH6-3（跳过 25%）、all@25%→top3、随机起点（20 次期望）。
- 种子：目录 seed 0；SH 冠军与 prior3 冠军补 seed 1、2（3 种子均值进头条）。
- 报 acc 与 macro-F1（`checkpoint_results` 两者都存）。

## 3. 判读（先写死）
| 门 | 检验 | 先验 |
|---|---|---|
| H1 | SH6-3-3 冠军 3 种子 acc ≥ 0.935 − 0.005 = 0.930（追平 GGS 锚） | 55% |
| H2 | 同上 ≥ TS2Vec 0.930（AutoCL Table 2） | 55% |
| H3 | seed 0：SH6-3-3 ≥ prior3 | 50% |
读数码：**HAR-WIN** = H1 ∧ H3；**HAR-PARITY** = H1 ∧ ¬H3；**HAR-DEAD** = ¬H1。描述性：目录 oracle 与 AutoCL 0.963 的差、GGS 本轮 seed 0 与锚 0.9348 的差（P4 型一致性）。

## 4. 预测
- P1：目录 21 条 100% 截点 acc 极差 < 0.06（贴顶靶，除 20be4f74 类破坏配置）（60%）。
- P2：GGS 本轮 seed 0 与 M1 锚 0.9348 差 ≤ 0.005（同机同口径，GPU 非确定性带内）（70%）。
- P3：冠军 ∈ {GGS, 9092e36e, e490b88e}（60%）。
- P4：25%→100% 排名 Spearman ≥ 0.6（长轨迹、大 N）（60%）。

## 5. 成本
M1 锚（4060）：2320 iters 167 s + 评估 21 s。每格 ≈ 167 + 3×21 ≈ 230 s；S1 21 格 ≈ 80 min；S2 ≤ 4 格 ≈ 16 min；合计 ≈ 1.7 h，零费用。
桌面 GPU 负载会让 4060 变慢且非确定（memory），跑时清负载。

## 6. 已知坑
- `EVAL_TIMEOUT_S=480` 沿用；legacy 网格无 C=∞，不预期超时。
- HAR 无 PA 参照、无切片 R'（W3 型读数不适用）；分析脚本对 HAR 走 H 门。
- `20be4f74` 在 HAR 上可能仍是垫底但不至于崩（大 N）。
