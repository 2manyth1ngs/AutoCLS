# 目标池预检协议：Epilepsy / FaceDetection 两尺度学习空间预检（判据先于数据）

日期：2026-08-12 ｜ 本机 RTX 4060 ｜ 用户拍板：四点确认全同意 + 顺序调整
（先排查一，坐实后预检）。本文件 commit 后判据/切分/预测逐字不改。

---

## 0. 排查一（前置闸，已完成）：UCI-HAR test 随机初始化基线

用户授权的第 1 次 test 触碰（`scripts/diag_har_test_random_baseline.py`，
`results/target_zero_shot/subject_protocol/test_random_baseline.json`）：

- 随机初始化 3 种子 test acc = **0.9131 / 0.9250 / 0.9169，均值 0.9183**
- GGS@2320 净增益：vs 同硬件锚 0.9348 → **+0.0165**；vs 原始锚 0.9372 → +0.0189
- 预注册分支（gain < 0.03 → 坐实）：**protocol_ok_verdict_confirmed**——
  实现/协议无问题，"UCI-HAR 不可评"终判直接验证坐实；test(0.918)−val(0.891)
  的差即受试者组难度差，与终判报告推论一致。**预检开闸。**

## 1. 范围与口径澄清

- 对象：目标池剩余两成员 **Epilepsy（UEA 版）** 与 **FaceDetection**。只跑预检，
  预检过了才谈完整 zero-shot 评估投入。
- ⚠️ 口径澄清（用户已拍板）：本项目 Epilepsy = **UEA Epilepsy**（275 例、3 通道
  腕部加速度计、L=206、4 类活动、6 参与者、HAR 域）。它**不是** AutoCL 原文用的
  UCI/Bonn Epileptic Seizure（11500×178 单通道 EEG，本地无数据）。
  "无受试者混杂/AutoCL 用过"的性质属于 Bonn 版，**不适用**于本对象——写报告时注意。
- 铁律 6：两数据集**首次触碰仅 train 侧**，官方 TEST 全程 0 触碰。预检不需要任务
  嵌入 E → 共享 TS2Vec 与 T-CLSC 完全不碰（连 load 都不需要）。

## 2. 结构调查（先于协议，实证依据）

`structure_probe.py` / `structure_probe2.py` + 四份 JSON（只读 TRAIN）：

| | Epilepsy | FaceDetection |
|---|---|---|
| 官方切分 | 跨参与者（3人/3人），**ID 已删除** | 跨受试者（subj 01-10 / 11-16），ID 删除 |
| 结构恢复 | **失败**：文件按类排序（4 类块），类内聚 k=3 得 22-28 run（成块应≈3）| **成功**：5890=10×589 恰好整除；589 网格 9 边界中 8 个跨界 z 分数 4.7-28.9，边界1 弱(1.46) 但整除+类占比+块纯度支持 |
| 孪生 | 逐次表演非滑窗；中位近邻余弦 0.51；**1 对完全重复 (#102,#104, cos=1.000)** + 1 对近重复 (#63,#65, 0.952) | 无（2000 抽样最大 0.727，0 对 >0.95）；类内随机交错 |

## 3. 切分协议（预注册）

| | 尺度1 任务级 | 尺度2 全量 val |
|---|---|---|
| **FaceDetection** | 任务内按受试者块（`idx//589`）**组切分**（`split_train_eval_by_group`，seed 由 task_id 派生，eval_ratio=0.3） | 留出块 **{8, 9}**（后两个受试者，n_val=1178） |
| **Epilepsy** | 分层随机切分（组切分实证不可行；canonical seed 同 UCI-HAR 首轮），eval_ratio=0.3 | 分层 80/20 × **5 个切分种子 {0..4} 取均值**（n_val=27 的 acc 量子 0.037 > 判据 0.03，必须多切分平均把分辨率压到 ~0.007；encoder 各档只训 1 次、SVM 按切分重评） |

- Epilepsy 预处理：**剔除 #104**（#102 的逐位重复，档案伪影；剔后 136 例）。代码
  fail-fast 校验该对余弦 >0.999 再剔，防文件顺序变化误伤。#63/#65（0.952）保留并记录。
- Epilepsy 泄漏方向声明（先写死）：受试者混杂只会抬高基线、压缩增益（UCI-HAR 机制）。
  若判"不可评"，结论在泄漏加持下成立，更强；若判"可评"，报告必须带
  "跨受试者部署口径未验证"警示。
- FaceDetection 组切分与官方 test（跨受试者）划分语义对齐，无此问题。

## 4. 判据（逐字，commit 后不改）

**尺度1 工作点标定**（判据与 UCI-HAR 两轮/源池探针完全同源，
`autocls.baseline_probe` 常数）：
- 网格 {1,2,5,10,20}ep × 10 shared CLS（探针目录前 10）× 前 2 任务 = 100 条 R'
- T1：探针任务基线（5 种子均值）≤ 0.90；T2：best_gain ≥ 0.05；
  T3：↑ ≥ ½ 且 ↑ ≥ 2×↓
- `select_workpoint`（margin=0.02）+ `no_workpoint_verdict`（量子=1/n_eval）

**尺度2 全量 GGS 增益诊断**：
- 随机初始化基线 3 种子（weight seed 0-2）；GGS（`c92058f2`，AutoCL Table 5）
  @10ep 与 @2320 iters（M1 anchor 同训练量）各 1 次（seed 0）
- gain = max(两档) − 随机基线均值（Epilepsy 一律先按切分取均值再算）
- **过闸线：gain ≥ +0.03**（理由独立于结果：UCI-HAR 的 +0.011/−0.004 在噪声内，
  对照 Handwriting best_gain +0.391；排查一实测 test 侧 +0.0165 同样定性为噪声带）

**三分判定**：
- **可评** = 尺度1 有工作点（T1+T2+T3 全过）且 尺度2 gain ≥ +0.03 → 值得投完整评估
- **边缘** = 单边过 → 报告细节，用户拍板
- **不可评** = 双空 → 不投评估，记入 limitation

## 5. 物料与工程纪律

- 切片：注册表唯一真源 `configs/source_pool.yaml`（本次补目标切片参数：
  Epilepsy `[3,4]` 类 ×8 任务、FaceDetection `[2,2]` ×8（allow_repeat 自动），
  通道子集 (⌈C/2⌉,C)，n_samples=1024/min_samples=32、slice seed=0）
- shared 10 CLS = `results/route_a_probe/probe_cls_catalog.json` 前 10（与源池
  label、UCI-HAR 标定逐键可比）；基线 = `inert_cls()` iters=0；GGS = m1 同源
- 训练：`configs/train.yaml` 的 batch_size/lr；iters 显式传参（铁律 2）；
  OMP=1 + threading 后端（loky 教训，勿删）
- 缓存：R' 逐条落盘 jsonl，键含 (dataset, task_id, cls_key, iters)，`--resume`
  幂等（铁律 3）；OOM 走 guarded_run 不写 null；device_name 100% 覆盖；
  种子幂等（run_seed_for 派生，与执行顺序无关）
- 泄漏自检：`load_source_dataset(目标)` 必须 TargetPoolLeakageError fail-fast；
  TEST 文件 0 读取（加载函数只认 `*_TRAIN.ts`）

## 6. 预测（先于数据；沿用"预测区间不预测点"传统）

| # | 预测 | 区间/先验 |
|---|---|---|
| P1 | Epilepsy 探针基线（分层口径） | [0.80, 0.98]，先验偏贴顶（文献强分类器 ~0.99、TS2Vec ~0.96；随机膨胀 CNN 特征在 HAR 域一贯接近协议天花板） |
| P2 | Epilepsy 情景 | 不可评(贴顶) 50% / 边缘 30% / 可评 20% |
| P3 | FaceDetection 探针基线（组切分口径） | [0.50, 0.65]（跨受试者 MEG 单 trial 二分类，DecMeg 冠军才 ~0.75；随机特征应显著低于它）——T1 大概率过 |
| P4 | FaceDetection 情景 | 不可评(学不动，全档≈chance) 45% / 边缘 30% / 可评 25% |
| P5 | 若两个都不可评 | 机制应可归因：Epilepsy→贴顶（源池同构）、FD→学不动（新机制，源池未见过 chance 型） |

## 7. 成本预算（4060 锚点：UCI-HAR prep 4-6 / 标定 13-15 / 全量诊断 4 min；L² 律）

| 阶段 | Epilepsy (T=206, ×2.6/iter) | FaceDetection (T=62, ×0.28/iter) |
|---|---:|---:|
| prep（8 任务 + 40 基线） | ~3 min | ~4-6 min |
| 标定 100 条 | ~8-10 min | ~10-12 min |
| 全量诊断 | ~8-10 min（2320 iters@T206 ≈6 min 支配） | ~4-6 min |
| **小计** | **~20-23 min** | **~18-24 min** |

合计 ~40-50 min GPU 墙钟。超预算 2× 即停下检查锚点（估时教训）。

## 8. 停机规则

- 每数据集独立判定，互不影响；一个数据集内尺度1 判 no_workpoint 不阻塞尺度2
  （尺度2 正是绕过切片伪影的独立证据，UCI-HAR 教训）。
- 两尺度跑完即出三分判定；**任何结果都不加跑测试集**——预检的输出只是"要不要投
  完整评估"的建议，投不投用户拍板。
