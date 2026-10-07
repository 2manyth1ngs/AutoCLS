# WP18-VP PROTOCOL：输入视图（对比在哪个域上做）上限探针

**背景（用户 2026-10-07）**：主张不变（T-CLSC 全空间搜索 + 任务感知），其余设计全部可改；目标 = 部署报表数字（官方 TEST，超 TS2Vec / InfoTS_s）+ 消融可见；编码器搜索代价过大不做。
UT（11 个未见 UEA 靶）读数：ME 0.583、ENS_ME 0.592、TS2Vec 0.580、InfoTS_s 0.631；差距集中在 EEG / MEG 类靶（FingerMovements、MotorImagery、HandMovementDirection、SCP2）。
十几轮实验说明 AutoCL 空间里数据集级的偏好弱（目录内任务感知余量 ≈ 0.01–0.03），E 在部署层看不见。

**假设**：“在哪个域上做对比”（时域 / 差分 / 频域 / 时频，TF-C、CoST、BTSF 一类）是 AutoCL 空间缺的一维；
它的好坏强烈依赖数据集（节律 / 平稳信号偏频域，形状类偏时域）→ ① 在 EEG 类靶上抬精度；② 给 E 一个大且数据集级的可学对象，消融可见。

## §0 已看过的东西（如实声明）
- UT / UT-ENS / AL 三轮已读过 11 个靶的 TEST（ME 选中配置的 TEST、并集候选的种子 0 TEST）。本探针**不读这 11 个靶的 TEST**，只用官方 TRAIN 内分层 3 折。
- 10 个 UEA 源的官方 TEST 历史上多次读过（WP3 / WP10 / S3 系列），本探针照常读。
- 冒烟（服务器，4 格）：BasicMotions tcv0 psd / spec 1.000；SWJ tcv0 diff 0.25；MotorImagery tcv0 spec 0.51–0.55。

## §1 设计
- 视图（`autocls/data/views.py`，逐样本确定性变换，变换后用 TRAIN 统计量逐通道标准化）：
  - raw；diff = raw ⊕ 一阶差分（2C 通道）；psd = 去均值 Hann 窗 |rfft|²、去直流、池化到 ≤ 256 频点、log10，频率轴当序列轴；
  - spec = STFT 对数功率（窗 ≈ T/8，2 的幂 8..256，半重叠），每通道 ≤ 16 频带（C × 频带 ≤ 4096），帧轴当序列轴。
- CLS：ts2v（TS2VEC_EQUIV + 官方 encoder 输入掩码）、ggs（AutoCL GGS 本空间合法版）。
- 训练：ts2vec_official（bs = min(8, N) = 官方 DataLoader 口径、drop_last、SWA、lr 1e-3），官方迭代数按**原始** TRAIN 规模（200 / 600），一条轨迹跑到 4×，截点 1× / 2× / 4×。
- 读出：max（官方 full_series）/ maxavg（max ⊕ 时间均值）。评估：官方 SVM 网格。
- 数据集：10 个 UEA 源（官方 TEST）+ 11 个未见靶（官方 TRAIN 分层 3 折，折种子 run_seed_for(d, "vp_tcv")，三折按样本数加权）。单种子 0，共 344 格。

## §2 读数与判读带（先于数据）
主读数：1× / max 下，每个数据集“最好非 raw 视图 − raw”（两 CLS 均值）。
- **VP-GO**：≥ 4 / 21 个数据集增益 ≥ +0.05，且 |Δ| ≥ 0.03 的 (数据集, 视图) 两 CLS 同号比例 ≥ 0.7 → 视图升级为 CLS 扩展字段，补源标签、重训比较器、全流程部署。
- **VP-WEAK**：2–3 个数据集 ≥ +0.05 或同号比例 0.5–0.7 → 只在增益集中的域上考虑。
- **VP-FLAT**：≤ 1 个 → 视图不升级。
副读数（不进门）：raw 视图下 2× / 4× 对 1×、maxavg 对 max 的均值差（训练量 / 读出作为候选字段的依据）。

## §3 预测
| # | 预测 | 先验 |
|---|---|---|
| P1 | psd 或 spec 在 ≥ 2 个 EEG / MEG 类数据集（SCP1、SCP2、FingerMovements、MotorImagery、HMD）上 ≥ +0.05 | 45% |
| P2 | psd / spec 在形状类（Handwriting、UWave、AWR、Cricket）上 ≤ −0.05 | 70% |
| P3 | 读数码 VP-GO | 35% |
| P4 | 4× 对 1×（raw、两 CLS、21 集均值）> 0 | 40% |
| P5 | maxavg 对 max 均值 > 0 | 50% |
