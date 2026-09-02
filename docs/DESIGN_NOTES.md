# DESIGN_NOTES — 三大设计问题审计（2026-09-02）

范围：对项目当前三个最重要的问题做**设计层面**的查明，允许质疑 CLAUDE.md 的既定结论。
每个问题分三节：问题定义 → 已排除的假设（带判负数字与出处）→ 候选解法（含成本、先验、与既定铁律的冲突）。
本文**不改代码、不跑实验、不改 CLAUDE.md**；对既定结论的质疑集中在 §5，改动留待拍板。
所有 `results/...` 路径相对仓库根；"@iters" 为训练量口径（铁律 2 增补：任何增益/噪声数字必带口径）。

---

## 0. 摘要（三句话判词）

1. **label 噪声**：噪声地板的中位数（≈0.010）与任务信号中位数（0.007–0.013）同量级，真正把任务信号剃光的是 **δ 取 P95 的硬阈值**（弃样 55%，存活 label 任务盲天花板 0.9525）。已排除的 12 条假设全是"换测量/换训练"，从未试过"换 label 模型"（软 label / 分位下调 / 自适应 k）。另有一条从未被质疑的口径：训练预算是 TS2Vec 官方默认的 12–186×。
2. **目标数据集受限**：这是"数据集 × 评估栈"的联合性质，不是数据集属性；候选池漏斗最后只剩 HW/UM。未充分排除的杠杆有三条：few-shot 下游协议（曾被预注册的 AWR 必要条件一票否决）、靶单位改回全量数据集标准协议（先修外部锚口径）、数据集轴放大到 UCR 128（全库零讨论）。
3. **E 惰性**：官方因果链只有 label 侧一条。审计发现特征侧有三条从未进报告的证据：E 过随机 SetTF 后余弦 ≥0.9875（区分力压掉 95%，验收数字引用错层）、zero-E 与 shuffle-E 巨大不对称（`set_fc(E)` 是常数偏置）、McNemar 全是 b=0/c=0（无可检验）。两条因果任一成立都得到"消融为零"，现有实验无法区分，修任一条都不够。

---

## 1. 问题 1：源池 label 仍有大量噪声

### 1.1 问题定义（把"噪声"拆成可分别归因的量）

- **R' 的定义**：(task, cls) 上 k=1 单条轨迹（weight_seed 由 (task, cls) 派生）预训练 → 时间维 max-pool → SVM-RBF(C∈{1,10,100}, cv=3) → ACC；eval 为任务内 30% 分层随机划分，n_eval≈307，HW 任务仅 125–155。
  出处 `autocls/downstream/linear_eval.py`、`autocls/labeling.py::split_task_train_eval`、`results/eval_size_probe/REPORT.md` §3。
- **噪声结构**：同 CLS 跨 seed 重复差 |ΔR'| = 计数成分（∝ n_eval^−0.502）+ 轨迹地板；地板 med ≈ **0.010**、P95 ≈ **0.030@120 / 0.042@30**（随 iters 呈 U 形，240 掉头）；地板的 90–98% 是 cls×seed 交互而非 init 主效应。
  出处 `results/eval_size_probe/REPORT.md`、`results/step0_um_eval_expansion/REPORT.md` §3、`results/step05_iters_floor/REPORT.md` §1、`results/shared_seed_probe/REPORT.md` §3。
- **任务信号**：同一 shared 对在各任务上的少数方向票之 |ΔR'| 幅度，p50 = 0.0072@30 → 0.0128@120（UM 单数据集口径，k=3 + n_eval 2000 干净尺子）。
  出处 `results/step0_um_eval_expansion/REPORT.md` §5、`results/step05_iters_floor/REPORT.md` §1。
- **δ 机制与代价**：δ := max(P95(重复差), 2×量子)，工作点口径 HW 0.110 / UM 0.075。生产 label 5320 候选对 → **弃样 2929（55%），全部由 δ 造成（平局弃样 0）**；存活 2391 对的任务盲天花板 **0.9525**（HW 内 1.000、UM 0.989；δ=0 时 0.747）；strict 破坏参与 HW 32% / UM 92%。
  出处 `results/labels_workpoint/summary.json`、`results/decisive_e_ablation/REPORT.md` §5、`configs/labels.yaml: workpoint`。
- **成分**：随机采样 CLS 65% 为破坏性（手挑探针 40%）；破坏率随 iters 单调涨（@290 0.24 vs 工作点 0.07）。存活 label 主体 = "破坏 vs 非破坏"的平凡对。
  出处 `results/decouple_random_cls/REPORT.md`、`results/destruction_matrix/REPORT.md`。
- **表述纠偏**：CLAUDE.md 的"任务信号 0.007–0.013 < 噪声地板 0.030"比的是**信号 p50 与噪声 P95**。以中位对中位（≈0.010 vs ≈0.010），信号不是被淹没，而是 z≈0.7、单对判对率 ≈0.75 的弱信号；硬 δ=P95 把 0.6<P<0.97 整段丢弃。这正是 C2 判定"δ 剃掉任务信号"的精确机制，也是 §1.3 候选 1 的出发点。

### 1.2 已排除的假设（带判负数字）

| # | 假设 | 判负数字 | 出处 |
|---|---|---|---|
| 1 | 共享种子配对（common random numbers） | 四口径降幅 +10.4% / −26.7% / +2.4% / +2.1%，全 <20% 红线；seed 主效应 SS 份额 0.02；Var(g) 占噪声方差 2–10% | `results/shared_seed_probe/REPORT.md` |
| 2 | SWA 权重平均 | HW P95 0.110→0.077 但中位纹丝不动（0.0289）；信号同步 −23%；δ_on 存活 label 任务盲天花板 1.000 | `results/swa_probe/REPORT.md` |
| 3 | per-CLS δ / trimmed δ | 天花板 0.9276 不过 0.90 线；放宽版新增 label 三种子同向率仅 66.7%；per-CLS 噪声跨任务 ρ=−0.08 不可迁移 | `results/outlet_y/REPORT.md` |
| 4 | 扩容 n_eval | 307→2000 后 δ_nd 0.075→0.045 且已是 n→∞ 渐近值（地板 0.042）；HW 全集 1000 无余量 | `results/step0_um_eval_expansion/REPORT.md` |
| 5 | iters 往上（欠训练） | 2320 vs 290：HW 15/20 变差（+0.161→+0.093），WISDM 更负（−0.032） | `results/full_iters_probe/REPORT.md` |
| 6 | iters 重排工作点 | δ_nd U 形谷底 0.030@120 > 0.025 亮灯线，240 掉头；@120 可及任务特异票 3 票/6 任务 | `results/step05_iters_floor/REPORT.md` |
| 7 | 换 encoder 家族（transformer） | "四问全败、地板不降"——**无 REPORT / analysis.json，仅 CLAUDE.md 一句** | `results/step1_encoder_probe/`（仅 PROTOCOL + 缓存） |
| 8 | 换任务类型（forecasting） | Ridge 全程确定性仍 rel P95 29.8%（红线 15%）；噪声上游锁定预训练轨迹 | `results/forecast_noise_probe/REPORT.md` |
| 9 | 切片泄漏 | 泄漏真实（2.8–9.9% eval 有孪生）但归因抬升 −0.007…+0.005；T2/T3 是配对量、结构免疫 | `results/slicing_diagnosis/REPORT.md` |
| 10 | 换下游分类器 | RBF→linear 差 0.003（"分类器族是空旋钮"） | `results/protocol_probe/REPORT.md` |
| 11 | 扩源池救 task-aware | 新源同样过按噪声标定的 δ，存活部分同样任务盲（推论） | `results/decisive_e_ablation/REPORT.md` §5 |
| 12 | encoder 实现 bug / init 过强 | 与 TS2Vec 官方逐项一致（init 均为 PyTorch 默认）；深度 10→1 基线不动 | `results/encoder_ablation/REPORT.md` |

唯一未被枪毙的：k>1 平均本身（数学有效、"不经济"）；以及"换 R' 定义使其对表征扰动不敏感"（`forecast_noise_probe` §6.2 列为未探索）。

### 1.3 候选解法（按"零 GPU → 便宜 → 贵"排序）

1. **噪声感知软 label 取代硬 δ（零 GPU，最优先）**。对每个配对，用已标定的两成分噪声模型（per-dataset、per-制式、重尾用 Student-t 或经验分布）算 P(a≻b)；训练 T-CLSC 用 soft BCE 或按 |2P−1| 加权，**不弃样**。与所有已判负 δ 变体（per-CLS / trimmed / 两制式）的本质区别：那些都是硬阈值，本方案保留 0.6<P<0.97 这一段——恰是任务信号所在。判读：在既有 28 任务 5320 对缓存上重训，看 (i) 留出准确率、(ii) **shuffle-E 消融是否首次非零**、(iii) 信息加权口径下的任务盲天花板是否 <0.95。若 (ii) 仍为零，"task-aware 判死"的证据升级为"与 δ 无关"，同样有价值。
2. **δ 分位下调 + 显式弃样上限（零 GPU，方案 1 的保守版）**。P95 这个分位从未被论证；HW 中位 0.029 vs P95 0.110（3.8×）由重尾撑起。试 P90 / P80 并给 `make_pairs` 加弃样率上限（≤30%），用 `route_a_probe` §2.2 的 label 正确率口径对账。
3. **直接测"label 噪声 → 比较器精度"的响应曲线（零 GPU，补证据链缺口）**。向干净 label 注入 x% 翻转；或用 k=3 共识 vs k=1 canonical 两套 label 各训一个比较器。全项目至今只有天花板这个上界代理，没有这条曲线；它决定候选 1/2/4 值不值得投。
4. **自适应复制（adaptive k / racing）**。不是全表 k=9，而是只给 |Δ| 落在噪声带内的配对补种子（successive halving on seeds）。约 40–60% 的对在 k=1 已判定（破坏 vs 非破坏），其余补到 k≈6–9，均摊成本 ≈3–4× 而非 9×。与方案 1 组合：软 label 的后验方差决定谁补种子。这是"Y-B 加强版"的经济版。注意 required-k 随口径与目标分位变化很大：@120 容 p50 (0.030/0.0128)²≈5.5；@30 容 p50 (0.042/0.0072)²≈34；@120 容 p90 (0.030/0.0168)²≈3.2。CLAUDE.md 的"k≈6–9"在任何 REPORT 中无推导。
5. **重审训练预算口径（与铁律 2 正面冲突，须拍板）**。TS2Vec 官方 `fit()` 默认 200 iters × bs 16（类默认；`train.py` 为 bs 8），即 1.6k–3.2k 样本遍历（`reference/AutoCTS_plusplus/NAS_Net/ts2vec.py:21-22,77`）。本项目 early 290×128=37k（12–23×）、full 2320×128=297k（93–186×）；在 n_train≈150–700 的切片上 early 已是 48–97 epoch，full 对全量 HW（150 样本）≈1980 epoch。已知后果：破坏率随 iters 单调涨、@2320 是 HW 的破坏段、外部锚 0.339 vs 论文 0.515。**噪声地板不会因此消失**（step05 已在 5–40 epoch 扫过，谷底 0.030），但 label **成分**会变（HW 20ep strict 5% vs 40ep 22%）。候选：按官方口径重标噪声/破坏率/spread 三件套，3 数据集 × shared10 × k3 ≈ 200 格。
6. **换 R' 的估计量而非阈值**：R' := 同一轨迹多截点（60/120/240）ACC 均值。截点间高度相关，对 cls×seed 交互压制有限（SWA 只压尾部的教训），先验低；但 SVM 评估几乎免费，可随方案 3 零成本顺带测。
7. **训练动力学稳定化（lr / bs / EMA）**：lr=1e-3 从未扫过；分叉在 60 iters 内发生暗示步长偏大。属于用户 2026-08-13 已关闭的"训练侧便宜选项"家族，仅登记不推荐。

### 1.4 审计发现的证据链缺口（不改设计也应补）

- `results/step1_encoder_probe/` 无 REPORT / analysis.json："三重排除"的第三重只存在于 CLAUDE.md 一句话，无法复核；写论文前须补 `--analyze`。
- 核心结论句四个成分来自四个不同 (数据集×工作点)：β=0.502 在 HW@60、两成分分解在 UM@30、方差分解在 HW@60、0.030 在 UM@120。"0.030 是 cls×seed 交互"未在同一设定上验证。
- 任务信号幅度只在 UM 测过（HW 无扩容余量）；跨数据集任务信号从未干净测量，CLAUDE.md 却写成全栈结论。
- 任务盲天花板只能在 shared-shared 对上算（358/2391 = 15% 的 label）；random CLS 参与的 85% 靠"同分布"假设，而 random 比 shared 更破坏（65% vs 40%），假设方向不利。零 GPU 对照：只用 shared-shared label 训一个 T-CLSC 看是否逼近 0.9525。
- 环境敏感 CLS 集合（20be4f74 / 8b87d66f / a1f67db4 / 9092e36e）三轮有效性闸 FAIL 都归因环境却从未定位；若它们同时是 P95 尾部主力，δ 有部分是环境伪影。
- `docs/TECHNICAL_DESIGN.md` 中 δ 出现 0 次，仍写 `y=1 若 R'a>R'b`；整个噪声机制在正式设计文档里不存在。
- 切片重叠泄漏判为"应修的卫生缺陷"，`split_train_eval_by_group` 已实现但生产路径未接入。
- `results/outlet_y` 推荐的 k=3 变体 B（28 任务 480 格，~1–2.5h）从未执行；"不跑"的理由 k≈11 建立在已作废的 0.023 幅度上。

---

## 2. 问题 2：目标数据集选取受限

### 2.1 问题定义

- 端到端展示（MX-2）的合格靶只有 Handwriting（常规靶·强：可评 gap 0.117 / 净空 0.067 @290）与 UniMiB-SHAR（常规靶·弱：0.059 / 0.057，两条都刚过 0.05 线）；PhonemeSpectra / Cricket 只能当展示靶（净空 0.010 / 0.016）；UWave（gap 0.036）/ Ethanol（gap 0.049，差 0.001）out。
  出处 `results/source_pool_v2/analysis.json`、`results/source_pool_v2/PROTOCOL.md` §4。
- 原目标池 {UCI-HAR, Epilepsy(UEA 版), FaceDetection} 三成员全灭，死法各一：贴顶（跨受试者基线 0.916 > 0.90）/ 小 N 分辨率（n_eval 30–40，零假设带 ±0.05–0.067）/ chance 型学不动（0.51，全量 GGS 增益 +0.027 < 0.03）。三者共同点：GGS 全量增益全落 ±0.03 带内。
  出处 `results/target_preflight/REPORT.md` §4、`results/target_zero_shot/subject_protocol/REPORT.md`。
- 候选池筛选漏斗：UEA 28（除目标）→ 21 卡 N<1000、3 卡 L<50 → 4 过初筛 → CT/SAD 贴顶、PS T3 不过 → **只剩 HW**。27 个新下载候选 → 11 卡 K<5、6 贴顶、7 个 @290 全 inert（UM 后靠工作点 @30 翻盘）。总账：换池没有降低贴顶比例（数据集级 31%→48%）。
  出处 `results/candidate_survey/REPORT_STEP{1,2,3}.md`。
- 根因是**联合性质**而非数据集属性：可评性 = 数据集 × 评估栈（随机初始化 10 层膨胀 CNN + max-pool + SVM-RBF）。随机特征在多数 UEA 数据集上已线性可分（AWR 25 类 iters=0 即 0.973）；预训练可测增益带极窄。反面案例 WISDM2019（headroom 0.614、n_eval 307、51 受试者）给足条件后最好增益仍只 +0.074。

### 2.2 已排除的假设

| 假设 | 判负证据 | 出处 |
|---|---|---|
| 贴顶是切片太简单，加难类别档能救 | 全类别只降 0.011–0.040，AWR 25 类仍 0.973 | `results/baseline_probe/REPORT.md` §5 |
| 贴顶是下游分类器造成的 | B′ 只换核差 0.003；线性探针反把贴顶 5→6、HW 基线 0.555→0.644 | `results/protocol_probe/REPORT.md` §3.1–3.2 |
| 换数据集来源（UEA→UCI/PhysioNet 27 候选） | 0/7 过 T1+T2+T3；贴顶比例反升 | `results/candidate_survey/REPORT_STEP3.md` |
| 训练量不够 | 8× 训练量下 WISDM 更负，HW 反降 | `results/full_iters_probe/REPORT.md` |
| 切片泄漏抬高基线导致 inert | 泄漏只动水平（T1），T2/T3 免疫 | `results/slicing_diagnosis/REPORT.md` |
| encoder 实现有误 / init 过强 | 与 TS2Vec 官方一致；深度 10→1 基线不动 | `results/encoder_ablation/REPORT.md` |
| 换 encoder 家族能离顶 | 随机 attention 在贴顶数据集上更强；贴顶跨架构家族成立 | `results/step1_encoder_probe/`（无 REPORT） |
| 换下游任务类型（预测） | 噪声继承自预训练轨迹 | `results/forecast_noise_probe/REPORT.md` |
| reserve 回填 / 补 UEA | reserve 三个全域 pass 率 ≤0.002 | `results/probe_sensitivity/REPORT.md` |
| UCI-HAR 是切分泄漏伪影 | 跨受试者再判 no_workpoint；全量 GGS@2320 −0.004 | `results/target_zero_shot/subject_protocol/REPORT.md` |

### 2.3 未充分排除 / 从未讨论的候选解法

1. **few-shot 下游协议重开**。`protocol_probe` 的 C_k5（每类 5 训练样本）把贴顶数据集 5→1、离顶任务 55→90，是唯一咬得动天花板的旋钮；被判 no_effect 只因预注册把 AWR 设为必要条件（AWR 0.967 免疫）。few-shot 同时把更多样本让给 eval（计数噪声降），且"标签稀缺时预训练才有价值"是 SSL 文献的常规舞台。**未测**：few-shot 下训练后的 CLS spread 是否越 ±2σ 带（闸门 2 从未跑）；抽样方差用 10 次 SVM 重复（便宜）。成本：3 数据集 × 8 任务 × shared10 @290 重编码 ≈240 格 ≈1.5h @5090。冲突：偏离 AutoCL B.1 口径，须在论文声明为"标签稀缺协议"。`protocol_probe/REPORT.md` §5.1 自己也把它列为"须另行预注册后才可采纳"。
2. **靶的单位从"切片任务"改回"全量数据集标准协议"**。切片是为造 label 发明的，用它当考场是人工舞台；外部锚（全量 HW 官方 split）才是可与文献对表的靶。当前锚 0.339 vs 论文 0.515 的差距归因为口径不对等（2320@bs128 对 150 样本 ≈1980 epoch）。先按 TS2Vec 官方默认口径重跑锚；若对齐，UEA 30 个数据集都可当靶（regret 相对论文公开数字），靶池受限被绕开一大半。主要是口径修正，零新设计。
3. **数据集轴放大一个数量级：引入 UCR 128 单变量档案**（全库零讨论）。encoder 天然支持 C=1；L≥50 的 UCR 数据集过百；每数据集只需 shared10 × k1 @官方口径 ≈10 格。同时喂饱问题 3（数据集级 n 从 16 → 100+）。约束：多数 UCR N<1000，只能作"数据集级三态/毒性"样本与全量靶，不能切多任务；下载走 aeon 的 Zenodo 镜像（与 UEA 同路）。
4. **半合成/构造靶**（全库零讨论）。用已知生成过程控制难度、类别几何与长度，制造"随机特征解不了、预训练能解"的靶，验证机制而非 SOTA。定位：analysis / 机制章，不作头条。
5. **接受受限并产品化为适用性闸门**（现行路线，pipeline v2）：金丝雀直测 (a=3, j=7)=21 格 ≈7 min 判可评性。诚实读法是"本方法适用带窄"——写进 limitation 而非解决问题。

### 2.4 审计发现的登记簿不一致（会误导下个会话）

- UniMiB-SHAR 不在"池唯一真源" `configs/source_pool.yaml`，只在 `configs/candidate_pool.yaml`；`tests/test_source_pool.py:57` 仍断言 source==14。`source_pool.yaml` 注释写"13 活跃"实际 14 条；引用的 `promoted_v2` 键不存在；ERing / MotorImagery / SCP1 等五成员 `role_v2` 未标注，处于"注册表说在、报告说不在"的悬空态。
- 两套"贴顶"定义并存未对账：旧 = 随机基线 >0.90（`autocls/baseline_probe.py:42`），v2 = 网格 max R' ≥0.99（`results/source_pool_v2/PROTOCOL.md` §4）。SAD 两套都死但死因不同（0.912 vs 0.993）。
- HW 净空 0.004（工作点）vs 0.067（@290）被全部归因于 iters，但任务集 12 vs 8、候选目录 20 vs 10 CLS、mean vs median 三个变量同时变了；20 CLS 的 oracle 天然更高，会同向压低工作点口径的净空。
- FaceDetection 的 +0.0272 只跑 1 个 GGS 种子、1 个切分（Epilepsy 是 5 切分平均）；报告给了翻案条件从未执行。
- Epilepsy 尺度 2 的"10ep"= 2 iters（136 样本 / bs128 → 2 steps/epoch），与 FaceDetection 的"10ep=470 iters"并列不可比。
- `propose_slice` 的类别档公式 [0.4K, 0.8K] 与 `source_pool.yaml` 实际档不符（HW K=26 公式 [10,21]，YAML [8,14]）。
- CLAUDE.md 的"eligible ≥ 5000/任务"判据在代码库里不存在。
- `docs/TECHNICAL_DESIGN.md` 无目标池 / LOO / 预检章节，仍写"第一阶段 HAR/Epilepsy"。

---

## 3. 问题 3：任务嵌入 E 无法参与模型（惰性特征）

### 3.1 问题定义

- **E 的构造**：共享单变量 TS2Vec（600 iters，只在源池 fit，冻结，不接触标签）→ 每任务采 100 条样本序列（分类语义无滑窗）→ 逐时间步嵌入 → 通道展开为独立实例、表征**跨通道取均值** → 两级 Set Transformer（SetTF，ln=False，输出无归一化）→ E∈R²⁵⁶。
  出处 `autocls/task_embed/__init__.py`（`encode_task_windows` / `embed_task`）、`autocls/task_embed/set_transformer.py`。
- **E 的消费**：`set_fc = Linear(256→128)+ReLU` 后与 CLS 支路 **concat**，再过一层 128 隐层 MLP 出 logit（`pred_fc`）。没有 FiLM / 双线性 / attention 之类显式调制通路，条件化只能靠那一层 ReLU 隐层隐式学。
  出处 `autocls/comparator/tclsc_model.py:47-80`。
- **训练配对量极小**：M3 路线 B 290 有向对；C2 存活 2391 对（训 420 记录）训一个 ~130k 参数 MLP + SetTF。
- **判负记录**：M3 shuffle-E 5 种子逐位全同（Δ=0.0000）；C2 七跑 × within/cross × 5 种子 Δ≤±0.0008；UCI-HAR 排名 shuffle τ=1.0；MX-1 E→毒性 LOO p=0.9945（方向反）；MX-1b E 在标签感知探针 P⊕G 之上零增量 p=0.963；MX-1c 先验收缩 p=0.353；MX-2a E 选源 vs 全量 p=0.187（vs 随机选源 p=0.024 显著）；宽面板 E 核插值 D1w 方向负 p=0.994。项目结论："E 表示得了任务差异（数据集内/跨距离 0.024 vs 0.178，7× 分离），但 δ 后 label 里没有需要 E 才能答对的题；不是模型/SetTF 失败"。

### 3.2 已排除的假设

| # | 假设 | 判负数字 | 出处 |
|---|---|---|---|
| 1 | E 在配对排序里被用上（任务级 / 数据集级留出） | Δ=0.0000（M3）、≤0.0008（C2 R1–R7） | `results/tclsc/report.json`、`results/decisive_e_ablation/REPORT.md` |
| 2 | E 能预测数据集毒性（16 折 LOO） | D1 p=0.9945，cross-E 反而更好 | `results/mx1_feasibility/REPORT.md` |
| 3 | E 在标签感知零训练探针之上有增量 | D3b p=0.963 | `results/mx1b_feasibility/REPORT.md` |
| 4 | E 作先验收缩 / 选源课程 | p=0.353 / p=0.187 | `results/mx1c_prior_shrinkage/`、`results/mx2a_source_selection/` |
| 5 | E 作任务级条件化核 | D1w 方向负 p=0.9944 | `results/wide_panel/REPORT.md` |
| 6 | 每任务重训 TS2Vec 造 E | 表征基任意性，区分力≈0 | `results/task_embed_sanity/` |
| 7 | 元特征（L/C/K/N）替代 E | B2 p=0.9807 | `results/mx1_feasibility/REPORT.md` |
| 8 | 毒性由 L 解释 | ρ=−0.2 n.s. | `results/destruction_matrix/REPORT.md` |

### 3.3 审计发现：三条从未进入任何报告的特征侧证据

1. **E 在进模型前已经退化成近常数**。`results/task_embed_sanity/sanity_results.json` 同一文件里：TS2Vec 级类别子集间余弦 0.6545，**过（随机初始化）SetTF 后同一对 = 0.98746**，全部 15 个非对角元 ∈ [0.9875, 0.9998]。被反复引用的验收数字 0.99/0.65/0.92（`TECHNICAL_DESIGN.md` 模块 5、`task_embed/__init__.py` 头注）全是 **SetTF 之前**的量；`results/mx1_feasibility/PROTOCOL.md` §4.1 却用它们给"冻结随机 SetTF 的 E 已有清晰区分力"背书——引用错了层。MX-1/1b 的 E 因此是"近常数向量的微小逐维扰动再 z-score"，其有效秩 / 条件数 / z-score 前方差谱从未被看过。
2. **zero-E 与 shuffle-E 的不对称**（`results/decisive_e_ablation/runs/R1_mixed_s0.json`）：normal loss 0.53712；shuffle（within/cross）loss 0.53714 / 0.53707，单向对上 **0 个**不一致预测（McNemar 每行 b=0, c=0，p=1.0 是"无可检验"不是"未通过"）；zero loss 0.45464、23 个不一致（b=8, c=15）。唯一自然解释：`set_fc(E)` 对所有任务输出几乎相同的向量——一个有幅度的常数偏置，不携带任务条件成分。报告把它当"消融确实生效"的旁证，没有当作特征侧诊断。
3. **within-derangement 空间极小**：HW 留出 3 任务只有 2 个 derangement，R1 的 5 个种子实际只有 3 种不同错位，"5 种子全同"部分是机械结果（cross 档不受此限且同为零，主结论未动摇，但"5/5"这个数需降级）。

另外四个结构性缺口：任务盲天花板只算得了 shared-shared 对（15% 的 label）；MX-1 折内 TS2Vec bs 128→32 后 E 的 7× 分离从未复测；E 到 L/K/N 等元信息的 probe 精度从未测过；AutoCTS++ Algorithm 2 的进化搜索从未实现（`autocls/search/` 只有 `rank.py`，`scripts/4_zero_shot_search.py` 不存在），E 的设计用途"在 3×10¹² 空间里定向"从未被构造出来，全程是 20–31 条小目录穷举，而穷举场景下全局排序天然够用。

### 3.4 设计层面的因果重排

项目官方因果链是单向的：噪声地板 → 大 δ → 存活 label 任务盲 → E 无梯度 → 惰性。审计后应改成**两条并行、互相遮蔽**的因果：

- **label 侧（已坐实）**：δ=P95 把任务信号带（|Δ|≈0.01）全部剃掉；
- **特征侧（此前免责）**：E 经随机 SetTF 后区分力压掉 95%，再经 concat 只提供常数偏置；即使 label 里有信号，这条通路也很难吃到。

两条任一成立都得到"消融为零"，现有实验**无法区分**它们；修任一条都不足以让 E 活，必须同时修。四次预注册（MX-1/1b/1c/2a）在 n=16 数据集上"信号真实但小"，也全部继承了同一个退化的 E。

### 3.5 候选解法（按"零 GPU → 便宜 → 贵"排序；标注与既定结论的冲突）

1. **先做特征侧诊断，再谈 E 是否惰性（零 GPU，最优先）**。用 `results/labels_workpoint/task_features/*.npz` 现成的 windows_repr：(a) 算 TS2Vec 级 vs E 级（联合训练后 SetTF 输出）的余弦矩阵、有效秩、z-score 前逐维方差谱；(b) 加载训好的 T-CLSC，量 `set_fc(E)` 跨 28 任务的方差 vs 均值幅度；(c) 复算 C2 时把 E 换成 **TS2Vec 级均值池化向量（绕过 SetTF）**，跑同样的 shuffle 消融。若 (c) 仍为零，"label 侧单因"成立且证据升级；若非零，E 惰性半数是 SetTF/融合的锅。
2. **修特征通路三件**（若 1 显示退化）：E 加 LayerNorm/L2；SetTF 先做自监督预训练（同任务不同窗口子集 → 同 E 的对比目标，只用源池数据，不违铁律 6）再联合训练；融合从 concat 换 FiLM（E 调制 CLS 支路的 scale/shift）或双线性项。全部零 GPU（比较器训练 CPU 分钟级）。
3. **与 §1.3 候选 1 绑定成一个实验**：软 label（保留任务信号带）+ 非退化 E + shuffle 消融。全项目第一次同时修两条因果链后再测 E。**冲突**：与 CLAUDE.md"细粒度 task-aware 排序判死勿复活"正面冲突——但原判决的证据基础（E 未退化 / label 侧单因）经审计已不完整，重测的是"判决前提"而非"复活旧协议"。须用户拍板。
4. **松绑"E 不得接触标签"（分类任务）**。该禁令（`autocls/task_embed/__init__.py:144-146`）继承自 AutoCTS++ 预测语义；分类下 R' 本身就是标签依赖量，目标数据集训练划分的标签在部署时本来就要给下游 SVM 用，读它不违铁律 6（铁律 6 管的是 TS2Vec 训练数据）。MX-1b 已证标签感知零训练探针 P⊕G 把毒性序信号从 0.248 提到 0.555（p=0.026）。候选：E ← [TS2Vec 分布特征 ⊕ P（iters=0 基线/chance/带）⊕ G（类别几何）]，在**排序目标**上重测——此前 P 只在毒性/条件化目标上试过。
5. **通道轴换聚合方式**。跨通道均值是 E 最看不见的轴（通道子集间 0.92，几乎等于同构 0.99），而 UEA 数据集间差异最大的正是通道数与语义。候选：对通道也做一层 set pooling（SetTF 天然适合）或 concat 到固定长度。MX-3 消融清单里没有这一项。零 GPU，sanity 上 15 分钟出结果。
6. **换 E 的单位和规模**：E 按数据集聚合（而非切片任务），数据集数 16 → 100+（UCR）。这是任何"学习型 E→X 映射"可辨识的唯一路径；n=16 下"测量优先于预测"四次独立出现是信息论层面的结论，不是 E 的错。与 §2.3 候选 3 同一件事。
7. **给 E 一个幅度足够大、且真随数据集变的目标**：把训练量（iters）纳入被搜索的配置。同一轨迹多截点评估只加 SVM 成本，(task, cls, iters) label 近乎免费；per-dataset 工作点差 19×、破坏率随 iters 单调涨，是已实测的大幅度数据集依赖信号。E 的任务变成"预测该数据集能吃多少训练"。**冲突**：改铁律 2 语义（iters 从常数变字段）。
8. **构造 E 的用武之地**：实现进化搜索（Algorithm 2），在大空间里比较"全局排序 + 进化" vs "E 条件化 + 进化"。不救 E，但把"E 无用"的结论范围说清楚（目前只在 ≤31 条目录上成立），否则审稿人会抓。

---

## 4. 三个问题的共同根因（设计层）

| 根因 | 影响问题 | 现状 / 来源 | 对应候选 |
|---|---|---|---|
| **A. 分析单位 = 数据集切片**。同数据集的切片是近似复本，任务间信号被构造性压小（p50 0.007–0.013）；真正大的差异（破坏率 0→0.75、工作点 19×）在数据集级，而 n 只有 8–16 | 1、3 | 切片继承自 AutoCTS++（那里切片确实改变预测任务） | §1.3-4、§3.5-6 |
| **B. 硬 δ=P95 弃样**。中位噪声 ≈ 中位信号 ≈ 0.010，本可当 P≈0.75 的软 label；P95 阈值把 0.6<P<0.97 整段丢弃，弃样 55%，存活 label 任务盲天花板 0.95 | 1、3 | 设计文档里 δ 出现 0 次 | §1.3-1/2 |
| **C. 训练预算口径**。early/full = TS2Vec 官方的 12–186×，切片上 48–2000 epoch；破坏率随 iters 单调涨，65% 随机 CLS 判破坏，label 主体变成"谁先训崩" | 1、2（外部锚） | 铁律 2 | §1.3-5、§2.3-2 |
| **D. 评估栈可测带窄**。随机卷积特征 + SVM-RBF 在多数 UEA 上已线性可分；预训练可测增益只在少数数据集出现 | 2、1（贴顶→δ 吃光） | "固定 encoder"决策（本文不挑战）；下游协议探针判 no_effect | §2.3-1（few-shot 重开） |
| **E. E 通路退化 + 标签盲禁令**。随机 SetTF 压掉 95% 区分力、concat 只给常数偏置、E 不许看标签而目标是标签依赖量 | 3 | 验收数字引用错层 | §3.5-1…5 |
| **F. 数据集数 n≈8–16**。任何数据集级学习映射不可辨识；四次预注册"信号真实但小"是 n 的结论 | 2、3 | 只用 UEA 多变量 30 个 | §2.3-3 |

---

## 5. 对 CLAUDE.md 既定结论的质疑清单

| CLAUDE.md 表述 | 质疑 | 建议改法 |
|---|---|---|
| "任务信号幅度 0.007–0.013 < 轨迹噪声地板 0.030" | 比的是信号 p50 与噪声 P95；中位对中位 ≈0.010 vs ≈0.010；且只在 UM 测过 | 改为"信号 p50 ≈ 噪声 med ≈ 0.010，< 噪声 P95 0.030（UM 口径）；硬 δ 下不可用，软 label 下未测" |
| "细粒度 task-aware 排序在本栈判死勿复活" | 判决的两个前提（E 未退化、label 侧单因）经审计不完整；random CLS 的 85% label 在天花板论证之外 | 降为"在硬 δ + 现行 E 通路下判负；两条前提修复后的重测为待办" |
| "E 表示得了它…不是模型/SetTF 失败"（decisive §5，CLAUDE.md 引用） | E 级余弦 ≥0.9875、zero/shuffle 不对称直接反证 | 删去"SetTF 没失职"，改为"特征侧退化未排除" |
| 铁律 2"跨任务训练量必须用 iters" | 在 n_train 150–700 的切片上等于 48–2000 epoch，与 TS2Vec 官方 200 iters × bs 16 差 1–2 个数量级；工作点探针实质已按 epoch 在调 | 保留 iters 计量，但预算按官方口径重标；或把 iters 升为搜索字段 |
| "transformer 分支已探针判死…勿再投入" | 判定无 REPORT / analysis，只有一句话 | 补 `--analyze` 出 REPORT 再引用 |
| "M3 留出 0.77–0.87" | 0.87 是路线 B 同分布留出（脚本自注"非 zero-shot 证据"），0.77–0.81 才是跨数据集 | 拆开写口径 |
| "k 放大 Y-B 搁置…k≈6–9 @120" | 6–9 无出处；outlet_y 的 k≈11 用的是已作废的 0.023 幅度；干净口径 required-k 随分位 3–34 | 改为"required-k 随口径/分位 3–34，需先定目标分位；自适应 k 可把均摊成本压到 3–4×" |
| "共享种子配对…均已枪毙勿重提" | 实测只共享 init 与种子，未做按增强类型键控的随机流；chaotic 分叉下先验仍低 | 保留判负，注明"部分 CRN 口径" |
| 铁律 6 引申的"E 不得接触标签" | 这不是铁律 6 的内容（铁律 6 管 TS2Vec 训练数据）；分类目标标签依赖，禁令是预测语义遗产 | 允许 E 含目标训练划分上的零训练探针统计量 |
| "eligible ≥ 5000/任务"入池判据 | 代码库中不存在 | 删除或实现 |
| "净空/增益是口径量…HW 0.004 vs 0.067" | 0.004 与 0.067 还差着任务集、目录大小、mean/median 三个变量 | 归因前拆解三项贡献 |

---

## 6. 推荐的决定性实验顺序（便宜优先，每步单独关闭一条假设）

| # | 实验 | 成本 | 关闭什么 | 与铁律冲突 |
|---|---|---|---|---|
| 1 | E 级退化诊断 + `set_fc` 常数性 + 绕过 SetTF 的 C2 复算 | 零 GPU，~1h CPU | 特征侧是否有锅 | 无 |
| 2 | 软 label（噪声感知 BCE）重训 T-CLSC + shuffle 消融；顺带 shared-shared-only 对照、label 翻转注入曲线 | 零 GPU，~2h CPU | δ 是否是任务信号唯一杀手；85% 同分布假设 | 触碰"勿复活"条款 |
| 3 | 官方口径（bs 16 × 200 iters）噪声/破坏率/spread 三件套重标，HW/UM/PS × shared10 × k3 | ~200 格，~1.5h @5090 | 铁律 2 预算是否放大破坏主导 | 铁律 2 |
| 4 | few-shot 协议闸门 2：3 数据集 × 8 任务 × shared10 @290 重编码 + C_k5 ×10 抽样 | ~240 格，~1.5h @5090 | 靶池受限能否由协议缓解 | AutoCL B.1 口径 |
| 5 | 自适应 k：只对噪声带内配对补种子，28 任务 | ~3–4× 560 格，~2–3h @5090 | task-aware 是噪声问题还是信号问题（Y-B 经济版） | "Y-B 搁置" |
| 6 | UCR 128 单变量 × shared10 × k1 @官方口径 + iters=0 基线 | ~1300 格 + 基线，半天 @5090 | 数据集级预测在 n=100+ 是否可辨识；靶池扩容 | 无 |

1、2 零 GPU 且互相独立，可立即做；3、4 各 1.5h 云；5、6 在 1–4 出结果后再决定。

---

## 附录 A：出处索引

**配置 / 代码**：`configs/labels.yaml`（δ、工作点、eval_ratio）、`configs/train.yaml`（290/2320/bs128/lr）、`configs/space.yaml`、`configs/source_pool.yaml`、`configs/candidate_pool.yaml`；`autocls/labeling.py`（make_pairs / split / seeds）、`autocls/workpoint.py`（退化判据）、`autocls/downstream/linear_eval.py`、`autocls/downstream/protocols.py`、`autocls/task_embed/*`、`autocls/comparator/tclsc_model.py`、`autocls/comparator/tclsc_engine.py`、`autocls/baseline_probe.py`（全部阈值常数）、`scripts/6_target_preflight.py`、`scripts/4b_decisive_e_ablation.py`、`reference/AutoCTS_plusplus/NAS_Net/ts2vec.py`（官方 fit 默认）。

**问题 1 证据链**：`results/step0_um_eval_expansion/`、`step05_iters_floor/`、`eval_size_probe/`、`shared_seed_probe/`、`swa_probe/`、`forecast_noise_probe/`、`decisive_e_ablation/`（REPORT + runs/*.json）、`outlet_y/`、`step1_encoder_probe/`（无 REPORT）、`destruction_matrix/`、`decouple_random_cls/`、`full_iters_probe/`、`iters_workpoint/`、`workpoint_confirm/`、`workpoint_followup/`、`workpoint_rank_transfer/`、`slicing_diagnosis/`、`protocol_probe/`、`encoder_ablation/`、`route_a_probe/`（label 正确率口径）、`labels_workpoint/`（summary.json、delta_calibration.json）、`delta_calibration/calibration.json`。

**问题 2 证据链**：`results/target_preflight/`、`target_zero_shot/`（含 subject_protocol/）、`baseline_probe/`、`protocol_probe/`、`candidate_survey/`（REPORT_STEP1/2/3）、`uea_survey/`、`route_a_pool/`、`source_pool_v2/`（PROTOCOL、REPORT、analysis.json、screen.json）、`mx2_endtoend/`、`mx2a_source_selection/`、`probe_sensitivity/`、`gate_v2/`。

**问题 3 证据链**：`results/task_embed_sanity/sanity_results.json`、`tclsc/report.json`、`decisive_e_ablation/`、`mx1_feasibility/`（REPORT、PROTOCOL、posthoc_diagnostics.json）、`mx1b_feasibility/`、`mx1c_prior_shrinkage/`、`mx2a_source_selection/`、`wide_panel/`、`tasklevel_precheck/`、`gate2_pilot/`、`gate2_pilot_b/`、`behavior_recon/`、`proxy_conditioning/`、`canary_validation/`、`anchor_sim/`。

**文档**：`CLAUDE.md`（主张与铁律唯一真源）、`docs/TECHNICAL_DESIGN.md`（2026-07-11，已过期：无 δ、无目标池、无 E 消融、M3 验收无 E 条件）、`docs/MX2_PREREGISTER_MEMO.md`、`出路X_更改主张.md`。
