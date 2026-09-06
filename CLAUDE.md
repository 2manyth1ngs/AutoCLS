# CLAUDE.md — AutoCLS

Zero-Shot NAS for Contrastive Learning Strategy Search。把 **AutoCL 的对比学习策略空间（CLS）** 作为搜索对象，先做**零样本排名**给出候选名单，再在目标数据集上用**逐次减半**的少量真实试跑定冠军，encoder 固定。

**主张定稿（MX-4，用户 2026-09-03 拍板，取代出路 X，勿回退）**：
① **零样本排名**（头条）：任务无关的 CLS 质量排序器 = **源池先验**（shared 探针在源池全部切片×预算上 within-task z 分数的均值）+ **字段结构化代理**（GBDT 在 CLS 字段编码上拟合 within-task z，`autocls/search/surrogate.py`，目录外候选由它打分）。7 折 leave-one-dataset-out：先验@1 regret 0.046、代理配对准确率 0.78 / regret@1 0.056，均显著优于 T-CLSC（0.074 / 0.735）与手工强默认 GGS（0.068）（`results/mx4_prior_sh`、`results/mx4_surrogate`）。
② **目标侧逐次减半验证**：零样本名单前 6 条各训到 @60 → 留 3 条到 @120 → 3 条到 b_final 取最好（SH6-3-3）。@290 regret 0.018、中位 0、垫底 0/42，成本 ≈3.6 次全跑（与 AutoCTS++ 式 top-3 真评估同档，后者 0.041）。**最后一级必须留 3 条**（只留 1 条 regret 翻倍）。
③ **任务感知靠测量不靠预测**：数据集级 CLS×数据集交互真实存在（R' 方差分解 @290：任务 73% / CLS 13% / 交互 14%，交互均方 = 噪声 140×，几乎全在数据集级），由目标侧 @60/@120 截点直接读出；E / SetTF / 可行性预测器全部关闭，T-CLSC 降为消融对照。
**两条判死**：(a) **细粒度 task-aware 排序**在配对 δ 标签 + ≤8 数据集口径下不可辨识（M3/C2/MX-1 系列/实验 1-2/MX-3-PQ 五轮判负；δ 存活对里好-好对只 1.6–4.3%，任务盲天花板按 shared–shared 对诚实算是 0.86–0.92 而非 0.97）；(b) **开放空间零样本外推**：任何代理对目录外候选系统性高估（进化 CLS 预测 z +0.52 vs 真 +0.05，到训练集字段汉明距离 8 vs 目录 3），bag-10 LCB 只砍 44%，信任域一刀切全砍——搜索只在源池已评估配置邻域提名，且必须过目标侧真评估（BO 式回灌才可立项）。

完整设计见 `TECHNICAL_DESIGN.md`（已过期，待按 MX-4 重写）；MX-4 计划与诊断见 memory `mx4-diagnosis-prior-beats-tclsc`。

本机python环境：Anaconda 下的名为pytorch的虚拟环境

## 核心概念
- **CLS**：一个对比学习策略 = 19 个离散字段的配置向量（AutoCL Table 1）。这是"要搜的架构"。
- **零样本排序器**（MX-4）：先验 + 字段代理，输入 CLS 向量（不看任务），输出质量分；`surrogate.pairwise_matrix` 把分数变成 `autocls/search/rank.py` 可消费的 P 矩阵，胜场/Borda/进化一行不改。
- **逐次减半（SH）**：同一条预训练轨迹的截点 {60,120,290} 各评一次 R'，逐级淘汰；部署选中的就是那条轨迹（继续训练即可），缓存模拟与部署零近似。
- **T-CLSC**：AutoCTS++ 式比较器 `(E, cls_a, cls_b)→y`，**只作消融对照**（MX-3-PQ 判 B-DEAD；本质是常数排序器，偏离常数时 regret 翻倍）。代码与铁律 7 保留，不再投入。
- **early-validation / 截点评估**：造 R' 时一条轨迹到 290 iters，@60/@120/@290 各评一次（`evaluate_rprime_trajectory`），逐位等于分开训。**只有 SH 幸存者跑到 b_final。**
- **固定 encoder**：架构固定、每个 CLS 各自重新预训练一份权重（不是权重冻结）。
- **字段主效应**（论文分析章）：分组置换重要性 temperature 0.28 ≫ sim_func 0.17 > norm 0.09 > freq_mask 0.08；temporal / cross_scale / pool / adj_neighbor / loss_type 平均惰性；ratio 字段档位效应非单调，只报字段级排名不报逐档。

## 关键决策（已定，勿改）
1. CLS 编码用 **向量**（方案 A），不用图/GIN。两种模式同出 `configs/space.yaml`：`scalar` 32 维（T-CLSC、旧缓存，逐位不变）/ `binned` 按档 one-hot（代理）。
2. encoder 两者可切换：`dilated_cnn`（默认，TS2Vec式）/ `transformer`，由 `configs/encoder.yaml` 控制。**transformer 分支已探针判死**（2026-08-29，`results/step1_encoder_probe`，无 REPORT 仅缓存，写论文前须补 `--analyze`）。
3. 第一阶段只做**分类**。转预测任务已探针判死（`results/forecast_noise_probe` 红灯）。
4. **k 放大（Y-B）搁置未关闭**；SH 的目标侧测量已承接大部分它想解决的问题，重启前先算账。
5. 原目标池 {UCI-HAR, Epilepsy, FaceDetection} **从主结果退役**（`results/target_preflight`）；全量 UCI-HAR 仍可当外部锚。
6. **头条臂 = SH6-3-3_prior**；SH10-4-3 只好 0.002 但贵 30%，不作默认。all20@60→top3 与 SH 同带但成本 1.8×——先验的价值是省一半探测预算。
7. **AutoCTS++ Algorithm 2 进化搜索**只保留代码，不作零样本适应度使用（D3 5/21 + WP2 S3 不过）。
8. **学习曲线外推 / 早截点外推淘汰已判死（WP5，2026-09-04，用户拍板关闭）**：任何形式的"从早截点预测终点分来做淘汰"（GBDT / ridge / 斜率 / 往先验收缩 / 分位回归 / 排序损失）都不再提、不再立项、不再作为"学习在哪"的答案；SH 一律按当前截点分淘汰、最后一级留 3 条。证据 `results/mx4_lc_extrapolator/REPORT.md`，代码保留不删。

## 铁律
1. `configs/space.yaml` 是策略空间**唯一真源**，代码不得硬编码取值集合。**增补（WP6）**：`extension: true` 的字段是 opt-in 扩展字段（现有 `input_mask`），`field_names` / 随机采样 / `cls_key` / 默认 `CLSEncoder`（32 维）/ 落盘序列化 `CLS.to_dict()` 对旧配置逐位不变；只有 `all_field_names`、`twin()`、`CLSEncoder(fields="all")`、`encoder_overrides()` 看得到它。新扩展字段照此模式加，不得改旧路径。
2. `cl_trainer` 的 `max_iters`（参数更新次数）必须显式传参，early-val 与最终评估禁止混用；跨任务训练量必须用 iters 而非 epoch 保证一致（early=290, full=2320，STD_BATCH_SIZE=128，固化在 `configs/train.yaml`；多保真截点 `budgets=[60,120,290]` 是同一轨迹的 checkpoint，非三次独立训练）。**增补**：任何 regret / 增益 / 净空数字必须带 @b_final 与实例数 n；多保真成本按 iters 逐级累加（k0·b0 + k1·(b1−b0) + …）并用缓存 `seconds` 增量换算 GPU 秒，三处（真值 / 裁判 / 外部锚）同口径。
3. 所有 `(task, cls, iters) → R'` 结果必须落盘缓存，脚本支持 `--resume`。这是最贵的一步。
4. `task_embed/` 相关代码必须先读 AutoCTS++ 真实源码再适配，注释标注对应原文件+公式号（现仅消融用）。
5. 先写 `tests/` 再写实现。与论文有出入的补全须注释标注"论文未给全，按 X 语义补全"。每个实验先写 PROTOCOL（判读带先于数据）+ 协议锁 commit 再跑；探索读数先看过的必须在 PROTOCOL §0 如实声明。
6. 先验 / 代理 / 共享 TS2Vec 仅在【源池训练侧】拟合；目标数据集绝不参与（fold_split 保证），只在考试时前向 / 短训。
7. **任务类型显式分发**：task 数据结构必须带显式 `task_type`；凡按任务类型不同而行为不同的逻辑必须通过 `task_type` 显式分发。任务无关模块（cls_space/cls_encoder/contrastive/encoders/comparator/search 含 surrogate）绝不能引入任务类型或数据集概念。`autocls/task_types.py` 是 task_type 取值唯一真源。
8. **R' 优劣方向抽象**：造 label / 排名时"更优"必须经 `is_better(r1, r2, task_type)` 判定；代理的目标 z 分数在脚本层按任务归一化，模块只见向量。

## 已知 limitation（encoder 适用边界与测量地板）
- **L≥50 是固定 encoder 的适用边界**（`results/route_a_probe`）：10 层膨胀 CNN 感受野为 L≥100 设计，短序列被 padding 主导，R' 与 CLS 质量零相关。L<50 不入源池；30–50 须过 SNR 检查。
- **测量分辨率是硬地板**：单跑 R' 噪声 sd 0.035–0.048（k=3 探针），δ 0.06–0.19；oracle 与第二名真差均值 0.020。非破坏半区在 k=1 下不可分辨（所有排序器在真 top-10 内配对 acc 0.43–0.51），臂间 0.01–0.02 的 regret 差在噪声内；"oracle"含赢家诅咒。
- **δ 非全局常数**：按 (数据集, 预算) 查表 `configs/labels.yaml: pq.delta_per_dataset_budget`（噪声在 @120 达峰）。
- **R' 噪声结构**：计数成分（∝1/√n_eval）+ 轨迹地板（cls×seed 交互）；共享种子配对 / SWA / per-CLS δ / 软 label 均已判负勿重提。
- **SH 的 G4 保留**：新鲜种子真值下 SH 只比先验@3 好 0.002（p=0.64，n=14）——头条量级是部署口径（同轨迹），"选出更好的配置"只能写"不差"。
- **源池 v2**（`results/source_pool_v2/REPORT.md`）：活跃源 8 = HW/UM/PS/UWave + Ethanol/Cricket/HMD + SCP2 低权（只训练不作折）；靶 = 7 折轮转。语音域仅 PS、短序列端空。
- **净空/增益是口径量**：HW 对 GGS 的净空 @工作点 0.004、@290 0.067；GGS 在 HW 近 oracle（regret 0.026）而先验 0.061，是先验唯一输 GGS 的数据集。
- **全量标准协议绝对值**（HW 0.34–0.39 vs TS2Vec 0.515）是训练口径差，由 `protocol-alignment` 分支处理（`configs/train.yaml: protocols.ts2vec_official`）；WP3 部署对表依赖它。

## 复用 AutoCTS++（repo: decisionintelligence/AutoCTS_plusplus）——现仅消融对照
- `Ts2vec_model/` → `autocls/task_embed/ts2vec_wrapper.py`
- `AHC/set_encoder/` → `autocls/task_embed/set_transformer.py`
- `AHC/ahc_engine.py` → `autocls/comparator/tclsc_engine.py`（GIN 支路换成 MLP）
- `exps/random_search.py` → `scripts/33_pq_comparator_loo.py` + `34_pq_search.py`（进化搜索 `autocls/search/evolution.py`，不作零样本适应度）

## 数据流（MX-4 版）
```
离线: 源数据→切片 t_i →采样 CLS（shared-10 探针 + 每切片 random-10）
     →对 (t_i, cls) 跑一条预训练轨迹，@{60,120,290} 各评一次 R'（缓存 (task, cls, iters)）
     →within-(t_i, b) z 分数 →先验（shared 均值）+ 字段代理（GBDT on CLSEncoder(binned) ⊕ b/290）
在线: 未见 D→零样本排名（shared 按先验、目录外按代理，只在信任域内提名）
     →前 6 条 @60 →留 3 条 @120 →3 条到 b_final →取最好 = cls*（≈3.6 次全跑）
展示: 7 折 leave-one-dataset-out（先验/代理只见训练侧，目标只考试）；
     外部锚 = 全量标准协议（UEA 官方划分）对 GGS / TS2Vec 公开数字（WP3）
```

## 里程碑
- ✅ M0 骨架 / M1 单 CLS 预训练（HAR 0.9372）/ M2 early-val 相关性（ρ=0.85）/ M3 T-CLSC 留出配对
- ✅ MX-1 系列（可行性预测器 / 先验收缩 / 选源课程）全部预注册判负——负翼
- ✅ MX-2 双向 LOO（T-CLSC 头条 p=6.3e-07；GGS 在 HW 近 oracle、@3 输 random@3、外部锚未达线）
- ✅ MX-3-PQ 7 折 × 3 预算：B-DEAD（E / 预算描述符零贡献）；T-CLSC 本身输给零训练先验（p=1.9e-06）
- ✅ **MX-4 WP1**（2026-09-03）：先验 + SH，SH-WIN（`results/mx4_prior_sh/REPORT.md`）
- ✅ **MX-4 WP2**：字段结构化代理，SUR-WIN，S3（开放空间）不过（`results/mx4_surrogate/REPORT.md`）
- ✅ **MX-4 WP3 全量部署**（2026-09-04，`results/mx4_deploy/REPORT.md`）：官方口径 SH6-3-3 冠军追平 TS2Vec **4/6**（UWave 0.918 / PS 0.256 / Cricket 0.986 / HMD 0.450 均超论文），**DEPLOY-DEAD**（预注册 W1 要 ≥5/6 且 HW 达标：HW 0.431 ✗、Ethanol 0.241 ✗）；W2 过（SH ≥ prior3 5/6）。HW 是**名单失败**：同目录 GGS 0.546 / e490b88e 0.542 被先验排到第 8/9，前 6 起跑没试；事后 SH10-4-3 选中 GGS（+33% 成本）。截点 25%→100% Spearman 中位 0.71（短轨迹早截点可用）；GGS seed 0 跨 torch 2.4/2.8 五靶逐位相同
- ✅ **MX-4 WP3b UCI-HAR 外部锚**（2026-09-04，本地 4060，`results/mx4_deploy_har/REPORT.md`）：先验/代理用全 8 源池拟合，`legacy_full` 口径截点 580/1160/2320；**HAR-WIN**——SH6-3-3 冠军 4f7d4135（先验第 4）3 种子 acc 0.935 / F1 0.936，追平 GGS 锚 0.935、超 TS2Vec 0.930；先验@3 0.918、先验@1 0.913；GGS 本轮 0.938（锚 0.935，P2 ✓）；目录 oracle 0.940 与 AutoCL 0.963 差 0.028 在目录内无解；25%→100% Spearman 0.86
- ✅ **MX-4 WP6 第 20 字段 `input_mask`**（2026-09-04–06，分支 `wp6-mask` 已合入，`results/mx4_mask/{PROTOCOL,REPORT,paper_table,FULL_TABLES}.md`）：TS2Vec encoder 侧二项时间步掩码作 opt-in 扩展字段；官方口径六靶目录 21 条各配孪生（S1 125 格 4090 + finals 46 格）。**字段成立**：目录天花板 6/6 由掩码开孪生给出（Ethanol 0.300→0.331）；淘汰赛"两版都量"按数据集自动选开关（UWave/PS/HMD 开、HW/Ethanol/Cricket 关）。**学习规则不成立**（S0/S1：顶部配置掩码符号随数据集反转；描述性探针只用配置字段 LODO 符号 0.70，加数据集描述量无增益，n=6 识别墙）。预注册读数码占位 **MASK-DEAD**（正式主臂 gbdt20 需 S2，挂起）；事后扫臂 **SH20-3**（名单 10 各带孪生，两级从 50% 起）3 种子六靶全超 TS2Vec 论文数（Ethanol 0.314±0.020），平均 0.577；掩码关的 **SH10-4-3 六靶追平 6/6（HW 0.531 / Ethanol 0.303）= 名单待办①坐实**。源侧标签（mx3_pq）不含掩码，先验/GBDT 不认识该字段；`--twins` 默认关、头条臂不变；S1b（K1，$1）/ S2（K2，$9）挂起
- **WP3 之后的名单侧待办（用户 2026-09-04 记录）**：① 起跑名单按先验置信度自适应（先验前 6 的 z 平坦时扩到 10；HW 上 SH10-4-3 事后选中 GGS 0.546）——**WP6 已在官方口径 3 种子坐实 SH10-4-3 六靶追平**；② 提名配置与 shared 混排进起跑名单（现行 hybrid 的 +10 偏置让提名从未起跑，HMD 上提名是目录最高 0.486）；③ 噪声带靶（Ethanol 类，候选全在 0.06 带内）3 种子复核后再宣布冠军。淘汰赛规则本身不动（关键决策 8）。
- MX-4 WP4（增强不变性画像作数据集级任务特征）默认不做，除非用户明确要保留"E 有用"的可能性
- ✅ **MX-4 WP5 学习曲线外推器**（2026-09-04，分支 `wp5-lc`）：**LC-DEAD**——SH 淘汰规则换成外推器预测，头条臂 0.0184→0.0183（4 胜/4 负/34 平），省档位 SH4-2-2/6-3-1 追不平（0.035）；增益 LODO R² 0.02 = 测量地板（`results/mx4_lc_extrapolator/REPORT.md`）——负翼
- 论文：正翼 = 零样本排名 + SH 验证 + 字段主效应 + 成本表；负翼 = task-aware 五轮判负 + 开放空间外推判死 + 学习曲线外推判死（analysis/limitations）

## 路径二：给学习部分加一个真有增益的组件（用户 2026-09-03 记录；WP5 已关闭，其他方法另开会话讨论）

**为什么**：现在的头条"先验 + 淘汰赛"审稿人会问"学习在哪"——先验 7 折第一名都是同一条配置（等于一个学来的默认），GBDT 在 top-1 上没超过先验，淘汰赛是 Hyperband 的常规组件。路径一（WP3 全量协议 + 候选扩到 50–100 + Hyperband/随机/BOHB 基线）把证据做实；路径二给"从源池学到、在目标上用"加一个有实证增益的模型。

**WP5 学习曲线外推器：已关闭（关键决策 8），本节只留存档，勿再展开讨论。**
读数 LC-DEAD（`results/mx4_lc_extrapolator/REPORT.md`，memory `mx4-wp5-lc-extrapolator-dead`）：头条臂 0.0184→0.0183（4/4/34 平），省档位追不平（0.035）；根因 = 增益 R'@290−R'@60 的 LODO R² 0.02，标签噪声地板淹没"晚熟"信号。
留给后续立项的两个硬事实：**SH6-3-3 的 0.0184 里 0.0118 是先验 top-6 不含 oracle 的不可约部分**（淘汰规则最多改 0.0067），**最后一级留 3 条是硬要求**（任何规则只留 1 条都翻倍）。

**次选：WP4 增强不变性画像**（机制性任务特征，与 CLS 字段同坐标，n=7–8 也可辨识；风险更高，先验 ~30%），设计见 memory `mx4-diagnosis-prior-beats-tclsc` 与计划文件 §5。
**WP6 之后的读法（2026-09-06）**：掩码字段把 task-aware 的目标缩成"每个数据集一个比特"（效应 0.03–0.1），但 `results/mx4_mask/mech_probe.md` 显示 n=6 下任何数据集描述量都不比"只用配置字段"更好——任务特征线仍不立项，除非数据集数量上去。

## 目录
`autocls/{space,contrastive,encoders,task_embed,comparator,downstream,search,data}`（`search/surrogate.py` 代理、`search/rank.py` 聚合、`search/learning_curve.py` 外推器（判死，保留）、`search/evolution.py` 仅保留）+ `scripts/`（MX-3-PQ：`32–34_pq_*`、`pq_common.py`；MX-4：`36_prior_sh_sim.py`、`37_surrogate_loo.py`、`38_lc_sh_sim.py`、`39_sh_deploy.py`（`--twins` 孪生目录）、`40_mask_field_sim.py`（WP6：s0/k1/c0/source/deploy）；`32_pq_generate_rprime.py --twins` 源侧孪生补标；pod 脚本 `pa_pod/wp6_*.sh`、`pq_pod/twins_*.sh`）+ `configs/*.yaml` + `tests/`。分支：`mx4`（MX-4 代码与结果，WP3 在 worktree `.claude/worktrees/mx4`）、`protocol-alignment`（全量协议对齐）、`wp5-lc`（WP5，已合入）、`wp6-mask`（WP6，已合入 d029eb7）。注意另有分支 `wp6-tf`（任务特征线，另一会话）也用了"脚本 40"编号，合入前须改名。
