# CLAUDE.md — AutoCLS

Zero-Shot NAS for Contrastive Learning Strategy Search。把 **对比学习策略空间（CLS；当前基线 = AutoCL 空间 + opt-in 扩展字段，空间本身仍在探索中）** 作为搜索对象，先做**零样本排名**给出候选名单，再在目标数据集上用**逐次减半**的少量真实试跑定冠军，encoder 固定。

## 本会话（2026-09-14 起）：重启 T-CLSC 的尝试会话（用户指令，最高优先级）

本会话的目的是**重新尝试最初的设想：T-CLSC 比较器 + 数据集特征 E → AutoCTS++ 式 task-aware 比较器**，作为论文主臂（承接下一节"用户正式主张 2026-09-13"）。本会话内的行为规则：
1. **立场是"怎么让它成立"，不是"它为什么失败过"。** 历史判负（M3 / C2 / 实验 1-2 / MX-3-PQ / WP6-TF / WP7-UCR / WP12）只作为**设计约束**引用，每条至多一句，不展开复述，不把结论推回"任务盲先验 + SH"，不重复论证不可行。
2. 用户提问先答问题本身；引用历史证据只挑与当前设计直接相关的那一条，并落到"所以重启时要怎么改"。
3. 本文件下方"T-CLSC 只作消融对照、不再投入""E / SetTF / 可行性预测器全部关闭"等表述**在本会话失效**；`autocls/comparator/`、`autocls/task_embed/`、`scripts/33_pq_comparator_loo.py` 是本会话的主要工作对象。
4. 重启的设计自由度全开：标签形式（δ 硬配对 / 连续 within-dataset z / 开关目标 / 基型偏好）、E 的来源（SetTF / 均值池化 / 目标上一次短训的表示 / 原始空间机制特征）、融合头（晚期 concat / FiLM / 双线性）、训练口径（切片 / 官方）、数据集数（UEA 8 / UCR 84 / 合池）都可改；改动照旧走 PROTOCOL + 单测 + 协议锁。
5. 重启必须绕开的三条硬约束（一句话版，细节在各 memory）：(a) 标签里要有数据集级信号——δ 硬配对在 ≤8 集上把它滤掉（存活对任务盲天花板 0.95，one-hot 数据集 ID oracle 也零）；(b) 预测目标要在数据集维上低秩——AutoCL 全目录"a 比 b 好"不是（8 字段 ICC≈0、残差 90% 需 22/86 维），ICC 过门的删除型开关（highpass05 0.91 / inst_norm 0.87）与基型偏好（B_I vs B_T ICC +0.72，尚无预测器）是；(c) 筛选口径 = 部署口径（增强偏好与温度方向随 batch 128 / 8 翻转）。
6. 结构事实（本会话已核）：T-AHC 的 GIN 只产架构支路 l_a / l′_a，E′ 走 `set_fc` 在 `pred_fc`（Linear+ReLU+Linear）处晚期 concat；我们的 `tclsc_model.py` 自 `cat([l_a,l_b])` 起与原码逐层对应，GIN→MLP 不触及 E′ 路径。融合头是弱交互归纳偏置（E′ 近似加性偏移），可作为重启的改动点之一，但须与带数据集级信号的标签一起改。
7. 重启路线图（用户 2026-09-14 拍板）：**L1** CLS 支路编码（scalar 32 → binned 116 = 每字段 embedding 表）× 目标函数（δ 配对 BCE → within-task z 回归）分解，协议 `results/tclsc_l1/PROTOCOL.md`、单测 `tests/test_tclsc_l1.py`、实现 = `33 --encoder-mode` + `scripts/56_tclsc_l1.py`（47–55 号已被各 worktree 占用）。**L1 已跑（2026-09-14，本地 4060，零新 R'，`results/tclsc_l1/REPORT.md`）：读数码 L1-FLAT**——G1 编码 Δ +0.012 ✗（5/7 折，p 0.12）、G3 mlp_z_binned 0.7644 ✗（差 0.0006）、G3′ 回归目标下 binned +0.037 ✓（p 2.8e-05）；差距账 main_pod 0.735 → cls_only 0.741/0.746 → cls_only_binned 0.758 → mlp_z_binned 0.764 → hgb 0.780（模型族 0.016 未分解）；binned 把 top-10 内配对 acc 从 0.44 抬到 0.53–0.54（事后，项目最高）；建议后续 T-CLSC 一律 binned（待拍板）；按协议下一步 = MLP 训练制度对照（零 GPU）再 L2；本地 4060 与 pod 5090 在 cls_only 上逐位复现；**L3** 融合头改 `s(l_a,E′) − s(l_b,E′)`（反对称、E′ 只能经交互起作用）。**L3 阶段 0 已跑（2026-09-14，S2 + oracle 数据集身份 token = 机制上界非 zero-shot，本地 4060，`results/tclsc_l3/REPORT.md`）：读数码 L3-HEAD-ONLY，G0/G-A/G-B/G-C 全过**——concat+hard（A8）身份 token 消融仍零（Δ_M1 +0.020 p 0.5），diff 头 + hard 标签 Δ_M1 +0.205（p 9e-4，12:2）、concat + zdiff 标签 +0.187（p 0.03）、diff+zdiff regret@1 0.027 vs 任务盲 0.080；七轮零消融的机制 = 晚期 concat 头 + δ 硬标签把任务向量逼成偏置，换任一即解开；film < diff；zdiff 伤任务盲臂（−0.03）。阶段 1 候选条件变量 = φ 元特征 / 真 E / 转导指纹。**阶段 1 φ 臂已跑（2026-09-14，零 GPU，7 折 LODO，`results/tclsc_s1phi/REPORT.md`）：PHI-DEAD**——diff 头 + 12 维 φ 在好半区 M1 消融 Δ +0.011 p 0.12、regret@3 0.031 vs 任务盲 0.029；但 M0/M2 cross 消融 +0.035/+0.026 p<1e-10（92:23）、M0 0.780 追平 hgb：φ 在 LODO 下被用于粗排序、对好半区/选择无增益（与 WP12 互证）；concat 对照仍零。**阶段 1 转导指纹臂已跑（2026-09-14，`results/tclsc_s1fp/REPORT.md`）：FP-DEAD**——ctx = 训练折先验前 6（= SH6 起跑名单）在目标 @60 的实测指纹，nonprobe 视图 cross 消融 Δ_M1 −0.042（目标自己的指纹比别的数据集的更差 = 过拟合 7 个训练数据集的指纹分布）、regret@3 0.031 vs 任务盲 0.027；同成本 `probe6_top3`（直接拿 6 个实测选前 3）regret@3 **0.015** 碾压经比较器外推的 0.030；逐折 UWave/Cricket 正、HW/PS 负。三种可迁移条件变量在 n=7 训练数据集下都只迁移粗排序或反向，差别不在头/标签，在训练侧数据集数；**n 放大已跑（2026-09-14，UCR 84 分组 7 折，δ 0.06，`results/tclsc_s1ucr/REPORT.md`）：UCR-PHI-DEAD / UCR-FP-DEAD**——n 7→72 证实机制（指纹反向消融 −0.042 → +0.018；φ zero 消融转显著、regret@3 0.0209 vs 0.0220 p 0.045）但价值仍小（好半区 Δ_M1 ≈ 0.02、regret 增益 0.001）；直接实测 probe6_top3 0.0125 = 任务盲的 58%，学到的 φ 映射只给 5%；concat 第三次仍零。读法：低维任务描述量对好半区偏好的信息量本身小，不是 n；预测式在比较器形态下成立的部分 = 粗排序可迁移 + 极小选择增益，SH 实测是主臂的测量式实现；**真 E 臂已跑（2026-09-14/15，pod nibk1kf2rjw4y9 4090 4.81 h ≈ $3.56 已停 3/3，`results/tclsc_s1e/REPORT.md`）：E-INERT**——SetTF E 在 binned 下两种头都不塌缩（diff 跨集余弦 0.58、集内 0.9999 = 学成数据集身份向量），diff 头依赖 E 的存在（zero 消融 M0 −0.028 p 7e-11）但不依赖内容（cross +0.002、M1 +0.003 p 0.88），regret@3 0.032 vs 任务盲 0.029；concat 下 E 连 zero 都零。阶段 1 四种条件变量（φ / 指纹 / n=72 / 真 E）总账：机制成立（L3）、粗排序可迁移（φ）、好半区与选择不可迁移，真 E 的失败模式 = 身份而非偏好；**自设 E 模块 A 臂 = 响应画像 E_A 已跑（2026-09-15，用户拍板“进行 A”，零 GPU，UCR 84 分组 7 折，`results/tclsc_s1ea/REPORT.md`）：EA-DEAD（预注册阶梯）**——E_A = 25 维不学习的机制特征（11 个 CLS 同坐标变换的标签保持率 Δinfo + 10 个无标签失真 + 难度 / 尺度 / 趋势，只用目标训练划分，`autocls/data/mech_features.py`），主臂 u_ea_diff_hard 好半区 M1 cross Δ +0.018 p 0.098 ✗（G-A1）；但 regret@3 0.0198 vs 任务盲 0.0220 p 0.009（G-A2 ✓，阶段 1 首次）、regret@1 0.0451 vs 0.0611（−26% p 8e-4），且**换成他集画像 regret 变差**（p 5e-4）= 五种条件变量里第一个“内容被用于选择”的；eaphi 臂 M0 0.794（项目最高）、cross Δ_M1 +0.041 p 0.002、vs φ regret@3 p 0.078 ✗；G-A4 机制对齐 0/4 且三字段反号（与 WP11 A1 同向）；probe6_top3 0.0125 仍碾压 0.0192。读法：阶梯把最吵的 M1 门放在了价值门之前；**A′ 确认版已跑（2026-09-15，用户拍板，零 GPU，`results/tclsc_s1ea_confirm/REPORT.md`）：EA-NOT-CONFIRMED+UCR-REP**——主考场 UEA 8 集 LODO（训练侧 7 集）eaphi regret@3 0.0354 > φ 0.0311 > 任务盲 0.0294（p 0.98）、换他集画像不伤（G-C1/G-C2 ✗），但 M0 0.793 最高（cross −0.040 p 5e-14）= 内容只用于粗排序；副考场 UCR 84 换种子 3/4/5 复现且更强：eaphi regret@3 0.0188 vs φ 0.0223（p 0.001）/ 任务盲 0.0197（p 0.003）、regret@1 0.043 vs 0.067、cross 消融 regret@3 → 0.0221（p 1e-4）、好半区 M1 cross Δ +0.061 p 0.002。读法：**分界变量是训练侧数据集数（n=72 成立且跨种子稳健，n=7 只迁移粗排序且选择反向），与 φ / 指纹在两个 n 下同形**；预测式主臂在比较器形态下的成立条件 = 训练侧 ≳ 70 集 + 响应画像 + diff 头。**A″ 跨语料部署检验已跑（2026-09-15，用户拍板，零 GPU，`results/tclsc_s1ea_xfer/REPORT.md`）：XFER-DEAD−CONTENT**——UCR 84 全训 → zero-shot 排 UEA 126 实例，u_eaphi regret@3 0.0352 / u_phi 0.0368 / u_blind 0.0374 全部输给 UEA 自训（n=7）任务盲 0.0294（G-X1 ✗ p 0.75），任务盲先验本身不跨语料（u_blind regret@1 0.107 vs 0.066，p 1.0）；画像相对同语料任务盲只在 top-1 显著（0.085 vs 0.107 p 0.002，5/7 靶 regret@3 更好、HMD 崩 0.10），G-X2 p 0.093 ✗、G-X3 ✗。读法：跨语料位移压过 n 的增益，预测式主臂在比较器形态下的成立域 = **同域语料 LODO、训练侧 ≳ 70 集**；UEA 官方口径部署不能靠 UCR 训练侧。**R 检验已跑（2026-09-15，用户提问“瓶颈是不是 E 被要求预测高秩对象”，零 GPU，`results/tclsc_s1ea_rank/REPORT.md`）：预注册码 E-OR-N，但阶梯外读数是本会话最强的正信号**——新 field 头（logit = s₀(l_a) − s₀(l_b) + ⟨A·B·t, x_a − x_b⟩，任务向量只给逐字段加性权重）：主臂秩 4 在 UEA n=7 上 0.0325 > 任务盲 0.0294 ✗（cross 不伤），在 UCR n=72 保值 0.0189 ✓；**对照臂不限秩（≤ ctx 维 40）在 UEA n=7 上首次过线：regret@3 0.0269 < 任务盲 0.0294（p 0.035）、< φ（p 0.017）、< eaphi_diff 0.0354（p 7e-5），换他集画像 → 0.0321（p 3e-4）、置零 → 0.0333（p 5e-4）= 有价值且依赖内容；UCR 上 0.0183 全项目最好（vs eaphi_diff p 0.025、regret@1 0.0385 vs 0.0611）**。读法：瓶颈确实是 E 被要求预测高秩对象（字段交互）；改成字段加性目标后 7 个训练集就够，但“低秩”要的是加性结构而非秩 4；n=7 好半区 M1 仍未打开、regret@1 未改善。§9 消融（用户要求）：fw 的增益里数据集特征在 UEA 上占全部（fwb 无特征 0.0304 ≈ 任务盲；置零 / 噪声输入回到 0.030）但只有 top-3 +0.009 acc、top-1 反向；UCR 上结构 : 特征 ≈ 2 : 1。**T 检验已跑（2026-09-15，用户拍板，`results/tclsc_s1et/REPORT.md`）：转导型响应画像 E_T（目标训练划分上 GGS 探针 290 iters 一条轨迹、@60/@290 表示空间的同 11 变换响应 22 维）+ field 头——预注册码 ET-EQUIV+CONTENT，实质为 n=7 上最强正读数**：ft60 regret@3 0.0207 vs 任务盲 0.0294（−30% p 0.001）/ 任务盲先验+GBDT 0.0313（p 2e-4）/ 原始空间 E_A 的 fw 0.0269（p 0.054 边缘未过 G-T1）；换他集 E_T → 0.0338（p 2e-7）、置零 → 0.0321；好半区 M1 0.562 首次超任务盲；逐集 top-3 acc @290 6/7 ≥ 任务盲、Handwriting 0.829 → 0.865（= oracle）；regret@1 不改善（先验 top-1 0.0465 仍最强）；@60 探针 ≥ @290；E_T ⊕ E_A 反而回到任务盲（n=7 装不下 59 维）；UCR n=72 上 u_ft60 0.0197 < u_blind p 5e-4 但 ≈ u_fw 0.0183（表示空间优势只在小 n）。**D 检验官方口径名单模拟已跑（2026-09-15，用户拍板，零新 R'，`results/tclsc_s1et_deploy/REPORT.md`）：预注册码 DEPLOY-ET-FLAT**——E_T T-CLSC 名单 + SH6-3-3 六靶均值 0.535 vs 先验 0.542（追平 TS2Vec 5/6 vs 4/6）；事后诊断 = 比较器把 10 条目录外提名配置排进前 6（5–6/6）而提名在官方口径系统性更差（HW 0.310 vs 0.400，oracle/GGS 被压到第 18–20）= 关键判负 (b) 的外推高估，与 E 无关；**限信任域（shared-10 + GGS）后 E_T 名单 + SH6-3-3 = 0.564 vs 先验 0.542（HW 0.542、Ethanol 0.297 = WP3 两个未追平靶），接近先验 SH10-4-3 0.567 而成本是 SH6**；任务盲 T-CLSC 限域 0.540 ≈ 先验；E_T 在官方全长序列上 |z| 到 15 需裁 ±4。**D′ 已跑（用户拍板“先限域跑一次 SH6”）：DEPLOY-ET-TR-WIN**——限信任域 + ctx 裁 ±4 + SH6-3-3，E_T T-CLSC 名单六靶 seed 0 0.564 / 3 种子 **0.575、追平 TS2Vec 6/6**（HW 冠军 e490b88ee785 3 种子 0.538 本地补跑、Ethanol 0.303），先验 SH6-3-3 0.547 / 4/6、先验 SH10-4-3 0.574 / 6/6（+33% 成本）；增益全在 HW（+0.107）与 Ethanol（+0.062），其余四靶冠军同先验。诚实项：信任域 / 裁剪规则在看过 D 的 seed-0 后加入（均为项目既有规则），新信息是 HW 3 种子。**U 检验已跑（2026-09-15，用户拍板上云，pod 5090 0.94 h ≈ $0.93 已停 3/3，`results/tclsc_s1et_uea30/REPORT.md`）：UEA30-ET-FLAT−PRIOR**——WP10 的 23 个靶外 UEA 数据集、all8 比较器、限信任域 SH6-3-3：E_T 名单 0.774 seed 0 / 0.766 3 种子 vs 任务盲 T-CLSC 名单 0.777 / 0.765（21/23 同冠军、MotorImagery −0.030、SCP2 −0.033、无一靶更好）vs 先验名单 **0.785 / 0.771**（hit@6 19 vs 15 / 13）；三名单追平 TS2Vec 均 21/23；三名单冠军全同 16/23。读法：E_T 的部署增益只在先验名单失败的靶（HW / Ethanol）上出现，靶外 23 靶先验 hit@6 已 19/23 无修正空间；all8 任务盲比较器本身弱于先验（DuckDuckGeese 0.500 vs 0.660）；论文只能写“预测式修正层修复先验名单失败的两靶、靶外与先验持平”，不能写普遍提升。**LC 检验已跑（2026-09-15，用户问“标签是不是太少”，零新 R'，4060 40 min，`results/tclsc_lc/REPORT.md`）：LC-SLICE**——E_T field 头 vs 任务盲 diff 的 regret@3 学习曲线三轴：配置轴（UEA 每任务 K 条）K6 0.0389 → K14 0.0213 → K20 0.0207 平（配对 acc 仍 +0.026）；数据集轴（UCR n 训练集）E_T 从 n=14 起平在 0.020、n=72 的显著优势是任务盲的噪声低点；**切片轴（每集 S 切片）S2 0.0330 / S4 0.0355 / S6 0.0207，E 增益 −0.005 → +0.009，S ≤ 4 时 E 学成身份、S6 才打开好半区**。读法：标签“数量”不是瓶颈，每集视角数是（= AutoCTS++ 11 集切 200 任务的对应物）；扩 shared 10→40 不值，要花钱花在每集加切片（S 6→12 ≈ $10，须预注册 S∈{6,9,12} 嵌套读数）。预测 2/6。事后换种子 3/4/5：切片 RISING 与配置 FLAT 均复现（6 种子 S 曲线 0.0368 / 0.0354 / 0.0242），但满量点 E_T 优势 −0.009 p 0.001 → −0.005 p 0.058（种子方差 0.007，T′ 部分答案：方向稳、量级六成、须 6 种子报），M1 好半区打开不复现。待做：R′/T′ 比较器新种子稳定性（T′ 已部分回答）；**L2** CLS 作流水线 DAG（`order` 字段 = 增强节点间的边）+ 共享 GIN——**已跑（2026-09-14，`results/tclsc_l2/REPORT.md`）：L2-FLAT**，gin_pipe_hub 0.761 vs binned MLP 0.758（p 0.21）、无边 none_mean 0.769 最高（p 0.026）但好半区 0.49–0.52 全低于 MLP 0.528，拓扑无信息、枢纽读出不如均值；**CLS 支路定为 binned MLP，关键决策 1 维持**。顺序 L1 → L3 → L2，每级零 GPU 先在 `results/mx3_pq` 缓存上判。注意：mx3_pq 的折特征 npz 与模型 .pt 不在本地（留在 pod 卷），凡吃 E 的臂都要先重跑 `33 --fit-folds`（GPU，须请示）。

## 用户正式主张（2026-09-13，覆盖本文件所有与之冲突的"主臂 / 放弃 / 判死"表述）

**论文主臂 = 从数据集特征得到 CLS 偏好（dataset-aware 预测式）。形式不限**：T-CLSC 式比较器、原始空间机制特征的规则、转导式（目标上一次短训的表示空间特征）、任何"输入数据集、输出该数据集的 CLS 偏好"的模型都算。
**默认排序器（源池先验 + SH）不是主臂**：能提供帮助最好（作辅助 / 兜底 / 对照），但不作为论文的主张；"追平 SH"不是目标函数，"预测式能否给出偏好"才是。
1. 此前所有"task-aware 判负 / 全线放弃 / 预测式关闭"的措辞只表示"某种形式在某口径下未成立"，**不改变主臂目标**；后续工作的任务是把预测式主臂做成立——找对的对象（哪个 CLS 维度的偏好是数据集级的）、对的口径（筛字段的训练口径必须等于部署口径）、对的特征（原始 / 转导 / 表示空间）。
2. 实验设计与读数码围绕"预测式主臂"设：门要问"预测器给出的偏好在留出集上是否成立、值多少"，不要问"是否追平先验 + SH"。
3. 已成立的正例（主臂证据链起点）：WP13c inst_norm ← level_scale_info（UCR 53 集，ρ −0.43，p 7e-4）；WP14 阶段 2 R1 highpass05 ← ret_highpass（20 集拟合、33 新集留出命中 73% vs 58%，ρ +0.62）；WP15 官方口径复现 ρ +0.67；事后待预注册：inst_norm ← ret_instnorm、rescale ← rep_ret_highpass05@60、polarity ← rep_ret_polarity@290。已知边界：增强偏好在 UEA 官方口径未兑现（H2-UEA），根因是筛字段口径 ≠ 部署口径，不是"预测式不成立"。
4. 汇报时不得把"预测式在某口径判负"说成"预测式路线关闭"；不得再提议把主臂换成任务盲默认。

**主张定稿（MX-4，用户 2026-09-03 拍板，取代出路 X；2026-09-13 起其中"头条 / 主臂"字样按上两节改读：零样本排名与 SH 是辅助与对照）**：
① **零样本排名**（头条）：任务无关的 CLS 质量排序器 = **源池先验**（shared 探针在源池全部切片×预算上 within-task z 分数的均值）+ **字段结构化代理**（GBDT 在 CLS 字段编码上拟合 within-task z，`autocls/search/surrogate.py`，目录外候选由它打分）。7 折 leave-one-dataset-out：先验@1 regret 0.046、代理配对准确率 0.78 / regret@1 0.056，均显著优于 T-CLSC（0.074 / 0.735）与手工强默认 GGS（0.068）（`results/mx4_prior_sh`、`results/mx4_surrogate`）。
② **目标侧逐次减半验证**：零样本名单前 6 条各训到 @60 → 留 3 条到 @120 → 3 条到 b_final 取最好（SH6-3-3）。@290 regret 0.018、中位 0、垫底 0/42，成本 ≈3.6 次全跑（与 AutoCTS++ 式 top-3 真评估同档，后者 0.041）。**最后一级必须留 3 条**（只留 1 条 regret 翻倍）。
③ **任务感知靠测量不靠预测**：数据集级 CLS×数据集交互真实存在（R' 方差分解 @290：任务 73% / CLS 13% / 交互 14%，交互均方 = 噪声 140×，几乎全在数据集级），由目标侧 @60/@120 截点直接读出；E / SetTF / 可行性预测器在"配对 δ 标签 + ≤8 数据集"口径下判负，T-CLSC 曾降为消融对照——**本会话起重启 T-CLSC（见顶部）**。
**两条已判负的具体形式**（判负的是口径，不是路线）：(a) **细粒度 task-aware 排序**在配对 δ 标签 + ≤8 数据集口径下不可辨识（M3/C2/MX-1 系列/实验 1-2/MX-3-PQ 五轮判负；δ 存活对里好-好对只 1.6–4.3%，任务盲天花板按 shared–shared 对诚实算是 0.86–0.92 而非 0.97）；(b) **开放空间零样本外推**：任何代理对目录外候选系统性高估（进化 CLS 预测 z +0.52 vs 真 +0.05，到训练集字段汉明距离 8 vs 目录 3），bag-10 LCB 只砍 44%，信任域一刀切全砍——搜索只在源池已评估配置邻域提名，且必须过目标侧真评估（BO 式回灌才可立项）。

完整设计见 `TECHNICAL_DESIGN.md`（已过期，待按 MX-4 重写）；MX-4 计划与诊断见 memory `mx4-diagnosis-prior-beats-tclsc`。

本机python环境：Anaconda 下的名为pytorch的虚拟环境

## 核心概念
- **CLS**：一个对比学习策略 = 19 个离散字段的配置向量（AutoCL Table 1）。这是"要搜的架构"。
- **零样本排序器**（MX-4）：先验 + 字段代理，输入 CLS 向量（不看任务），输出质量分；`surrogate.pairwise_matrix` 把分数变成 `autocls/search/rank.py` 可消费的 P 矩阵，胜场/Borda/进化一行不改。
- **逐次减半（SH）**：同一条预训练轨迹的截点 {60,120,290} 各评一次 R'，逐级淘汰；部署选中的就是那条轨迹（继续训练即可），缓存模拟与部署零近似。
- **T-CLSC**：AutoCTS++ 式比较器 `(E, cls_a, cls_b)→y`，**本会话的重启对象**（见顶部）。旧口径读数：MX-3-PQ 判 B-DEAD、在 δ 配对标签下退化为常数排序器（配对 acc 0.735 LODO / 0.866 同分布），偏离常数时 regret 翻倍——这些是重启要绕开的约束，不是不投入的理由。代码 `autocls/comparator/{tclsc_model,tclsc_engine,soft_labels,mean_pool_encoder,e_diagnostics}.py` + `autocls/task_embed/` 齐全，铁律 7 照旧。
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
- **源池规模不是瓶颈**（`results/mx4_source_size/REPORT.md`，2026-09-06）：7 折 × k=1..7 全部子集，先验前 6 + SH6-3-3 与 regret@3 在 ≤2 个源即到地板（regret@3 @290 对 k 平坦 0.031–0.034；SH k=4 0.0231 vs k=7 0.0184），只有单默认 top-1 到 6–7 源才稳定（9092e36e vs cffb5503 平局，regret@1 0.082→0.052）；逐靶方向相反（HW/UniMiB 加源变好，Ethanol/PS/Cricket 变差）。论文只写"先验在 ≤4 个同域源上即收敛"，不写"加源提升精度"。
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
- ✅ **WP10-UEA30 靶外 23 靶目标侧全量部署**（2026-09-06/07，分支 `wp10-uea30` 自 mx4，协议锁 a5b168f，`results/uea30/{PROTOCOL,REPORT,table30}.md`，脚本 `45_uea30_deploy.py` + 真源 `configs/uea30.yaml`）：UEA 30 里 6 个 LODO 靶之外的 23 个按官方口径部署（InsectWingbeat 用户拍板不跑），**作可评性分类的预注册检验**（A 贴顶 7 / B 短序列 3 / C n_test≤50 3 / D chance-EEG 5 / E 有净空 5）。每靶 22 格 = shared-10 + GGS + `input_mask=binomial` 孪生，不提名；先验对非池靶全 8 源、SCP2 LODO，排序器不重训；EigenWorms 训练侧官方 `max_train_length=3000` 分段（`uea.segment_train`）。**UEA30-WIN+TWIN**：主臂 SH10-4-3_prior（掩码关，5.0 当量）3 种子追平 TS2Vec **23/23**（全 30 含 LODO 6 + HAR = 30/30），U1 E 类 5/5（PEMS-SF 0.771 / SCP1 0.881 / ERing 0.947 / NATOPS 0.950 超论文 0.02–0.09，EigenWorms 0.822 刚追平）、U2 A/B/D 带内 14/15（FaceDetection 0.597 **向上**出带，TS2Vec 0.501）、U3 副臂 SH20-3_both Δ +0.001 / 20/23；全 29 均值 0.733 vs TS2Vec 0.712；对目录 oracle22 regret 中位 0.001；冠军先验第 1 名 7 靶、第 7–10 名 6 靶（名单 10 再坐实：主臂 ≥ SH6-3-3 于 21/23）。预测 5/8（P5 冠军更分散、P7 墙钟 6.9 h 低于预估、P8 均值高出带 0.001 均为"更好"方向）。成本 4090 pod 8.24 h ≈ $6.1，22 格锚总偏差 7%（逐靶 ×0.3–×5）。**D 类"chance 型"预测在官方口径下偏保守**（FaceDetection / Heartbeat 有信号）写 limitation；AutoTCL/InfoTS 逐靶数字待补录
- **WP3 之后的名单侧待办（用户 2026-09-04 记录）**：① 起跑名单按先验置信度自适应（先验前 6 的 z 平坦时扩到 10；HW 上 SH10-4-3 事后选中 GGS 0.546）——**WP6 已在官方口径 3 种子坐实 SH10-4-3 六靶追平**；② 提名配置与 shared 混排进起跑名单（现行 hybrid 的 +10 偏置让提名从未起跑，HMD 上提名是目录最高 0.486）；③ 噪声带靶（Ethanol 类，候选全在 0.06 带内）3 种子复核后再宣布冠军。淘汰赛规则本身不动（关键决策 8）。
- MX-4 WP4（增强不变性画像作数据集级任务特征）默认不做，除非用户明确要保留"E 有用"的可能性
- ✅ **MX-4 WP5 学习曲线外推器**（2026-09-04，分支 `wp5-lc`）：**LC-DEAD**——SH 淘汰规则换成外推器预测，头条臂 0.0184→0.0183（4 胜/4 负/34 平），省档位 SH4-2-2/6-3-1 追不平（0.035）；增益 LODO R² 0.02 = 测量地板（`results/mx4_lc_extrapolator/REPORT.md`）——负翼
- ✅ **MX-4 WP6-TF 任务特征线**（2026-09-04，分支 `wp6-tf`，`results/mx4_task_feature/REPORT.md`）：**TF-DEAD**——数据集级任务特征进零样本打分器，A 预测型（12 维 φ_meta/base/sig ⊕ CLS 字段残差 GBDT，LODO）入前 6 @290 17/42 vs hybrid 26、regret@3 持平（p=0.60），5 个 φ 置换臂把真臂包在中间；B 转导型（前 6 条 @60 实测分当 φ_B，追加 2 条 = SH8-3-3）0.0182 vs 同成本对照 0.0200 但入 8 只 +2、vs 无 φ_B p=0.30，且 +2 任意候选的精确随机期望 0.0181 同值；同数据集身份上界存在（regret@3 0.031→0.014）但跨数据集学不到、6 个实测数读不出——交互是数据集特异的，n=7 不可辨识；官方口径旁证：追加先验第 7–8 名在 HW 带进 GGS（SH8 命中 0.546，成本 1170），提名靠代理分混排进前 6 则 5/6 靶更差——负翼
- ✅ **MX-4 WP7-UCR 任务感知模块**（2026-09-05，分支 `wp7-ucr`，`results/ucr_module/REPORT.md`）：**UCR-U1-DEAD**——把数据集数从 8 放大到 84（UCR 单变量语料，剔除 9 个同源；84 × 20 CLS × 三截点 = 5040 行 R'，5090 16.2 GPU-h），用 14 维数据集特征训 f(CLS, φ)→z，留一数据集下 hgb regret@3 0.0209 vs 任务盲 pooled 0.0199（p=0.41），与 5 个 φ 置换臂同带；孪生匹配 0.0186（p=0.083）；语料交互份额 9.6%（含 k=1 噪声）、26/84 贴顶；U2/U3 按协议未跑。**预测型零样本 task-aware 在本栈关闭（n=8 六轮 + n=84 一轮）**——负翼
- ✅ **MX-4 WP8-BO 目标侧提名回灌搜索**（2026-09-05，分支 `wp8-bo`，`results/mx4_bo/REPORT.md`）：**BO-DEAD**——判死 (b) 留的唯一出口（信任域汉明 ≤ 3 提名 + 靶侧 @25% 真评估回灌，bo_ucb vs 同成本 rand_nbr 各 3 轮 × 5 条，官方口径 6 靶 + HAR，本地 4060 7.4 GPU-h）：B1 头顶空间 1/6（只有 Ethanol）、B2 BO-SH 3 种子 vs SH6-3-3 均值差 −0.011（4 胜但 HMD −0.153 / PS −0.052）、B3 引导 3/6；瓶颈 = 回灌信号是早截点（池内 ρ(@25%, @100%) 0.22–0.79，@100% 最高的提名被第一级淘汰，代理只追 @25% 赢家），是关键决策 8 的同一地板。旁证：Ethanol 目录外 0.373（3 种子）> TS2Vec 0.308（目录非天花板）；HW BO-SH 0.497 过 TS2Vec 线但 SH10-4-3 拿 GGS 0.546 更便宜（名单问题）。**开放空间搜索关闭**——负翼
- 论文：正翼 = 零样本排名 + SH 验证 + 字段主效应 + 成本表；负翼 = task-aware 五轮判负 + 开放空间外推判死 + 学习曲线外推判死（analysis/limitations）

## 路径一（WP9，下一步，用户 2026-09-06 拍板新开会话执行）与 路径二（已全部关闭）

**路径一 = 把证据做实**：官方口径 6 靶（+HAR 可选）每靶候选池扩到 80–100 条（shared-10 + GGS + 现有一次性提名 10 + 空间均匀随机补足，seed 0 完整 25/50/100% 轨迹），在缓存上零 GPU 模拟同成本对照：先验前 6 + SH6-3-3（3.75 全跑当量）vs Hyperband（η=2，三 rung = 三截点）/ 随机搜索 / 随机起点 SH / BOHB-TPE 式；冠军 3 种子；UCR 84 语料先验 + SH regret 表（零 GPU）；消融主表；AutoCL 搜索预算换算进成本表。估时 4090 ≈ 5.4 h / 5090 ≈ 3.5 h；PS 评估须 480 s 上限。交接细节见 memory `paper-positioning-and-path-one-handoff`。

**论文定位（2026-09-06；其中"task-aware 全线放弃"已被 2026-09-13 用户正式主张与本会话 T-CLSC 重启覆盖，下文只作当时口径存档）**：task-aware 全线放弃。"首次"只能写成 "first to *learn* a task-agnostic CL strategy default from cross-dataset real evaluations, validated by target-side SH"，AutoML 前人必引（Pfisterer *Learning Multiple Defaults*、Winkelmolen *zero-shot HPO*、Feurer *Auto-sklearn 2.0* portfolio+SH、Probst *Tunability*）；"快"须等路径一的同成本基线；最硬的贡献是负翼（七轮预注册 task-aware 判负 + 噪声地板 + 开放空间早截点机制）。投稿：现证据 → CIKM/ECML-PKDD/ICDM（CCF-B）、AutoML Conf、TMLR；路径一后 → AAAI/IJCAI；R' 语料打包 → NeurIPS D&B。

### 路径二存档（用户 2026-09-03 记录；WP5/WP6-TF/WP7-UCR/WP8-BO 全部判负关闭）

**为什么**：现在的头条"先验 + 淘汰赛"审稿人会问"学习在哪"——先验 7 折第一名都是同一条配置（等于一个学来的默认），GBDT 在 top-1 上没超过先验，淘汰赛是 Hyperband 的常规组件。路径一（WP3 全量协议 + 候选扩到 50–100 + Hyperband/随机/BOHB 基线）把证据做实；路径二给"从源池学到、在目标上用"加一个有实证增益的模型。

**WP5 学习曲线外推器：已关闭（关键决策 8），本节只留存档，勿再展开讨论。**
读数 LC-DEAD（`results/mx4_lc_extrapolator/REPORT.md`，memory `mx4-wp5-lc-extrapolator-dead`）：头条臂 0.0184→0.0183（4/4/34 平），省档位追不平（0.035）；根因 = 增益 R'@290−R'@60 的 LODO R² 0.02，标签噪声地板淹没"晚熟"信号。
留给后续立项的两个硬事实：**SH6-3-3 的 0.0184 里 0.0118 是先验 top-6 不含 oracle 的不可约部分**（淘汰规则最多改 0.0067），**最后一级留 3 条是硬要求**（任何规则只留 1 条都翻倍）。

**次选：WP4 增强不变性画像**（机制性任务特征，与 CLS 字段同坐标，n=7–8 也可辨识；风险更高，先验 ~30%），设计见 memory `mx4-diagnosis-prior-beats-tclsc` 与计划文件 §5。
**WP6 之后的读法（2026-09-06）**：掩码字段把 task-aware 的目标缩成"每个数据集一个比特"（效应 0.03–0.1），但 `results/mx4_mask/mech_probe.md` 显示 n=6 下任何数据集描述量都不比"只用配置字段"更好——任务特征线仍不立项，除非数据集数量上去。

## 目录
`autocls/{space,contrastive,encoders,task_embed,comparator,downstream,search,data}`（`search/surrogate.py` 代理、`search/rank.py` 聚合、`search/learning_curve.py` 外推器（判死，保留）、`search/evolution.py` 仅保留）+ `scripts/`（MX-3-PQ：`32–34_pq_*`、`pq_common.py`；MX-4：`36_prior_sh_sim.py`、`37_surrogate_loo.py`、`38_lc_sh_sim.py`、`39_sh_deploy.py`（`--twins` 孪生目录、`execute(loader=)`）、`40_task_feature_scorer.py`（WP6-TF）与 `40_mask_field_sim.py`（WP6 掩码：s0/k1/c0/source/deploy）——两个文件同用编号 40、文件名不同不冲突；`41_ucr_rprime.py` / `42_ucr_module.py`（WP7-UCR）、`43_bo_nominate.py`（WP8-BO，含 Windows 评估超时 `install_windows_eval_timeout`）、`44_source_size_curve.py`（源池规模曲线，零 GPU 缓存模拟范式）、`45_uea30_deploy.py`（WP10：build-candidates/check/dry-run/smoke/catalog/analyze/finals/table30，真源 `configs/uea30.yaml`）、`46_p1_pool.py`（WP9-P1 路径一，真源 `configs/p1.yaml`）；`32_pq_generate_rprime.py --twins` 源侧孪生补标；pod 脚本 `pa_pod/wp6_*.sh`、`pa_pod/uea30_{bundle,setup,run}.sh` + `uea30_fetch.py`、`pa_pod/p1_*.sh`、`ucr_pod/*`、`pq_pod/twins_*.sh`）+ `configs/*.yaml` + `tests/`。分支：**`wp9-p1` 是当前超集**（wp8-bo 系：`mx4` / `protocol-alignment` / `wp5-lc` / `wp6-tf` / `wp7-ucr` / `wp8-bo`，+ `wp10-uea30`（自 mx4，含 `wp6-mask` 掩码字段与 UEA30）已合入）；worktree：`.claude/worktrees/mx4`（WP3）、`.claude/worktrees/wp10-uea30`（`data` 是指向主目录 `data` 的 junction）保留。`results/route_a_probe/` 未跟踪，新 worktree 须从主目录拷贝否则 ~110 个单测 FileNotFound。
