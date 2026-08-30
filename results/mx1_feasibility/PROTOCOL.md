# MX-1 协议：可行性预测器——数据集毒性水平的 leave-one-dataset-out 可预测性（判读带先于计算写死）

日期：2026-08-30 ｜ 脚本：`scripts/15_feasibility_labels.py`（零 GPU）+
`scripts/16_feasibility_loo.py`（轻 GPU：16 折 TS2Vec） ｜ 单测：
`tests/test_feasibility_predictor.py` ｜ 本协议与测试、实现先 commit（协议锁），
之后才允许跑数据出报告。

## §1 问题与主张边界

破坏矩阵（`results/destruction_matrix/REPORT.md`）已坐实**存在性**：同 10 条 shared
CLS @290，数据集破坏率 0→0.75（cls_only 零模型 p<0.001）。本实验把它升级为
**可预测性**：训练 (E, cls) → 三态 {提升/惰性/破坏} 的预测器，16 折
leave-one-dataset-out 检验它能否预判**未见数据集**的空间毒性水平，且 E 有增量。

主张边界（先写死）：
- 头条主张只到**数据集级毒性水平/三态分布预测**。(task,cls) 级细粒度翻转图
  未坐实（destruction REPORT §0.3，2 任务/格功效不足），其结果如实报但不当头条。
- **三态都记分**。inert 数据集（MotorImagery/SCP1/HMD）破坏率 0 是"训不动"不是
  "安全"——若退化成二分类破坏检测，这三个会被误读为安全区。主指标取三态分布
  距离而非破坏率单项。
- **label 绑 @290 口径**。破坏率是 iters 的函数（@290 0.24 vs 工作点 0.07，
  destruction REPORT limitation 3），所有主张限定 @290，行级记录带 `iters` 字段。

## §2 数据源与配对（先于计算写死；全部现成缓存，label 生成零 GPU）

面板 = 16 UEA 数据集 × 前 2 任务（`<ds>_00/_01`）× shared 10 CLS = **320 cell**。
**样本量如实声明**：全部信息只有这 ~320 格（每折训练 300 例）。因此模型保持
极小（§4.2 小 MLP）、主判据放数据集级（§5）；若判别力不足，"宽任务面板"扩容
是预留升级分支（§6 R4），**不默认跑**。

- **R'（trained）**：
  - k=3 均值优先（13 数据集×2 任务×10 CLS×3 种子 = 780 行）：
    `results/stage1_rerun_20260804/results/step0_snr_gate/noise_cache.jsonl`，
    复用 `scripts/14_destruction_matrix.py::collect_trained_k3`（恰 3 rep 才成 cell，
    <3 rep 丢弃计数，**不静默回退 k1**）。
  - k=1 canonical 补齐 3 个骨干（Cricket/PhonemeSpectra/SpokenArabicDigits，60 行）：
    `results/baseline_probe/trained_fill.jsonl`，复用 `collect_trained`（iters=290）。
    行级 `r_source ∈ {k3, k1}` 声明；k1 单种子噪声为已知局限（§6 R1 敏感性覆盖）。
    **并表规则写死**：k1 只用于 k3 零覆盖的数据集（即 3 骨干）；k3 有覆盖的数据集
    出现 k1 行一律拒收计数，禁止同数据集混源。
- **基线**：`results/baseline_probe/baseline_cache.jsonl`（iters=0，5 种子，剔
  task_id 含 "+c"），复用 `collect_baselines`。
- **配对校验**逐字复用 scripts/14 的实现：同 task_id、classes 逐字段一致、
  split_seed == canonical（`run_seed_for(tid,"train_eval_split")`）、cls_key ∈
  shared 10（`results/route_a_probe/probe_cls_catalog.json` 前 10）、iters=290。
  拒收计数如实入 `label_audit.json`。预期 320/320 满格（destruction REPORT 同源）；
  缺 cell 照跑并报告，某数据集有效 cell <10 则该折剔除（如实报）。

## §3 三态 label（与 scripts/14 逐字同式）

任务基线 mean/σ 取 ≥3 种子；band = 2·max(σ, 1/n_eval)（`summarize_task` 同带，
quantum 取该任务基线行 1/n_eval 的最大值）：

- **破坏** ⇔ R' < mean − band（复用 `destruction_flag`，严格小于）
- **提升** ⇔ R' > mean + band（镜像式，严格大于，与 `summarize_task` 的
  n_trained_above_null 同判）
- **惰性** ⇔ 其余（含落在带上，浮点边界不做特判——commit f2da547 教训：
  0.8−0.1 非精确二进制，测试夹具用带公差断言）

落盘 `labels.jsonl`：{dataset, task_id, cls_key, classes, iters:290, r_source,
r_prime, base_mean, band, label}（classes 供 16 脚本再切片时逐字段校验任务定义；
band 已含 max(σ, 量子) 信息，不另存 σ）。label 分布审计表随 audit 出，但
**判读带（§5）不依赖分布数值**——分布本身就是被预测对象。

## §4 预测器与 16 折 LOO（管线先于计算写死）

### §4.1 按折任务特征（铁律 6 按折生效）

每折留 1 个数据集：**该折共享 TS2Vec 只在其余 15 个数据集上 fit**
（`fit_shared_ts2vec`，600 iters / embed_seed=0 / 每数据集单变量实例封顶 4096，
参数与 `configs/labels.yaml: route_a` 同口径；实例池构造复用 2b
`build_ts2vec_pool`——采样种子按数据集名派生，与折无关）。留出数据集只做
**冻结前向编码**（推理非训练）。折缓存按目录隔离：
`results/mx1_feasibility/folds/<heldout>/`，`shared_ts2vec.pt` + meta 记
fit_datasets 指纹（复用 2b `get_shared_ts2vec` 的指纹校验语义，防半池漂移）。

**折间单变量原则（写死）**：SetTF 初始化（`set_encoder_init.pt` 全局一份）与
embed_seed=0（窗口采样/编码种子）跨 16 折固定；**每折唯一被允许变动的量 =
TS2Vec 的 fit 池**（及由此产生的编码器权重）。否则折间方差混入 init 抖动，
稀释 D1 的 Wilcoxon 功效。实现以断言+单测固定。

每折对全部 32 个任务（15×2 训练 + 1×2 留出）产特征，一次前向同时出两种：
- **E**（主特征，nz=256）：`embed_task`，SetTF 用全折共享的同一份初始化
  （torch.manual_seed(embed_seed)，落盘 `set_encoder_init.pt`，冻结推理）；
- **E_mean**（敏感臂特征）：windows_repr 对 (窗口,时间) 取均值（TS2Vec 级基线，
  task_embed_sanity 同款）。

与 2b `ensure_task_features` 的偏差声明：**不落盘 windows_repr**（SCP2 级任务
单个 118 MB × 32 任务 × 16 折不可行），只存 E/E_mean 两个 256 维向量。

**SetTF 不联合训练**（与 tclsc 骨架的偏差，先声明理由）：联合训练需每 epoch
对 30 任务的 (100, T≤1152, 256) 窗口表征做 SetTF 前向，16 折×5 种子×3 消融臂
的量级不可行；且 task_embed 定稿诊断表明冻结 init SetTF 的 E 已有清晰区分力
（0.99/0.65/0.92），M3 亦证明联合训练不是 E 生效的充分条件。风险如实记录：
若冻结随机 SetTF 恰好抹掉毒性相关信息，本设计**只会低估** E——伤 P2 不伤
主判有效性；E_mean 敏感臂（§6 R2）提供第二条通路。

### §4.2 预测器（从简，容量让位于消融干净）

小 MLP：输入 z ⊕ v_cls（z = E 经折内标准化——按该折 30 个训练任务的逐维
mean/σ z-score，σ 下限 1e-6，**留出任务用训练折统计量**，无泄漏；v_cls =
`CLSEncoder.encode`，现成），隐层 [128, 64]+ReLU，3 类 softmax 头。
训练：Adam lr=1e-3，weight_decay=1e-3，全批，300 epoch 固定（**无 early
stopping**——不允许任何指标窥视留出折），类别加权 CE（权重 = N/(3·count)，
count 下限 1、权重上限 10，按折训练 label 算）。**CPU 训练**（模型微小，换
逐位确定性；GPU 非确定性 memory 教训）。种子 S=5（折×臂×种子派生），cell 预测
= 5 种子 softmax 均值。折训练集 = 15 数据集 × 20 cell = 300 例。

### §4.3 对照臂（全部预注册，同折同种子）

| 臂 | 定义 | 用途 |
|---|---|---|
| **B0 任务盲·全局率** | 该折 300 训练 cell 的三态全局率（常数向量） | **主判据 D1 的对照**——数据集级"任务盲"即常数预测 |
| **B1 任务盲·per-CLS 率** | 该折每 CLS 30 训练 cell 的三态率 | cell 级次判据对照（数据集级聚合后与 B0 重合，因面板均衡） |

**任务盲基线必须按折算（显式断言）**：B0/B1 只允许从该折 15 个训练数据集的
300 个 cell 统计，留出数据集的 label 绝不参与——否则 D1 的对照自身泄漏。
实现层断言 heldout ∉ 基线统计源；单测以"改动留出 label 不改变 B0/B1"验证。
| **B2 元数据臂** | 同 §4.2 MLP，E 换 4 维元数据 [log10 l_enc, log10 C_task, n_classes_task, log10 n_train]（折内 z-score） | "元数据可替代"归因诊断，不做门 |
| **A1 cross-E 消融（主消融）** | 真模型 + 评估时留出任务的 E 换成其他数据集随机训练任务的 E（4b `cross_assignment` 同语义，5 个 assignment 种子×5 模型种子取均值） | **D2 的对照**——杀数据集身份 |
| **A2 mean-E 消融（辅助）** | 评估时 E 置折训练均值（标准化空间零向量） | 辅助，不做门 |
| **A3 cls-only 重训（诊断）** | E 通道整个去掉重训 | 量化 cls 泛化单独贡献；数据集级应塌缩为常数（一致性自检） |

注：数据集身份**只经 E 进入模型**（每数据集拿到同样 10 CLS×2 任务），故数据集级
凡赢 B0 必然吃了 E——D2 是一致性/幅度检验而非独立关卡；真正风险方向是
"E 携带信息但 15 数据集训不出来"（D1 失败）。

## §5 量、检验与判读带（先于计算写死）

**数据集级（主）**：折 d 的观测三态率 p(d) = 20 个留出 cell 的 label 频率
（3 维向量）；预测率 p̂(d) = 同 20 cell 预测概率向量的均值（**概率均值，非
argmax**）。主误差 = 总变差 err_TV(d) = ½·Σ_s |p̂_s − p_s|。头条可读数 =
破坏率单项 |p̂_破坏 − p_破坏| 与 16 点 Spearman(p̂_破坏, p_破坏)（报告用，不做门）。

**cell 级（次）**：320 个 LOO cell 预测的三分类多类 Brier（proper、容忍退化折）
逐折均值对比 B1；池化 macro-F1/混淆矩阵描述性报告。

**err_TV 噪声地板参照线（预先声明）**：留出折"真实分布"由 20 格估计，自带
多项抽样噪声。报告须逐折附 oracle 参照线——已知真分布的理想预测器在 20 格
多项抽样下的期望 err_TV（以观测率代入，10⁴ 次多项模拟均值，种子写死）。
模型/基线的**绝对** err_TV 对着该地板读（谁也到不了 0）；D1 只做相对比较，
模型与基线面对同一地板，公平性不受影响。

**检验**（16 折配对，exact Wilcoxon signed-rank，单侧，α=0.05）：
- **D1**：err_TV(模型) < err_TV(B0)，p<0.05。
- **D2**：err_TV(模型) < err_TV(A1)，p<0.05。
- S1（次）：Brier(模型) < Brier(B1)，p<0.05——只对 P3 记账，不做门。

**D2 逐折功效异质性（预先声明）**：cross-E 换入的 donor E 多来自毒性接近
全局均值的数据集——毒性极端折（NATOPS 0.75、三个 inert 0）预期退化大，
中游折（HW/CT ≈0.2）预期退化天然小。这不是缺陷：**D2 判定只用跨折聚合
检验，单折读数不作"消融失败"解读**。诊断随报告出：逐折退化幅度
Δ(d)=err_A1−err_model 与折毒性极端度 |p_破坏(d)−全局破坏率| 的 Spearman。

**判读带**：

| 条件 | 判定 |
|---|---|
| D1 ∧ D2 | **成立**：MX-1 主结果（任务感知毒性水平预测），进论文正翼；E 消融预注册达成 |
| D1 ∧ ¬D2 | **E 增量未证实**分支：如实报告；按 B2/A3 归因——若元数据臂或 cls-only 同样过 D1 对应检验，主张降格为"毒性可预测但驱动信号待归因" |
| ¬D1 且（p_D1 < 0.15 或 E_mean 敏感臂过 D1） | **边缘**：交用户拍板（候选出路：宽任务面板扩折内分辨率） |
| 其余 | **不成立**：毒性差异存在（p<0.001 存在性保留）但 15 数据集训不出可迁移预测器；MX-1 降级为存在性结果 + limitation |

## §6 稳健性与诊断（预注册，随主判一并出，不改判读带）

- **R1 k3-only 重跑**：剔除 3 个 k1 骨干，13 折重算 D1——方向须一致，翻转则如实
  报并拍板（k1 单种子噪声声明的兑现）。
- **R2 E_mean 敏感臂**：主管线整套换 E_mean 重跑 D1/D2（冻结随机 SetTF 风险的
  第二通路）。
- **R3 inert 三数据集读法自检**：MI/SCP1/HMD 三折的预测分布应为惰性主导而非
  提升主导（"破坏率 0 ≠ 安全"的直接检验），见 P6。
- 诊断：p̂_破坏 与 L 的 Spearman（已知观测端 ρ≈−0.2 n.s.，模型若强烈跟 L 走
  则怀疑吃了长度混淆）。
- **R4 宽任务面板升级分支（预留，不默认跑）**：判读带落"边缘"或用户拍板后
  才立项——4–6 个跨脆弱度数据集 × 8–16 任务 × shared+random CLS × k≥1
  （destruction REPORT §4.2 设想，成本与老造 label 同量级），同时解锁细粒度
  翻转图的功效问题。本协议任何判定不依赖该分支。

## §7 预测（先验，先于计算写死）

| # | 预测 | 先验 |
|---|---|---|
| P1 | D1 成立（模型逐折赢任务盲常数基线，p<0.05） | 55% |
| P2 | D2 成立（cross-E 消融显著退化） | 50% |
| P3 | S1 成立（cell 级赢 per-CLS 任务盲基线） | 35% |
| P4 | Spearman(p̂_破坏, p_破坏) ≥ 0.5（16 点） | 45% |
| P5 | B2 元数据臂**不过** D1 对应检验（E 信息超出元数据） | 55% |
| P6 | MI/SCP1/HMD 三折预测惰性率 > 预测破坏率且 > 预测提升率 | 60% |

## §8 执行、成本与产物

- **顺序**：本协议 + tests + 两脚本实现 → **协议锁单独 commit** → 15（零 GPU，
  labels.jsonl + label_audit.json）→ 16 --fit-folds（GPU）→ 16 --train/--analyze
  （CPU）→ REPORT.md。
- **成本（4060 本机口径，锚点带硬件标签）**：折 TS2Vec fit 600 iters ≈2–5 min
  ×16；折特征编码 32 任务（MotorImagery 64ch×100 窗为最重）≈5–10 min×16；合计
  **≈1.5–3 h GPU**。预测器 16 折×6 臂×5 种子全 CPU，分钟级。
- **GPU 规程**：开跑前清桌面负载（Wallpaper Engine 等）+ nvidia-smi 核对（评估集
  探针 §7 规程）；折 meta 记 device_name。长任务 Start-Process 脱离 + `--resume`
  幂等（逐折落盘，指纹校验防半池漂移；收官重验折数/文件完整性，勿信中途计数）。
- **产物**：`results/mx1_feasibility/{PROTOCOL.md, labels.jsonl, label_audit.json,
  set_encoder_init.pt, folds/<ds>/{shared_ts2vec.pt, shared_ts2vec.meta.json,
  task_features/*.npz}, predictions.jsonl, analysis.json, REPORT.md}`。
- 铁律 6 自检随 16 脚本跑：每折断言留出数据集 ∉ fit_datasets；目标池
  （UCI-HAR/Epilepsy/FaceDetection）全程零参与（不在 16 面板内）。

## §9 MX-2 预埋备忘（本步不执行）

MX-2（端到端展示）的已拍板决定另立备忘 `docs/MX2_PREREGISTER_MEMO.md`
（与 memory `outlet-x-confirmed-roadmap` 同款）：扩源对照臂（UM-only vs
UM+15 UEA）、按任务 28 实例 regret、外部锚（TS2Vec 论文公开数字）、同机铁规。
届时 MX-2 协议必含，本协议不承载其判据。
