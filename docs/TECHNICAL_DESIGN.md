# AutoCLS 技术设计文档

**Zero-Shot NAS for Contrastive Learning Strategy Search**

结合 AutoCL（对比学习策略空间）与 AutoCTS++（零样本 NAS 搜索机制），实现一个面向时序对比学习策略（CLS）的零样本神经架构搜索框架。

---

## 1. 项目定位

**一句话定义**：给定一个未见过的时序数据集与下游分类任务，无需为它做任何对比预训练搜索，直接用一个预训练好的比较器（T-CLSC）从策略空间中排出最优的对比学习策略（CLS），再用固定 encoder 按该 CLS 预训练一次即可。

**与两篇源论文的对齐关系**：

| 维度 | AutoCTS++（预测 / 有源码） | AutoCLS（本项目 / 对比学习） |
|---|---|---|
| 搜索对象 | arch-hyper（ST-block 架构 + 超参） | **CLS**（对比学习策略，AutoCL Table 1） |
| 搜索空间编码 | 双 DAG + Hyper 节点 → 邻接/特征矩阵 | **CLS 向量 → MLP 编码**（方案 A） |
| 任务表示 | TS2Vec + 两层 Set-Transformer | **完全复用**（时序通用） |
| 比较器 | T-AHC：(task, ah₁, ah₂) → 二值 | **T-CLSC**：(task, cls₁, cls₂) → 二值 |
| 训练标签 | early-validation（k=5 epoch 训练预测模型） | **early-validation**（early_iters=290 对比预训练 + 下游评估） |
| 被评估的模型 | 完整 ST-block 预测模型 | **固定 encoder + CLS 预训练 → 线性/SVM 下游评估** |
| 零样本目标 | 未见数据集 / 预测设置 | 未见数据集 / 下游任务 |

**核心洞察**：AutoCTS++ 能做到零样本，是因为 T-AHC 学到了"任务特征 → 架构排名"的映射。本项目复现的是"任务特征 → CLS 排名"的映射。数据端（TS2Vec 任务嵌入）可原样复用，改动集中在**搜索空间编码**与**候选评估的 label 生成流程**。

**关键设计决策（已确认）**：
1. CLS 编码采用**方案 A：向量 + MLP**（不做图 + GIN）。理由：CLS 本质是配置向量，向量编码建模诚实、落地快、风险隔离好；novelty 落在"首个用于对比学习策略的零样本 NAS"，与编码方式无关。
2. 固定 encoder **两者可切换**：TS2Vec 式 dilated CNN（与 AutoCL 一致）与轻量 Transformer，由 config 控制。
3. 第一阶段**只做分类任务**（HAR / Epilepsy），最快验证核心假设。

---

## 2. 数据流全景

```
【离线：预训练 T-CLSC】
源数据集集合 ──切片──> 大量任务 t_i = (D_i, 分类任务)
                          │
      ┌───────────────────┼────────────────────┐
      │                   │                    │
   TS2Vec+SetTF        采样 CLS 候选        对每个(t_i, cls)算 label
   得任务嵌入 E_i      cls∈策略空间         固定encoder按cls预训练290 iters
      │                   │                → 下游acc = R'(cls)  (early-val)
      │                   │                    │
      └──────► 组装样本 (E_i, cls_a, cls_b, y) ◄┘   y=1 若 R'(cls_a)>R'(cls_b)
                          │
                  BCE 训练 T-CLSC
                          │
                  得到 N* (预训练好的比较器)

【在线：零样本搜索（未见任务 t）】
未见数据集 D ──TS2Vec+SetTF──> E
策略空间 ──进化算法+N* 两两比较──> CLS 排名 ──取 top-K──> 每个真正预训练+评估──> 选最优 cls*
```

**必须牢记**：T-CLSC 训练阶段用 early-validation（只预训练 early_iters=290，=full 的 1/8）快速造 label；只有最终输出的 top-K 才做完整预训练。这是 AutoCTS++ 效率的核心，不能省。

---

## 3. 目录结构

```
AutoCLS/
├── README.md
├── CLAUDE.md                 # Claude Code 项目约定（精简）
├── requirements.txt
├── configs/
│   ├── space.yaml            # CLS策略空间定义（唯一真源）
│   ├── encoder.yaml          # encoder选择: dilated_cnn / transformer
│   ├── pretrain_tclsc.yaml   # T-CLSC预训练超参
│   └── search.yaml           # 零样本搜索超参（进化算法等）
├── autocls/
│   ├── space/
│   │   ├── cls_space.py       # CLS定义、采样、合法性约束
│   │   └── cls_encoder.py     # 【方案A核心】CLS向量化
│   ├── contrastive/
│   │   ├── augmentations.py   # 6种数据增强 + 5种顺序（AutoCL附录A）
│   │   ├── emb_transform.py   # 嵌入jitter/mask/norm
│   │   ├── pair_construction.py # instance/temporal/cross-scale + 层级pooling
│   │   ├── losses.py          # InfoNCE / Triplet × Dot/Cosine/Euclidean × temp
│   │   └── cl_trainer.py      # 给定cls+固定encoder，跑对比预训练
│   ├── encoders/
│   │   ├── dilated_cnn.py      # TS2Vec式10层dilated CNN
│   │   ├── transformer.py      # 轻量transformer encoder
│   │   └── registry.py         # 按config切换
│   ├── task_embed/            # 【从AutoCTS++搬运】
│   │   ├── ts2vec_wrapper.py   # 封装其Ts2vec_model
│   │   └── set_transformer.py  # 搬其AHC/set_encoder
│   ├── comparator/
│   │   ├── tclsc_model.py      # T-CLSC: (E, cls_a, cls_b)->y 【核心创新点】
│   │   └── tclsc_engine.py     # 改自AutoCTS++ ahc_engine.py
│   ├── downstream/
│   │   └── linear_eval.py       # 分类下游: SVM(RBF) 或 linear probe
│   ├── search/
│   │   ├── evolutionary.py      # 进化算法+Round-Robin（AutoCTS++ 3.3）
│   │   └── zero_shot_search.py  # Algorithm 2 对应实现
│   └── data/
│       ├── datasets.py          # HAR/Epilepsy加载
│       └── task_slicer.py       # 源数据切片造任务（AutoCTS++ 3.2.4）
├── scripts/
│   ├── 1_prepare_data.py        # 下载/预处理HAR/Epilepsy
│   ├── 2_generate_labels.py     # 造(t,cls_a,cls_b,y)样本（最耗时）
│   ├── 3_pretrain_tclsc.py      # 预训练比较器
│   ├── 4_zero_shot_search.py    # 未见任务上搜索
│   └── 5_evaluate.py            # 对比baseline
└── tests/
    ├── test_cls_space.py
    ├── test_cls_encoder.py
    ├── test_augmentations.py
    └── test_end2end_tiny.py
```

**AutoCTS++ 源码复用对照**（仓库 `decisionintelligence/AutoCTS_plusplus`）：
- `Ts2vec_model/` → `autocls/task_embed/ts2vec_wrapper.py`
- `AHC/set_encoder/` → `autocls/task_embed/set_transformer.py`
- `AHC/ahc_engine.py` → `autocls/comparator/tclsc_engine.py`（把 GIN 支路换成 MLP）
- `exps/random_search.py` → `scripts/3_pretrain_tclsc.py` + `scripts/4_zero_shot_search.py`

---

## 4. CLS 策略空间（AutoCL Table 1）

一个 CLS 是 19 个字段组成的定长配置。字段清单（严格照 AutoCL Table 1）：

**数据增强（6 个增强各一个比率参数 + 顺序）**
- resize, rescale, jitter, point_mask, freq_mask, crop：取值 {0.0, 0.1, …, 0.9, 0.95}（11 档，0.0=不启用）
- order ∈ {0,1,2,3,4}（照 Table 8 的 5 种预设顺序）

**嵌入变换**
- emb_jitter ∈ 11 档，emb_mask ∈ 11 档
- norm ∈ {None, LayerNorm, l2}

**对比对构造**
- instance ∈ {True}（恒 True，最不可或缺）
- temporal ∈ {True, False}，cross_scale ∈ {True, False}
- kernel ∈ {0,2,3,5}，pool ∈ {Avg, Max}，adj_neighbor ∈ {True, False}

**对比损失**
- loss_type ∈ {InfoNCE, Triplet}
- sim_func ∈ {Dot, Cosine, Euclidean}
- temperature ∈ {1e-2, 1e-1, 1e0, 1e1, 1e2}

空间规模约 3×10¹²，用作 sanity check。

**合法性约束**：
- instance 恒 True；至少一个对比维度开启（instance 已保证）。
- kernel / pool / adj_neighbor 仅在 temporal 或 cross_scale 开启时有意义；关闭时置为"无效档"而非随机值，避免比较器学到噪声。

---

## 5. CLS 向量编码（方案 A 核心）

把一个 CLS 编码成定长实数向量供 MLP 使用：

| 字段类型 | 字段 | 编码方式 |
|---|---|---|
| 数值/序数 | 6 个增强比率、emb_jitter、emb_mask | 数值线性归一化到 [0,1] |
| 对数数值 | temperature | 取 log10 后归一化（跨 5 个数量级，必须对数化） |
| 类别 one-hot | norm(3)、pool(2)、loss_type(2)、sim_func(3)、order(5)、kernel(4) | one-hot |
| 布尔 | temporal、cross_scale、adj_neighbor | 0/1 |

输出总维度约 45–55 维，固定不变。提供 `encode(cls)->np.ndarray`、`dim` 属性、`decode(vec)->cls`（近似逆，仅用于调试与可解释性分析）。

---

## 6. 各模块任务规格

### 模块 1：CLS 策略空间 `cls_space.py` + `space.yaml`
用 `space.yaml` 作为唯一真源。提供 `CLS` dataclass、`sample_random()`、`validate(cls)`、`space_size()`。
**验收**：空间规模数量级正确；随机采样 10000 个全部通过 validate。

### 模块 2：CLS 向量编码 `cls_encoder.py`
按第 5 节规格实现。
**验收**：维度固定；encode→decode 能还原类别字段；归一化范围正确。

### 模块 3：对比学习组件 `contrastive/`
AutoCL 无源码，照论文正文 2.2 节 + 附录 A 从零实现。
- `augmentations.py`：6 个增强（jitter=`x+N(0,p)`；point mask=`x⊙m`, `m~Bernoulli(1-p)`；freq mask=DFT→掩频→IDFT；crop=带共享子序列的两段裁剪；resize/rescale=幅度/长度乘 `1+N(0,p)` 类缩放）+ order（Table 8 的 5 种顺序）。**附录 A 中 resize/rescale 公式被截断，需按论文语义补全并在注释标注。**
- `pair_construction.py`：instance/temporal/cross-scale 三类对比 + 层级 pooling（avg/max × kernel size）+ temporal 可选纳入相邻 embedding。
- `losses.py`：InfoNCE / Triplet，相似度 Dot/Cosine/负 Euclidean，温度参数。
- `cl_trainer.py`：输入 `(cls, encoder, D_train)` 输出预训练好的 encoder。**必须支持 `max_iters` 参数**（参数更新次数；early-val 传 `early_iters=290`，最终评估传 `full_iters=2320`，见 `configs/train.yaml`；基准为 M1 实测 bs=128）。计量用 iters 而非 epoch：切片任务大小不同时 epoch 截断会让训练量不一致，污染跨任务 label（与 AutoCL 附录 B.3 一致）。
**验收**：各增强输出形状正确；p=0 恒等；p 边界不崩。

### 模块 4：固定 encoder `encoders/`
`dilated_cnn.py`（10 层 dilated CNN，hidden=64，output=320，照 TS2Vec/AutoCL）与 `transformer.py`（轻量），`registry.py`+`encoder.yaml` 切换。
**语义澄清**："固定"指架构固定、每个 CLS 各自重新预训练一份权重，不是权重冻结。README 需写明。

### 模块 5：任务嵌入 `task_embed/`
适配 AutoCTS++ 的 `Ts2vec_model/` 与 `AHC/set_encoder/`：`E = SetTransformer(TS2Vec(sliding_windows(D)))`。
分类任务的"窗口"直接用样本序列长度（不同于预测里的 S=P+Q），保留接口但简化窗口逻辑并注释差异。
**要求**：先读 AutoCTS++ 真实源码再适配，不要凭论文臆造 Set-Transformer 细节。

### 模块 6：比较器 T-CLSC `comparator/`（核心创新）
改造 T-AHC，输入 `(E, cls_a, cls_b)` 输出二值 y。结构对齐 AutoCTS++ Fig.4：
- CLS 支路：`encode`→MLP→`l_a`；两个 cls 共享 MLP，concat 成 `L_a`（Eq.15-16）。
- 任务支路：`E`→FC→`E'`（Eq.18）。
- 融合：concat→两层 FC + Sigmoid→BCE（Eq.17-21）。
**唯一实质改动**：把 arch-hyper 的 GIN 编码换成 CLS 的 MLP 编码，其余引擎（动态配对、curriculum、shared samples，AutoCTS++ 3.2.4）沿用。
**验收**：前向输出 ∈ [0,1]；BCE 能下降。

### 模块 7：Label 生成 `scripts/2_generate_labels.py`（最耗算力）
照 AutoCTS++ 3.2.4：
1. `task_slicer.py` 从 HAR/Epilepsy 切子任务（时间连续切片 + 随机采样通道）。
2. 每任务算 E_i。
3. 采样 CLS：L 个 shared（所有任务共用）+ L 个 random（每任务独立），支持 curriculum。
4. 对每 (task, cls) 用 `cl_trainer` 做 **early_iters=290** early-val 预训练 → `linear_eval` 得 R'。
5. 两两配对成 y。
**工程要求**：支持断点续跑、结果落盘缓存（每个 (task,cls) 的 R' 存下复用）、可并行。缓存设计决定迭代速度。

### 模块 8：搜索 `search/` + `scripts/4_zero_shot_search.py`
照 AutoCTS++ Algorithm 2：未见任务→算 E→进化算法用 N* 两两比较排名→Round-Robin 定 top-K→top-K 完整预训练+评估→选 cls*。
进化参数照原文（crossover 0.8、mutation 0.2、population 10、top-3）；先过滤非法 CLS；Round-Robin 处理 T-CLSC 非传递性。

### 模块 9：评估 `scripts/5_evaluate.py`
- **上界**：目标任务上直接跑 AutoCL 式搜索（来不及则用随机搜索 N 次取最优代替）。
- **主 baseline**：InfoTS、TS2Vec、TS-TCC。
- **消融**：w/o TS2Vec、w/o Set-Transformer、w/o shared samples（AutoCTS++ 4.2.3）。
- **核心指标**：目标任务 ACC/F1 + 搜索耗时对比（零样本 vs 直接搜索），后者是相对 AutoCL 的主要卖点。

---

## 7. 分阶段里程碑

| 阶段 | 目标 | 交付 | 验证问题 |
|---|---|---|---|
| M0 | 骨架 | 目录+configs+空间/编码器+全部单测 | CLS 采样与向量化正确 |
| M1 | 单 CLS 可评估 | contrastive+encoder+linear_eval，HAR 上给定 CLS 能预训练+出 acc | 复现 AutoCL 单点量级（HAR ACC≈0.9+） |
| M2 | Label 流水线 | task_slicer+2_generate_labels，产小批 (t,cls_a,cls_b,y) | **early-val(k=5) 与完整训练的 acc 排名相关性（Spearman）** |
| M3 | 比较器 | task_embed+T-CLSC+3_pretrain | T-CLSC 在留出任务配对准确率显著 >0.5 |
| M4 | 零样本搜索 | search+4/5 脚本 | AutoCLS 接近"直接搜索"上界，且耗时低一到两个数量级 |
| M5 | 消融+成文 | 三项消融+耗时表 | 各组件必要性成立 |

**两个关键闸口**：
- **M1** 验证"对 AutoCL 的复现是对的"。
- **M2** 验证"early-validation 排名可信"——这是整套方法能成立的隐含前提。**若 k=5 的排名相关性太低，需增大 k 或改评估协议。这是全项目风险最高、最该优先跑的实验。** 建议 M1+M2 打通验证核心假设后，再投入比较器开发。

---

## 8. 全局约定

1. `space.yaml` 是策略空间唯一真源，任何代码不得硬编码取值集合。
2. `cl_trainer` 的 `max_iters`（参数更新次数）必须显式传入，禁止在 early-validation 和最终评估间混用；跨任务训练量必须用 iters 而非 epoch 保证一致（early=290, full=2320，STD_BATCH_SIZE=128，基准为 M1 实测 bs=128 而非早期 bs=64 误算，固化在 `configs/train.yaml`）。
3. 所有 (task, cls) → R' 结果必须落盘缓存，脚本支持 `--resume`。
4. 任务嵌入相关代码必须先读 AutoCTS++ 真实源码再适配，注释标注对应原文件与公式编号。
5. 每个模块配套 `tests/`，M0 阶段先写测试再写实现。
6. 与原论文有出入的补全必须在注释显式标注"论文未给全，此处按 X 语义补全"。

---

## 9. 参考

- AutoCL: *Automated Contrastive Learning Strategy Search for Time Series*, CIKM '24.
- AutoCTS++: *zero-shot joint neural architecture and hyperparameter search for correlated time series forecasting*, VLDB Journal 2024. 代码：https://github.com/decisionintelligence/AutoCTS_plusplus
