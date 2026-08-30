# CLAUDE.md — AutoCLS

Zero-Shot NAS for Contrastive Learning Strategy Search。把 **AutoCL 的对比学习策略空间（CLS）** 作为搜索对象，用 **AutoCTS++ 的零样本比较器机制** 来搜，encoder 固定。

**主张定稿（出路 X，用户 2026-08-30 拍板，勿回退）**：① 通用 CLS 质量排序（任务无关先验，留出 0.77–0.87/跨数据集 0.69–0.71）+ top-K 真评估；② 任务感知**可行性/脆弱度预测**（新模型，(E, cls)→三态 提升/惰性/破坏；数据集毒性 0→0.75 已坐实 p<0.001）；③ 端到端指标 = 双向 leave-one-dataset-out（UM↔HW，按任务报 28 个搜索实例的 regret+成本表）+ 全量 HW/UCI-HAR 标准协议对 TS2Vec 论文公开数字的外部锚。**细粒度 task-aware 排序在本栈判死勿复活**：任务信号幅度 0.007–0.013 < 轨迹噪声地板 0.030（评估扩容/iters 扫描/换族探针三重排除，见 memory 与 results/step0*、step1_encoder_probe、destruction_matrix）。

完整设计见 `TECHNICAL_DESIGN.md`。

本机python环境：Anaconda 下的名为pytorch的虚拟环境

## 核心概念
- **CLS**：一个对比学习策略 = 19 个离散字段的配置向量（AutoCL Table 1）。这是"要搜的架构"。
- **T-CLSC**：比较器，输入 `(任务嵌入 E, cls_a, cls_b)` 输出二值 y（哪个 CLS 在该任务上更好）。负责排序；其配对 label 样式**不变**。
- **可行性预测器**（出路 X 新增）：输入 `(E, cls)` 输出三态（提升/惰性/破坏）。看门层：预判未见数据集的空间毒性与安全边界；label 须绑 iters 口径（@290 起步）。二者分工：预测器剪雷区/判可评性，比较器排序，top-K 完整评估定冠军。**训练侧不过滤破坏配置**（它们是 avoid-worst 的粮草），过滤只在推理侧。
- **early-validation**：造训练 label 时只对比预训练 575 iters（≈5 epoch 等量）就评估，用来快速排名。**只有最终 top-K 才做完整预训练。**
- **固定 encoder**：架构固定、每个 CLS 各自重新预训练一份权重（不是权重冻结）。

## 关键决策（已定，勿改）
1. CLS 编码用 **向量 + MLP**（方案 A），不用图/GIN。
2. encoder 两者可切换：`dilated_cnn`（默认，TS2Vec式）/ `transformer`，由 `configs/encoder.yaml` 控制。**transformer 分支已探针判死**（2026-08-29，`results/step1_encoder_probe` 四问全败：随机 attention 特征在贴顶数据集上比随机卷积**更强**——贴顶跨架构家族成立；HW 对照崩 3/20；地板 0.030 不降。勿再投入调参）。
3. 第一阶段只做**分类**（HAR / Epilepsy）。转预测任务已探针判死（噪声继承自预训练轨迹，`results/forecast_noise_probe` 红灯）。
4. **k 放大（Y-B 加强版）搁置未关闭**：k≈6–9 @120 工作点在数学上可压 δ 至信号之下，用户曾表态不在乎成本；X 期间不投入，若重启从 memory `outlet-x-confirmed-roadmap` 进入。
5. 原目标池 {UCI-HAR, Epilepsy, FaceDetection} **从主结果退役**（三种死法判不可评，`results/target_preflight`），写进 limitations/预检案例；全量 UCI-HAR 标准协议仍可当外部锚（区别于切片舞台）。

## 铁律
1. `configs/space.yaml` 是策略空间**唯一真源**，代码不得硬编码取值集合。
2. `cl_trainer` 的 `max_iters`（参数更新次数）必须显式传参，early-val 与最终评估禁止混用；跨任务训练量必须用 iters 而非 epoch 保证一致（early=290, full=2320，STD_BATCH_SIZE=128，固化在 `configs/train.yaml`。基准为 M1 实测 bs=128（40 epoch=2320 iters，ACC 0.9372），非早期 bs=64 误算；early:full 保持 M2 验证的 1:8 比例）。
3. 所有 `(task, cls) → R'` 结果必须落盘缓存，脚本支持 `--resume`。这是最贵的一步。
4. `task_embed/` 相关代码必须先读 AutoCTS++ 真实源码再适配，注释标注对应原文件+公式号。
5. 先写 `tests/` 再写实现。与论文有出入的补全须注释标注"论文未给全，按 X 语义补全"（如附录 A 的 resize/rescale 公式被截断）。
6. 共享 TS2Vec 仅在【源池】训练、冻结后编码所有任务；目标池/未见数据集绝不参与 TS2Vec 训练，防 zero-shot 数据泄漏。API 上由 `fit_shared_ts2vec(source_data)` 显式传入训练数据，`embed_task` 只接受已冻结的单变量编码器。
7. **任务类型显式分发**：项目支持多种下游任务类型（当前仅 `classification`，将来加 `forecasting` 等）。task 数据结构必须带显式 `task_type` 字段（当前恒为 `classification`）；凡按任务类型不同而行为不同的逻辑，必须通过 `task_type` 显式分发，禁止隐式假设"就是分类"。已知分发点：task_slicer 切片策略、task_embed 窗口逻辑、downstream 评估器、early-val 的 R' 优劣方向。任务无关模块（cls_space/cls_encoder/contrastive/encoders/comparator/search）绝不能引入任务类型假设，保持纯粹。新增任务类型时应是"在各分发点加 case"，而非重构核心。`autocls/task_types.py` 是 task_type 取值唯一真源。
8. **R' 优劣方向抽象**：early-val 的 R' 优劣方向随任务类型不同（分类 ACC 越大越优；预测 MSE 越小越优）。造 label 时 y 的定义禁止写死方向，必须经统一比较函数 `is_better(r1, r2, task_type)`（`autocls/task_types.py`）判定，把"什么叫更优"交给任务类型。当前 classification: 大者优；forecasting: 小者优的分支留 stub 接口。

## 已知 limitation（encoder 适用边界）
- **L≥50 是固定 encoder 的适用边界**（路线 A 探针实证，`results/route_a_probe`）。根因：默认
  encoder 是 10 层膨胀 CNN，dilation 逐层翻倍到 2^9=512，感受野为 **L≥100** 设计；序列过短时
  被 padding 主导，任何 CLS（含破坏性增强）都收敛到同一贴顶表征，R' 与 CLS 质量**零相关**
  （PenDigits L=8：CLS spread SNR 0.39，比正常长度窄 ~20×）。**L<50 的数据集只产噪声 label，
  一律不入源池**（编码长度口径：变长数据集 pad 到 ≥50 且 SNR>1 的可留，如 SpokenArabicDigits
  真实 4–93 → pad 93 → SNR 2.24）。L 在 30–50 之间须先过 SNR 检查再入池。写论文时列为 limitation。
- **δ 非全局常数**：R' 噪声随 n_eval 变小 / R' 离天花板而涨（探针实测 0.0075→0.0495，4× 跨度），
  δ=0.02 是 HAR 特调值。改 **per-dataset δ**（`configs/labels.yaml: delta_per_dataset`）。
- **R' 噪声结构定稿（2026-08-29）**：噪声 = 计数成分（∝1/√n_eval，β=0.502）+ 轨迹地板
  （cls×seed 交互，UM 谷底 δ_nd=0.030@120 iters，U 形曲线 240 掉头）。两制式 δ（退化判据
  隔离破坏 CLS 后标定）白拿 ~40%。**旧的任务信号幅度估计（outlet_y 的 0.023–0.033）已作废
  勿引**——k=1 差值噪声膨胀 3–4×，干净口径 p50=0.0072。共享种子配对/SWA/per-CLS δ 均已
  枪毙勿重提。
- **端到端目标三档阶梯**：①未见任务（多）；②可行性预测目标（16 UEA 轮转 LOO，贴顶数据集
  够格）；③端到端搜索目标（须过预检闸，现仅 HW 强 / UM 弱）。③的资格独立于②，新数据集
  过 `6_target_preflight` 即可入③。
- 源池重排（L≥50 剔除 PenDigits/LSST 后）见 `results/route_a_pool/REPORT.md`。
- **长序列端与 EEG 域在源池耦合**：EthanolConcentration（唯一非 EEG 长锚，Spectro/L1024）
  降为 reserve 后，源池长序列端（L>800：SCP1 896 / SCP2 1152 及 MotorImagery 512）**全为
  EEG/MEG 域**。"长序列"与"EEG 域"两个因子在源池中共线，比较器可能难分辨它偏好的是**长度**
  还是**域**。写论文时列为 limitation；若需解耦，从 reserve 回填 Ethanol 补一个非 EEG 长锚。
  源/目标/reserve 池组成的唯一真源为 `configs/source_pool.yaml`（δ 仍在 labels.yaml）。

## 复用 AutoCTS++（repo: decisionintelligence/AutoCTS_plusplus）
- `Ts2vec_model/` → `autocls/task_embed/ts2vec_wrapper.py`
- `AHC/set_encoder/` → `autocls/task_embed/set_transformer.py`
- `AHC/ahc_engine.py` → `autocls/comparator/tclsc_engine.py`（GIN 支路换成 MLP）
- `exps/random_search.py` → `scripts/3_pretrain_tclsc.py` + `4_zero_shot_search.py`

## 数据流（出路 X 版）
```
离线: 源数据→切片成任务 t_i →(冻结共享TS2Vec+SetTF 得 E_i);(采样CLS)
     →对(t_i,cls)跑预训练+下游评估得 R'
     →配对 (E_i,cls_a,cls_b,y) →BCE训T-CLSC（排序）
     →三态 (E_i,cls,y∈{提升/惰性/破坏}) →训可行性预测器（看门）
在线: 未见D→冻结encoder前向得E（推理，非训练，铁律6）
     →可行性预测器判毒性/剪雷区→T-CLSC两两比较排名→top-K完整预训练真评估→选 cls*
展示: leave-one-dataset-out——目标数据集从离线一切训练中抠除（TS2Vec fit/label/δ标定），
     仅考试时前向编码。UM→HW 主方向、HW→UM 反方向。
```

## 里程碑（出路 X 版，2026-08-30 重排）
- ✅ M0 骨架 / M1 单 CLS 预训练（HAR 0.9372）/ M2 early-val 相关性（ρ=0.85）
- ✅ M3 T-CLSC 留出配对准确率（0.77–0.87；跨数据集 0.69–0.71）
- **MX-1 可行性预测器**：三态 label（@290 破坏矩阵数据起步，零 GPU）→ 16 折
  leave-one-dataset-out 预测数据集毒性 vs 任务盲基线；E 消融在此目标重新预注册 ← **下一步**
- **MX-2 端到端展示（论文头条指标）**：双向 leave-one-dataset-out（UM→HW 主 / HW→UM 副），
  按任务 28 个搜索实例报 regret（vs 直接搜索上界/GGS/随机/垫底）+ 成本表（1–2 数量级）；
  外部锚 = 零样本选中 CLS 在全量 Handwriting（+可选 UCI-HAR）标准协议 vs TS2Vec 论文公开数字
- MX-3 消融（w/o TS2Vec / SetTF / shared samples）+ 耗时表 + 论文
  （正翼=排序/脆弱度/端到端；负翼=三重排除证据链降格为 analysis/limitations）

## 目录
`autocls/{space,contrastive,encoders,task_embed,comparator,downstream,search,data}` + `scripts/1-5_*.py` + `configs/*.yaml` + `tests/`
