# 预测任务 R' 噪声探针协议（先于任何 GPU 数据 commit；2026-08-13）

## 0. 目的与判定对象

分类路线已收官判死：任务特异信号 0.023–0.033 vs 工作点单种子噪声 P95 0.110（HW），
δ 按噪声标定后存活 label 任务盲天花板 0.95+ → E 零边际（decisive C2）；协议/口径/
数据集/k=3 预演/per-CLS δ/trimmed/SWA 全部排除，实现/评估路径/encoder 三层排查排除。
**噪声是根因。** 预测路线的核心假设：R' 走线性回归 + MSE（AutoCL 附录 B.1），没有
"训练轨迹随机 → 表征不同 → SVM 决策边界跳动"那层放大器，噪声应显著更小。
这是推理不是实测（本项目元教训：两轮探针 9/10 预测错）。**本探针只测一件事：
预测 R' 跨 weight_seed 的重复噪声的相对水平。** 信号侧（GGS 增益/任务特异性）
不在本轮，即使噪声达标也只解锁下一步。

铁律 6 声明：本轮不定预测任务的源/目标池；用 ETTh1 只因它是 AutoCL 用过、有论文
对照数字的标准数据集，不构成入池承诺。任务嵌入（TS2Vec/SetTF）不参与本探针。

## 1. 协议确认（铁律 4：先读真实源码，逐行核对后固定）

对照源：yuezhihan/ts2vec（main，2026-08-13 拉取）+ AutoCL 论文（reference/AutoCLS.pdf）。

**表征提取（与分类协议本质不同，逐行确认）**：
- 分类：encoder 逐时间步输出 → 时间维 **max-pool**（full_series）→ 实例向量。
- 预测：**逐时间步因果表征，无任何池化**——`model.encode(data, causal=True,
  sliding_length=1, sliding_padding=200, batch_size=256)`（tasks/forecasting.py
  L21-30）。时间步 t 的表征 = 编码"以 t 结尾、长 201 的因果窗"取**末位**
  （ts2vec.py L242-256：l=i−padding, r=i+1, slicing=slice(padding, padding+1)）。
  因果性由窗口构造保证，与 encoder 自身是否因果无关。
- 预测协议没有独立的"P"：Ridge 输入是单时间步表征 r_t，其感受上下文 =
  sliding_padding+1 = **201 步**（官方常数 200，固定沿用）。

**下游回归（AutoCL B.1 "linear regression with l2 norm" = TS2Vec Ridge 协议原文）**：
- 样本对齐：r_t 预测 data[t+1 .. t+Q]（generate_pred_samples，forecasting.py L5-12）；
  train 侧 drop=200（填充窗污染的表征丢弃，L46），valid/test 不 drop。
- Ridge alpha ∈ {0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000}，**验证集**
  score=RMSE+MAE 选 alpha，best alpha 在 train 上重拟合（_eval_protocols.py L81-109）。
  全程确定性：无 GridSearchCV、无 loky、无随机下采样（样本量 < MAX_SAMPLES）。
- 指标：MSE/MAE，**norm（标准化）尺度为论文报告口径**；R' = norm MSE@Q。

**任务定义**：
- ETTh1 **单变量**（AutoCL Table 4 口径）：目标 = OT 列；encoder 输入 8 通道 =
  7 日历协变量（各自标准化）+ z 后 OT（datautils.py L137-181 逐行）。
- Q=48：AutoCL B.3 预测的验证指标 = "MSE (48 horizons)"。单 horizon。
- 切片：train/valid/test = 前 12/4/4 月（8640/2880/2880 步，官方 L151-154）；
  StandardScaler **只 fit train 切片**。冻结 encoder 事后编码全序列（valid/test 的
  窗口上下文是过去数据，非泄漏，官方同款）。
- 论文对照锚点（ETTh1 univar @48, norm MSE/MAE）：AutoCL 0.052/0.173、
  GGS 0.054/0.177、TS2Vec 0.062/0.189。本探针 iters 远小于论文（论文训至收敛），
  MSE 只会更高；锚点仅作量级 sanity（band：MSE ∈ (0.03, 1.5)，破坏性 CLS 可到
  ~1.0+ = 方差水平）。

**预训练**：cl_trainer 原样复用（对比预训练与下游无关）。实例构造为本项目适配
（官方长序列直喂 + max_train_length=3000 切段 → N=2、batch=2；AutoCL 上限 2000；
本项目 cl_trainer 是实例批语义、STD_BATCH_SIZE=128 铁律 2）：train 切片滑窗
INST_LEN=256（路线 A CROP=256 口径）、STRIDE=24（1 天）→ **350 实例**，窗口绝不
越出 train 切片。

**与官方的全部偏差**（其余逐行同式；详见 forecast_eval/forecast_loader docstring）：
1. 左侧缺失上下文零填充（官方 NaN 填充 + encoder 内置零）——只影响全序列前 200 步
   表征，全部落在 train 且被 drop=200 丢弃，不影响任何入判数值；
2. raw 指标逆标准化用显式广播（官方对 4 维数组调 sklearn，依赖旧版行为；数学同式；
   raw 只作诊断，入判用 norm）；
3. np.float/weekofyear 等已移除 API 换等值现代写法；
4. 预训练实例构造如上（官方协议未覆盖本项目 trainer 语义，属必要适配而非偏离评估协议）。

## 2. 实验设计（口径对齐分类噪声探针，保证可比）

- 数据：ETTh1（data/ETT/ETTh1.csv，md5 8381763947c85f4be6ac456c508460d6，
  17420 行，源 zhouhaoyi/ETDataset main/ETT-small）。data/ 在 .gitignore，不入库。
- 任务：TASK_ID = ETTh1_univar_Q48（单任务单 horizon）。
- CLS：**10 shared 探针 CLS**（results/route_a_probe/probe_cls_catalog.json 前 10，
  与 labels_workpoint/swa 探针同一目录同一批——跨任务类型可比），含已知破坏性配置。
- 重复：k=3 weight_seed/cell。rep0 = run_seed_for(TASK_ID, cls_key)；
  rep1/2 = run_seed_for(TASK_ID, f"{cls_key}:noise{rep}")——派生式与 2c/7_swa 逐字
  一致，幂等无重放风险。无 split_seed（时间切片固定，无切分随机性）。
- iters：扫 **{30, 60, 290}**（预测工作点未知：30/60 = 分类两数据集实测工作点，
  290 = early 全局口径；官方 TS2Vec ETTh1 univar 默认 200 iters 落带内）。
- 合计 90 条。缓存键 (dataset, task_id, cls_key, iters, seed)，--resume 贯穿，
  OOM 走 RprimeOOM 不落盘，device_name 落盘。本机 4060、OMP=1；无 threading 后端
  （Ridge 无 loky，教训不适用，脚本注明）。
- 执行：scripts/8_forecast_noise_probe.py --run --resume；分析 --analyze。
  收官校验：90 行、键唯一、零 null（NaN 条目如实报告）。

## 3. 测量与报告

每 (cls, iters) cell 的 3 个 MSE（norm@48）：
- 绝对重复差：3 个两两 |ΔMSE|（每档 10 cell × 3 = 30 对）→ med/p90/**P95**/max
  （分位公式与 2c calibrate 逐字同式：sorted 后取 round(q·(n−1)) 索引）；
- **相对重复差 = |ΔMSE| / cell 三 rep 均值**（预注册分母：cell 均值，比配对均值稳）
  → 同四分位；MSE 有量纲，相对量才能对照分类 ACC 的 0.110；
- CV = std(ddof=1)/mean per cell → med/max（次要口径）；
- MAE 同口径一并报告（次要）。

诊断（不参与判定）：噪声随 iters 的走向；per-CLS 噪声差异（Spearman(cell mean MSE,
cell CV)——UM 曾见"越破坏越吵"ρ≈0.52）；重尾指标 rel P95/median（分类 HW 为
0.110/0.029 ≈ 3.8）。

## 4. 判据（先写死，commit 后开跑）

**主判据**：每档的 **相对重复差 P95**；总判定取三档中最低者（将来工作点会定在
噪声最低且信号可用的档）。边界值归中间档（宁保守不放行）：

| rel P95（最优档） | 判定 |
|---|---|
| < 5% | **绿灯**：噪声比分类好 2–4×，即使信号只有 10% 量级也在噪声之上 → 进信号侧探针 |
| 5–15%（含边界） | **中间档**：单看噪声定不了，进第二小步（信号侧）后合并判定 |
| > 15% | **红灯**：与分类同量级或更差，预测路线核心假设不成立 → 重新评估投入 |

阈值理由（独立于结果，2026-08-13 写死）：分类 ACC 噪声 P95 0.110 相对 ACC 水平
（0.55–0.95）≈ 12–20% 相对幅度；<5% = 好 2–4×，>15% = 无改善。

**有效性检查**（不改判据，只判实验是否成立）：
- NaN（训练发散）条目如实报告；若 >20%（>18 条）→ 实验无效，先排查再重跑；
- MSE 量级 sanity：cell 均值应落 (0.03, 1.5) band（超出 → 检查实现，不静默采信）；
- rep0 与 rep1/2 无系统性位移（canonical 种子无特殊地位）。

**预注册预测**（对账用，元教训：外推常错，写下以校准直觉）：
- P1：最优档 rel P95 落中间档 5–15%（先验 45%；绿 30%、红 25%）。
  理由：Ridge 消掉评估层随机性，但预训练轨迹随机（分类噪声的上游）仍在。
- P2：290 档噪声 ≤ 30/60 档（训练更久轨迹更收敛）。
- P3：噪声与 CLS 破坏性正相关（Spearman ρ > 0.3）。

## 5. 成本估算（4060，估时锚点带硬件标签）

单条 = 预训练（iters × ~0.08–0.15s/iter，B=128×T≤256（resize 后 ≤512）×C=8）
+ 因果滑动编码 14400 窗（57 × (256,201,8) 前向 ≈ 2–4s）
+ Ridge 13 alpha × (8392×320) 单线程 BLAS ≈ 3–6s + 常数开销。

| 档 | 条数 | 单条估 | 小计 |
|---|---|---|---|
| 30 | 30 | ~10–15s | 5–7.5 min |
| 60 | 30 | ~12–18s | 6–9 min |
| 290 | 30 | ~30–45s | 15–22.5 min |

**点估 ~30 min / 保守 ~90 min**（保守含首条 CUDA 预热、resize 重尾、诊断重跑）。
smoke（首条）后校准 ETA；显存峰值远低于分类长序列端（T≤512、编码窗 201），
OOM 风险低但防护全开。实现工作量（本 commit）：4 新文件 + 3 分发点改动 +
35 新单测。
