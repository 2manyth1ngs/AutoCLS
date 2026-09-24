# WP17-ACTS S3-B PROTOCOL：任务嵌入 E（per-task TS2Vec / 折共享 TS2Vec × SetTF / 均值池化）预测轴坐标 a(t)

**会话定位**：
- S3-A 读数码 **S3A-GO**（③ land25 ρ 0.81、regret@3 0.006；④ 手选 4 探针 0.010、模型增量 p 0.041）。
- S3-A′ 三项全负：PROBE-DOMINATED / SPACE-MU-FAIL / HEADROOM-UNIFORM。
- 按方案 `docs/AUTOCTS_STRICT_REPRO_PLAN.md` §3.3，第二段 = **S3-B**：用 AutoCTS++ 式任务嵌入 E 做 S3-A 的同一件事——LOSO 预测 a(t)，代回低秩模型读 regret@3。
- 用户 2026-09-24 拍板：
  - 保留字面门（E 对 ③）；
  - 加副门"E 替代 ④ 的 4 条探针"；
  - 加对照组"E ⊕ 4 条探针"。

产物 `results/acts/s3b/`；代码 `scripts/86_acts_s3b.py`；单测 `tests/test_acts_s3b.py`。
GPU 段（E 抽取 + SetTF 训练）在实验室服务器跑，上服务器前请示用户。

## §0 已看过的读数（如实声明）
1. S3-A / S3-A′ 全部读数，包括：
   - ③ land25：ρ_a1 0.81，regret@3 0.0058；
   - ④ 手选：0.0098，ρ 0.55；
   - 任务盲 μ：0.021；oracle-a：0.000；
   - M25：0.005；
   - 贪心探针 k2–12：ρ 0.43–0.69；
   - φ 模型对名单外配置排序：Spearman 0.20。
2. 历史 E 读数（一句话设计约束）：
   - S1E 与 EV1：SetTF E 学成数据集身份向量，集内余弦 0.95–0.9999、跨集 0.15–0.58；
   - `task_embed/__init__.py` 诊断：每任务独立重训的 TS2Vec 有基任意性，区分力 ≈ 0。
   
   所以本协议同时跑 per-task（严格复现）与折共享两种 TS2Vec。
3. **没看过**：本语料任何任务的 E、SetTF 的任何训练或读数。

## §1 E 的构造
### 1.1 窗口与编码（两种 TS2Vec 共用）
- **任务数据**：`ac.task_arrays_cv` 的 X，即 S2 标签同一份物化切片、按通道 z-score，不含标签。
  - 训练池 160 任务；另加 HMD / SCP2 各 8 任务，只作表征读数。
- **窗口**：
  - 每任务按 `seed = run_seed_for(task_id, "s3b_windows")` 采 min(100, N) 个实例（分类：窗口 = 样本序列）；
  - 通道当单变量实例各自编码，再取跨通道均值（Eq.10，`encode_task_windows`）；
  - 时间轴 > 152 时用 `pool_time_axis` 池化到 152；
  - 存成 fp16 的 `(m, T′, 256)`。
- **E_pt（严格复现）**：每任务在自己的 X 上 fit 一份单变量 TS2Vec。
  - 通道展开成 (N·C, T′, 1)，`seed = 0`（固定初始化，AutoCTS++ 原味）；
  - repr 256、hidden 64、depth 10、batch 16（TS2Vec 原码默认）；
  - iters 用 TS2Vec 原码规则：pool.size ≤ 1e5 → 200，否则 600；
  - fit 完只编码本任务。
- **E_sh（消融臂，方案 §3.3 的"共享 TS2Vec 留作消融"）**：每个 LOSO 折（17 个）在该折 16 个训练源的**整集数据**上 fit 一份。
  - 训练池用 `build_ts2vec_pool`：每源封顶 4,096 个单变量实例，NaN 右填充；
  - iters 600、batch 32、seed 0，与 MX-3-PQ `--fit-folds` 同口径；
  - 训完冻结，编码全部 176 任务。
  - 另 fit 一份 17 源全训版（"ALL"），只用于 HMD / SCP2 的表征读数。
  - 留出源、HMD、SCP2 都不进任何一折的 TS2Vec 训练（铁律 6）。
- **成本记账**：每任务记 fit / encode 的 GPU 秒与设备名，换算成全跑当量 = 秒数 ÷ 该任务 S2 shared 格的中位秒数。
  - E_pt 的成本 = fit + encode；E_sh 只计 encode。

### 1.2 聚合器
- **mean**：E_mean = windows_repr 对时间与窗口取均值（256 维）。
  - 接 S3-A 同一条管线：`84.loso_evaluate`，α 与 K 在内层 LOSO 选，零训练。
- **settf**：`TaskSetEncoder(dim_input=256, nz=256)`（intra sabPF + inter sabP，原码取值），后接头 Linear(256 → 3)，直接回归 a(t) 的 3 个分量。
  - 目标：每折训练任务的 `ctx["A"][:, :3]`，按训练侧均值 / 标准差标准化；损失 MSE。
  - 优化：Adam，lr 1e-4（AutoCTS++ 默认）；每步 16 个任务。
  - **每任务每步随机子采 32 个窗口**：方案 §8 第 6 项显存策略取 (i)，是相对 AutoCTS++ 的偏差；评估时用全部窗口。
  - 最多 100 epoch；早停 patience 10，监控验证源 MSE。
  - 验证源 = 该折训练源中按池顺序紧跟留出源的那一个（循环），它的任务不参与梯度，只用于早停与 K 选择。
  - 3 个 seed（0 / 1 / 2）平均 â。
  - K ∈ {1, 2, 3} 在验证源上按 regret@3 选（平局取小）。
  - 16 个非留出源的 μ / B 用 `84._fold(Z, tr=全部 16 源)`，与 S3-A 同一份。
- **⊕ 探针（用户对照组）**：S3-A ④ 的 5 维探针特征（折内 μ 前 2 + b₁ 两端，@25%）与 E 拼接。
  - mean 臂：ridge on [E_mean, ④]；
  - settf 臂：头改为 Linear(256 + 5 → 3)，④ 特征按训练侧标准化。

### 1.3 臂
{E_pt, E_sh} × {mean, settf} × {E, E ⊕ ④} = 8 臂。
- **cross-E 消融**（settf 与 mean 的 E-only 臂）：留出源每个任务改用循环下一个源同 OA 行号任务的 E（行号不存在取该源第 0 行），再读 regret@3。
- **身份诊断 C1**：E_mean 与 settf 输出向量的集内 / 跨集平均余弦。

## §2 读数与统计（同 S3-A：160 任务 / 17 源 LOSO，源聚类 bootstrap 10,000 次，p 按 17 个源均值）
- 每臂报：ρ_a1 [CI]、ρ 源均值、ρ 源内、regret@1、regret@3 [CI]、Δ vs 任务盲、K 分布。
- 另报配对差：对 ③ land25（ρ 差 [CI]、regret 差 [CI] p）、对 ④（regret 差 [CI]、上界）、E ⊕ ④ 对 ④（regret 差 [CI] p）。
- cross-E：regret@3(cross) − regret@3(E)，[CI] p。

## §3 门（先于数据）
- **G-S3B（字面主门）**：存在一个 **E-only 臂**（4 个里任一个）同时满足两条 → **S3B-E-WIN**：
  - ρ_a1 − ρ_land25 的 95% 源聚类 CI 下界 > 0；
  - regret@3 显著低于 ③：源级 Wilcoxon，Holm 校正到 4 臂后 p < 0.05，且点估计差 > 0。

  否则 **S3B-E-NOT-BETTER**（结论写"E 不比便宜特征多给信息"，S4 用便宜特征）。
- **SG1（副门，替代 ④）**：存在一个 E-only 臂满足 regret@3(E) − regret@3(④) 的 95% CI 上界 ≤ +0.005（非劣效）→ 后缀 **+REPLACE**。
  - 同时报该臂的目标侧成本：E_pt = fit + encode 的当量；E_sh = encode 的当量；④ = 1.0 当量 @25%。
- **SG2（对照组，E ⊕ ④ 有增量）**：存在一个 E ⊕ ④ 臂满足 regret@3(④) − regret@3(E ⊕ ④) > 0，且源级 Wilcoxon p < 0.05（该门内 4 臂 Holm）→ 后缀 **+ADD**。
- 读数码示例：`S3B-E-NOT-BETTER+REPLACE`。
- 只报不设门：
  - C1 身份（集内余弦 ≥ 0.9 记作"身份型 E"）；
  - cross-E；
  - HMD / SCP2 的 â₁（ALL 版 E_sh + 17 源全训的 settf / ridge，E_pt 同理）。

## §4 C0 自检（先修再读）
- ③ / ④ / 任务盲在同一代码路径下复现 S3-A 的 0.0058 / 0.0098 / 0.0213（逐位）。
- 每任务窗口数 = min(100, N)；npz 形状、dtype 正确；E_sh 每折的 meta 记录的 fit 池不含留出源。
- SetTF 冒烟：2 折 × 1 seed × 3 epoch 可跑通，loss 下降。

## §5 预测（先验）
| # | 预测 | 先验 |
|---|---|---|
| P1 | S3B-E-WIN | 10% |
| P2 | +REPLACE（任一 E-only 臂对 ④ 非劣效） | 25% |
| P3 | +ADD（任一 E ⊕ ④ 臂优于 ④） | 20% |
| P4 | mean 聚合下 E_sh 的 ρ_a1 ≥ E_pt | 65% |
| P5 | 同一 E 下 settf 的 ρ_a1 ≥ mean | 40% |
| P6 | settf 输出集内余弦 ≥ 0.9（身份型） | 60% |
| P7 | 任一 E-only 臂 ρ_a1 ≥ 0.50 | 30% |
| P8 | E_pt 的 fit + encode 中位成本 ≥ 0.5 全跑当量 | 50% |

## §6 执行
1. PROTOCOL + 单测先红 → 协议锁 commit。
2. 实现 86：
   - `--extract-pt / --fit-shared / --train-settf / --analyze / --smoke`，支持 `--parts / --part` 与 `--resume`；
   - 单测绿；
   - 本地 4060 冒烟：2 个任务的 E_pt、1 折 E_sh（小 iters）、SetTF 2 折 × 3 epoch；
   - 零 GPU 部分（mean 臂、C0）可在本地跑。
3. **请示用户后**上实验室服务器，跑 E 抽取与 SetTF。
4. 拉回 → `--analyze` → REPORT → commit → CLAUDE.md / memory。

## §7 铁律对照
- 3：E npz 落盘，`--resume`。
- 4：TS2Vec / SetTF 沿用 `task_embed/` 已注释的 AutoCTS++ 对应项。
- 5：先红；§0 已声明。
- 6：E_sh 每折只在训练源 fit；E_pt 只用本任务无标签数据；SetTF / ridge / K / α 只在训练源。
- 7：`task_windows` 显式分发 task_type。
- 8：acc 越大越好。

## §8 执行台账（跑后填写）
