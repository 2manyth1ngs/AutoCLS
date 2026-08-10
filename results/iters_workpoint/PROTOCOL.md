# iters 工作点扫描 —— 协议锁（判据/网格/预测先于任何测量）

日期：2026-08-10 ｜ 本机 RTX 4060 ｜ **本文件写死的判据与预测跑完不调**
｜ 上游：`results/full_iters_probe/REPORT.md`（判定 D：加训练量不是出路；
290 iters 在切片任务上实为 36–97 epoch，"early=5 epoch"的本意从未被执行）
｜ `results/candidate_survey/REPORT_STEP3.md`（7 候选 @290 全判 inert，
其中 UniMiB-SHAR 与 InsectWingbeatSound 各 9/20 差 1 条）

---

## 0. 要回答什么

之前 7 个候选全部在**过训练点**（290 iters = 36–97 epoch）被判 inert。full-iters
实验证明加训练没用（判定 D），但减训练可能有用——Handwriting 在 30 iters（10 ep）
是 18↑/0↓，比 290 的 15↑/4↓ 更干净。本扫描检验：**在正确的工作点上，那些 inert
判定会不会翻盘。**

**核心方法修正（本会话最重要的一条）**：固定全局 iters 会让各数据集被评估在自己
R'–iters 曲线的不同位置上，把"离工作点多远"这个**数据集属性**混进 label——那正是
task-aware 排序最不该被污染的地方。所以扫描按 **epoch 对齐**设计，同时报告
iters 与 epoch 两种口径；若结论不同，以 epoch 口径为准。

### 0.1 写协议时的实测发现：epoch 对齐退化为 per-dataset iters

从既有缓存取全部相关任务的 `n_train`（非新测）：候选池任务一律 716–718
（`MAX_SAMPLES=30000` 下切片取 1024、eval_ratio=0.3 所致）→ **6 steps/epoch**；
Handwriting 前 2 任务 360/289 → **都是 3 steps/epoch**。数据集内 steps/epoch 恒定，
故 `iters = epoch × ⌈n_train/128⌉` 的逐任务换算恰好等于"每数据集一个 `--fill-iters`
值"——**不需要逐任务变 iters 的代码改动，也不需要双份跑两种口径**：同一批 run
每条既有 iters 值也有 epoch 值，只是横轴换算；跨数据集"按 iters"的对比由已有的
30/290/2320 历史臂提供。

---

## 1. 扫描网格（写死）

Epoch 网格 **{1, 2, 5, 10, 20}**，Handwriting 额外加 **40**。理由：

- **1、2 ep**：从未测过的真 ultra-early。直接检验 WISDM2019"上确界是 0"
  （full-iters Limitation 2 明说 <30 iters 未验证），并检验破坏是否在几步内就发生。
- **5 ep**：M1 本意的 early 点（HAR 7352 样本 × 5 ep ÷ 128 = 290 iters 的原始出处），
  本扫描的中心。
- **10、20 ep**：接到已有 48.3 ep（=290）；Handwriting 在 10 ep 最干净。
- **Handwriting 加 40 ep**：其 realized 峰在 10–96.7 ep 的上升段内，10→96.7 之间
  只有 20 ep 一个点不够定峰；40 ep 使网格成近似 2× 对数间隔。候选不加 40 ep
  （=240 iters，与已有 290=48.3 ep 几乎重合）。
- **PAMAP2 进核心组**（用户已确认）：第三个过 T2 的候选（+0.056），败在 T3 的
  1↑/9↓——而"↓ 多"恰是低 iters 最可能改善的量（破坏随训练量单调涨是
  full-iters §2.2 三数据集已证模式）。USC-HAD / ChestAccel / ElectricDevices 的
  best_gain 仅 0.010–0.043，离 T2 太远，不列入翻盘检验（默认不跑）。

| 数据集 | steps/ep | 已有臂（iters@ep） | **新跑臂（iters@ep）** | 新增条数 |
|---|---:|---|---|---:|
| WISDM2019 | 6 | 30@5, 290@48.3, 2320@386.7 | 6@1, 12@2, 60@10, 120@20 | 80 |
| InsectWingbeatSound | 6 | 30@5, 290@48.3 | 6@1, 12@2, 60@10, 120@20 | 80 |
| UniMiB-SHAR | 6 | 290@48.3 | 6@1, 12@2, 30@5, 60@10, 120@20 | 100 |
| PAMAP2 | 6 | 290@48.3 | 6@1, 12@2, 30@5, 60@10, 120@20 | 100 |
| Handwriting（正对照） | 3 | 30@10, 290@96.7, 2320@773.3 | 3@1, 6@2, 15@5, 60@20, 120@40 | 100 |

合计新跑 **23 点 × 20 条 = 460 条**。每点 20 条 = 10 shared CLS × 前 2 任务 × 1 种子，
与第3步/full-iters 逐字段同口径。

## 2. 测量口径（唯一变量是 iters）

与 full-iters 协议 §1 逐项相同：CLS = `probe_cls_catalog.json` 前 10（同序）；任务 =
各数据集前 2（注册表默认档）；`weight_seed = run_seed_for(task_id, cls_key)`（不含
iters）；`split_seed` = canonical；threading 后端 + `OMP_NUM_THREADS=1`。

由 `pretrain_encoder` 的单次 seed 性质，**同 (任务, CLS) 的所有 iters 点是同一条
训练轨迹的不同截点**——整条 R'–iters 曲线不含初始化/数据顺序/增强抽样方差。

**基线复用不重测**：iters=0 基线与训练长度无关，±2σ 零假设带与第3步/full-iters
是**同一条带子**，全部臂的 ↑/↓ 计数是同一把尺子量的。

## 3. 落盘与隔离

- 独立 outdir：`results/iters_workpoint/{candidates,uea}/`（两个注册表分开）。
- resume 去重键与 `_trained_by_task` 配对均含 iters（full-iters 已修，有单测）；
  新 iters 值 {3,6,12,15,60,120} 与历史 {30,290,2320} 不可能相撞。
- 基线 `baseline_cache.jsonl` 从 `results/candidate_survey/baseline/`（candidates）与
  `results/full_iters_probe/uea/`（Handwriting，其源为 `results/baseline_probe/`）
  **复制**进来，outdir 自包含；provenance 记在此。
- **Handwriting@290 的 R' 来自 5090**（stage1_rerun，±1 样本抖动已知，与
  full-iters 分析同源同口径）；其余全部本机 4060。本扫描起所有新记录带
  `device_name` 字段（§8）。
- 铁律 6：目标池（UCI-HAR / Epilepsy / FaceDetection）全程不加载。

## 4. 判据（先写死，commit 后跑；T1/T2/T3 定义逐字不动）

T1 = 基线 ≤0.90；T2 = best_gain ≥0.05；T3 = 超出 ≥10/20 且 超出 ≥2×跌破。
±2σ 带 = 各任务 5 个 iters=0 种子（量子 1/n_eval 兜底），逐 (数据集, iters 点) 判，
判定代码 `classify_verdict` 原样计算。

1. **曲线**：每数据集报 realized gain / best_gain / ↑↓带内 对 epoch（并列 iters 轴），
   并入全部历史臂。
2. **最优工作点定义（用户拍板）**：峰 = **best_gain 的 argmax**——下游造 label 需要
   的是可分辨幅度（best_gain ≥ δ 才有存活配对），不是平均净收益；realized gain 峰
   作**辅助诊断**并列报告（它回答"训练是否普遍有益"，与 label 产量是两个问题）。
   峰落在网格端点时记为**截尾**（真峰可能在网格外），只报"≥/≤ 端点"，不报点值。
3. **峰值一致性**（阈值写死）：对"至少一个臂 best_gain ≥0.05"的数据集，取峰值
   epoch 的 max/min 比：**≤2 → 一致**（全局 iters 可保，只是值要改）；**≥4 →
   不一致**（铁律 2 应改 per-dataset 标定）；之间或有截尾峰参与 → **存疑**。
   网格是 ~2× 对数间隔，更细的分辨率不许声称。
4. **翻盘**：候选（UniMiB-SHAR / InsectWingbeatSound / WISDM2019 / PAMAP2）在任一
   扫描点过 T1+T2+T3。
   - **翻盘（稳）**：该点 ↑≥11/20，或相邻两个网格点都过 T3；
   - **翻盘（弱）**：单点恰 10/20——Wilson 90% 含 0.5，且 4–5 点多重比较放大假
     阳性，只算"需加测确认"，**不作入池依据**。
   不对称性预先注册：既防事后否认，也防单点撞线庆祝。
5. **无工作点确认**（对每个候选，重点 WISDM2019；用户拍板补 best_gain 维度）
   ——两个维度**都**空才算确认：
   - (a) **净收益维**：所有扫描点 realized gain ≤ 0 + 1/n_eval（=0.0033 一个 ACC 量子）；
   - (b) **label 产量维**：没有任何扫描点同时过 T2+T3（即不存在"最好 CLS 显著拉开
     ≥δ"的工作点）。
   若 (a) 空但 (b) 有 → 记为"**有 label 工作点但训练无净收益**"——这对造 label
   是可用的（label 只需要可分辨幅度），单独登记，不与"无工作点"混为一谈。
6. **Handwriting**：按判据 2 定峰；并单独报"5 ep 是否优于当前 290 口径"
   （best_gain、realized gain、↑/↓ 干净度三个量并列）。
7. **铁律 2 建议的触发条件**（只报告不改）：判据 3 判"不一致"，或任一候选峰
   ≤5 ep 而 Handwriting 峰 ≥10 ep → 建议"per-dataset 工作点标定进源池准入"
   （同 δ 的 per-dataset 做法）；判"一致"且峰 ≠48.3 ep → 建议 early 改为按 epoch
   的全局定义。附成本变化估算。

## 5. 结果解读（用户原文照抄，先于数据写死）

- **有候选翻盘** → 之前的 inert 判定是工作点错误的伪影，源池筛选要在正确工作点重做
- **无候选翻盘、但 Handwriting 峰值明显** → 确认"候选根本没有工作点"，
  源池问题坐实，转向动 encoder 或改主张
- **各数据集峰值位置差异大** → 铁律2 的"全局固定 iters"要改成按 epoch 或按数据集标定，
  这本身是重要发现

## 6. 预注册预测（全部从算术/既有数据导出，不用领域直觉；跑完对账）

1. **best_gain 在 1→48.3 ep 区间随 iters 单调不减**（每个数据集）——full-iters §2.2
   三数据集"best_gain 随 iters 单调涨"模式的内插。推论：候选的 best_gain 峰将
   截尾在网格右端（≥48.3 ep），Handwriting 的峰在 96.7 ep（既有三点 0.249/0.397/0.372
   的内部极大）。
2. **WISDM2019 全点 realized gain ≤ +0.01，无 T3 通过**（三档单调下行 + 上确界 0）。
3. **InsectWingbeatSound 与 UniMiB-SHAR 最好点的 ↑ ∈ [7,12]/20**——仍在统计边缘带，
   不出现 ≥13 的干净通过。
4. **Handwriting 峰不在 ≤5 ep**（10→96.7 ep 是 realized 上升段），且 5 ep 仍判 usable。
5. **无候选达成"翻盘（稳）"**——主风险预测；若错，源池筛选须在正确工作点重做，
   且本轮全部 inert 结论作废重验。

## 7. 成本（锚点全部为本机 4060 缓存逐条 `seconds` 实测，同数据集同硬件同后端）

已知两个偏差机制在此结构性不踩：跨硬件误用（锚点全 4060 本机）；短 iters 超线性
偏差（那是"短跑估长跑会低估"，本次是短跑锚点估短跑，新点全 ≤120 iters）。
模型：iters≤30 取 t(30) 为下限（SVM 评估+启动开销主导）；30–120 线性内插
t(30)→t(290)。

| 数据集 | 锚点 mean s/条 | 新点 | 估时 |
|---|---|---:|---:|
| WISDM2019 | t(30)=6.4, t(290)=31.9 | 4 | ~13 min |
| InsectWingbeatSound | t(30)=16.2, t(290)=62.6 | 4 | ~29 min |
| UniMiB-SHAR | t(290)=40.4，t(30)≈10（同类比率估） | 5 | ~21 min |
| PAMAP2 | t(290)=36.1，t(30)≈9（同上） | 5 | ~19 min |
| Handwriting | t(30)=4.9；斜率取 t(2320)=767 反推上界 0.33 s/iter | 5 | ~14–21 min |

**点估计 ~1.7 h；预算按 ×2 报 3.5 h**（本项目三次估时事故的处置惯例）。
参照：整个扫描 ≈ WISDM 单数据集 @2320 的成本（3.7 h）。

**显存**：无新风险——full-iters §1.4 实测峰值不随 iters 标度（iters×10 → +0.15%），
新点全部 ≤120 iters 且 L≤256，同 (任务,CLS) 已在 290 下零 OOM 跑满。
`guarded_run` OOM 防护保持生效。

## 8. 顺带修正（本扫描一并落地）

1. **`evaluate_rprime` 落盘加 `device_name`**（cuda → `torch.cuda.get_device_name`，
   cpu → 平台标识）。防跨机误用耗时/R' 锚点（full-iters §1.6 教训：Handwriting 锚点
   串用 5090 致估时 5.6× 事故）。铁律 5：先写测试。旧记录无法回填，从本扫描起全部
   新记录带标签。
2. **铁律 2 修改建议**按判据 7 触发条件出报告，**不动 configs**。

## 9. 铁律遵守

- 铁律 2：iters 显式传参、入缓存键、独立 outdir，早/晚臂绝不混算 ✅
- 铁律 3：逐条落盘 + `--resume` ✅
- 铁律 5：device_name 与分析判定逻辑先写 tests ✅
- 铁律 6：目标池全程不加载（`--datasets` 白名单 + 脚本内断言双重防护）✅
