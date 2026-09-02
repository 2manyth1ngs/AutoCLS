# 实验 1 协议锁：E 级退化诊断 + `set_fc` 常数性 + 绕过 SetTF 的 C2 复算

日期：2026-09-02 ｜ 本机 RTX 4060 Laptop（桌面负载未清，GPU 非确定性 ±0.01 记 limitation）｜
来源：`docs/DESIGN_NOTES.md` §3.3 / §3.5-1 / §6 实验 1 ｜ 判据/预测**先于数据**写死。
数据：`results/labels_workpoint/`（28 任务 windows_repr + 560 R' + set_encoder_init.pt），零新增 R'。
脚本：`scripts/30_e_degradation_diag.py`；共享模块 `autocls/comparator/mean_pool_encoder.py`。

## 0. 问题

C2（`results/decisive_e_ablation/`）把 E 消融全零归因于 label 侧单因（δ 剃掉任务信号），并写下
"SetTF/TS2Vec 支路没有失职"。DESIGN_NOTES §3.3 指出三条特征侧证据从未进报告：
(1) E 过随机 SetTF 后余弦 ≥0.9875（验收数字引用错层）；(2) zero-E 与 shuffle-E 巨大不对称
暗示 `set_fc(E)` 是常数偏置；(3) 现有实验无法区分"label 侧单因"与"特征侧退化"。
本实验回答：**特征侧有没有锅**。三问独立回答，判读表 §4。

## 1. 诊断 (a)：三层表征的区分力（零训练）

对 28 任务分别取三层向量：
- **L0 TS2Vec 级**：`windows_repr` (100, T, 256) 对窗口与时间取均值 → (256,)。
- **L1 E 级 @init**：`set_encoder_init.pt`（随机 SetTF）输出（npz 内 `E` 字段；脚本复算校验逐位相等）。
- **L2 E 级 @trained**：R1（mixed，seed 0，C2 同超参）重训后联合训练过的 SetTF 输出。
  C2 未保存权重，本实验在本机重训一份并落盘 `tclsc_R1_s0.pt`（GPU 非确定性 ±0.01，不影响读数量级）。

每层报：28×28 余弦矩阵；数据集内 / 跨数据集平均余弦与欧氏距离；**分离比** =
跨数据集平均距离 / 数据集内平均距离；**有效秩**（协方差谱的 participation ratio
(Σλ)²/Σλ²）；**相对离散度** = mean_t‖x_t − x̄‖ / ‖x̄‖（"常数 + 微扰"的直接读数）；
z-score（跨 28 任务逐维标准化）前后的分离比。

## 2. 诊断 (b)：`set_fc(E)` 常数性（零训练，用 L2 的模型）

设 c_t = set_fc(E_t)（128 维），c̄ = 跨 28 任务均值，c₀ = set_fc(0)（zero 消融时的取值）。
- **常数性比** ρ_c = mean_t‖c_t − c̄‖ / ‖c̄ − c₀‖。ρ_c ≪ 1 ⇔ 任务间差异远小于"有 E vs 无 E"的差异，
  即 set_fc(E) 是带幅度的常数偏置——精确对应 zero 有 23 个不一致对而 shuffle 有 0 个的现象。
- **支路相对幅度**：跨任务 std 向量范数 ‖std_t(c_t)‖ 对比 CLS 支路跨配对的 ‖std_pairs(cls_fc(L))‖
  （用 R1 测试对计算）——E 支路给融合头带来的变异 vs CLS 支路。
- **梯度灵敏度**：测试对上 mean‖∂logit/∂E‖·‖E‖ 与 mean‖∂logit/∂cls_a‖·‖cls_a‖（乘以输入范数得无量纲相对灵敏度）。

## 3. 诊断 (c)：绕过 SetTF 的 C2 复算

把 T-CLSC 的 set_encoder 换成 **MeanPoolTaskEncoder**（零参数，E := L0 均值池化向量，nz=256；
`set_fc` 仍学习）。其余与 C2 完全一致：同 21/7 任务划分、同 δ 硬 label、同超参
（lr 1e-4 / bs 64 / 200 ep / curriculum 40）、同两档 shuffle（within/cross × 5 assignment 种子）+ zero、
同 McNemar 口径、同达标线（Δ ≥ 0.02 且 ≥4/5 种子 p<0.05）。
- **主臂 bypass-raw**：E = L0 原样。R1–R3（seed 0/1/2）+ R4（仅 HW）+ R5（仅 UM）。
- **副臂 bypass-z**：E = L0 经逐维 z-score（统计量只用**训练任务**，留出任务套用），排除"尺度太大/太小"的解释。同 5 跑。
- 前提检查：bypass 的 E 必须**可证非退化**——L0 跨数据集平均余弦 < 0.95 且 28 任务有效秩 ≥ 3；不满足则 (c) 无判读力，如实报。

## 4. 判读表（先于数据写死）

| 问 | 判定量 | 阈值 | 读法 |
|---|---|---|---|
| F1 E 退化 | L1 最小余弦、L1 相对离散度、L0→L1 分离比变化 | L1 min cos ≥ 0.98 且相对离散度 < 0.10 → **退化确认**；L2 仍满足 → **联合训练修不好** | DESIGN_NOTES §3.3-1 成立/不成立 |
| F2 常数偏置 | ρ_c | ρ_c < 0.10 → **常数偏置确认**；0.10–0.30 灰区；>0.30 不成立 | §3.3-2 成立/不成立 |
| F3 绕过 SetTF | bypass-raw / bypass-z 的 within Δ 与 cross Δ（R1–R3 中位数种子，与 C2 同法） | 任一臂 within 达标 → **特征侧是半个锅**；任一臂仅 cross 达标 → 域级信号原本被 SetTF 埋没；两臂双零 → **label 侧单因证据升级** | 决定 §3.5-2/3 是否值得投 |

综合：F1∧F2 成立且 F3 双零 = "E 通路确实退化，但即使修好特征侧，硬 δ label 下 E 仍无梯度"
——与实验 2（软 label）交叉后才是完整回答（交叉臂 soft × bypass 在实验 2 脚本内）。

## 5. 预测（先于数据；带宽给足）

1. F1：L1 min cos ∈ [0.97, 0.999]、相对离散度 < 0.10（先验 75%）；L2 相对离散度 < 0.15（先验 65%）。
2. F2：ρ_c < 0.10（先验 70%）。
3. F3：bypass-raw 与 bypass-z **双零**（|Δ| ≤ 0.005，n_sig ≤ 1）（先验 60%）；任一臂 within 达标（先验 15%）；仅 cross 达标（先验 20%）；其他 5%。
4. bypass-raw 的 normal acc ∈ [0.72, 0.85]（与 C2 0.77–0.80 同带；E 改变不应大动通用排序）。
5. L0 前提检查通过（跨数据集 cos < 0.95）（先验 80%）。

## 6. 成本

GPU（4060，桌面负载在）：R1 重训 ≈ 5–10 min；bypass 10 跑无 SetTF，每跑 ≤ 2 min；诊断 CPU 秒级。
合计 ≤ 40 min，与实验 2 并行共用 GPU（峰值 <1 GB/进程）。
