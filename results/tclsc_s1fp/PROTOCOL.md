# WP16-TCLSC 阶段 1（转导指纹臂）PROTOCOL：diff 头 + 目标侧探针实测指纹，7 折 LODO（零 GPU）

**会话定位**：重启 T-CLSC 第四步。φ 臂读数 PHI-DEAD（`results/tclsc_s1phi/REPORT.md`）：12 维元特征在 LODO 下被 diff 头用于粗排序
（M0 cross Δ +0.035 p 3e-11）但对好半区 / 选择无增益——低维数据集描述量映不到高秩的好半区偏好上（WP12）。
本协议换条件变量：**转导指纹** = 目标到达后把源池先验前 K 条（= SH6 的起跑名单）各训到 @60、拿实测分当指纹。
WP12 显示目标自测是唯一在好半区非零的信息源（Spearman 0.13–0.6 vs 先验 0.000–0.005），且 K=6 时它的成本就是 SH6-3-3 第一级
本来要花的 6×60 iters——比较器在这里的角色是"用 6 个实测数把其余 14 条排好"。

数据 `results/mx3_pq/` 只读，零新 R'（铁律 3）。分支 wp9-p1。产物 `results/tclsc_s1fp/`。

## §0 已看过的读数

- L3 阶段 0：diff 头 + oracle 身份 token Δ_M1 +0.205；φ 臂：diff 头 + φ LODO 下 M1 +0.011 p 0.12、M0 +0.035 p 3e-11、regret@3 0.031 vs 任务盲 0.029。
- WP12（`results/mx4_land`，零 GPU）：好半区自测 @25% Spearman 0.133（UCR）/ 0.600（UEA 官方）vs 先验 0.005 / 0.000；
  `model_p5_2_1`（5 探针 @25% → 岭回归提名 1 条）+0.002 在噪声内、赢先验槽不赢随机槽；`late_promote` 更好的指纹不换来更好的提名。
  一句话约束：指纹进**岭回归打分器**没读出选择增益；本协议换的是形态（diff 比较器条件化整张目录），指纹本身与 WP12 同源。
- WP1：SH6-3-3 regret@3 @290 0.018；SH6-3 0.011；prior3 0.031；all20@60→top3 0.017。
- **没看过**：任何以实测分作 ctx 的比较器；`probe6_top3` 参照臂的逐预算数字。

## §1 假设

H-FP：目标上 6 条探针 @60 的实测形状（相对高低）含有该数据集对 CLS 的偏好信息；diff 头能把它迁移到其余候选的排序上——
LODO 下 cross 消融掉分（用了），且**非探针候选**的 regret@3 低于同头任务盲臂（有价值）。
对立：指纹只重述"这 6 条谁好"，对未测候选无外推力（WP12 的 late_promote 读法），表现为 all20 视图有增益、nonprobe 视图零。

## §2 材料与口径

- 划分：S1 = 7 折 LODO，考侧 126 实例（6 切片 × 3 预算 × 7 靶），候选 = 目录 20 条。
- **探针 P**（每折在**训练侧**定，铁律 6）：训练折全部行 `zscore_within` → `prior_scores`（36 号）→ shared-10 中先验前 **K = 6**（= SH6 起跑名单；同分 key 升序）。
- **指纹**（每切片一份，@60 一次测量，三个预算共用）：m = 6 条探针在该切片 @60 的 R'（缓存 `(slice, key, 60)`）；
  f = [(m − mean(m)) / sd(m)]（6 维形状，sd 下限 1e-6）⊕ [mean(m), sd(m)]（2 维水平）= 8 维；再按**训练切片**的 μ/σ 逐维标准化（存 meta）。
  ctx = f_std(8) ⊕ 预算 one-hot(3) = **11 维**。训练切片的指纹同法（自己的 @60 实测）。
- 头 diff（主）/ concat（对照）；标签 hard（主）/ zdiff（副臂）；binned 116；超参不变。
- 排名：3 种子多数票；**消融**：cross = 指纹换成**他数据集训练切片**的（3 种子）；zero = 指纹置 0。
- **两个候选视图**（关键）：
  - `nonprobe`：把 6 条探针从候选与真值里剔掉，只评其余 14 条（干净的"外推到未测候选"）——**主门视图**；
  - `all20`：全部 20 条（部署视图；@60 实例里探针的真值已在指纹里，不作门）。
- 指标同前：M1 好半区（nonprobe 视图下真值前 7 = 14 的一半，取 `top_k=7`；all20 取 10）、M2、M0、regret@1/@3、Spearman。
- 同成本参照臂（零训练，缓存直算）：`probe6_top3` = 6 条探针按 @60 实测取前 3 的 regret@3（真值取该实例预算）；只在 @120/@290 实例（84）上有意义。

## §3 臂（`scripts/33` 独立表 `FP_ARMS`，S1、3 种子）

| 臂 | 任务支路 | ctx | 头 | 标签 | 用途 |
|---|---|---|---|---|---|
| `blind_diff_hard`（复用 PHI_ARMS 规格，独立重训） | none | — | diff | hard | 同头任务盲对照 |
| `fp_concat_hard` | ctx | 指纹⊕b | concat | hard | 头对照 |
| **`fp_diff_hard`** | ctx | 指纹⊕b | diff | hard | **主臂** |
| `fp_diff_zdiff` | ctx | 指纹⊕b | diff | zdiff | 副臂（慢，可后补） |

## §4 门（先于数据）

- **C0**：`blind_diff_hard` M0 与 φ 臂的 blind 0.756 差 ≤ 0.01（同规格重训，同机）。
- **G-F1（用了指纹，nonprobe）**：`fp_diff_hard` 在 nonprobe 视图 Δ_M1 ≥ **0.03** 且 p < 0.05；zero 同向只报。
- **G-F2（对未测候选有价值，nonprobe）**：`fp_diff_hard` nonprobe regret@3 < `blind_diff_hard` nonprobe regret@3，配对 Wilcoxon p < 0.05。
- **G-F3（同成本部署价值，all20，@120/@290 的 84 实例）**：`fp_diff_hard` all20 regret@3 < `probe6_top3` regret@3，配对 Wilcoxon p < 0.05。
- G-F4（头对照）：`fp_concat_hard` nonprobe Δ_M1 零——只报。

**读数码**：

| G-F1 | G-F2 | G-F3 | 码 | 含义 / 下一步 |
|---|---|---|---|---|
| ✓ | ✓ | ✓ | **FP-ALIVE** | 转导条件化比较器在 LODO 成立且同成本胜实测：进官方口径 + 与 SH6-3-3 合并成部署臂 |
| ✓ | ✓ | ✗ | **FP-EXTRAPOLATES** | 对未测候选有外推力，但不比"直接用 6 个实测"好：与 SH 融合（实测 + 比较器排其余）作部署臂再判 |
| ✓ | ✗ | — | **FP-USED-NO-GAIN** | 用了指纹但只重述探针本身（WP12 late_promote 读法）：转导路线在比较器形态下也到顶，剩真 E |
| ✗ | — | — | **FP-DEAD** | 指纹进不了 diff 头：真 E 是最后一条，先做 E 诊断 |

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 带内 | 85% |
| P2 | G-F1 过 | 45% |
| P3 | G-F2 过 | 30% |
| P4 | G-F3 过 | 20% |
| P5 | all20 视图 Δ_M1 > nonprobe 视图 Δ_M1（探针自身重述占一部分） | 75% |
| P6 | fp 臂 M0（nonprobe）≥ φ 臂 M0 0.780 | 40% |
| P7 | concat 对照 nonprobe Δ_M1 零 | 70% |

## §6 成本

- 63 跑 × ~13 s ≈ 15 min（4060）；zdiff 副臂 21 × 5 min 可后补；排名/分析分钟级；零新 R'。
- 部署口径：指纹 = SH6 第一级（6 × 60 iters），零额外成本。

## §7 执行顺序与协议锁

1. 本文件 + `tests/test_tclsc_s1fp.py`（先红）→ commit（协议锁）。
2. 实现：33 `FP_ARMS` / `ctx_kind="fp_budget"`（探针选择、指纹、折内标准化、meta）；`scripts/59_tclsc_s1fp.py` → 单测绿 → commit。
3. `59 --prepare --train` → `59 --rank --analyze` → REPORT；zdiff 副臂视时间后补。

## §8 铁律对照

- 铁律 3 零新 R'（指纹取自缓存里已有的 @60 格）；铁律 5；铁律 6 探针选择与标准化只用训练折，目标只贡献自己的实测；
  铁律 7 指纹只在脚本侧装配；铁律 8 方向经 better_fn。**关键决策 8**（学习曲线外推判死）不触及：指纹不预测终点分，只作条件变量。

## §9 执行台账（跑后填写）

- 2026-09-14 协议锁 b36fc08（13 例先红）→ 实现 33 FP_ARMS / fp_budget / 指纹函数 + 59 号 → 13 例绿 + 邻近 60 绿。修两处自身 bug：
  同分探针的形状项浮点残差（改显式 0）、shared 不足 K 时静默维度错（改 fail-fast）。
- `--train` 三段：后台首启 5/63 被系统按内存杀；脱离重启（LZTray/LeAppOM 占满 CPU，每跑 90–104 s）16/63 处 CUDA illegal memory access；
  用户清理内存后 `--resume` 每跑 13–20 s 跑完 63。模型键唯一，无重复。
- `--rank --analyze` → **FP-DEAD**（G-F1 Δ_M1 −0.042 ✗ / G-F2 ✗ / G-F3 probe6_top3 0.015 vs fp 0.030 ✗ / G-F4 ✓），预测 4/7。详见 REPORT.md。
