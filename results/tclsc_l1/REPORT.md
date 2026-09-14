# WP16-TCLSC-L1 报告：T-CLSC CLS 支路的 编码 × 目标函数 分解（2026-09-14，零新 R'）

协议：`PROTOCOL.md`（协议锁 3e80e1b，先红 27 例单测）。脚本：`scripts/33_pq_comparator_loo.py --encoder-mode` +
`scripts/56_tclsc_l1.py`（`--c0 / --prepare / --train / --rank / --analyze`）。产物：`c0.json`、`{scalar,binned}/`（各 21 个模型 +
runs + rankings.json）、`analysis.json`、`tables.md`、`train.log`。硬件：本地 RTX 4060 Laptop，42 跑 × 8.4–10.8 s。

## 0. 一句话结论

**读数码 L1-FLAT（按预注册门）**：G1 编码门不过（Δ +0.012 < 0.020，折数 5/7 达标，Wilcoxon p 0.12）、G3 目标函数门不过
（`mlp_z_binned` 0.7644 vs 门 0.765，**差 0.0006**）。但两条与门无关的事实必须一起读：
1. **对 MLP 而言编码是真的**：同一 MLP 换成 z 回归目标后，binned − scalar = **+0.037（p 2.8e-05，73 胜 36 负）**——WP2 里
   ridge / hgb 对编码无差别（树能自己切档），MLP 从一维标量里学非单调档位效应确实吃力（H-ENC 在 MLP 族成立）。
2. **好半区分辨力（top-10 内配对 acc）是本轮最大的意外**：binned 两臂 0.528 / 0.543，scalar 与 pod 全部 0.38–0.45，WP2 所有代理
   0.39–0.51；配对 Wilcoxon p 0.010（cls_only）/ 0.018（mlp_z），n=51（事后读数，未设门）。这是 WP12 说"先验信息量归零"的那个区域。

0.735 → 0.780 的差距账（pooled 配对 acc，rank→score 口径）：

| 步 | 臂 | acc | 增量 | 归因 |
|---|---|---:|---:|---|
| 0 | main_pod（E+ctx，3 种子投票） | 0.735 | — | WP2 的 T-CLSC |
| 1 | cls_only_pod（无 E，1 种子） | 0.741 | +0.006 | E 支路在 δ 标签下略负（与 MX-3-PQ "cls_only 只低 0.017（模型输出口径）"同向异号——口径不同） |
| 2 | cls_only_scalar（本地 3 种子投票） | 0.746 | +0.005 | 种子投票（C0-local 带内） |
| 3 | **cls_only_binned** | **0.758** | **+0.012** | 编码（G1 ✗：量不够；5/7 折） |
| 4 | **mlp_z_binned** | **0.764** | +0.006 | 目标函数 δ-BCE → z 回归（p 0.08） |
| 5 | hgb_binned（WP2） | 0.780 | +0.016 | 模型族（树 vs MLP）+ 训练制度——本轮未分解 |

## 1. C0 硬门（`c0.json`）：PASS

- `rankings.json["main"]` 复算 pooled 配对 acc **0.7349** / regret@1 **0.0745**（门 0.735 / 0.074 ± 0.001）。
- ref 五臂（ridge_scalar 0.773044 / ridge_binned 0.765163 / hgb_scalar 0.787522 / hgb_binned 0.780181 / tclsc 0.734864）与
  WP2 `analysis.json` **逐位一致**（≤1e-9）。
- 本地重训 Handwriting/cls_only/seed 0（scalar）normal acc **0.7300201477501679 = pod 逐位**（BCE 第 7 位起不同）：4060 与
  5090 在这个小模型上逐位可复现。C0-local：cls_only_scalar 0.746 vs cls_only_pod 0.741，Δ +0.005（带 0.02 内，P6 ✓）。

## 2. 主表（126 实例 = 7 靶 × 6 切片 × 3 预算；`tables.md`）

| 臂 | 编码 | 目标 | 配对 acc | top-10 内 acc | Spearman | regret@1 | regret@3 |
|---|---|---|---:|---:|---:|---:|---:|
| main_pod | scalar | δ-BCE（E+ctx） | 0.735 | 0.385 | 0.351 | 0.074 | 0.038 |
| cls_only_pod | scalar | δ-BCE | 0.741 | 0.434 | 0.375 | 0.081 | 0.033 |
| cls_only_scalar | scalar | δ-BCE | 0.746 | 0.437 | 0.376 | 0.074 | 0.033 |
| **cls_only_binned** | binned | δ-BCE | **0.758** | **0.528** | 0.402 | **0.066** | 0.032 |
| mlp_z_scalar | scalar | z 回归 | 0.727 | 0.449 | 0.354 | 0.116 | 0.039 |
| **mlp_z_binned** | binned | z 回归 | **0.764** | **0.543** | **0.414** | 0.074 | **0.030** |
| hgb_binned（WP2 ref） | binned | z 回归 | 0.780 | 0.508 | 0.448 | 0.056 | 0.033 |
| random | — | — | — | — | — | 0.134 | 0.052 |

逐折配对 acc（binned / scalar，cls_only 本地 3 种子投票）：HW **0.783**/0.716、UM **0.843**/0.816、PS **0.784**/0.760、
Ethanol **0.519**/0.473、HMD **0.758**/0.723、Cricket 0.820/**0.855**、UWave 0.786/**0.855**。binned 输的两折正是 scalar 已在
0.85+ 的两折（天花板效应）；赢的五折含 T-CLSC 最差的 PS / Ethanol（P7 ✓）。

逐预算（配对 acc，binned / scalar）：@60 0.720/0.702、@120 0.791/0.779、@290 0.764/0.756——三档同向。

模型输出口径（33 号 run json 的 normal acc，双向对，种子均值）：binned **0.743** vs scalar 0.716（+0.027；HMD +0.096、
Ethanol +0.058、HW +0.049、UM +0.044、PS +0.040；UWave −0.071、Cricket −0.027）。rank→score 口径把 +0.027 压成 +0.012——
两个口径并列报，不互相替代。

## 3. 门与预测对账

| 门 | 值 | 过 |
|---|---|---|
| C0 | 0.7349 / 0.0745，ref 逐位 | ✓ |
| G1 编码（cls_only） | Δ +0.0124，5/7 折，Wilcoxon p 0.125（56 胜 53 负 12 平） | ✗（量不够） |
| G2 主臂 | 阶段 2 未跑 | n/a |
| G3 目标函数 | mlp_z_binned 0.7644 vs 0.765 | ✗（差 0.0006） |
| G3′ 回归目标下的编码边际 | +0.0375，p 2.8e-05（73 胜 36 负 12 平） | ✓ |

预测 **4/7**：P1 ✓、P2 ✗（先验 40%）、P3 ✗（先验 65%，差 0.0006）、P4 ✓（先验 30%——失手方向：低估了编码对 MLP 的作用）、
P5 ✗（先验 25%）、P6 ✓、P7 ✓。

## 4. 读法（按 §4 表：L1-FLAT → L2 前先做 MLP 训练制度对照）

- **编码**：对 MLP 族是真实效应（回归下 +0.037 强显著；比较器下 +0.012 同向不显著），且从未在 pooled 上更差。
  **建议后续 T-CLSC 一律用 binned**（零成本）——这是判断而非门的结论，写在这里供拍板。
- **目标函数**：z 回归比 δ-BCE 只多 +0.006（p 0.08）。δ 过滤丢的行没有想象中致命；L3 改标签的动机主要是**给 E′ 留数据集级信号**
  （C2 的机制），不是靠它涨通用排序 acc。
- **模型族 / 训练制度 0.016**：MLP（binned + 回归）0.764 vs hgb 0.780，是当前最大的未分解项。按协议应先做一轮 MLP 训练制度
  对照（lr / epoch / curriculum / 宽度，零 GPU 分钟级），再进 L2。
- **好半区**：binned 把 top-10 内配对 acc 从 0.44 抬到 0.53–0.54，是本项目所有排序器里最高的一格（事后，n=51 实例）。
  它直接关联 SH 名单的质量（好配置之间谁更好）；建议在训练制度对照里把它列为正式次级读数并预注册。
- **不能写的**：任何"E′ / task-aware"结论——本轮没有 E 臂（折特征不在本地）；main_binned 待阶段 2。

## 5. 复现

```
python scripts/56_tclsc_l1.py --c0                                   # 硬门 + ref 逐位
python scripts/56_tclsc_l1.py --prepare --train --modes scalar binned --seeds 0 1 2 --device cuda --resume
python scripts/56_tclsc_l1.py --rank --analyze --seeds 0 1 2         # rankings.json ×2 + analysis.json / tables.md
```
