# WP16-TCLSC-L2 报告：CLS 流水线图 + 共享 GIN（2026-09-14，零 GPU / 零新 R'）

协议：`PROTOCOL.md`（协议锁 ce996cf，17 例先红 → 绿）。代码：`autocls/comparator/cls_graph.py`、`TCLSC(cls_encoder="graph")`、33/34/56 graph 支持、
`scripts/61_tclsc_l2.py`。产物：5 臂各 `results/tclsc_l2/<arm>/{runs,rankings.json}`（模型 gitignore）、`analysis.json`、`tables.md`、`train.log`。
硬件：本地 4060，105 跑 × 41–75 s（与 φ / 指纹阶段的测试并行时变慢），脱离启动零事故。

## 0. 一句话结论

**读数码 L2-FLAT**：流水线拓扑不加信息。主臂 `gin_pipe_hub` 配对 acc 0.761 vs binned MLP 0.758（Δ +0.003，p 0.21，4/7 折）；
拓扑对照 `gin_full_mean` 0.756、**无边的 `gin_none_mean` 0.769 反而最高**（vs MLP +0.011，p 0.026，57 胜 42 负）——但所有 GIN 臂的
好半区 top-10 acc（0.49–0.52）都低于 MLP 的 0.528。**结论：CLS 支路定为 binned MLP（关键决策 1 维持）**；"逐字段 embedding + 均值池化"
这个结构在 δ 对全域上有 +0.01 的边际，好半区更差，不采纳。

## 1. C0

L1 `cls_only_binned` 复算 0.7579（= 0.758 ✓）；结构对照 `gin_none_mean` 0.769 在 0.03 带内 ✓。

## 2. 主表（126 实例，7 折 LODO cls_only）

| 臂 | 拓扑 / 读出 / 标量 | 配对 acc | top-10 内 acc | Spearman | regret@1 | regret@3 |
|---|---|---:|---:|---:|---:|---:|
| mlp_binned（L1，ref） | — | 0.758 | **0.528** | 0.402 | **0.066** | 0.032 |
| gin_pipe_hub（主臂） | pipe / hub / ✓ | 0.761 | 0.499 | 0.420 | 0.080 | 0.029 |
| gin_pipe_mean | pipe / mean / ✓ | 0.763 | 0.515 | 0.415 | 0.069 | 0.031 |
| gin_full_mean | full / mean / ✓ | 0.756 | 0.495 | 0.411 | 0.072 | 0.030 |
| gin_none_mean | none / mean / ✓ | **0.769** | 0.487 | 0.416 | 0.073 | 0.032 |
| gin_pipe_hub_noscalar | pipe / hub / ✗ | 0.764 | 0.507 | **0.426** | 0.081 | 0.031 |

逐折：pipe_hub 只在 Ethanol / UWave / UM 高于 MLP；none_mean 在 Cricket / HW / PS / UM 高。配对 Wilcoxon（vs MLP）：pipe_hub p 0.21、pipe_mean 0.18、
none_mean **0.026**、noscalar 0.056、full 0.75。pipe_hub vs full_mean +0.005 p 0.21；vs none_mean −0.008。

## 3. 门与预测

| 门 | 读数 | 过 |
|---|---|---|
| C0 | 0.7579 / none 0.769 | ✓ |
| G-L2a（图 vs MLP +0.02 且 5/7） | +0.003，4/7 | ✗ |
| G-L2b（拓扑 vs 完全图 +0.01 且 5/7） | +0.005，4/7 | ✗ |

预测 1/6（P1 ✓；P2 ✗ 先验 30%；P4 ✗ hub < mean；P5 ✗ 标量 −0.003；P6 ✗）。

## 4. 读法

1. **19 个固定节点上关系先验无信息**：MLP 已经看见全部字段；把 sim_func–temperature–loss 与增强顺序写成边，学不到 MLP 学不到的东西。
   `order` 字段作边 vs 作 one-hot 也无差别（pipe vs none）。
2. **有一点点信号的是"逐字段 embedding + 池化"**（none_mean +0.011 p 0.026）——它与 binned MLP 的差别是每字段 embedding 经**共享** MLP 再平均
   （置换不变的 DeepSets），比 binned MLP 的"拼接后一层 Linear"多了参数共享。但它在好半区更差（0.487 vs 0.528），regret 不动——不值得换。
3. **论文式的枢纽读出不如均值**（P4 ✗），标量序数无边际（P5 ✗）。
4. **对重启路线的含义**：CLS 支路到此定型 = binned MLP；比较器侧剩余差距（vs hgb 0.016）归模型族 / 训练制度，不再投。真 E 臂按 binned 跑。

## 5. 复现

```
python scripts/61_tclsc_l2.py --prepare --train --device cuda --resume     # 105 跑 ≈ 70 min
python scripts/61_tclsc_l2.py --rank --analyze
```
