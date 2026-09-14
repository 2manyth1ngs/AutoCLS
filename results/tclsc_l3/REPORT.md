# WP16-TCLSC-L3 阶段 0 报告：融合头 × 标签形式，oracle 数据集身份 token（2026-09-14，零新 R'）

协议：`PROTOCOL.md`（协议锁 1f88493，先红 32 例 → 33 例绿）。代码：`autocls/comparator/tclsc_model.py`（`head` ∈ concat/diff/film）、
`autocls/comparator/zdiff_labels.py`、`scripts/33`（`L3_ARMS` / `arm_spec` / zdiff 采样 / meta.head）、`scripts/34`（按 meta 重建 head）、
`scripts/56`（`rank_local_models(ctx_rows=)`）、`scripts/57_tclsc_l3.py`。产物：`c0.json`、`binned/`（24 模型 + runs + `rankings_l3.json`）、
`analysis.json`、`tables.md`、`train.log`。硬件：本地 RTX 4060；hard 臂 10–16 s/跑、zdiff 臂 280–370 s/跑（全对 45k 对/epoch）。

**定位（先于任何数字）**：S2 划分 + oracle 身份 token = **机制上界**——8 个数据集都参与训练（各 4 切片），考侧是同数据集的留出切片；
它回答的是"给了完美任务信号，比较器的头能不能用"，**不是 zero-shot**。zero-shot 只在阶段 1（真 E + LODO）。

## 0. 一句话结论

**读数码 L3-HEAD-ONLY（按预注册判序）；但 G-A / G-B / G-C 全过**：反对称 diff 头在 δ 硬标签下就能用身份 token
（Δ_M1 **+0.205，p 9e-4**，12 胜 2 负），z 差分标签在 concat 头下也能（+0.187，p 0.03），两者叠加在 M2 / regret 上最好
（`l3_diff_zdiff` regret@1 **0.027** vs 任务盲 0.080，regret@3 0.016 vs 0.035）。concat + hard（= A8 在 binned 下）仍零（+0.020，p 0.5）。
**这是本项目七轮 task-aware 零消融之后，任务信号消融第一次离开零**——机制是"晚期 concat 头 + δ 硬标签"这一对组合把任务向量逼成偏置，
换掉任一个都解开。

## 1. C0：PASS

本地重训 `onehot_oracle`（scalar，pod 同臂）normal acc 0.7922 vs pod 0.7932（带 0.005 内）；cross Δ̄ −0.003 zero（A8 复现）。
中途事故：初版 `head` 改动把 `set_fc` 构造挪到 `cls_fc` 之前 → 同 seed 初始化变、C0 失守 0.008；恢复构造顺序 + 加"同 seed 权重逐位金标"
单测后过（memory `module-construction-order-is-rng-order`）。

## 2. 主表（45 实例 = 15 留出切片 × 3 预算；M1 只有 17 个实例在真值前 10 内有 δ 存活对）

normal（各切片自己的数据集 token）：

| 臂 | 头 | 标签 | M1 好半区 acc | M2 \|Δz\| 加权 acc | M0 δ 对 acc | Spearman | regret@1 | regret@3 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| l3_none_hard | — | hard | 0.766 | 0.725 | 0.770 | 0.393 | 0.080 | 0.035 |
| l3_none_zdiff | — | zdiff | 0.737 | 0.711 | 0.739 | 0.372 | 0.064 | 0.029 |
| l3_concat_hard（A8） | concat | hard | 0.780 | 0.721 | 0.769 | 0.388 | 0.084 | 0.034 |
| l3_concat_zdiff | concat | zdiff | 0.771 | 0.731 | 0.779 | 0.426 | 0.046 | 0.019 |
| **l3_diff_hard** | diff | hard | **0.821** | 0.747 | 0.788 | 0.438 | 0.062 | 0.027 |
| **l3_diff_zdiff** | diff | zdiff | 0.816 | 0.751 | 0.803 | **0.457** | **0.027** | **0.016** |
| l3_film_hard | film | hard | 0.806 | 0.749 | 0.808 | 0.447 | 0.056 | 0.020 |
| l3_film_zdiff | film | zdiff | 0.797 | **0.753** | **0.809** | 0.455 | 0.036 | 0.019 |

消融 Δ = normal − cross（数据集 token 错位到别的数据集，3 个错位种子均值；Wilcoxon H1 normal > cross）：

| 臂 | Δ_M1（n=17） | p | 胜/负/平 | Δ_M2（n=45） | p | Δ_M0 | p | 33 号 zero_ctx Δ̄（token 置零） |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| l3_concat_hard | +0.020 | 0.50 | 1/2/14 | +0.002 | 0.009 | +0.002 | 0.016 | −0.000 |
| l3_concat_zdiff | +0.187 | 0.030 | 9/4/4 | +0.127 | 5e-07 | +0.173 | 1e-06 | +0.067 |
| l3_diff_hard | **+0.205** | **9e-04** | 12/2/3 | +0.035 | 1e-06 | +0.039 | 2e-05 | +0.044 |
| l3_diff_zdiff | +0.189 | 0.023 | 10/3/4 | +0.122 | 2e-06 | +0.170 | 7e-08 | +0.077 |
| l3_film_hard | +0.101 | 0.018 | 8/2/7 | +0.068 | 2e-08 | +0.098 | 1e-07 | +0.052 |
| l3_film_zdiff | +0.130 | 0.037 | 9/4/4 | +0.114 | 3e-07 | +0.163 | 4e-07 | +0.066 |

zero_ctx（token 置零而非错位）列说明效应不只是"错 token 伤人"：去掉 token 同样掉分。

## 3. 门与预测对账

| 门 | 读数 | 过 |
|---|---|---|
| C0 | 0.7922 vs 0.7932 | ✓ |
| G0（concat+hard 零） | Δ_M1 +0.020，p 0.5 | ✓ |
| G-A（头单独换） | diff +0.205 p 9e-4；film +0.101 p 0.018 | ✓ |
| G-B（头 + 标签） | diff +0.189 p 0.023；film +0.130 p 0.037 | ✓ |
| G-C（标签单独换） | concat +0.187 p 0.030 | ✓ |

预测 **5/8**：P1–P5 ✓；P6 ✗（diff > film，先验 50%）；P7 ✗（任务盲臂换 zdiff 后 M0 掉 0.031，ctx 臂反而 +0.01；zdiff 的价值只在有任务
token 可条件化时出现）；P8 ✗（任务盲 zdiff M1 −0.029 于 hard——L1 的"回归目标下 binned 更好"不等于"配对软标签下任务盲比较器更好"）。

## 4. 读法

1. **七轮零消融的机制现在有了直接证据**：同一批数据、同一个任务 token，concat+hard 零，换头或换标签都不零。C2 / 实验 1-2 / MX-3-PQ 当时
   同时踩着两个因子（晚期 concat 头 + δ 硬标签 + 退化的 E），任何单独的消融都看不到出口——这不推翻它们的读数，但把"任务信号不可学"
   改写成"在那个头和那种标签下不可学"。
2. **头是更便宜的杠杆**：diff 头 + hard 标签 10 s/跑就拿到最大的 Δ_M1；zdiff 每跑 5 min，叠加只在 M2 / regret 上再加一点。
   阶段 1 主臂 = **diff + hard**（按读数码），diff + zdiff 作副臂。
3. **film 不如 diff**（P6 ✗）：调制 CLS 编码本身比在打分层交互差 0.05–0.1；但 film 的 M2 / M0 最高——它"用得多、排得平"。
4. **regret 是最实用的一格**：diff_zdiff 把 S2 上的 regret@1 从 0.080（任务盲）压到 0.027、regret@3 0.035 → 0.016。这是上界口径，但
   量级已经到了 SH6-3-3 在 LODO 上的水平（0.018）。
5. **不能写的**：任何 zero-shot / task-aware 成立的主张。身份 token 在 LODO 下对未见数据集没有定义；阶段 1 要问的是
   "学出来的 E（SetTF over TS2Vec 窗口）在 diff 头 + LODO 下能不能替代身份"——实验 1 看到 E 训成常向量，
   一个可检验的解释是 concat 头没有给 E 保持信息的梯度理由，diff 头有。这是阶段 1 的假设，不是结论。

## 5. 阶段 1 需要什么

真 E 的折特征（`folds/<target>/task_features/*.npz`）不在本地，须 `33 --fit-folds`（7 折 + all8 共享 TS2Vec 拟合 + 窗口编码，GPU，
估时待标定）；之后 `33 --train --arms <diff 版 main> --encoder-mode binned` 7 折 × 3 种子（含 SetTF 联合训练，每跑 ≈ 3 min 4060）。
开 GPU 前按 `wp-stage-reports-and-pod-consent` 请示；阶段 1 协议另立（LODO、E 消融 shuffle / zero、主指标同 M1、对照 = 先验 + SH）。

## 6. 复现

```
python scripts/57_tclsc_l3.py --c0 --device cuda
python scripts/57_tclsc_l3.py --prepare --train --device cuda --resume      # 24 跑 ≈ 65 min（zdiff 慢）
python scripts/57_tclsc_l3.py --rank --analyze
```
