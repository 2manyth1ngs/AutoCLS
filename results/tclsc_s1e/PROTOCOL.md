# WP16-TCLSC 阶段 1（真 E 臂）PROTOCOL：diff 头 + SetTF 任务嵌入 E，7 折 LODO（GPU：折特征 + 联合训练）

**会话定位**：重启 T-CLSC 的阶段 1 最后一条可迁移条件变量。前三轮（φ / 转导指纹 / UCR n 放大）的共同形状：diff 头能用可迁移变量、
粗排序可迁移、好半区增益 0.001 量级；oracle 身份 token 在同分布下 +0.205。真 E（AutoCTS++ 式 9–12：目标无标签窗口经共享 TS2Vec →
两级 Set Transformer）是维度最高、与比较器**联合训练**的条件变量——也是最初设想里的那一个。实验 1 曾看到 E 在 concat 头下训成常向量
（跨数据集余弦 0.9999）；一个可检验的解释：concat 头没有给 E 保持信息的梯度理由，diff 头有。本协议同时问两件事：**E 在 diff 头下是否
仍退化**（诊断）与 **E 是否给出可迁移的偏好**（LODO 消融 + 价值）。

材料：`results/mx3_pq/` 只读；折特征 `folds/<target>/task_features/*.npz` 与 `all8` 不在本地 → pod 上 `33 --fit-folds`（每折 TS2Vec ≈ 20 s，
整段 < 15 min，见 memory `mx3pq-pod-run-20260902`）。用户 2026-09-14 授权：L2 收官后自主开 pod、跑完必须停（[[runpod-api-and-stop-rule]]）。
分支 wp9-p1。产物 `results/tclsc_s1e/`。

## §0 已看过的读数

- WP2 / MX-3-PQ A0 `main`（scalar + concat + SetTF E）：LODO 配对 acc 0.735、E cross 消融 7/7 折零；实验 1：E 余弦 0.9999、范数 ×21、logit 灵敏度 13%。
- L1：binned cls_only 0.758；φ 臂 blind_diff_hard 0.756 / regret@3 0.029（LODO 同口径，作任务盲基线）；φ_diff M0 cross Δ +0.035 p 3e-11、M1 +0.011。
- L3 阶段 0：diff 头 + 身份 token Δ_M1 +0.205。
- **没看过**：任何 diff 头 + 真 E 的比较器；E 在 diff 头下的余弦 / 范数。

## §1 假设

H-E1（诊断）：diff 头下联合训练的 E 不再塌缩——跨数据集余弦 < 0.99 且数据集内余弦 > 跨数据集余弦。
H-E2（迁移）：LODO 下 E 的 cross 消融正向（M1 ≥ 0.03，p < 0.05）且 regret@3 低于任务盲 diff 头。
对立：E 在 diff 头下不塌缩但只承载粗排序（与 φ 同形）；或仍塌缩（SetTF 在 7 数据集下无可学的区分）。

## §2 材料与口径

- 划分：S1 7 折 LODO，126 实例；E 来自**该折**的共享 TS2Vec（只在源池 7 个数据集上 fit，铁律 6）对目标切片窗口的编码；SetTF 与比较器联合训练；
  `settf_windows` / `settf_t_max` / `nz` 按 `configs/labels.yaml` 原样（与 MX-3-PQ 同）。
- CLS 编码 = **L2 读数决定**：L2-GRAPH-WIN → graph（gin_pipe_hub）；否则 binned。头 diff（主）/ concat（对照）；标签 hard；超参不变。
- 排名（pod 上 `62 --rank`，需窗口 npz）：3 种子各自的 E（SetTF 与比较器联合训练，种子间 E 不同）→ `rank_candidates` 多数票；
  **消融**：cross = E 换成**他数据集训练切片**的窗口经同一模型 SetTF 的 E（3 种子错位）；zero = E 置 0；预算 one-hot 不动。
- E 诊断（seed 0 模型，每折）：45 切片的 E → 跨数据集平均余弦 / 数据集内平均余弦 / 范数中位；与实验 1 的 F1 同式。
- 指标同前（M1 好半区 / M2 / M0 / regret@1/@3），126 实例；任务盲基线 = φ 阶段的 `blind_diff_hard` 逐实例行（`results/tclsc_s1phi/analysis.json`）。

## §3 臂（`scripts/33` 独立表 `E_ARMS`，S1，3 种子，`e_source="fold"`，铁律 6 合规）

| 臂 | 任务支路 | 头 | 用途 |
|---|---|---|---|
| `e_concat_hard` | E+ctx（SetTF） | concat | = MX-3-PQ `main` 换 binned 编码：头对照 + C0 |
| **`e_diff_hard`** | E+ctx（SetTF） | diff | **主臂** |

42 跑；每跑含 SetTF 联合训练 ≈ 3 min（4090，每进程 ~10 GB 显存，两并行）≈ 1.1 h；`--fit-folds` ≈ 15 min；总 ≈ 2 h。

## §4 门（先于数据）

- **C0**：`e_concat_hard` 的 E cross 消融（33 号 run json，δ 对口径）7 折全 zero（复现 A0），且 pooled M0 ∈ [0.70, 0.80]。
- **G-E0（不塌缩，诊断）**：`e_diff_hard` seed 0 的 E 跨数据集平均余弦 < **0.99** 且 数据集内余弦 − 跨数据集余弦 ≥ 0.01（7 折中位）。
- **G-E1（用了 E）**：`e_diff_hard` 排名口径 cross 消融 Δ_M1 ≥ **0.03** 且 p < 0.05；zero 同向只报。
- **G-E2（有价值）**：`e_diff_hard` regret@3 < `blind_diff_hard`（φ 阶段），126 实例配对 Wilcoxon p < 0.05。
- G-E3（头对照）：`e_concat_hard` 排名口径 Δ_M1 零——只报。

**读数码**：

| G-E0 | G-E1 | G-E2 | 码 | 含义 |
|---|---|---|---|---|
| ✓ | ✓ | ✓ | **E-ALIVE** | 最初设想成立（diff 头 + 真 E 在 LODO 有选择价值）：进官方口径、写主臂 |
| ✓ | ✓ | ✗ | **E-USED-NO-GAIN** | E 不塌缩、被用、但只到粗排序（与 φ 同形）：n=7 的结构性上界，写 analysis |
| ✓ | ✗ | — | **E-INERT** | E 不塌缩但比较器不用（信息与偏好不对齐） |
| ✗ | — | — | **E-COLLAPSED** | diff 头也救不了 E 的塌缩：SetTF 在 7 数据集下无区分，真 E 路线关闭 |

## §5 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 过 | 85% |
| P2 | G-E0 过（不塌缩） | 55% |
| P3 | G-E1 过 | 35% |
| P4 | G-E2 过 | 20% |
| P5 | e_diff_hard M0 cross Δ 显著（p<0.01，δ 对或排名口径任一） | 60% |
| P6 | e_diff_hard pooled M0 ≥ e_concat_hard + 0.01 | 55% |

## §6 成本与 pod 规矩

- 4090（$0.74/h）优先，EU-RO-1 卷 `hgj73ti51t`（MX-3-PQ 的 `/workspace/wp/CLS` 含 8 数据集与旧折特征，若在且 meta 一致可直接复用）；无货则 5090（$0.99/h）。
- 上传：代码 bundle（HEAD）+ mx3_pq 四件 + `configs/labels.yaml`；数据集缺则上传 197 MB（7 UEA .ts + UniMiB npz/json）。
- 顺序：setup（pip pin + 解包 + LF + md5 + CUDA 闸）→ `33 --fit-folds`（缺则）→ `33 --train --arms e_concat_hard e_diff_hard --encoder-mode <L2 决定> --resume`
  两进程分折并行 → `62 --rank --device cuda` → 打包拉回（runs / models / rankings_s1e.json / e_diag.json）→ **stop + confirm-stopped ×3** → 本地 `62 --analyze` → REPORT。
- 所有 pod 调用显式 `--device cuda` + 启动前 CUDA 闸（[[runpod-nvml-dropout-mid-run]]）。

## §7 执行顺序与协议锁

1. 本文件 + `tests/test_tclsc_s1e.py`（先红）→ commit。
2. 实现 33 `E_ARMS`、`scripts/62_tclsc_s1e.py`（rank / diag / analyze）、`scripts/pa_pod/s1e_{bundle,setup,run}.sh` → 单测绿 → commit。
3. pod 全流程 → REPORT → commit。

## §8 铁律对照

铁律 3 零新 R'（折特征是 TS2Vec 表征不是 R'）；铁律 4（task_embed 代码沿用已对照 AutoCTS++ 的实现）；铁律 5；铁律 6 折内 TS2Vec 只 fit 源池；
铁律 7 E 只是浮点向量；关键决策 8 不触及。

## §9 执行台账（跑后填写）

（空）
