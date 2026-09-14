# WP16-TCLSC 阶段 1 真 E 臂报告：diff 头 + SetTF 任务嵌入 E，7 折 LODO（2026-09-14/15，pod 4090）

协议：`PROTOCOL.md`（协议锁 a10dc52，6 例先红 → 绿）。代码：33 号 `E_ARMS`、`scripts/62_tclsc_s1e.py`、pod 三件套 `pa_pod/s1e_{bundle,setup,run}.sh`。
产物：`binned/{runs,rankings_s1e.json,set_encoder_init.pt,analysis.json,tables.md}`（42 模型 .pt 与 out_s1e.tar.gz 279 MB gitignore；折特征留 pod 容器盘已随 pod 停止消失，可再生）。
**pod**：`nibk1kf2rjw4y9`（s1e-realE-4090，RTX 4090 $0.74/h，EU-RO-1，卷 hgj73ti51t）2026-09-14 20:50 本地开机，主流程 19:07–23:32 UTC，
2026-09-15 01:38 本地 `stop` + `confirm-stopped 3/3 EXITED`，uptime 17319 s ≈ 4.81 h ≈ **$3.56**。fit-folds 7 折 ≈ 8 min（每折 TS2Vec 28–33 s）；
E 臂每跑 ≈ 720–740 s（SetTF 联合训练 200 epoch，两进程并行占 21.6 GB）。CLS 编码 binned（L2 读数 FLAT → 不用 graph）。

## 0. 一句话结论

**读数码 E-INERT**：E 在两种头下都**不塌缩**（G-E0 ✓：diff 头跨数据集余弦中位 0.58、数据集内 0.9999；concat 头 0.17–0.51），
比较器**依赖 E 的存在**（zero 消融 M0 −0.028 p 7e-11、M2 −0.022）却**不依赖 E 的内容**（cross 消融：M1 +0.003 p 0.88、M0 +0.002）；
regret@3 0.032 vs 任务盲 0.029（p 0.84）。**学出来的 E 是"数据集身份向量"（同数据集切片余弦 ≈ 1），diff 头把它当尺度/归一化输入用，
而不是偏好选择器**——换成别的数据集的身份向量排名不变。最初设想（T-CLSC + 真 E → task-aware）在它最好的形态下（diff 头、binned、
非塌缩 E、LODO）给出的是：E 有信息、比较器用了、但用的不是可迁移的偏好。

## 1. C0

`e_concat_hard` M0 0.758 ∈ [0.70, 0.80] ✓；δ 对口径 cross 档 7 折：zero / ambiguous 为主，**Cricket 一个种子 pass**（+0.0097）→ 严格"全 zero"未过（形式 ✗），
但 pooled 排名口径 cross Δ_M0 +0.000（p 0.48）——A0 的"E 消融零"在 binned 下实质复现。

## 2. 主表（126 实例，7 折 LODO；M1 只算有存活对的 51 实例）

| 臂 | M1 好半区 acc | M2 | M0 δ 对 acc | Spearman | regret@1 | regret@3 |
|---|---:|---:|---:|---:|---:|---:|
| blind_diff_hard（φ 阶段，任务盲） | 0.499 | 0.731 | 0.756 | 0.406 | **0.066** | **0.029** |
| e_concat_hard | 0.501 | 0.730 | 0.758 | 0.399 | 0.074 | 0.033 |
| e_diff_hard | 0.492 | **0.742** | **0.768** | **0.427** | 0.079 | 0.032 |

消融（排名口径）：

| 臂 | 消融 | Δ_M1（n=51） | p | Δ_M2（n=126） | p | Δ_M0 | p | 胜/负/平（M0） |
|---|---|---:|---:|---:|---:|---:|---:|---|
| e_concat_hard | cross | −0.007 | 0.90 | +0.001 | 0.41 | +0.000 | 0.48 | 39/39/43 |
| e_concat_hard | zero | −0.008 | 0.83 | +0.000 | 0.44 | +0.000 | 0.34 | 34/28/59 |
| e_diff_hard | cross | +0.003 | 0.88 | +0.001 | 0.07 | +0.002 | 0.032 | 23/16/82 |
| **e_diff_hard** | **zero** | +0.008 | 0.35 | **+0.022** | **1e-10** | **+0.028** | **7e-11** | **71/21/29** |

逐折 M0（e_diff / blind）：Cricket 0.797/0.804、Ethanol 0.535/0.518、HMD 0.735/0.745、HW 0.811/0.796、PS 0.802/0.783、UWave 0.828/0.790、UM 0.849/0.839——
e_diff 5/7 折更高（+0.01–0.04），regret@3 不动。逐预算 M0（e_diff / blind）：@60 0.734/0.723、@120 0.809/0.787、@290 0.762/0.758。
33 号 δ 对口径 cross（e_diff）：HW +0.015、UM +0.011、PS +0.010、HMD +0.016 等，档位 detectable / ambiguous，无 pass。

E 诊断（seed 0，45 切片）：e_diff 跨集余弦 HW 0.58 / UM 0.56 / PS 0.60 / UWave **1.00** / Ethanol 0.43 / Cricket 0.56 / HMD 0.58，集内 0.999–1.000，
范数 80–519；e_concat 跨集 0.17–0.51。（实验 1 在 scalar + concat 下曾见跨集 0.9999——本轮 binned 下两种头都不塌缩，塌缩不是头的问题。）

## 3. 门与预测

| 门 | 读数 | 过 |
|---|---|---|
| C0 | M0 0.758；cross 档 6/7 折 zero/ambiguous、Cricket 一种子 pass | ✗（形式）/ ✓（实质） |
| G-E0 不塌缩 | cross 0.58 < 0.99，within − cross 0.42 | ✓ |
| G-E1 用了 E（M1 cross） | +0.003，p 0.88 | ✗ |
| G-E2 有价值 | regret@3 0.032 vs 0.029，p 0.84 | ✗ |
| G-E3 concat 零 | −0.007 | ✓ |

预测 2/6（P2 ✓ 不塌缩、P1 形式 ✗；P3–P6 ✗；P6 差 0.0003：e_diff M0 比 concat 高 0.0097）。

## 4. 读法

1. **E 学成了"我是哪个数据集"**：数据集内余弦 ≈ 1、跨集 0.4–0.6——SetTF 在 7 个源上学到的是身份，不是"这个数据集偏好什么"。
   这和 L3 阶段 0 的 oracle 身份 token 在同分布下 +0.205 并不矛盾：那里比较器在训练集里见过每个身份对应的偏好；LODO 下目标的身份向量是新的，
   比较器对它没有查表项，只能把它当一个"尺度大小合理"的输入——所以 zero（尺度归零）伤、cross（换成另一个合理尺度）不伤。
2. **头的结论不变**：concat 下 E 连 zero 消融都零（真的没用）；diff 下 E 被用（zero −0.028）。diff 头"能用任务向量"跨四种条件变量复现。
3. **阶段 1 四臂总账**（LODO，diff 头，同 126 实例）：

   | 条件变量 | 对新数据集 | cross 消融（用了内容？） | zero 消融（用了存在？） | regret@3 vs 任务盲 0.029 |
   |---|---|---|---|---|
   | oracle 身份（S2，非 LODO） | 无定义 | +0.205 | — | 0.027 vs 0.080 |
   | φ 元特征 | 有 | M0 +0.035 p 3e-11 | +0.007 | 0.031 |
   | 探针指纹 | 有 | −0.042（反向） | −0.036 | 0.031 |
   | UCR n=72 φ / 指纹 | 有 | +0.013 / +0.018 | +0.010 / +0.003 | 0.021 vs 0.022（p 0.045）/ 0.021 |
   | **真 E（SetTF）** | 有 | **+0.002** | **+0.028** | 0.032 |

   可迁移的偏好信息：φ 有一点（粗排序）、指纹反向、真 E 零。选择价值全部为零（除 n=72 的 φ +0.001）。
4. **对主臂的含义**：预测式 dataset-aware 在比较器形态下的边界已画完——机制成立（L3）、粗排序可迁移（φ）、好半区与选择不可迁移
   （四种变量 × 两种 n），真 E 的失败模式是"身份而非偏好"。SH 的目标侧实测（probe6_top3 0.015 / 0.0125）仍是唯一拿到选择价值的路。
5. **不能写的**：真 E task-aware 成立；E 塌缩是头造成的（本轮否定）。**可以写的**：E 在 diff 头下非塌缩且被依赖但内容无关——这是 AutoCTS++ 式
   T-AHC 在跨数据集 zero-shot 下的一个明确失败模式，可作 analysis 章。

## 5. 复现

```
# pod：bash scripts/pa_pod/s1e_setup.sh /workspace/upload_s1e && MODE=binned nohup bash scripts/pa_pod/s1e_run.sh
# 本地：tar -xzf results/tclsc_s1e/out_s1e.tar.gz -C results/tclsc_s1e && python scripts/62_tclsc_s1e.py --analyze --outdir results/tclsc_s1e/binned
```
