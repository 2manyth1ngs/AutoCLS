# WP16-TCLSC 阶段 1 A″ 跨语料部署检验报告：UCR 84 全训 → zero-shot 排 UEA，XFER-DEAD（2026-09-15，本地 4060，零 GPU 新 R'）

协议：`PROTOCOL.md`（协议锁 24c69bb，5 例先红 → 绿 834f7af）。代码：`scripts/65_tclsc_s1ea_xfer.py`。产物：`runs/fold0`（9）、`rankings_xfer.json`（8 靶 45 切片）、
`analysis.json`、`tables.md`（模型 gitignore）。训练 9 跑 × 19–22 s ≈ 3.5 min（84 集 5040 记录）；排名 + 分析 CPU ≈ 3 min。C0 三条参照臂 regret@3 逐位。

## 0. 一句话结论

**读数码 XFER-DEAD−CONTENT**：UCR 84 全训的比较器在 UEA 126 实例上全部输给 UEA 自训（n=7）的任务盲——`u_eaphi` regret@3 **0.0352** vs blind(UEA) **0.0294**
（G-X1 ✗，p 0.75，45:53）、`u_blind`(UCR) 0.0374（p 0.97 反向，35:56）、`u_phi` 0.0368；`u_eaphi` 相对同语料任务盲有改善但不显著（G-X2 ✗，0.0352 vs 0.0374 p 0.093；
regret@1 0.085 vs 0.107 **p 0.002**）；换他集画像 regret 不变（G-X3 ✗）。
**读法：跨语料位移（单变量 UCR → 多变量 UEA）压过训练侧数据集数的增益**——连任务盲先验都从 84 集迁不到 UEA（regret@1 0.107 vs 0.066），
"n 是瓶颈"只在同语料内成立。预测式主臂在比较器形态下的成立域到此画完：**同语料 LODO、训练侧 ≳ 70 集**；官方口径部署（UEA 靶）不能靠 UCR 训练侧。

## 1. 主表（UEA 7 靶 126 实例）

| 臂 | 训练侧 | M1 | M2 | M0 | regret@1 | regret@3 |
|---|---|---:|---:|---:|---:|---:|
| blind_diff_hard | UEA 7 折（n=7） | 0.499 | 0.731 | 0.756 | **0.0657** | **0.0294** |
| phi_diff_hard | UEA 7 折 | 0.499 | 0.748 | 0.780 | 0.0771 | 0.0311 |
| eaphi_diff_hard | UEA 7 折 | 0.521 | 0.756 | **0.793** | 0.0791 | 0.0354 |
| u_blind_diff_hard | UCR 84 | 0.486 | 0.733 | 0.749 | 0.1069 | 0.0374 |
| u_phi_diff_hard | UCR 84 | 0.508 | 0.729 | 0.744 | 0.0909 | 0.0368 |
| **u_eaphi_diff_hard** | UCR 84 | 0.518 | 0.741 | 0.757 | 0.0851 | 0.0352 |

配对（H1 b < a）：blind(UEA) → u_eaphi regret@3 p 0.75、regret@1 p 0.99；**u_blind → u_eaphi regret@1 0.107 → 0.085 p 0.002（32:18）**、regret@3 p 0.093（31:19）；
u_blind → u_phi regret@1 p 0.019；blind(UEA) → u_blind(UCR) regret@1 0.066 → 0.107 p 1.0（33:78）。
消融（UCR 训练臂）：u_eaphi cross Δ_M0 +0.009（p 2e-4）、Δ_M1 +0.008；regret@3 cross 0.0363 / zero 0.0329（不伤）；u_phi regret@1 cross 0.110 / zero 0.114（p 0.002 / 4e-4）——
φ 在 top-1 上被用，画像在 top-3 上没被用。
逐靶 regret@3（blind(UEA) / u_eaphi）：Cricket 0.015/0.009、Ethanol 0.043/0.032、HMD 0.030/**0.102**、HW 0.052/0.034、PS 0.030/0.028、UWave 0.018/0.015、UM 0.018/0.027——
u_eaphi 5/7 靶更好，HMD 一靶（EEG 10 通道、UCR 里没有同类）崩到 0.10 把均值拉翻；SCP2 附录（9 实例）u_eaphi 0.0244 < u_phi 0.0302 < u_blind 0.0341。

## 2. 门与预测

| 门 | 读数 | 过 |
|---|---|---|
| C0 | 0.0294 / 0.0311 / 0.0354 逐位 | ✓ |
| G-X1 部署价值（< blind UEA） | 0.0352 vs 0.0294，p 0.75 | ✗ |
| G-X2 画像价值（< u_blind UCR） | 0.0352 vs 0.0374，p 0.093 | ✗ |
| G-X3 内容依赖 | cross 0.0363 vs 0.0352，p 0.42 | ✗ |

预测 3/7：P1 C0 ✓、P5 u_blind(UCR) ≥ blind(UEA) ✓、P6 M0 ≥ 0.75 ✓；P2 / P3 / P4 / P7 ✗。

## 3. 读法

1. **位移在任务盲层就发生了**：84 个单变量数据集学到的"什么 CLS 普遍好"在多变量 UEA 上比 7 个 UEA 数据集学到的差（regret@1 +0.04）——与 WP15 的
   "非增强轴偏好随基 / 语料翻转"一致。响应画像在这个位移之上只能修回一部分（u_blind → u_eaphi regret@1 −0.022 p 0.002，5/7 靶 regret@3 更好），修不回 HMD 这类 UCR 里没有的域。
2. **"n 是瓶颈"的读法要收窄**：A′ 证明的是"同语料内 n=72 成立、n=7 不成立"；A″ 证明"换语料补 n 不行"。合起来：预测式 dataset-aware 需要**同域**的几十个训练集。
3. **可以写的**：(a) UCR 84 LODO 上机制特征作任务嵌入的比较器有可复现的选择价值（A/A′）；(b) 跨语料 zero-shot 失败模式 = 任务盲先验本身不迁移（analysis 章）；
   (c) 画像相对同语料任务盲在 top-1 上有显著改善（p 0.002）。**不能写的**：训练侧换 UCR 即可部署到 UEA；n 是唯一瓶颈。
4. **对官方口径部署的含义**：UEA 6 靶若要用预测式主臂，训练侧必须是多变量同域语料且 ≳ 数十个数据集——现有 R' 缓存只有 8 个（mx3_pq）；WP10 的 23 个靶外 UEA
   数据集是官方口径全训（每靶 22 格），口径不同不能直接当训练侧。

## 4. 下一步候选（待拍板）

- **A‴ 合池训练**（零 GPU，63 跑 ≈ 25 min）：UEA 7 折 LODO 里训练侧 = 该折 7 个 UEA 集 + UCR 84 集（合池 91 集，δ 各按语料），考 UEA 126 实例——问"在同域 7 集之上再加 84 个异域集是否有帮助"；
  这是 XFER-DEAD 之后唯一还能把 n 补上去的零 GPU 路径。过线 = eaphi(合池) regret@3 < blind(UEA 自训) 且 < blind(合池)。
- 若 A‴ 也不过：预测式主臂定稿为 UCR 84 语料的同语料结论 + 三条失败模式（身份 E / n=7 / 跨语料），官方口径部署维持 SH 实测；B（度量学习 E）在 UCR 语料上作可选增强。

## 5. 复现

```
python scripts/65_tclsc_s1ea_xfer.py --train --device cuda     # 9 跑 ≈ 3.5 min
python scripts/65_tclsc_s1ea_xfer.py --rank --analyze
```
