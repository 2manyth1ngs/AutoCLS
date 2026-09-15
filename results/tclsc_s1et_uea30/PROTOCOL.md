# WP16 U 检验 PROTOCOL：官方口径 UEA 靶外 23 靶——E_T T-CLSC 名单 + SH6-3-3 vs 任务盲 T-CLSC 名单（消融）vs 先验名单（零新目录 R'；新冠军补 3 种子 finals 上云）

**会话定位**：D′ 在六个 LODO 靶上读 DEPLOY-ET-TR-WIN（3 种子 0.575 / 6/6，先验 SH6 0.547），但 4/6 靶冠军与先验相同——用户担心审稿人读成"E 没用"。
本协议把同一部署形态搬到 WP10 的 23 个靶外 UEA 数据集（`results/uea30/`：每靶 22 格官方口径 = shared-10 + GGS + 孪生，25/50/100% 截点 seed 0，
主臂 SH10-4-3_prior 冠军已有 3 种子），把名单对比从 n=6 扩到 n=23，并按用户要求做**任务盲 T-CLSC 名单的消融**。比较器不含这 23 靶（训练侧 = 源池 8 集全部切片，
all8 模型），E_T 在各靶官方 TRAIN 上用 GGS 探针算。分支 wp9-p1。产物 `results/tclsc_s1et_uea30/`。

## §0 已看过的读数

- WP10：SH10-4-3_prior 3 种子 23/23 追平 TS2Vec、全 29 均值 0.733；SH6-3-3 未作主臂（`uea30_arms` 有 SH6-3-3_prior 的 seed 0 读数）。
- D′（六靶）：E_T 限域名单 SH6-3-3 seed 0 0.564 / 3 种子 0.575；先验 0.542 / 0.547；任务盲 T-CLSC 限域 0.540。
- **没看过**：E_T 在 23 靶官方 TRAIN 上的值；all8 比较器对 23 靶目录的任何排名；23 靶 SH6-3-3 三名单的冠军。

## §1 假设

H-U1：23 靶上 E_T T-CLSC 名单 + SH6-3-3 的冠军 acc（seed 0）均值高于任务盲 T-CLSC 名单（同 SH6-3-3），且不低于的靶数 ≥ 15/23——E 的贡献在更多靶上可见。
H-U2：E_T 名单不低于先验名单（SH6-3-3）均值，hit@6（前 6 含目录 oracle）更多。
对立：23 靶多为贴顶 / chance 型（WP10 分类 A/B/C/D 共 18 个），三名单冠军几乎全相同——那就是"E 只在有净空的靶上起作用"的证据（E 类 5 靶）。

## §2 材料与口径

- 目录 / 真值：`results/uea30/runs.jsonl` s1_catalog（23 靶 × 22 格 seed 0）；名单**限信任域** = shared-10 + GGS 的 11 条（孪生不进名单，比较器不见 input_mask）；
  3 种子 finals：`results/uea30/runs.jsonl` s2_finals 已有的 + 本协议补跑的。
- 比较器：**all8** = 在源池 8 集全部 45 切片上训练（3 种子；63/60 号同式，无折）——`ft60_hard`（E_T@60 ⊕ φ12，field 头）与 `blind_diff_hard`（任务盲 diff）。
  用 all8 而非某个 LODO 折：23 靶都不在源池，训练侧越全越好；两臂同一训练侧。
- E_T：各靶官方 TRAIN（`uea30_arrays`，scaler=train；有 `max_train_length` 的靶（EigenWorms）截前 3000 步）上 GGS 探针 290 iters 轨迹取 @60，`rep_profile` 22 维；
  φ12：40 号 `phi_meta` + `signal_stats` 在官方 TRAIN 上算（与 `official_phi` 同式）；23 靶没有 iters0 基线，`base_mean` / `base_band` 置 None → 标准化时按 all8 训练侧均值填补（= 0），记入产物；
  ctx 标准化用 all8 训练侧统计，裁 ±4。
- 三名单：`prior`（`candidates.json` prior_order = 先验只见源池 8 集的 shared 排序 + GGS）、`blind_tclsc`（all8 任务盲）、`et_tclsc`（all8 + E_T）；SH6-3-3（36 号同式）。
- 冠军 3 种子：三名单冠军缺 seeds {1,2} 的 (靶, 配置) → picks.json → 45 号 `--finals`（官方口径 ts2vec_official）在 pod 上补跑；pod 用完必停 + 3 次确认。

## §3 门（先于数据；23 靶）

- **C0**：先验 SH6-3-3 seed 0 冠军与 `results/uea30/analysis.json` 的 `SH6-3-3_prior` 逐位。
- **G-U1（消融，主）**：E_T 名单 SH6-3-3 seed 0 均值 ≥ 任务盲 T-CLSC 名单 + 0.005，且不低于的靶数 ≥ 15/23。
- **G-U2（vs 先验）**：E_T 名单 SH6-3-3 seed 0 均值 ≥ 先验名单，且 hit@6 数 ≥ 先验。
- **G-U3（3 种子）**：E_T 冠军 3 种子均值 − 任务盲 T-CLSC 冠军 3 种子均值，23 靶配对 Wilcoxon（H1 E_T 更高）p < 0.05。
- 只报：追平 TS2Vec 论文数（−0.03）靶数（三名单，按 uea30.yaml 的 paper 值，缺论文数的靶不计）；按 WP10 A–E 类分组的均值；E 类 5 靶单列；oracle 位次；E_T ctx 出界诊断；成本。

**读数码**：G-U1 ✓ G-U3 ✓ → **UEA30-ET-WIN**；G-U1 ✓ G-U3 ✗ → **UEA30-ET-BETTER**（seed 0 赢、3 种子未显著）；G-U1 ✗ → **UEA30-ET-FLAT**。后缀 G-U2 ✓ `+PRIOR` / ✗ `−PRIOR`。

## §4 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 逐位 | 90% |
| P2 | G-U1 过 | 45% |
| P3 | G-U2 过 | 50% |
| P4 | G-U3 过 | 35% |
| P5 | 三名单冠军全相同的靶数 ≥ 12/23 | 60% |
| P6 | E 类 5 靶上 E_T ≥ 任务盲 T-CLSC 的靶数 ≥ 4/5 | 45% |

## §5 执行

1. 本文件 + `tests/test_tclsc_s1et_uea30.py`（先红）→ commit。
2. `scripts/72_tclsc_s1et_uea30.py`（--train-all8 / --features / --simulate / --analyze；finals 走 45 号）→ 绿 → commit。
3. 本地：--train-all8（6 跑）；上云（4090）：--features → --simulate（写 picks.json）→ 45 --finals → 打包拉回；本地 --analyze → REPORT → stop + confirm ×3 → memory。
   （执行中：曾在本地 4060 跑 --features，EigenWorms 3000 步把显存溢到系统内存、整机换页，19/23 后中止并作废，全部改在 pod 上从零重算——协议口径不变。）

## §6 铁律对照

铁律 2 冠军 acc 带官方 iters / n；铁律 3 零新目录 R'（只补冠军 finals，与 WP10 同口径）；铁律 5；铁律 6 all8 训练侧不含 23 靶、E_T 只用官方 TRAIN；铁律 7 ctx 只是浮点向量；死规定 pod 用完必停。

## §7 执行台账（跑后填写）

- 2026-09-15 协议锁 8c40ef8（4 例先红）→ 实现 → 绿 d419a93；本地 --train-all8 6 跑（10–12 s）。本地 4060 --features 在 EigenWorms 显存溢出（19/23 作废），改全流程上云。
- pod `iaake2zulxlwsu`（u23-uea30，RTX 5090 $0.99/h，EUR-NO-1 卷 tudo4ntlot，4090 两卷无货）：setup 全过（23 靶 .ts 在卷上 md5 全对）；主流程 15:58–16:47 UTC
  （features 23 靶 ≈ 15 min → simulate → finals 28 行 25 格 ≈ 33 min）；out_u23.tar.gz 24 KB 拉回；18:52 本地 stop + confirm-stopped 3/3，uptime 3370 s ≈ **$0.93**。
- 读数 **UEA30-ET-FLAT−PRIOR**（C0 ✓；G-U1 ✗ 0.774 vs 0.777；G-U2 ✗ vs 先验 0.785、hit@6 13 vs 19；G-U3 ✗ p 0.5），预测 3/6。详见 REPORT.md。
