# WP16-TCLSC 阶段 1 D 检验 PROTOCOL：官方口径零样本名单模拟——E_T T-CLSC（ft60）给六靶官方目录排名 → SH，与任务盲先验 + SH、TS2Vec 论文数同表（零新 R'）

**会话定位**：T 检验在切片考场上把 E_T T-CLSC 排到了任务盲先验前面（regret@3 0.0207 vs 0.0313）。用户 2026-09-15 要求把它放进 TS2Vec 对照表。
六靶的官方口径目录（WP3 `results/mx4_deploy`：shared-10 + GGS + 提名 10 = 21 条，每条一条 25% / 50% / 100% 截点的官方全训轨迹，seed 0）都在缓存里，
所以只要给这 21 条一个新排名、走同一套 SH 模拟，就能得到 E_T 名单的冠军 acc——零新 R'。E_T 在官方 TRAIN 划分上用 GGS 探针 @60 算（6 条短训）。
分支 wp9-p1。产物 `results/tclsc_s1et_deploy/`。

## §0 已看过的读数

- WP3 官方口径（seed 0 / 3 种子）：prior3 与 SH6-3-3_prior 冠军 HW 0.419/0.431、UWave 0.906→0.931/0.918、PS 0.237→0.251/0.256、Ethanol 0.285→0.274/0.241、
  Cricket 0.986/0.986、HMD 0.378/0.450；目录 oracle（seed 0）HW 0.546（GGS）、UWave 0.931、PS 0.251、Ethanol 0.300、Cricket 1.000、HMD 0.486（提名）；
  SH10-4-3_prior 3 种子 HW 0.531 / Ethanol 0.303（6/6 追平）。TS2Vec 论文：0.515 / 0.906 / 0.233 / 0.308 / 0.972 / 0.338。
- T 检验切片考场：ft60 regret@3 0.0207、top-1 不优于先验。
- **没看过**：E_T 在任何官方 TRAIN 划分上的值；ft60 对官方目录的任何排名。

## §1 假设

H-D1：E_T 名单 + SH6-3-3 的冠军 acc（seed 0，同缓存）在六靶均值上不低于先验名单 + SH6-3-3（0.719 seed 0 六靶均值），且 HW 名单前 6 含 GGS（先验把它排到第 8）。
对立：切片考场的增益不转到官方口径（口径 ≠ 部署口径，WP14/15 教训）；或 E_T 在全长官方序列上的分布与切片训练时不同，ctx 出界。

## §2 材料与口径

- 目录 / 真值：`results/mx4_deploy/runs.jsonl` s1_catalog（六靶 × 21 条 × 截点 [25%, 50%, 100%]，seed 0）；孪生（WP6）不进名单（比较器不见 input_mask 字段）。
- 名单：ft60 = `results/tclsc_s1et/uea/models/<靶>/ft60_hard_s{0,1,2}.pt`（该靶 LODO 折模型，训练侧不含该靶）；ctx = E_T@60（官方 TRAIN，`load_uea_pair(scaler="train")`，
  GGS 探针 batch 128、290 iters 轨迹取 @60）⊕ φ12（`results/ucr_module/phi_official.json`）⊕ 预算 one-hot——预算档取 **290**（切片口径的最高档，对应"最终训练量"）；
  三种子多数票（`rank_local_models`）。对照名单：`candidates.json` 的 `prior_order`（先验，与 WP3 同）、`blind_diff_hard` 折模型（任务盲 T-CLSC）。
- SH：36 号 `successive_halving` + `sh_pick`，档位 SH6-3-3（主）与 SH10-4-3（副），预算 = 三截点；冠军 acc = 缓存 100% 截点 seed 0 的 acc；
  若冠军有 s2_finals 3 种子行则一并报。成本按 iters 逐级累加。
- 读数：六靶冠军 acc、六靶均值、追平 TS2Vec 论文 − 0.03 的靶数、名单前 6 是否含目录 oracle、regret vs 目录 oracle。

## §3 门（先于数据）

- **C0**：先验名单 + SH6-3-3 在本模拟里的六靶冠军（seed 0）= WP3 逐位（0.419 / 0.931 / 0.251 / 0.274 / 1.000 / 0.378）。
- **G-D1（部署价值，主）**：E_T 名单 + SH6-3-3 六靶均值 acc ≥ 先验名单 + SH6-3-3 六靶均值 + 0.005，且不低于先验的靶数 ≥ 4/6。
- **G-D2（追平）**：E_T 名单 + SH6-3-3 追平 TS2Vec（论文 − 0.03）的靶数 ≥ 5/6（先验 SH6-3-3 是 4/6）。
- 只报：SH10-4-3 两名单；任务盲 T-CLSC 名单；名单前 6 含 oracle 的靶数；regret；E_T 出界诊断（官方 E_T 标准化后 |z| 最大值）。

**读数码**：G-D1 ✓ G-D2 ✓ → **DEPLOY-ET-WIN**；G-D1 ✓ G-D2 ✗ → **DEPLOY-ET-BETTER**；G-D1 ✗ → **DEPLOY-ET-FLAT**（切片增益不转官方口径）。

## §4 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | C0 逐位 | 90% |
| P2 | G-D1 过 | 40% |
| P3 | G-D2 过 | 30% |
| P4 | HW 名单前 6 含 GGS（先验排第 8） | 45% |
| P5 | E_T 名单 SH6-3-3 ≥ 任务盲 T-CLSC 名单 SH6-3-3（六靶均值） | 60% |

## §5 执行

1. 本文件 + `tests/test_tclsc_s1et_deploy.py`（先红）→ commit；2. `scripts/71_tclsc_s1et_deploy.py`（--features / --simulate）→ 绿 → commit；3. 跑 → REPORT → commit → memory。

## §6 铁律对照

铁律 2 冠军 acc 带 @100% 官方 iters 与 n；铁律 3 零新 R'（探针 @60 不评 R'）；铁律 5；铁律 6 折模型不含该靶、E_T 只用官方 TRAIN；铁律 7 ctx 只是浮点向量。

## §7 执行台账（跑后填写）

- 2026-09-15 协议锁 b76fb1c（5 例先红）→ 实现 71 号 → 绿；`--features` 六靶官方 TRAIN GGS 探针 17–335 s（Ethanol T=1751 最慢）；`--simulate` CPU < 1 min。
- 读数 **DEPLOY-ET-FLAT**（C0 逐位；G-D1 ✗ 0.535 vs 0.542、不低于 2/6；G-D2 ✓ 5/6），预测 3/5。
  事后：比较器把提名配置排进前 6（5–6/6）而提名在官方口径系统性更差（外推高估，关键判负 b）；限域到 shared-10 + GGS 后 E_T 名单 + SH6-3-3 = 0.564（HW 0.542、Ethanol 0.297）vs 先验 0.542——须 D′ 预注册复验。详见 REPORT.md。

## §8 D′ 预注册变体（用户 2026-09-15 拍板"先限域跑一次 SH6 档"；先于新数据）

- 变体：比较器只排**信任域**（`prior_order` 的 11 条 = shared-10 + GGS，与先验同集合）；E_T ⊕ φ 标准化后裁到 ±4（A″ 护栏）；SH6-3-3；其余同 §2。
- 已看过：§7 事后限域（未裁剪）seed 0 六靶 = 0.542 / 0.931 / 0.251 / 0.297 / 0.986 / 0.378，均值 0.564。本变体的新信息 = 裁剪是否改名单、冠军 3 种子均值。
- 冠军补 3 种子：seed 0 之外缺 {1,2} 的 (靶, 冠军) 用 39 号 `--finals` 在 `results/tclsc_s1et_deploy/finals/` 补跑（本地 4060，官方口径 ts2vec_official）——与 WP3 冠军 3 种子同口径。
- 门：**G-D′1** E_T 限域名单 SH6-3-3 六靶均值（seed 0）≥ 先验 SH6-3-3 0.542 + 0.01；**G-D′2** 3 种子均值追平 TS2Vec（论文 − 0.03）≥ 5/6 且 HW 3 种子 ≥ 0.485。
  读数码：两过 **DEPLOY-ET-TR-WIN**；只过 G-D′1 **DEPLOY-ET-TR-BETTER**；否则 **DEPLOY-ET-TR-FLAT**。
- 预测：P6 G-D′1 过 70%；P7 G-D′2 过 45%；P8 裁剪不改任何靶的前 6 名单 60%；P9 HW 冠军 3 种子 ≥ 0.50 50%。
