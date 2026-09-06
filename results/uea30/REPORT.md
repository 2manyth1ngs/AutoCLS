# WP10-UEA30 报告：UEA 30 靶外 23 靶目标侧全量部署——**UEA30-WIN+TWIN**（判读带原样执行）

日期：2026-09-06/07 ｜ 协议：`PROTOCOL.md`（协议锁 a5b168f，判读带与逐靶预测先于数据写死）｜ 判定机：
`scripts/45_uea30_deploy.py --analyze / --table30` ｜ 数据：`runs.jsonl`（586 行：S1 目录 506 + S2 finals 80，**全部 RTX 4090
单 pod `mvqetmtt1t4azg`**，torch 2.8.0+cu128）｜ 表：`tables.md`（23 靶臂表）、`table30.md`（全 30 + HAR）。

## 0. 一句话结论

**零样本先验名单 + 目标侧逐次减半在 23 个从未见过的 UEA 靶上按官方口径全部部署成功且无一 INCOMPLETE**：主臂 SH10-4-3_prior
（掩码关，5.0 全跑当量）3 种子 **追平 TS2Vec 论文数 23/23**（含 6 LODO 靶 + HAR 为 30/30），**E 类 5 靶全部追平、4 靶显著超过**
（PEMS-SF +0.089 / ERing +0.073 / SCP1 +0.069 / NATOPS +0.022），A/B/D 类 15 靶 **14 落在预注册预测带内**（唯一出带的 FaceDetection
是**向上**出带：0.597 vs 带顶 0.547，TS2Vec 0.501）。副臂 SH20-3_both（两版都量）配对 Δ 均值 +0.001、20/23 在 −0.01 内 → **+TWIN**。
全 UEA 30（已跑 29）主臂均值 **0.733 vs TS2Vec 0.712**，可评集 12 靶 0.729 vs 0.695。8 条预测命中 5：三条未中全是"比预测更好 / 更便宜"
（P5 冠军来源比先验前 3 更分散、P7 墙钟 6.9 h 低于预估下界、P8 均值高出预测带 0.001）。

## 1. 执行台账（铁律 2：锚点带硬件标签）

- pod `mvqetmtt1t4azg`（RTX 4090 23 GB，120 vCPU，$0.74/h，镜像 runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404）：15:31 UTC 开 →
  setup（依赖 / Zenodo 直下 23 靶 2.3 GB md5 全对 / 单测 97 通过 / `--check` 23 靶 / GPU 冒烟 14 格）→ **主流程 15:42:58 开跑**：
  catalog 506 格 → 22:35:46（**6.88 h**）→ analyze → finals 80 格 → 23:34:42（0.98 h）→ analyze → table30 → 23:34:49 done；
  **23:45 UTC 停机，`confirm-stopped` 3/3 EXITED，uptime 29646 s = 8.24 h ≈ $6.1**。
- **收官重验全绿**：586 行 / 586 唯一键 / 0 error 行 / 0 空 acc / 全部 3 截点 / device 全 RTX 4090 / 23 靶各 22 格 S1 / finals 80
  （两臂同冠军的靶 2 格、不同冠军 4 格）/ `inflight.txt` 空 / 无 `oom_failures.jsonl`；本地 `runs.jsonl` md5 `ee7fedaa…` 与 pod 逐位同；
  本地 `--analyze` 复现：`picks.json` / `tables.md` 与 pod 逐字节相同，`analysis.json` 仅两处 `cost_seconds` 浮点末位（196.9 vs
  196.89999999999998）不同，读数与 verdict 相同。
- 主机侧：RunPod 于 18:45 UTC 发 "Possible Pod Issue"（网络抖动通知），登 pod 核对进程 / 进度 / GPU 均正常，未中断。
- **成本对锚**（22 格 catalog，预训练 + 评估秒；锚 = yaml `anchor_4090`）：合计实测 **28671 s = 7.96 h** vs 锚预估 26778 s = 7.44 h
  （比 **1.07**，落在 PROTOCOL §5 的 8.84 h 基准之内；墙钟 6.88 h 更低是因为 `pretrain_seconds` 含截点评估重叠）。逐靶偏差两头：
  LSST ×5.0（14 类 RBF-OvO 网格）、DuckDuckGeese ×3.8（1345 通道）、MotorImagery ×1.55、EigenWorms ×1.53 偏贵；PenDigits ×0.32、
  SpokenArabicDigits ×0.39、FaceDetection ×0.82 偏便宜（120 vCPU 的 loky 网格比标定 pod 快）。finals 80 格 4016 s。
- 部署口径当量：主臂 SH10-4-3 = **5.0** 全跑（10·25% + 4·25% + 3·50%），副臂 SH20-3 = **11.5**（20·50% + 3·50%），对照 SH6-3-3 3.75。

## 2. 门（PROTOCOL §3；主门用 3 种子均值，23/23 靶完整）

| 门 | 判据 | 读数 | 判 |
|---|---|---|:--:|
| U1 | E 类 5 靶主臂 ≥ TS2Vec − 0.03 于 ≥ 3/5 | **5/5**（ERing 0.947 / NATOPS 0.950 / PEMS-SF 0.771 / SCP1 0.881 / EigenWorms 0.822） | ✓ |
| U2 | A/B/D 15 靶主臂落在预测带 ≥ 12/15 | **14/15**（A 7/7、B 3/3、D 4/5；FaceDetection 0.597 **高于**带顶 0.547） | ✓ |
| U3 | Δ = 副 − 主：均值 ≥ 0 且 Δ ≥ −0.01 于 ≥ 60% | 均值 **+0.001**，20/23 在容差内（SCP2 −0.056、NATOPS −0.013、EigenWorms −0.010 出容差） | ✓ |
| C | 只报 | AF 0.200 ± 0.094（TS2Vec 0.200）、SWJ 0.444 ± 0.083（0.467）、DDG 0.640 ± 0.016（0.680，1.96·sd 0.13） | — |

**读数码：UEA30-WIN+TWIN**。不完整靶 0；error 行 0。

## 3. 逐靶（3 种子 mean ± std；完整表见 `table30.md`，臂表见 `tables.md`）

**E 类（可评，主表）**

| 靶 | TS2Vec | 主臂 SH10-4-3 | 副臂 SH20-3 | 目录 oracle22 | 主臂冠军（先验位次） | 预测区间 |
|---|---:|---:|---:|---:|---|---|
| PEMS-SF | 0.682 | **0.771 ± 0.076** | 0.767 ± 0.028† | 0.803 | 65627517（10） | 0.70–0.78 ✓ |
| SelfRegulationSCP1 | 0.812 | **0.881 ± 0.020** | 0.881 ± 0.020 | 0.901 | 9092e36e（1） | 0.82–0.87（略高） |
| NATOPS | 0.928 | **0.950 ± 0.016** | 0.937 ± 0.007 | 0.956 | a1f67db4（2） | 0.93–0.96 ✓ |
| ERing | 0.874 | **0.947 ± 0.005** | 0.948 ± 0.006† | 0.952 | 9092e36e（1） | 0.93–0.95 ✓ |
| EigenWorms | 0.847 | 0.822 ± 0.048 | 0.812 ± 0.019† | 0.863 | a1f67db4（2） | 0.85–0.90（低；追平线 0.817 刚过） |

**A/B/D 类**：全部与 TS2Vec 同带或更好。值得单列的三条：**FaceDetection 0.597**（D 类"chance 型"预测错——官方口径下有可学信号，
超 TS2Vec 0.501 近 0.10，也超上一会话记的表内最高 0.581）；**Heartbeat 0.741**（vs 0.683，带内）；**MotorImagery 主 0.537 / 副 0.593†**
（孪生 +0.057，是掩码开最大的正增量）。A 类 7 靶主臂 0.969–0.994 与 TS2Vec 差 |Δ| ≤ 0.009，全部带内。

**孪生（掩码开）**：副臂在 14/23 靶选中孪生；目录天花板 oracle_on > oracle_off 于 10/23；配对 Δ 正向最大 MotorImagery +0.057、
FingerMovements +0.030、BasicMotions +0.017，负向最大 SCP2 −0.056、NATOPS −0.013。§4 预测"选孪生"的 8 靶命中 6（SCP1 与 StandWalkJump
两版同分并列）；预测"不选"却选中孪生的 8 靶集中在 A 类贴顶靶（差值 ≤ 0.004，噪声内）与 FaceDetection / AF。

**冠军来源（P5 相关）**：先验第 1（9092e36e）7 靶、第 2 两靶、第 3–5 五靶、**第 7–10 六靶**（FingerMovements 9 / Heartbeat 7 /
PEMS-SF 10 / LSST 7 / MotorImagery 10 / PenDigits 8）、GGS 2 靶（AF、DDG，C 类）。**名单 10 的价值再次坐实**：主臂 ≥ SH6-3-3 于 21/23，
主臂追平 23/23 vs SH6-3-3 22/23；主臂 ≥ prior1 22/23、≥ prior3 21/23、≥ random1 23/23；对目录 oracle22 的 regret 中位 **0.001**、
最大 0.050。25%→100% 截点排名 Spearman 中位 0.563（min 0.01 BasicMotions，n_test 40 大量同分）。3 种子均值与 seed 0 的 |Δ| 中位
0.009、最大 0.133（PEMS-SF，n_test 173、std 0.076）。

## 4. 预测对账（PROTOCOL §4，先验概率括号内）

| # | 预测 | 读数 | 判 |
|---|---|---|:--:|
| P1 | U1 过（55%） | 5/5 | ✓ |
| P2 | U2 过（60%） | 14/15 | ✓ |
| P3 | U3 过（65%） | +0.001，20/23 | ✓ |
| P4 | 孪生冠军 ≥ 6/23 且预测"是"8 靶命中 ≥ 5（50%） | 14/23；6/8 | ✓ |
| P5 | 主臂冠军 ∈ {GGS} ∪ 先验前 3 于 ≥ 14/23（60%） | **12/23** | ✗（冠军更分散：6 靶落在先验第 7–10） |
| P6 | 25%→100% Spearman 中位 ≥ 0.5（60%，描述性） | 0.563 | ✓ |
| P7 | catalog 墙钟 ∈ [8.8, 14.3] h（60%） | **6.88 h**（成本秒 7.96 h / 锚 7.44 h） | ✗（更快：120 vCPU 评估） |
| P8 | 全 29 主臂均值 ∈ 0.712 ± 0.02（60%） | **0.733**（+0.021） | ✗（高出带 0.001） |

逐靶带：A 7/7、B 3/3、D 4/5（FaceDetection 向上出带）；E 类预测区间 3/5 落入（SCP1 略高、EigenWorms 略低）。

## 5. 读法与 limitation

1. **这是"可评性分类"的预注册检验，不是 30 靶排行榜**：策略选择在 E 类移动了精度（4/5 超 TS2Vec 0.02–0.09），在 A/B/C 类与 TS2Vec
   同带（差 |Δ| ≤ 0.02，噪声 sd 0.002–0.13）。全 30 均值 0.733 vs 0.712 的 +0.021 里，E 类 5 靶贡献 +0.228/29 = +0.008，D 类
   （FaceDetection +0.096、Heartbeat +0.058、MotorImagery +0.027、SCP2 +0.015、FingerMovements +0.033）贡献 +0.229/29 = +0.008——
   **D 类"chance 型"的预测在官方口径下偏保守**（切片口径 @290 判"学不动"的 FaceDetection / Heartbeat 在全量 600 iters 下有信号）。
2. **与 AutoTCL / InfoTS 的比较只能描述**：本轮只转录了 TS2Vec Table 10 的六列；上一会话 §5.1 记录的"表内最高"（AutoTCL Table 12）
   在 E 类：PEMS-SF 0.838 > 0.771、SCP1 0.891 > 0.881、EigenWorms 0.901 > 0.822、ERing 0.953 ≈ 0.947、NATOPS 0.944 < 0.950；
   FaceDetection 0.581 < 0.597。全 30 均值不会居首（与 §5.1 预测一致）。逐方法数字待用户截图补录进 `literature_extra`。
3. **EigenWorms 训练侧分段是本仓库对官方 `max_train_length` 的近似**（补零而非补 NaN，评估全长）：0.822 ± 0.048 低于预测区间
   下沿 0.85，目录 oracle22 0.863 > 论文 0.847——差在名单 / 噪声（std 0.048）而非天花板。
4. **成本锚**：4090 的 22 格锚在 120 vCPU pod 上总体偏差 7%，但逐靶两头各 ×3–5（评估重的靶更快、宽通道 / 多类靶更慢）——
   估时必须按 (L, C, n_train, 类数) 四维，不能只按 L。
5. **孪生的净价值仍在噪声内**（Δ +0.001，14/23 选中但只有 MotorImagery / FingerMovements 的 +0.03–0.06 超出 1.96·sd），
   与 WP6 六靶结论一致；两版都量成本 11.5 vs 5.0 当量。
6. InsectWingbeat 未跑（用户拍板；L≤22 低于 encoder 适用边界、评估占 28% 成本）；全 30 表按 29/30 报。

## 6. 交接

- 主表建议 = 可评集 12（E 5 + LODO 6 + HAR）：主臂追平 12/12、超 TS2Vec 11/12（Ethanol 0.303 vs 0.308 追平未超）；附录 = 全 30 按类。
- 论文措辞（与 `paper-positioning` 一致）："on 23 held-out UEA datasets the learned task-agnostic portfolio with target-side SH matches
  or exceeds TS2Vec on 23/23 at 5 full-pretraining equivalents; gains concentrate on the 5 datasets our pre-registered evaluability
  screen predicted to have headroom; on ceiling / short-sequence / tiny-test-set datasets every method is within noise."
- 待办：① AutoTCL / InfoTS 逐靶数字补录（`configs/uea30.yaml: literature_extra`，零 GPU）；② D 类"chance 型"预测偏保守——切片口径
  的可评性判定在官方口径下须重标（FaceDetection / Heartbeat），写进 limitation；③ 分支 `wp10-uea30` 合入 `mx4` 待用户拍板。
