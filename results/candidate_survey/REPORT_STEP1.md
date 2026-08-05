# 换源池调研 · 第1步 —— 候选统计与初筛

日期：2026-08-05 ｜ 本机 RTX 4060 ｜ 协议锁：`PROTOCOL.md`（commit 3686719，判据/阈值/
预测先于数据）｜ 数据：`data/candidates/*.npz` + `*.json`（**全部实际加载确认，非论文数字**）
｜ 复现：`python scripts/prepare_candidates.py --table --plan`

> 本报告只覆盖**第1步（统计初筛）与第2步成本估算**。基线（第2步）与"训练能拉动"
> （第3步）尚未测量，本报告不对任何候选的可用性下结论。

---

## 0. TL;DR

1. **UEA 档案在"还没测基线"这一步就已经出局**：28 个（除目标池 2 个）里只有 **4 个**
   过初筛——**21 个卡 N<1000**，另 3 个 N 够但 L<50。此前那 16 个被测过基线的数据集中，
   **12 个本来就不该进源池**。选 UEA 的问题不止是贴顶，**档案规模本身撑不起 R' 的
   噪声要求**。
2. **新候选把每任务 n_eval 拉到一律 307**：UEA 那 137 个任务实测 n_eval **24–309、
   中位 122，其中 36%（50 个）不足 100**（Cricket 低到 24 → R' 噪声 0.102 ≈ **2δ**）。
   候选的每个任务都吃满 1024 采样上限 → 噪声 **0.029 ≈ 0.6δ**（UEA 中位 0.045、最差 0.102）。
   这是 UEA 那 7/13 个 δ 兑现不了标定目标的直接原因，换池首先修好的是这个。
3. **27 个新候选实际下载并加载成功，16 个过初筛入源、11 个降 reserve**。主力是
   N=5k–30k、L=128、C=3–54、K=6–33 的**多受试者**传感器数据（4–51 人）——与 UCI-HAR
   （N=10299/L=128/C=9/K=6）同型，与 UEA 失败的那批（几百样本、L 几十）相反。
   源池任务总数 **256**，远超 floor=100。
4. **成本不是约束**：第2步基线 **5.5 min**；第3步（训练能拉动）**仅源角色 7.2 h / 全部
   11.5 h**，且这是双重上界（reserve 可不跑、贴顶的直接出局）。真正的风险是
   PROTOCOL §6 预测 5——**至少 3 个候选同时过 T1+T2+T3**。若它错，换数据集这条路也
   不成立，须重议项目范围。

---

## 1. 候选统计表

完整表见**附录 A**。口径：

- **全部经 `prepare_candidates.py` 实际下载 + 解析 + 开窗 + 落盘后统计**，不引用文献数字
- 流式传感器数据统一开窗 **128 步 @原生采样率 / stride 64**（PROTOCOL §1）
- 已切好样本的档案（UniMiB 151 / UCR）保留原生长度
- N 是**剔稀类（<64）+ 分层降采样（上限 30000）之后**的落盘值；原始 N 记在 meta 的
  `n_samples_raw`

### 1.1 读表的三个陷阱（实测发现，不看会误判）

1. **PAMAP2 的不平衡比 19.95 是丢窗伪像**。含 NaN 的窗被丢弃（协议 §1，不插值），
   而 NaN 与活动强相关——无线 IMU 在剧烈活动时掉帧多：walking 丢 **80.9%**、
   rope_jumping 79.7%、running 72.5%，而 lying 只丢 3.5%。**按纯段口径的不平衡比只有
   4.88**（meta `imbalance_ratio_pure`）。单通道最大 NaN 步占比仅 0.46%，但 128 步的窗
   把它放大成 33% 的丢窗率。
2. **null/transient 段吃掉的比例很大**：MHEALTH 有 **71.8%** 的时间步是 null
   （每人 ~40 min 记录里只有 ~12 min 在做标注动作），PAMAP2 的 transient 占 32.4%。
   这解释了为什么 MHEALTH 只有 5223 条——不是数据集小，是有标注的部分小。
3. **UCR 的三个语义陷阱**（都会影响对比学习的增强语义，入池前必须处理）：
   - `UWaveGestureLibraryAll` 的 L=945 是 **X/Y/Z 三轴首尾拼接**（315×3），不是单条连续
     信号——`crop` / `jitter` / `resize` 会跨轴边界作用。且它与 UEA 的
     `UWaveGestureLibrary` 同源，不算独立候选。
   - `InsectWingbeatSound` 的 L=256 是**频谱**不是时域波形，时间轴增强无物理意义。
   - `ECG5000` 全部 4976 条来自**同一个病人**的 20 小时记录——零受试者多样性，
     与"跨受试者变异"这个我们指望的难度来源正相反。

---

## 2. 初筛结果

判据 PROTOCOL §2（H1 L≥50 / H2 N≥1000 / H3 K≥5，H4 目标池排除），阈值先于数据写死。

### 2.1 新候选

27 个新候选全部实际下载并加载成功（HHAR 例外，见 §4），**16 个入源 / 11 个降 reserve /
0 个被 L 或 N 淘汰**——这与 UEA 的 4/28 形成直接对照。完整表见附录 A。

**降 reserve 的 11 个全部只因 K<5**，且几乎全是 UCR 单变量档案（9/11）：
UCR 以二分类为主，这是档案的结构性属性，不是加载问题。K=2–4 降 reserve 而非淘汰，
是因为它们每条基线只要 0.2 s，测了能独立回答"贴顶是不是普遍现象"；但它们撑不起
任务多样性——**切片试算实测 Daphnet / FordA / FordB / Wafer / Yoga / FreezerRegularTrain
各只切得出 1 个任务**（K=2 → 类别组合数 = 1），这就是 H3 的量化理由。

### 2.2 UEA 存量：24/28 出局

| 出局原因 | 个数 | 数据集（括号内为实测值） |
|---|---:|---|
| **N < 1000** | **21** | AWR(575) / AtrialFibrillation(30) / BasicMotions(80) / Cricket(180) / DuckDuckGeese(100) / ERing(300) / EigenWorms(259) / Ethanol(524) / FingerMovements(416) / HMD(234) / Heartbeat(409) / JapaneseVowels(640) / Libras(360) / MotorImagery(378) / NATOPS(360) / PEMS-SF(440) / RacketSports(303) / SCP1(561) / SCP2(380) / StandWalkJump(27) / UWaveGestureLibrary(440) |
| L < 50（**且 N 够**，即只卡长度） | 3 | InsectWingbeat(L22, N=50000) / LSST(L36, N=4925) / PenDigits(L8, N=10992) |
| （L<50 但 N 也不够，已计入上行） | 3 | JapaneseVowels(L29) / Libras(L45) / RacketSports(L30) |
| **过初筛** | **4** | CharacterTrajectories(2858) / Handwriting(1000) / PhonemeSpectra(6668) / SpokenArabicDigits(8798) |

出局合计 24 = 21 + 3。**主因压倒性是样本量**：UEA 里 N≥1000 的一共只有 7 个
（含目标池 FaceDetection），其中 3 个还卡在 L<50。

**这四个的后续命运是已知的**（baseline_probe 已测）：CharacterTrajectories 0.981 与
SpokenArabicDigits 0.912 判 **ceiling**；PhonemeSpectra 2↑/7↓ 判 **inert**；
只有 Handwriting 判 **usable**。即：**UEA 档案里同时满足"规模够"与"空间被用上"的，
从头到尾只有 Handwriting 一个**。

### 2.3 一处必须点明的紧张

`UWaveGestureLibrary`（N=440）是 baseline_probe 判定的**第二个 usable**，但它过不了
本轮的 H2。这不是判据打架——H2 管的是 R' 噪声下限（它的任务 n_eval 只有 ~90，
噪声 0.053 ≈ 1.06δ），而 usable 管的是信号大小。**它的 usable 靠信号勉强压过噪声，
不是噪声低**。同理 Handwriting（N=1000）恰在 H2 阈值上，靠的是 best_gain +0.397 远大于
噪声 0.044。本轮的立场是：**噪声下限与信号大小两条都要过**，不接受"信号大到能盖住
噪声"的侥幸。

---

## 3. 为什么这批候选与 UEA 不同（三个可量化的差别）

| 维度 | UEA 那 16 个 | 本轮过初筛的候选 | 为什么重要 |
|---|---|---|---|
| **每任务 n_eval** | **24–309，中位 122，36% 不足 100**（Cricket 24–48、HMD 32–69、AWR 42–175） | **一律 307**（全部吃满 1024 采样上限） | R' 噪声 0.5/√n_eval：中位 0.045、最差 **0.102（=2δ）** → 一律 **0.029（=0.6δ）**。UEA 有 7/13 个 δ 兑现不了标定目标，根因在此 |
| **样本来源** | 单次采集、几百条 | 多受试者（4–51 人）连续记录开窗，1.6 万–3 万条 | 类内方差来自真实受试者差异，不是同一批录音的切片 |
| **形态** | L 8–17984 极端分散、K 2–39 | L=96–151（少数 500）、K 5–18 | 与唯一正样本 UCI-HAR 同型 |

**但要诚实**：以上没有一条能保证基线不贴顶。协议探针的教训正是"机制推断会低估"
——我预测线性探针能压低天花板，实测是零效应甚至反向。**第2步之前不下任何结论。**

---

## 3.1 数据完整性体检（花算力之前先做）

对每份落盘 npz 实测：**零 NaN、零常值通道、按通道 z-score 后 |mean| < 2e-9 且
|std−1| < 8e-9、标签全部为连续 0..k-1**——全部通过，无一例外。

切片试算同样通过：**每个候选都切满规划任务数，每任务都吃满 1024 采样上限
→ n_eval 一律 307**（UEA 那边中位 122、最低 24）。

---

## 3.2 领域覆盖（对目标池的初步含义，**不是建议改动**）

过初筛的 16 个源角色候选按域分布：

| 域 | 候选 | 任务数 |
|---|---|---:|
| HAR（传感器） | USC-HAD, WISDM2019, ChestAccel, MotionSense, PAMAP2, WISDM, UniMiB-SHAR, REALDISP, MHEALTH | 144 |
| ECG | PTB-XL, NonInvasiveFetalECGThorax1 | 32 |
| **EEG** | **SleepEDF** | 16 |
| Device / Image / Motion / Sensor（UCR 单变量） | ElectricDevices, MixedShapesRegularTrain, UWaveGestureLibraryAll, InsectWingbeatSound | 64 |

**与目标池的关系**：当前目标池 UCI-HAR + Epilepsy + FaceDetection。

- UCI-HAR（HAR）与 Epilepsy（HAR 域的 UEA 条目）在新源池里**有充足同域支撑**（144 个任务）。
- FaceDetection 是 **EEG/MEG**。新源池的 EEG 支撑只有 SleepEDF 一个（16 任务），
  且它是**睡眠分期**（30 s epoch、单通道 Fpz-Cz），与 FaceDetection 的 62 步 ×144 通道
  MEG 差异极大。**这个域距离问题依然存在，且比 UEA 时期更突出**（旧源池有 4 个
  EEG/MEG 源）。
- **本报告不改目标池**（PROTOCOL §5 第 4 条）。第3步判完 usable 后再给建议——
  若 SleepEDF 判 inert（预测 4 就是这么猜的），FaceDetection 将完全没有同域源支撑，
  届时要么换目标池，要么把"跨域 zero-shot"明确写成论文的一个主张而非缺陷。

**一个可低成本改善的点**：SleepEDF 现在的 `n_subjects=5`——取的是"前 10 份记录"，
而 sleep-cassette 每人 2 晚，故只覆盖 5 人。改成每人取 1 晚（SC4001/SC4011/SC4021/…）
可得 **10 名受试者，下载量不变**。受试者多样性正是我们指望的难度来源，建议第2步前改掉。

---

## 4. 未能获取的候选（如实报告）

| 数据集 | 状态 | 原因 |
|---|---|---|
| **HHAR**（Heterogeneity Activity Recognition，UCI 344） | **未获取** | 服务器 chunked 传输无 Content-Length、忽略 Range 请求（无断点续传）；实测吞吐 84–92 KB/s，主下载在 271 s / 44.4 MB 处被服务器掐断。全量 ~1.1 GB 推算需 ~3.3 h 且不能断线，超 15 min 预算。adapter 已写好并通过逻辑冒烟（合成数据只在 scratchpad），网络条件允许时一条命令即可补齐 |

### 4.0 超预算但仍完成的三个（如实标注）

下载预算是"单个超 15 min 或超 1.5 GB 就停下报告"。以下三个超了但已跑完，
未中途丢弃已下载的进度：

| 数据集 | 实际 | 预算 |
|---|---|---|
| REALDISP | **2.67 GB zip / 43.5 min**（UCI 页面声称 ~200 MB，**差 13 倍**） | 1.5 GB / 15 min |
| PAMAP2 | 656 MB / **28 min**（4 个并行任务抢带宽，中途降到 ~200 KB/s） | 15 min |
| SleepEDF | 504 MB / **31 min**（PhysioNet 单连接限速 ~120 KB/s，用 4 路并发才拉到 270–430 KB/s） | 20 min |

### 4.1 两个必须记下来的数据源坑

1. **aeon 1.3.0 的 `tsc_zenodo` 映射有错条目**：`NonInvasiveFetalECGThorax1 → 11197817`，
   而该 Zenodo record 实际是 **MoteStrain**（正确的是 11197821）。本次没有静默串档案
   ——URL 带 `<name>_TRAIN.ts` 文件名，record 不对就是 404，坑是这么暴露的。但已额外
   加一道 `@problemName` 头部核对（14/14 通过），因为"下错档案还能跑通"是最坏的失败模式。
2. **aeon 的默认下载地址已挂 reCAPTCHA（401）**：`load_classification(name, extract_path=...)`
   一句调用在当前网络下跑不通，必须走 Zenodo 镜像（沿用 `results/uea_survey/scripts` 的
   既有做法）。`.ts` 纯 Python 解析比下载慢得多（ElectricDevices 227 s、FordA 361 s）。

---

## 5. 成本估算（第2步 / 第3步）

实测锚点（PROTOCOL §7）：基线 `iters=0` **0.22 s/条**（685 条 / 2.5 min，摊薄值）、
训练 `iters=290` **65 s/条**（60 条 / 65 min）。

完整表见**附录 B**。要点：

- **第2步（基线）= 5.5 min**，27 个候选 × 各自任务数 × 5 种子。几乎免费，
  **reserve 一起测**：它们能独立回答"贴顶是不是普遍现象"这个问题。
  （通路冒烟实测单条冷启动 2.1 s，含一次性 CUDA/模型初始化；0.22 s 是摊薄率。）
- **第3步（训练能拉动）：仅源角色 7.2 h / 全部 11.5 h**，双重上界：
  ① reserve（K<5）无论如何进不了源池，可不跑；② 只对过 G1（基线 ≤0.90）的候选跑。
  若基线阶段淘汰掉一半，第3步再减半。
- 长序列修正：iters=290 的耗时随 L 近似线性。成本大头是 4 个长序列候选——
  **SleepEDF(L=3000) 单个就占 1.1 h**，StarLightCurves / MixedShapes(L=1024)、
  PTB-XL(L=1000) 各 0.55 h。L≤300 一律按 ×1.0（保守：UEA 那 65 s 锚点均 L≈190，
  而本批 HAR 主力是 128，实际应更快）。

**建议的执行顺序**：先跑第2步全量（5.5 min）→ 看有多少过 G1 → 再决定第3步跑谁。
若长序列候选在第2步就贴顶出局，第3步能压到 3–4 h。

算力不是本轮的约束，与阶段1 造 label（8.1 h 单卡、必须上云）完全不同——**可以本机
一晚跑完，不需要上云**。

---

## 6. 尚未回答的问题（第2/3步才有答案）

1. 这批候选的随机初始化基线是不是也贴顶？（PROTOCOL §6 预测 2：流式 HAR 落 0.70–0.90）
2. UCR 单变量是不是整体更贴顶？（预测 1：≥60% 基线 >0.90）
3. 低基线的是不是"训不动"？（预测 4：SleepEDF 判 inert）
4. **至少 3 个候选同时过 T1+T2+T3 吗？**（预测 5，本调研的核心风险）

---

## 附录 A · 候选统计与初筛（`--table` 原始输出）

```
==================================================================================================================================
新下载候选（实际加载确认，非论文数字）
==================================================================================================================================
数据集                        域               N     L    C   K     不平衡     猜测     fs/受试  初筛  未过条件
ElectricDevices            Device      16637    96    1   7    3.41  0.143    None/?  ✔源  
ECG5000                    ECG          4976   140    1   4   30.41  0.250    None/1  △备  K=4 < 5（类别子集撑不起任务多样性）→ reserve
PTB-XL                     ECG          4000  1000   12   5    1.62  0.200  100/3833  ✔源  
NonInvasiveFetalECGThorax1 ECG          3765   750    1  42    1.31  0.024    None/?  ✔源  
SleepEDF                   EEG         10589  3000    1   5    4.79  0.200     100/5  ✔源  
USC-HAD                    HAR         30001   128    6  12    3.73  0.083    100/14  ✔源  
WISDM2019                  HAR         29998   128    6  18    1.05  0.056     20/51  ✔源  
ChestAccel                 HAR         29848   128    3   7   13.12  0.143     52/15  ✔源  
MotionSense                HAR         21539   128   12   6    2.70  0.167     50/24  ✔源  
PAMAP2                     HAR         20195   128   27  12   19.95  0.083     100/9  ✔源  
Daphnet                    HAR         17090   128    9   2   11.28  0.500     64/10  △备  K=2 < 5（类别子集撑不起任务多样性）→ reserve
WISDM                      HAR         16057   128    3   6    8.89  0.167     20/36  ✔源  
UniMiB-SHAR                HAR         11771   151    3  17   12.97  0.059     50/30  ✔源  
REALDISP                   HAR         10319   128   54  33   10.73  0.030     50/17  ✔源  
Opportunity                HAR          6332   128   45   4    9.65  0.250      30/4  △备  K=4 < 5（类别子集撑不起任务多样性）→ reserve
MHEALTH                    HAR          5223   128   23  12    3.13  0.083     50/10  ✔源  
Yoga                       Image        3300   426    1   2    1.16  0.500    None/?  △备  K=2 < 5（类别子集撑不起任务多样性）→ reserve
MixedShapesRegularTrain    Image        2925  1024    1   5    1.82  0.200    None/?  ✔源  
UWaveGestureLibraryAll     Motion       4478   945    1   8    1.00  0.125     100/?  ✔源  
StarLightCurves            Sensor       9236  1024    1   3    4.01  0.333    None/?  △备  K=3 < 5（类别子集撑不起任务多样性）→ reserve
Wafer                      Sensor       7164   152    1   2    8.40  0.500    None/?  △备  K=2 < 5（类别子集撑不起任务多样性）→ reserve
FordA                      Sensor       4921   500    1   2    1.06  0.500    None/?  △备  K=2 < 5（类别子集撑不起任务多样性）→ reserve
FordB                      Sensor       4446   500    1   2    1.03  0.500    None/?  △备  K=2 < 5（类别子集撑不起任务多样性）→ reserve
ChlorineConcentration      Sensor       4307   166    1   3    2.31  0.333    None/?  △备  K=3 < 5（类别子集撑不起任务多样性）→ reserve
FreezerRegularTrain        Sensor       3000   301    1   2    1.00  0.500    None/?  △备  K=2 < 5（类别子集撑不起任务多样性）→ reserve
InsectWingbeatSound        Sensor       2200   256    1  11    1.00  0.091    None/?  ✔源  
TwoPatterns                Simulated    5000   128    1   4    1.09  0.250    None/?  △备  K=4 < 5（类别子集撑不起任务多样性）→ reserve

新候选 27 个：入源 16，其余见「未过条件」列

==================================================================================================================================
UEA 存量对照（results/uea_survey 的实测统计，同一把尺子）
==================================================================================================================================
数据集                        域               N     L    C   K     不平衡     猜测     fs/受试  初筛  未过条件
ArticularyWordRecognition  Motion        575   144    9  25    1.00  0.040        已测  ✘   N=575 < 1000（n_eval 不足 100，R' 噪声超 δ）
AtrialFibrillation         ECG            30   640    2   3    1.00  0.333        未测  ✘   N=30 < 1000（n_eval 不足 100，R' 噪声超 δ）；K=3 < 5（类别子集撑不起任务多样性）→ reserve
BasicMotions               HAR            80   100    6   4    1.00  0.250        未测  ✘   N=80 < 1000（n_eval 不足 100，R' 噪声超 δ）；K=4 < 5（类别子集撑不起任务多样性）→ reserve
CharacterTrajectories      Motion       2858   182    3  20    1.56  0.050        已测  ✔源  
Cricket                    HAR           180  1197    6  12    1.00  0.083        已测  ✘   N=180 < 1000（n_eval 不足 100，R' 噪声超 δ）
DuckDuckGeese              Audio         100   270 1345   5    1.00  0.200        未测  ✘   N=100 < 1000（n_eval 不足 100，R' 噪声超 δ）
ERing                      HAR           300    65    4   6    1.00  0.167        已测  ✘   N=300 < 1000（n_eval 不足 100，R' 噪声超 δ）
EigenWorms                 Motion        259 17984    6   5    4.40  0.200        未测  ✘   N=259 < 1000（n_eval 不足 100，R' 噪声超 δ）
EthanolConcentration       Other(Spectro)    524  1751    3   4    1.01  0.250        已测  ✘   N=524 < 1000（n_eval 不足 100，R' 噪声超 δ）；K=4 < 5（类别子集撑不起任务多样性）→ reserve
FingerMovements            EEG/MEG       416    50   28   2    1.00  0.500        已测  ✘   N=416 < 1000（n_eval 不足 100，R' 噪声超 δ）；K=2 < 5（类别子集撑不起任务多样性）→ reserve
HandMovementDirection      EEG/MEG       234   400   10   4    1.30  0.250        已测  ✘   N=234 < 1000（n_eval 不足 100，R' 噪声超 δ）；K=4 < 5（类别子集撑不起任务多样性）→ reserve
Handwriting                HAR          1000   152    3  26    1.96  0.038        已测  ✔源  
Heartbeat                  Audio         409   405   61   2    2.59  0.500        已测  ✘   N=409 < 1000（n_eval 不足 100，R' 噪声超 δ）；K=2 < 5（类别子集撑不起任务多样性）→ reserve
InsectWingbeat             Audio       50000    22  200  10    1.00  0.100        未测  ✘   L=22 < 50（encoder 适用边界）
JapaneseVowels             Audio         640    29   12   9    2.19  0.111        未测  ✘   L=29 < 50（encoder 适用边界）；N=640 < 1000（n_eval 不足 100，R' 噪声超 δ）
LSST                       Other(Astro)   4925    36    6  14  111.00  0.071        未测  ✘   L=36 < 50（encoder 适用边界）
Libras                     Motion        360    45    2  15    1.00  0.067        未测  ✘   L=45 < 50（encoder 适用边界）；N=360 < 1000（n_eval 不足 100，R' 噪声超 δ）
MotorImagery               EEG/MEG       378  3000   64   2    1.00  0.500        已测  ✘   N=378 < 1000（n_eval 不足 100，R' 噪声超 δ）；K=2 < 5（类别子集撑不起任务多样性）→ reserve
NATOPS                     HAR           360    51   24   6    1.00  0.167        已测  ✘   N=360 < 1000（n_eval 不足 100，R' 噪声超 δ）
PEMS-SF                    Other(Traffic)    440   144  963   7    1.14  0.143        未测  ✘   N=440 < 1000（n_eval 不足 100，R' 噪声超 δ）
PenDigits                  Motion      10992     8    2  10    1.08  0.100        未测  ✘   L=8 < 50（encoder 适用边界）
PhonemeSpectra             Audio        6668   217   11  39    1.01  0.026        已测  ✔源  
RacketSports               HAR           303    30    6   4    1.26  0.250        未测  ✘   L=30 < 50（encoder 适用边界）；N=303 < 1000（n_eval 不足 100，R' 噪声超 δ）；K=4 < 5（类别子集撑不起任务多样性）→ reserve
SelfRegulationSCP1         EEG/MEG       561   896    6   2    1.01  0.500        已测  ✘   N=561 < 1000（n_eval 不足 100，R' 噪声超 δ）；K=2 < 5（类别子集撑不起任务多样性）→ reserve
SelfRegulationSCP2         EEG/MEG       380  1152    7   2    1.00  0.500        已测  ✘   N=380 < 1000（n_eval 不足 100，R' 噪声超 δ）；K=2 < 5（类别子集撑不起任务多样性）→ reserve
SpokenArabicDigits         Audio        8798    93   13  10    1.00  0.100        已测  ✔源  
StandWalkJump              ECG            27  2500    4   3    1.00  0.333        未测  ✘   N=27 < 1000（n_eval 不足 100，R' 噪声超 δ）；K=3 < 5（类别子集撑不起任务多样性）→ reserve
UWaveGestureLibrary        HAR           440   315    3   8    1.00  0.125        已测  ✘   N=440 < 1000（n_eval 不足 100，R' 噪声超 δ）

UEA 28 个（除目标池 2 个）：过初筛 4 个
```

## 附录 B · 切片规划与成本（`--plan` 原始输出）

```
数据集                     角色           类别档   任务    任务N n_eval     R噪声     L    步骤2 s    步骤3 s  n_eval
USC-HAD                 source    [5,10]   16   1024    307   0.029   128       18     1300  ✔
WISDM2019               source    [7,14]   16   1024    307   0.029   128       18     1300  ✔
ChestAccel              source     [3,6]   16   1024    307   0.029   128       18     1300  ✔
MotionSense             source     [2,5]   16   1024    307   0.029   128       18     1300  ✔
PAMAP2                  source    [5,10]   16   1024    307   0.029   128       18     1300  ✔
Daphnet                 reserve    [2,2]    1   1024    307   0.029   128        1     1300  ✔
ElectricDevices         source     [3,6]   16   1024    307   0.029    96       18     1300  ✔
WISDM                   source     [2,5]   16   1024    307   0.029   128       18     1300  ✔
UniMiB-SHAR             source    [7,14]   16   1024    307   0.029   151       18     1300  ✔
SleepEDF                source     [2,4]   16   1024    307   0.029  3000       18     3900  ✔
REALDISP                source   [13,26]   16   1024    307   0.029   128       18     1300  ✔
StarLightCurves         reserve    [2,3]    4   1024    307   0.029  1024        4     1997  ✔
Wafer                   reserve    [2,2]    1   1024    307   0.029   152        1     1300  ✔
Opportunity             reserve    [2,3]   10   1024    307   0.029   128       11     1300  ✔
MHEALTH                 source    [5,10]   16   1024    307   0.029   128       18     1300  ✔
TwoPatterns             reserve    [2,3]   10   1024    307   0.029   128       11     1300  ✔
ECG5000                 reserve    [2,3]   10   1024    307   0.029   140       11     1300  ✔
FordA                   reserve    [2,2]    1   1024    307   0.029   500        1     1493  ✔
UWaveGestureLibraryAll  source     [3,6]   16   1024    307   0.029   945       18     1921  ✔
FordB                   reserve    [2,2]    1   1024    307   0.029   500        1     1493  ✔
ChlorineConcentration   reserve    [2,3]    4   1024    307   0.029   166        4     1300  ✔
PTB-XL                  source     [2,4]   16   1024    307   0.029  1000       18     1974  ✔
NonInvasiveFetalECGThorax1source   [17,34]   16   1024    307   0.029   750       18     1733  ✔
Yoga                    reserve    [2,2]    1   1024    307   0.029   426        1     1421  ✔
FreezerRegularTrain     reserve    [2,2]    1   1024    307   0.029   301        1     1301  ✔
MixedShapesRegularTrain source     [2,4]   16   1024    307   0.029  1024       18     1997  ✔
InsectWingbeatSound     source     [4,9]   16   1024    307   0.029   256       18     1300  ✔

步骤2（基线，iters=0，5 种子）：5.5 min
步骤3（训练能拉动，iters=290，每候选 20 条）：全部 11.5 h / **仅源角色 7.2 h**
  ← 两重上界：① reserve 候选（K<5）无论如何进不了源池，可不跑；② 只对过 G1（基线≤0.90）的候选跑，贴顶的直接出局

源池任务合计 256（floor=100，过闸）——注意这是**初筛后**的上界，步骤3 判完 usable 还会再掉
```
