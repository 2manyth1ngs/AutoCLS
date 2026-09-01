# 源池 v2 定稿终判报告（283 格 @290，判读带原样执行）

日期：2026-09-01 ｜ 协议：`PROTOCOL.md`（锁 4e91057，判据先于补格数据写死）｜
判定机：`scripts/27_source_pool_v2.py --analyze` ｜ 数据：`rprime_cache.jsonl`
283 格 @290 k=1 canonical（**全部 RTX 5090 单机**）+ 既有基线 2325 键复用。

## 0. 一句话结论

**源池 v2 = 8 个数据集**：核心 4（Handwriting / UniMiB-SHAR / PhonemeSpectra /
UWaveGestureLibrary）+ 晋级 3（**EthanolConcentration / Cricket /
HandMovementDirection**）+ EEG 低权 1（SelfRegulationSCP2）。
**靶池：常规靶 Handwriting、UniMiB-SHAR；展示靶 PhonemeSpectra、Cricket。**
两项预期外结果：**SAD 因 6 任务口径贴顶（max 0.993）落选**（n=2 初筛 0.964
未暴露）、**NATOPS 确认腰斩 0.58→0.32 落选**；**HW @290 净空 0.067 站上常规靶**
（PS4 判错，工作点口径的 0.004 不外推到 @290）。

## 1. 执行台账

- 协议锁后单遍执行，无期中偷看。283 格 = 确认 210（5 候选 ×6 任务 ×10 CLS +
  SCP2 3×10）+ GGS 73。基线 0 补测（既有 2325 键全复用）。
- **收官重验全绿**：283 行 / 283 唯一键 / 零 null / 全 `iters=290` /
  device_name 全为 RTX 5090 / `inflight.txt` 0 B / **零 OOM** /
  逐数据集配额与协议逐格一致；本地 md5 与 pod 端逐位一致
  （`fc6d56aa346fc54e95d37ecef09d4977`）。
- **硬件与墙钟（锚点带硬件标签）**：RTX 5090 单卡单进程 283 格 **136.7 min**
  ——短 L（NATOPS 51）~9 s/格、Cricket 256 ~25 s、HMD 400 ~24 s、
  Ethanol 1024 ~34–144 s、SCP2 1152 ~90 s。
- **一次失败的前史（如实记）**：首轮在 RTX 4090（24 GB）跑，273/283 后
  **6 格因 resize 重尾 OOM 跑不出**（Ethanol_02..05 与 SCP2_02 的重尾 CLS）；
  `expandable_segments` 重试梯子只救回 4 格即零进展。**24 GB 不足以覆盖本工单的
  长序列端**，32 GB 全程零 OOM。4090 的 277 行因跨卷不可达而弃用——本报告
  一切数字来自 5090 单机全量重跑，无混机。

## 2. 晋级判（协议 §4：存活率 ≥40% ∧ max<0.99 ∧ 全距 ≥0.10）

| 数据集 | n | 配对 | 确认存活率 | δ | max R' | 全距 | 判定 |
|---|---|---|---|---|---|---|---|
| SpokenArabicDigits | 6 | 270 | 0.574 | 0.03 | **0.993** | 0.732 | ❌ 贴顶落选（锁位未生效） |
| Cricket | 6 | 270 | 0.530 | 0.08 | 0.950 | 0.637 | ✅ 晋级 |
| EthanolConcentration | 6 | 270 | 0.474 | 0.05 | 0.692 | 0.444 | ✅ 晋级（锁位） |
| HandMovementDirection | 6 | 270 | 0.448 | 0.06 | 0.688 | 0.412 | ✅ 晋级 |
| SelfRegulationSCP2 | 3 | 135 | 0.407 | 0.05 | 0.596 | 0.149 | ✅ EEG 低权入池 |
| NATOPS | 6 | 270 | **0.322** | 0.06 | **1.000** | 0.519 | ❌ 双重落选 |

- **n=2 初筛 vs n=6 确认的对账**（协议 §1 明令初筛不得作晋级依据，本表印证）：
  SAD 58→57（存活率稳，但 max 0.964→**0.993** 暴露贴顶任务）、
  Cricket 51→53、Ethanol 54→47、HMD 52→45、SCP2 36→41、
  **NATOPS 58→32（腰斩）**。存活率的 n=2 估计既能高估也能低估，且
  **贴顶判据只有在更多任务上才暴露**——"n=2 禁止免确认晋级"这条预注册条款
  在本轮救下两个错误晋级。
- 晋级 3 个（协议上限 4）：锁位 Ethanol 合格入选；SAD 锁位因贴顶判据失效；
  备选按存活率降序取 Cricket、HMD（Cricket 0.530 > HMD 0.448）。

## 3. EEG 低权判与源池 v2 组成

SCP2 存活率 0.407 ≥ 0.30、全距 0.149 ≥ 0.10 → **以低权身份入池**：只供
label / 全局毒性先验粮草（破坏配置是 avoid-worst 的粮草，训练侧不过滤），
**不作 LOO 折、不入靶池**（协议 §4 写死）。

**源池 v2（8）**：Handwriting、UniMiB-SHAR、PhonemeSpectra、
UWaveGestureLibrary、EthanolConcentration、Cricket、HandMovementDirection、
SelfRegulationSCP2(低权)。
**退役（5）**：CharacterTrajectories、ArticularyWordRecognition（δ 存活率
19%/23% + 顶 0.995/1.000 贴顶）、Heartbeat（0%，全距 0.04 死平）、
SpokenArabicDigits、NATOPS（本轮确认落选）。

配置写回：`configs/source_pool.yaml`（唯一真源）——Ethanol 由 reserve 升回
source，逐条加 `role_v2` 审计字段；`tests/test_source_pool.py` 的池大小与
Ethanol 角色断言同步更新（源 14 / reserve 2；其中 v2 活跃 8，退役者留在
注册表并由 `role_v2` 标注，便于复算历史缓存）。

## 4. 靶池判（可评 = oracle−中位 ≥0.05 ∧ max<0.99；净空 = oracle−GGS@290 ≥0.05）

| 数据集 | 任务 | 可评 gap | 净空 | max R' | 档 |
|---|---|---|---|---|---|
| Handwriting | 8 | 0.117 | **0.067** | 0.944 | **常规靶** |
| UniMiB-SHAR | 8 | 0.059 | 0.057 | 0.893 | **常规靶** |
| PhonemeSpectra | 8 | 0.072 | 0.010 | 0.657 | 展示靶（净空不足） |
| Cricket | 6 | 0.066 | 0.016 | 0.950 | 展示靶（净空不足） |
| UWaveGestureLibrary | 8 | **0.036** | 0.059 | 0.982 | out（不可评：区分度不足 + 近贴顶） |
| EthanolConcentration | 6 | **0.049** | 0.100 | 0.692 | out（可评 gap 差 0.001） |

- **HW 的净空在 @290 是 0.067，不是工作点口径的 0.004**——PS4（先验 85%）
  判错。含义：`source-pool-screen-20260831` 的"HW 对 GGS 净空仅 0.004"是
  **工作点（60 iters）口径**的结论，**不外推到 @290**；MX-2v2 的靶池与真值
  口径必须一致（铁律 2 v2 增补），引用净空时必须带 iters 口径。
- Ethanol 以 0.049 vs 0.05 之差落在 out——**边界值如实报，不四舍五入救活**；
  它净空 0.100 全场最高，但可评判据先行。若 MX-2v2 需要更多靶，按协议须补
  `_06/_07` 网格重算（本轮 n=6 口径下不改判）。
- UWave 从核心源变成"不可评靶"：源池资格（供 label）与靶池资格（供搜索展示）
  是两件事，本表是二者分离的又一实例。

## 5. 预测对账（协议 §5）

| # | 预测 | 先验 | 实际 | 判 |
|---|---|---|---|:--:|
| PS1 | SAD 确认 ≥40% | 70% | 0.574（但贴顶落选） | ✅ |
| PS2 | Ethanol 确认 ≥40% | 60% | 0.474 | ✅ |
| PS3 | 晋级总数 ≥2 | 75% | 3 | ✅ |
| PS4 | HW @290 净空 <0.05 | 85% | **0.067** | ❌ |
| PS5 | SCP2 确认 ≥30% | 60% | 0.407 | ✅ |
| PS6 | 常规靶 ≥2 | 60% | 2（HW/UM） | ✅ |

PS1 命中却仍落选，说明**存活率与贴顶是两个独立闸**——协议把两者写成合取是
对的。PS4 是本轮唯一判错，且错在"把工作点口径的数字当数据集属性外推"，与
`cost-anchors-need-hardware-labels` / `iters-workpoint-scan-flips` 同类教训。

## 6. 对 MX-2v2 的交接

1. **源池 v2 定稿可作 LOO 折的 7 个**（8 减低权 SCP2）；靶池常规 2 + 展示 2。
2. **长序列 × EEG 共线 limitation 解除**：Ethanol（Spectro，L=1024）回到源池，
   长端不再全是 EEG/MEG。CLAUDE.md 的该条 limitation 可改写为"v1 期存在，
   v2 已由 Ethanol 回填解除"。
3. **域覆盖新缺口**：SAD 退役后**语音域仅剩 PhonemeSpectra**；NATOPS 退役后
   短序列端（L=51）清空——两条写入 limitations。
4. **成本锚点**（带硬件标签）：5090 单卡 283 格 136.7 min；**24 GB 卡不适用于
   本工单长序列端**（4090 实测 6 格 OOM 不可救）。
5. 引用口径提醒：本报告一切数字 @290 iters、shared 10 CLS、k=1 canonical、
   δ 查 `configs/labels.yaml`（确认轮未重定 δ）、GGS = `scripts/m1_ggs_har.GGS`。
