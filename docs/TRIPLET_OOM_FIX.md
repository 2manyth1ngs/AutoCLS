# Triplet 分块修复 + 阶段1 那 69 条 OOM 的补跑方案

阶段1（2026-08-03，1×RTX 5090）跑出 243 canonical + 728 噪声 R'，另有 **69 条
(task, cls) 因显存不足没跑成**（`results/step0_snr_gate/oom_failures.jsonl`，
按 `oom_guard` 规矩 2 未写缓存）。本文记录修复、等价性验证、显存验证与补跑操作。

代码：`autocls/contrastive/losses.py`；测试：`tests/test_triplet_chunking.py`。

---

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| 语义等价 | float32 最坏相对偏差 **loss 2.78 eps / grad 1.90 eps**；float64 同为 ~2 eps（证明只是浮点结合序） |
| 端到端回归 | 6 条 triplet canonical 条目重算，R' 与阶段1 缓存值 **逐位相同**（Δ=0.0000） |
| 显存 | 69 条的峰值全部降到预算内（最坏 175.9 → **29.2 GiB** < 31.4） |
| 吞吐 | 损失步 ~1.2–1.3×；整条流水线 CPU-bound，端到端影响远小于此 |
| 全套测试 | **464 passed** |

---

## 2. 对阶段1 报告的两处订正

1. **不是"69 条全部 triplet"**：68 条 triplet + **1 条 info_nce**
   （`SelfRegulationSCP2/a1f67db4`）。分块修复对 info_nce 无效，那条能装下靠的是
   第 3 节里的其他几项（掩码/nonzero/τ）。
2. **不是"triplet × 大 crop"这么简单**：真正的放大器是 **`resize`**。
   `resize` 的 p 是高斯 σ，`T_new = round(T·(1+ε))` 上限 `2T` —— SCP2 的编码长度
   `n=2T` 在 290 个 iteration 里从 1152 跳到 **4378**（p50 才 2276）。
   `sim` 是 `(B, n, n)`，n 翻 2 倍它就翻 4 倍：单个 sim 从 2.5 GiB 涨到 **9.1 GiB**。
   所以 OOM 是**随机尾部事件**（ε≥0.9 的概率约 6.7%，290 iters 里必然发生若干次），
   这也解释了为什么同一条 CLS 在短数据集上没事、在长数据集上必崩。

---

## 3. 改了什么（五项，全部是纯内存，无一改变数值语义）

爆点原本记作 `losses.py:91` 的 `hinge=(P, n)`。分块把它降到 `O(chunk × n)` 后，
**原先被 hinge 掩盖的另外四项与 `sim` 同形的开销就成了新峰值**，必须一并处理，
否则"分了块也照样 OOM"（实测中间态 39.3 → 35.1 → 32.4 GiB，都还超预算）：

| # | 改动 | 省掉的东西（SCP2 最坏形状 B=128, n=4378） |
|---|---|---|
| ① | hinge 按 P 分块累加 | `(P, n)` 的 4~5 个中间张量，**27.4 GiB × 数份** |
| ② | 分块体改成自定义 `autograd.Function`，反向手写、逐块 `index_put_(accumulate=True)` 进同一块梯度缓冲 | 高级索引的反向会为**每块**分配一个与 `sim2` 同形的零张量，**9.1 GiB/块** |
| ③ | `/τ` 与 `neg = valid ∧ ¬pos` 移进分块体 | 全尺寸的 `s = sim/τ`（9.1）+ 其梯度缓冲（9.1）+ `neg`（2.3） |
| ④ | 三元组计数在块内累加 | 对 `(B,m,n)` bool 整块 `.sum(-1)` 会先升 int64 → **18.3 GiB**（分块后最大的残余热点） |
| ⑤ | 正样本下标按前导维分片取 `nonzero` | `pos&valid` 的全尺寸 bool（2.3）+ CUDA nonzero 按 numel 计的前缀和缓冲（2.4） |

②只用 `torch.utils.checkpoint` 是**不够**的：checkpoint 的反向是"重放前向再走
autograd"，`sim2[b_i, a_i]` 的反向仍会分配全尺寸零张量。实测 checkpoint 版
32.4 GiB > 预算，手写反向版 29.0 GiB。

分块阈值 `TRIPLET_CHUNK_ELEMS` 默认 2^26 元素（268 MiB/张量），
由 `configs/runtime.yaml: env.AUTOCLS_TRIPLET_CHUNK_ELEMS` 注入到每个 worker
（与 `OMP_NUM_THREADS` 同一条链：`cloud_plan.worker_env`）。

---

## 4. 等价性验证（红线）

### 4.1 单测：与改动前实现逐项比对

`tests/test_triplet_chunking.py` 把**改动前的 `_masked_triplet` 原样留作 oracle**
（`reference_masked_triplet`），85 个用例覆盖单/多 batch、三种相似度、chunk=1 到
不分块、分块整除与不整除、无正样本/无负样本边界、以及真实的 temporal+adj_neighbor
调用形状；比对**损失值与对 `sim` 的梯度**两项。

270 组（dtype × 形状 × 相似度 × seed × chunk）的最坏相对偏差：

```
float32   loss 3.312e-07 = 2.78 eps    grad 2.265e-07 = 1.90 eps
float64   loss 4.365e-16 = 1.97 eps    grad 4.415e-16 = 1.99 eps
float32 梯度逐位相同元素：6581214/6884352 = 95.60%
```

**float64 的偏差同样只有 ~2 eps** —— 若语义有变，偏差不会随精度一起缩小。
这是"差异纯粹来自浮点结合序"的判据。

反向是手写的，另有独立的 `torch.autograd.gradcheck`（float64 数值差分）——
与 oracle 一致只能排除浮点层面的问题，排不掉"公式本身抄错"。

### 4.2 端到端回归：用阶段1 的 canonical 缓存复算

同 task / cls / weight_seed / split_seed / iters，只换实现：

```
数据集/任务         cls        loss       cache   旧@本机   新@本机      新-旧    旧-cache
NATOPS/00      9092e36e   triplet      0.9537    0.9537    0.9537   +0.00e+00   +0.0000
NATOPS/00      65627517   triplet      0.9259    0.9259    0.9259   +0.00e+00   +0.0000
NATOPS/00      8b87d66f   triplet      0.9722    0.9722    0.9722   +0.00e+00   +0.0000
ERing/00       cffb5503   triplet      0.8778    0.8778    0.8778   +0.00e+00   +0.0000
ERing/00       8c4aa283   triplet      0.9111    0.9111    0.9111   +0.00e+00   +0.0000
ERing/00       9092e36e   triplet      0.9111    0.9111    0.9111   +0.00e+00   +0.0000
```

6 条 triplet 条目**全部逐位复现**阶段1 在 5090 上的 R'（本机 4060）。
CPU（完全确定）上另跑一条，`新-旧 = 0.000e+00`。

> **一条 info_nce 对照条目的插曲**：`NATOPS/20be4f74`（info_nce，本次改动未触及）
> 出现 |Δ|=0.046。**同实现在 GPU 上重复跑 4 次**得 `[0.3704, 0.5833, 0.3426, 0.5463]`，
> 极差 **0.2407** —— 该条目本身就是 run-to-run 噪声（R'≈0.35~0.58，表征基本塌了），
> 与本次改动无关。同一对照里 triplet 条目 4 次全同（极差 0.0000）。
> ⚠️ 这条本身值得记一笔：它的 R' 不携带信息，δ 过滤能不能挡住这种条目值得复查。

---

## 5. 显存验证

本机 RTX 4060 Laptop 只有 8 GiB，装不下 B=128 的真实 batch，故实测**小 B 的
峰值-B 斜率**再线性外推到 128（主导项 `sim=B·n²`、`hinge=k·B·n²`、encoder 激活
都严格正比于 B；线性拟合残差一并列出）。被测路径 = 一个 iteration 的
`encoder 前向 → emb_transform → total_contrastive_loss → backward`。
每条取该 (数据集, cls, task) 在 290 iters 里**最坏一次 resize** 的 n。

| 数据集/cls | 条 | n=2T | 旧 B=128 | 新 B=128 | 判定 |
|---|---:|---:|---:|---:|---|
| SelfRegulationSCP2/9092e36e | 8 | 4378 | 175.9 | **29.2** | ✅ |
| SelfRegulationSCP2/8c4aa283 | 8 | 4378 | 30.5 | 29.1 | ✅（余量薄，见下） |
| SelfRegulationSCP2/cffb5503 | 8 | 2152 | 44.8 | 11.7 | ✅ |
| SelfRegulationSCP2/a1f67db4 | 1 | 2304 | 21.3 | 21.9 | ✅（info_nce，未改动） |
| SelfRegulationSCP2/8b87d66f | 1 | 1382 | 20.6 | 8.0 | ✅ |
| EthanolConcentration/9092e36e | 8 | 3892 | 140.1 | 24.5 | ✅ |
| EthanolConcentration/cffb5503 | 8 | 1976 | 38.2 | 10.5 | ✅ |
| EthanolConcentration/8c4aa283 | 4 | 3892 | 25.3 | 26.8 | ✅（余量薄） |
| SelfRegulationSCP1/9092e36e | 8 | 3404 | 107.9 | 22.4 | ✅ |
| SelfRegulationSCP1/cffb5503 | 7 | 1748 | 30.3 | 8.8 | ✅ |
| MotorImagery/9092e36e | 8 | 1946 | 37.2 | 10.3 | ✅ |

旧值 175.9 GiB 与阶段1 实际报错（"26.57 GiB in use, tried to allocate 7.10 GiB"）
一致 —— 进程是在往那 175.9 GiB 爬的路上死的。

**余量薄的两条（`8c4aa283`，temporal=False）**：它们的 instance 对比 sim 是
`(T, 2B, 2B)`，**按 B² 增长**，线性外推会低估约 1 GiB，真实值可能到 ~30 GiB。
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 已加进
`configs/runtime.yaml: env`（阶段1 报错里有 7.58 GiB "reserved but unallocated"，
张量尺寸随 resize 浮动必然造成分段碎片）。**补跑时这几条务必独占整卡**（worker=1，
收尾段本来就是）。

---

## 6. 吞吐影响

损失步（GPU 侧）实测：

| 形状 | 块数 | 旧 ms/it | 新 ms/it | 倍率 |
|---|---:|---:|---:|---:|
| L=51  B=128 | 1 | 29.3 | 35.3 | 1.21× |
| L=315 B=128 | 3 | 196.4 | 258.1 | 1.31× |
| L=1152 B=12 | 3 | 219.2 | 277.1 | 1.26× |

代价来自手写反向要重算一遍 hinge。整条流水线是 **CPU-bound**（增强在 CPU 上做，
长序列单 worker GPU 利用率仅 5.7%，见 `GATE_5090_REPORT §1.9`），端到端影响
远小于 1.3×。

---

## 7. 补跑操作

### 7.1 前置检查（已在本机验证）

69 条在 `--resume` 眼里全部是"未跑"：
- canonical 缓存 243 条（无重复），噪声缓存 728 条；
- 69 条 OOM 记录**两处皆无**命中 → `is_done` 全判 False ✅
- 种子分布 `s0:17 s1:17 s2:17 s3:18`（s=0 进 `labels_route_a/rprime_cache.jsonl`，
  s≥1 进 `step0_snr_gate/noise_cache.jsonl`）。

### 7.2 命令

```bash
# ① 先把旧的 OOM 清单归档 —— 它是 append-only 的审计记录，不归档的话
#    补跑成功后 run_stage1.py 仍会念"69 条显存不足"，判读时会误导
mv results/step0_snr_gate/oom_failures.jsonl \
   results/step0_snr_gate/oom_failures_20260803.jsonl

# ② 只跑收尾段（worker=1 串行 + --on-oom record）+ 判读，跳过并发段
python scripts/run_stage1.py --skip-concurrent 2>&1 \
  | tee -a results/step0_snr_gate/stage1_rerun.log

# ③ 核对：新的 oom_failures.jsonl 不应再出现（若出现，逐条看是哪个形状）
wc -l results/step0_snr_gate/oom_failures.jsonl 2>/dev/null || echo "无新 OOM ✅"
```

`--skip-concurrent` 之后 `run_stage1.py` 会自动接 `--analyze`，
那 4 个数据集的 SNR/δ 会**基于完整证据重算**（阶段1 的判读是在
SCP2 7/10、Ethanol 8/10、SCP1 9/10、MotorImagery 9/10 的残缺覆盖上做的）。

### 7.3 ETA

用阶段1 **实测**的秒/条（缓存里的 `seconds` 字段），worker=1 串行：

| 数据集 | 待补 | 实测中位 s/条 | 串行小时 |
|---|---:|---:|---:|
| SelfRegulationSCP2 | 26 | 62 | 0.45 |
| EthanolConcentration | 20 | 71 | 0.39 |
| SelfRegulationSCP1 | 15 | 70 | 0.29 |
| MotorImagery | 8 | 47 | 0.10 |
| **合计** | **69** | | **1.24 h** |

这 69 条是各数据集里最贵的（resize/crop 最大 → T 最长），中位数偏低；
按各数据集 P90 秒/条算上界 **1.88 h**，再乘损失步 ~1.3× 的上界仍在 **~2 h** 内。
另需算上收尾段对其余 9 个数据集的空转扫描（每个几分钟，`--resume` 直接跳过）。

---

## 8. 对阶段2 的影响（同样生效）

`scripts/2b_generate_labels_route_a.py` 与 step0 走**同一条**
`evaluate_rprime → pretrain_encoder → total_contrastive_loss → losses.py` 路径，
同 `early_iters` 同 `batch_size`，故修复对阶段2 **自动生效**。

而且阶段2 更需要它：step0 每任务只跑 10 个 shared 探针 CLS，阶段2 还要加
**每任务 L 个 random CLS**（`sample_random_cls`），从 19 维空间里随机采样必然
撞上更多 "triplet × 大 resize × 长序列" 的组合。不修的话阶段2 会重演，且这次
是在更大的分母上丢 label。

---

## 9. 遗留 caveat

- **`resize` 能把 T 放大到 2×，显存按 T² 走** —— 这是长序列源数据集的结构性风险，
  本次只是把 triplet 侧的常数压下去了。若将来源池再加更长的数据集（或 batch_size
  变大），`sim = (B, 2T, 2T)` 本身会再次成为墙。彻底的解法是把相似度矩阵也分块
  （改动面覆盖 InfoNCE），本次**未做**。
- 上面所有 B=128 的显存数字是**小 B 实测 + 线性外推**，不是 B=128 实测
  （本机 8 GiB 装不下）。拟合残差已随表列出；`8c4aa283` 两条因 instance 项按 B²
  增长，外推会低估约 1 GiB。
