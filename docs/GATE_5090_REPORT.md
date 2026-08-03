# 第0步闸门执行报告：1×RTX 5090 兼容性 smoke

> 执行日期 2026-08-01（UTC 06:40–08:00，单卡约 1.3 h）
> 对应手册 `docs/RUNPOD_5090_STAGE1_RUNBOOK.md` §1，代码 commit `6debed3`
> 产物存档 `results/gate_5090_20260801/`，canonical R' 已归位 `results/labels_route_a/rprime_cache.jsonl`

## 结论

**闸门通过，按 §1.8 应放行 §2。** 但 §1.9 标定推翻了 §0.7 的成本模型与 4 卡选型前提，
§2 部署前必须先改配置（见「必须改的三件事」）。

---

## 1. 环境指纹（阶段2 必须复用同一镜像）

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，cc **(12, 0)**，31.4 GiB，driver 570.211.01 |
| torch | **2.8.0+cu128** |
| arch_list | `['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` ← 含 sm_120 |
| CUDA / cuDNN | CUDA_VERSION 12.8.1 / cuDNN 91002 |
| OS | Ubuntu 24.04.3，python 3.12.3 |
| CPU | AMD EPYC 9J14，cgroup 配额 **54.4 核**（宿主 384 线程，Community Cloud 共享） |
| 依赖 | numpy 2.1.2 / scipy 1.15.3 / pandas 2.3.3 / numba 0.61.2 / sklearn 1.7.2 / **aeon 1.3.0** / joblib 1.5.3 |

> ⚠️ 镜像 tag 全名**未记录**（部署时未抄，pod 内 env 无此信息）。阶段2 重开 pod 时
> 须用上表指纹逐项核对，尤其 torch 2.8.0+cu128 与 cuDNN 91002。

与本机基线唯一差异：numpy 2.1.2 vs 2.2.6（同为 2.x 小版本差，按 §0.2 判据无所谓）。

## 2. §1.8 判定表

| 检查 | 标准 | 实测 | |
|---|---|---|---|
| §1.4① wheel cu128+/torch≥2.7 | 两项 PASS | 2.8.0+cu128 | ✅ |
| §1.4② arch_list 含 sm_120 | PASS | 含 | ✅ |
| §1.4③ 认出 5090 / cc 12.x | PASS | cc (12,0) | ✅ |
| §1.4④ matmul+膨胀Conv1d+反向+AdamW 实跑 | 无 `no kernel image` | 通过 | ✅ |
| §1.3 装依赖后 torch 指纹不变 | 逐字符相同 | 完全相同，pip 未触碰 torch | ✅ |
| §1.6 逐条 \|ΔR'\| | 中位≤1、最大≤4 样本 | **中位 0.00、最大 2.00** | ✅ |
| §1.6 ΔR' 带符号均值 | \|·\|≤0.01 | **+0.0001** | ✅ |
| §1.6 Spearman | ≥0.90 | **0.9980** | ✅ |
| §1.6 决策级 | verdict 一致、SNR 差≤0.30、R̄ 差≤0.02 | **全同，差值均 0** | ✅ |
| §1.6 NaN | 零条 | 0 | ✅ |
| §1.7 resume | 待跑 ≪ 计划 | 计划 30 / 待跑 2（kill 前 28 条零重算） | ✅ |
| §1.7 SVM 兜底 | 无池损坏或被接住 | 零 `BrokenProcessPool` | ✅ |

数值一致性细节：90 条中 **88 条与 4060 逐位相同**，2 条各差 1–2 个评估样本，
方向一正一负。决策级 verdict/SNR/R̄ 与 4060 完全相同：

| 数据集 | verdict | SNR | R̄ |
|---|---|---|---|
| NATOPS | `fail_unstable`（4060 同） | 1.23（4060 同） | 0.9154（4060 同） |
| ERing | `fail_mismatch`（4060 同） | 0.71（4060 同） | 0.9644（4060 同） |

§1.7 演练证据：watchdog 识别 `rc=137` → `--resume` 续跑；
`watchdog_state.tsv` 记毒条目 `ERing_00/f48e5da1d899 iters=30` 第 1 次（未达
FAIL_LIMIT=3，未隔离，行为正确）。SIGKILL 确实留下 55 个 loky 孤儿，
但未导致 `BrokenProcessPool`，串行兜底未被触发。

## 3. §1.9 标定：R 不是常数

### 3.1 `OMP_NUM_THREADS=1` 是硬性启动条件

`torch.get_num_threads()` 在 54 核容器里默认 **192**。对 128×51×24 这种小张量，
OpenMP 同步开销压倒计算：

| 配置 | 秒/条（NATOPS@290） | R vs 4060(33.5 s/条) |
|---|---|---|
| 默认（192 线程） | 53 | **0.63×（比 4060 笔记本还慢）** |
| **`OMP_NUM_THREADS=1`** | **12.6** | **2.67×** |

根因：`cl_trainer.py:50-51` 明确写着「数据增强在 CPU 上执行」，GPU 只做 encoder
前向 → 本负载是 **CPU-bound**，单 worker 下 GPU 利用率仅 **5.7%**（75 W / 上限 600 W）。

### 3.2 单卡并发扩展性（NATOPS L=51，真实 `evaluate_rprime`）

| 并发 N | 墙钟 | 吞吐 | 等效 s/条 | 相对 N=1 |
|---|---|---|---|---|
| 1 | 61.9s | 2.9 条/分 | 20.6 | 1.00× |
| 2 | 56.6s | 6.4 条/分 | 9.4 | 2.19× |
| 4 | 62.7s | 11.5 条/分 | 5.2 | 3.96× |
| 8 | 94.8s | 15.2 条/分 | 3.9 | **5.24×**（GPU 91%, 10.9 GB, 200 W） |

**1 张卡 8 个 worker（15.2 条/分）> 4 张卡各 1 个（11.6 条/分）。**

### 3.3 长序列撞显存墙（合成 SCP2 形状 380×1152×7）

| N | 结果 |
|---|---|
| 1 | 123.6s，正常（单 worker R ≈ 5.4× vs 4060 估算 757.8 s/条） |
| 2 | **1 个 worker CUDA OOM** |
| 4 | **2 个 worker CUDA OOM** |

爆点 `losses.py:91`：`hinge = torch.relu(margin + rows - s_pos.unsqueeze(-1)) * neg_rows`
—— triplet 损失物化 (n,n) 相似度矩阵，L=1152 时单进程峰值 **15.8 GB**，
31.4 GB 装不下两份。

**每卡 worker 数必须按 L 分档**：

| 序列长度 | 单卡 worker 数 | 依据 |
|---|---|---|
| L ≤ ~150（NATOPS/ERing/Handwriting/ArticularyWord/CharacterTrajectories） | **8** | 实测 GPU 91%，5.2× 吞吐 |
| L ~300–500（UWave 315/HandMovement 400/MotorImagery 512） | **2（保守）** | **未实测**，开跑时看显存再调 |
| L ≥ ~900（SCP1 896 / SCP2 1152） | **1** | 实测 2 个必 OOM |

## 4. 必须改的三件事（§2 部署前）

1. **所有 worker 必须带 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`**，否则慢 4.2×。
2. **每卡 worker 数按 L 分档**，不能全局取一个值。
3. **OOM 会静默丢 label**：`evaluate_rprime` 把异常吞成 `r_prime=null` 写缓存，
   `--resume` 不重试；看门狗还会把连崩条目主动隔离。长序列开多 worker =
   大面积永久丢 label。这是会静默毁数据的配置错误。

## 5. 手册需订正处（本次实证）

| 位置 | 手册写的 | 实际 |
|---|---|---|
| §1.2 | `winget install runpod.runpodctl` | winget 仓库无此包，须从 GitHub Releases 取（v2.8.0） |
| §1.2 pod 侧 | `mkdir -p data results/route_a_probe` | 漏 `results/step0_snr_gate`，§1.6.3 的 nohup 重定向会先于 `cloud_run.sh` 的 mkdir 求值 → 整跑静默失败 |
| §1.3 | `pip install "aeon==1.3.0" ...` | Ubuntu 24.04 PEP 668 拦截，须加 `--break-system-packages`（落点 `/usr/local/lib/python3.12/dist-packages`，与 torch 同处，不碰发行版目录） |
| §1.9 | NATOPS 标定「80 条，10–18 min」 | 290 iters 下**未触发**加难重探，确为 80 条；实测 16 min。但 smoke（30 iters）会触发 2 轮，手册未区分两者 |
| §0.7 成本表 | R=2.5/3/4/6 四行 | 四行都基于「GPU 是瓶颈」的错误前提；真实 R 随 L 与并发数变化，见 §3 |

## 6. 产物

| 路径 | 内容 |
|---|---|
| `results/labels_route_a/rprime_cache.jsonl` | **20 条 NATOPS canonical R'@290**（正式产物，阶段1 重传后 `--resume` 逐键复用，省 16 min） |
| `results/gate_5090_20260801/noise_cache.jsonl` | 60 条 NATOPS 噪声种子 R'@290 |
| `results/gate_5090_20260801/analysis.json` | NATOPS@290 正式判读 |
| `results/gate_5090_20260801/smoke_cache.jsonl` | 90 条 5090 smoke@30（**勿与 `results/step0_snr_gate/` 的 4060 基线混淆**） |
| `results/gate_5090_20260801/bench_*.py` | 并发扩展性基准脚本 |

`results/step0_snr_gate/` 保持 4060 基线原样未动。

## 7. 附带发现：NATOPS 在真 iters 下的判读

`analysis.json`（290 iters，非 smoke）：

```
SNR=4.01  R̄=0.8546  σ=0.0392  spread=0.1571  符号稳定=0.589  单调=0.067
建议 δ≈0.118（注册表 0.06）
判定: 符号稳定率=0.59<0.7：成对方向被噪声翻转（须 k 折降噪或大 δ）
```

- SNR 从 smoke 的 1.23 涨到 **4.01**，CLS 区分力在真 iters 下很强（spread 0.157，是 smoke 的 8 倍）
- 但**符号稳定率 0.589 < 0.7** 仍判 fail
- **建议 δ 0.118 是注册表值 0.06 的约 2 倍** —— `configs/labels.yaml` 的 per-dataset δ 初值可能整体偏小

这是 §8 判读要处理的实质问题，不是闸门问题。单数据集 n_seeds=4，样本有限，
阶段1 全量跑完后应重新审视 δ 标定。
