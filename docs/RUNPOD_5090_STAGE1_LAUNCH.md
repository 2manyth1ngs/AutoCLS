# 阶段1 定稿执行卡：1×RTX 5090 单卡跑 step0 SNR 闸

> 取代 `docs/RUNPOD_5090_STAGE1_RUNBOOK.md` 的 §0.7 / §2 / §4 / §6 / §7（那四节的 4 卡
> 前提已被闸门标定推翻）。**§1（Blackwell 闸门）、§3（环境与数据）、§8（判读）、
> §9（结束与成本）仍然照那份做。**
>
> 定稿依据 `docs/GATE_5090_REPORT.md`（2026-08-01，闸门全绿；原始产物在
> `results/gate_5090_20260801/`）。

---

## 0. 为什么是单卡

| 事实（实测） | 推论 |
|---|---|
| 增强在 CPU 上做（`cl_trainer.py:50`），GPU 只跑 encoder 前向；单 worker GPU 利用率 **5.7%** | 本负载 **CPU-bound**，不是 GPU-bound |
| 1 卡 8 worker **15.2 条/分** vs 4 卡各 1 worker **11.6 条/分** | 加卡不如加 worker |
| CPU 是 cgroup 共享的 **54.4 核**（Community Cloud） | 4 卡会互抢同一份 CPU，边际收益更差 |
| `OMP_NUM_THREADS` 默认 192 → 53 s/条（**0.63×**，比 4060 笔记本还慢）；置 1 → 12.6 s/条（2.67×） | 线程数是**硬性启动条件**，已写进 `cloud_run.sh` 与 `run_stage1.py`，不靠手动 export |
| L=1152 单进程显存峰值 **15.8 GB**（`losses.py:91` 的 triplet (n,n) 矩阵），31.4 GiB 装不下两份 | worker 数必须**按 L 分档**，且必须有 OOM 防护 |

**嫌慢再加卡**：三个长数据集（SCP2/Ethanol/SCP1）互斥串行 7.4 h，占墙钟 91%。
加第二张卡的唯一正确用法是把长数据集分到另一张卡上，能砍到 ~4.5 h。

---

## 1. 开 pod

| 项 | 选择 |
|---|---|
| GPU | **RTX 5090 ×1** |
| Cloud Type / Pricing | Community Cloud / **On-Demand**（非 spot：无抢占，看门狗只需处理进程级崩溃） |
| Template | **与闸门同一个镜像**（判据见下） |
| Container Disk | 30 GB |
| Volume / Mount | 30 GB / `/workspace` |

### 环境指纹逐项核对（阶段2 也要对同一张表）

闸门产出的 canonical R' 会被后续逐键复用、**不重算**。两次跑若数值环境不同，
同一批 label 里会混进系统性偏移。开机第一件事：

```bash
python - <<'PY'
import torch, platform
print("torch      ", torch.__version__)                    # 期望 2.8.0+cu128
print("cuda/cudnn ", torch.version.cuda, torch.backends.cudnn.version())  # 12.8 / 91002
print("arch_list  ", torch.cuda.get_arch_list())           # 必须含 sm_120
print("gpu        ", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))  # 5090 (12,0)
print("python/os  ", platform.python_version(), platform.platform())      # 3.12.3 / Ubuntu 24.04
PY
nvidia-smi --query-gpu=driver_version,memory.total --format=csv   # 570.211.01 / 32607 MiB
```

**任一项对不上就停**，按 RUNBOOK §1.5 处置或换镜像重开。依赖版本基线：
`numpy 2.1.2 / scipy 1.15.3 / pandas 2.3.3 / numba 0.61.2 / sklearn 1.7.2 / aeon 1.3.0 / joblib 1.5.3`。

> **镜像 tag 记录**：闸门当天漏抄。这次**开 pod 时立刻抄下 Template 全名**填进本节表格。
> 已经开了忘了抄，用下面任一条找回：
> ```bash
> runpodctl get pod $RUNPOD_POD_ID --allfields        # 输出含 IMAGE NAME（需先 runpodctl config --apiKey <key>）
> curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
>   -H 'content-type: application/json' \
>   -d '{"query":"{myself{pods{id name imageName}}}"}' | python -m json.tool
> ```
> 网页端：Pods → 该 pod 卡片上的模板名，或 Templates 页对应条目。

---

## 2. 传代码与数据（RUNBOOK §3 全套 + 闸门产物复用）

RUNBOOK §3.1–3.3 照做（`git archive` 打包、16 个 uea_cache 数据集 0.86 GB、
`probe_cls_catalog.json` 单独传、铁律 6 双保险自检）。**额外三件**——闸门跑出的
NATOPS 80 条 R'，不带上去就要白跑 ~17 分钟：

```powershell
# 本地：把闸门产物打一个包
cd D:\DevelopDoc\CLS
tar -czf "$env:TEMP\gate_reuse.tar.gz" `
    results/labels_route_a/rprime_cache.jsonl `
    results/gate_5090_20260801/noise_cache.jsonl `
    results/gate_5090_20260801/tasks_meta.json
runpodctl send "$env:TEMP\gate_reuse.tar.gz"
```

```bash
# pod：解到正确位置（noise 与 tasks_meta 要改路径！）
cd /workspace/repo/CLS
mkdir -p results/step0_snr_gate results/labels_route_a
tar -xzf /workspace/data/gate_reuse.tar.gz -C /tmp/
cp /tmp/results/labels_route_a/rprime_cache.jsonl        results/labels_route_a/
cp /tmp/results/gate_5090_20260801/noise_cache.jsonl     results/step0_snr_gate/
cp /tmp/results/gate_5090_20260801/tasks_meta.json       results/step0_snr_gate/
wc -l results/labels_route_a/rprime_cache.jsonl results/step0_snr_gate/noise_cache.jsonl
# 期望 20（NATOPS canonical @290）+ 60（NATOPS 噪声 @290）
```

> ⚠️ **别把 `results/gate_5090_20260801/smoke_cache.jsonl` 放进 `results/step0_snr_gate/`**
> ——那是 30 iters 的 smoke，混进去会污染判读。

### 单卡版 preflight

RUNBOOK §3.4 的 `preflight.sh` 里那句 `assert n == 4` 改成：

```python
n = torch.cuda.device_count()
assert n == 1, f"!!! 期望 1 卡，实际 {n} !!!"
```

其余五项（compat_check / 单测 / 目标池 fail-fast / 探针目录 / canonical 就位）不变。
单测在 pod 上必须全绿：`python -m pytest tests/ -q` → **408 passed**。

---

## 3. 启动前预检（不花 GPU 时间，先看表）

```bash
cd /workspace/repo/CLS
python scripts/run_stage1.py --dry-run
```

预期输出（本地算过的基线，pod 上待跑条数会因复用而不同）：

```
数据集                          l_enc   分档    峰值GiB    秒/条    待跑       预计  告警
SelfRegulationSCP2            1152    1     15.8    124    80     2.7h
EthanolConcentration          1024    1     14.1    111    80     2.5h
SelfRegulationSCP1             896    1     12.4     98    80     2.2h
MotorImagery                   512    2      7.4     59    80     1.3h
Heartbeat                      405    2      6.0     48    80     1.1h
HandMovementDirection          400    2      5.9     48    80     1.1h
UWaveGestureLibrary            315    2      4.8     39    80     0.9h
CharacterTrajectories          182    7      3.1     26    80     0.6h  分档 8 × 3.1 GiB > 预算，降到 7
Handwriting                    152    8      2.7     23    80     0.5h
ArticularyWordRecognition      144    8      2.6     22    80     0.5h
ERing                           65    8      1.6     14    80     0.3h
NATOPS                          51    8      1.4     13     0     0.0h   ← 复用了闸门产物才是 0
FingerMovements                 50    8      1.4     12    80     0.3h
单 worker 串行总量 14.1 h → 按分档并发预计墙钟 **8.1 h**
长杆：SCP2, Ethanol, SCP1 互斥串行 7.4 h（占墙钟 91%）
```

**NATOPS 那行"待跑"必须是 0**，否则闸门产物没放对位置（回 §2 检查）。

> **前期只有 2 个进程在跑是正常的**，不是调度器坏了：SCP2 一个就吃掉 15.8 GiB，
> 24 GiB 预算里只塞得下再加一个 MotorImagery。墙钟的 91% 由三个长数据集互斥串行
> 决定，其余 11 个数据集全部躲在它们的影子里跑完——挤进更多并发也省不到 1 小时，
> 却要把显存吃到 90% 以上。

---

## 4. 启动

```bash
cd /workspace/repo/CLS
mkdir -p results/step0_snr_gate results/labels_route_a          # 漏建 = nohup 重定向失败 = 整跑静默失败
tmux new -d -s stage1 \
  'python -u scripts/run_stage1.py 2>&1 | tee -a results/step0_snr_gate/stage1.log'
tmux ls
```

**怎么停**（本机实证：直接 kill 调度器**不会**带走 worker——它们是独立进程，
会继续跑到自己那份完为止）：

```bash
tmux kill-session -t stage1                       # 只停调度器（不再起新数据集）
pkill -f "scripts/cloud_run.sh"; pkill -f "step0_snr_gate.py --run"   # 停掉在跑的 worker
```

停了随时可以原样再 `python -u scripts/run_stage1.py` 起来：所有已落盘的 R'
按缓存键 `--resume` 复用，一条不重算。

调度器做三段：
1. **并发段**：按 `configs/runtime.yaml` 分档 + 显存预算准入，每数据集一个
   `cloud_run.sh` 看门狗进程（崩了自动 `--resume` 重拉、毒条目连崩 3 次永久隔离）；
   OOM 用 `--on-oom skip`（不写缓存、不计毒条目）。
2. **收尾段**：**worker=1 串行**逐数据集再跑一遍 `--resume --on-oom record`——
   并发段被 OOM 跳过的条目在这里独占显存重试；仍 OOM 才算终局失败，写
   `results/step0_snr_gate/oom_failures.jsonl`。
3. **判读**：`--analyze` 出 `analysis.json` + 任务数下限闸。

---

## 5. ⚠️ 启动后必做：确认真的在产出

闸门当天有过"整跑静默失败"（目录没建 → nohup 重定向失败 → 什么都没发生但看着像在跑）。
**下面两个检查点不做完，不要离开终端。**

### 检查点 A：启动后 3 分钟

```bash
cd /workspace/repo/CLS
echo "=== ① 调度器起了哪些数据集 ==="
tail -12 results/step0_snr_gate/stage1.log
echo "=== ② worker 进程数（应 = 上面"在跑 N 个"）==="
pgrep -af "step0_snr_gate.py --run" | wc -l
echo "=== ③ 线程环境（不是 1 就慢 4.2×）==="
grep -h "线程环境" results/step0_snr_gate/wd/*/watchdog.log
echo "=== ④ GPU 真在算 ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
echo "=== ⑤ worker 认领了工作量 ==="
grep -h "计划" results/step0_snr_gate/wd/*/worker_*_r0.log
```

**预期**：

| # | 预期输出 | 不对就是 |
|---|---|---|
| ① | `[stage1 ...] 起 SelfRegulationSCP2（L=1152，待跑 80 条，占用 15.8/24.0 GiB，在跑 1 个）` 之类 2 行 | 调度器没起来 → 看 stage1.log 尾部报错 |
| ② | **2**（与①的"在跑 N 个"一致） | 进程没起 / 起完就死 → 看对应 `wd/<ds>/watchdog.log` |
| ③ | 每数据集一行 `[watchdog] 线程环境: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` | **红线**：不是 1 就停，全程会慢 4.2× |
| ④ | 显存 ≈ ①报的"占用"（±20%，即 ~23 GiB）；利用率 >0 | 显存 ~0 = 根本没在跑；显存远超预估 = 立刻停，模型估错了 |
| ⑤ | 每数据集一行 `[base] 设备 cuda:0 \| iters=290 \| 计划 80 条，待跑 80` | `待跑 0` 全场 = 缓存被误当成跑完了（检查有没有把 smoke_cache 放错位置） |

### 检查点 B：启动后 15 分钟（真的有 R' 落盘）

```bash
cd /workspace/repo/CLS
echo "完成行数："; grep -hc "R'=" results/step0_snr_gate/wd/*/worker_*.log
echo "缓存行数："; wc -l results/labels_route_a/rprime_cache.jsonl results/step0_snr_gate/noise_cache.jsonl
a=$(cat results/step0_snr_gate/noise_cache.jsonl results/labels_route_a/rprime_cache.jsonl 2>/dev/null | wc -l)
sleep 300
b=$(cat results/step0_snr_gate/noise_cache.jsonl results/labels_route_a/rprime_cache.jsonl 2>/dev/null | wc -l)
echo "5 分钟新增 $((b-a)) 条（前期只有 SCP2 124s/条 + MotorImagery 59s/条 两个进程，预期 5-9 条）"
```

**预期**：15 分钟时完成行数 ≥ 10（`rprime_cache.jsonl` 从 20 涨到 ~30）；
5 分钟增量 **≥3 条**。

**红线（见到就停）**：

- 增量为 **0** 且进程还在 → 卡死；看 `wd/<ds>/worker_*.log` 尾部。
- `results/step0_snr_gate/oom_failures.jsonl` 在**并发段就出现** → 不该有（并发段是
  skip 模式不写它）；出现说明有人手滑用了 `--on-oom record`。
- 某个 `wd/<ds>/` 下 `worker_*_r3.log` 及以上（看门狗反复重拉）→ 该数据集在崩，
  `cat wd/<ds>/watchdog.log` 找 rc 和毒条目计数。
- `grep -c "OOM" results/step0_snr_gate/wd/*/worker_*.log` 持续增长 → 中档 worker=2
  的保守值不够，把 `configs/runtime.yaml` 的 `gpu.budget_gib` 调小重启调度器
  （已完成的条目全部 `--resume` 复用，不浪费）。

---

## 6. 持续监控（每隔几小时看一眼）

```bash
cd /workspace/repo/CLS
python scripts/progress_route_a.py          # ①过闸 ②存活任务数 ③label 进度 ④看门狗/OOM
tail -5 results/step0_snr_gate/stage1.log   # 调度器：谁跑完了、还剩几个
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

`progress_route_a.py` 的④已经把 `wd/<数据集>/` 的重启次数、毒条目、inflight 残留和
`oom_failures.jsonl` 条数汇总在一起——**这四个数任何一个非 0 都要看一眼原因**。

**预计完成时间：约 8.1 小时**（+ 天花板加难重探每触发一轮 +80 条/数据集；
最坏全触发 ≈ +2 h）。前 2.7 h 主要在 SCP2，看到"起 EthanolConcentration"就说明
过了第一根长杆。

---

## 7. 收尾

1. 调度器自己会跑收尾段和 `--analyze`。看到 `stage1.log` 里 `判读 --analyze` 之后
   打印出各数据集 verdict 即完成。
2. **先确认 `oom_failures.jsonl` 为空或已处置**——里面的条目是没有 R' 的空洞，
   会影响 SNR 判读的样本量。
3. 判读按 RUNBOOK §8（逐数据集分类 → 恢复档写回 `configs/source_pool.yaml` →
   存活任务数 ≥100 闸 → per-dataset δ 微调）。
4. 备份按 RUNBOOK §9.2（**方案 C 下是强制步骤**），产物 <5 MB：

```bash
cd /workspace/repo/CLS
tar -czf /workspace/stage1_artifacts.tar.gz \
    results/step0_snr_gate/{analysis.json,noise_cache.jsonl,tasks_meta.json,tasks_meta.d,oom_failures.jsonl,stage1.log} \
    results/labels_route_a/rprime_cache.jsonl \
    results/step0_snr_gate/wd/*/watchdog.log
runpodctl send /workspace/stage1_artifacts.tar.gz
```

5. **本地收到并解开确认无误后**，再 terminate pod。

## 8. 成本

`8.1 h × 单价`（Community 5090 on-demand，部署页现看现填）。闸门那趟 ~1 h 已花。
重探全触发的最坏情况 ~10 h。
