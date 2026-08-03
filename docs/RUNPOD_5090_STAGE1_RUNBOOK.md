# RunPod RTX 5090 阶段1 执行手册（含 Blackwell 兼容性闸门）

> # ⛔ 2026-08-03 订正：**4 卡前提已被闸门标定推翻，阶段1 走单卡**
> #
> # `docs/GATE_5090_REPORT.md` §1.9 实测：本负载 **CPU-bound**
> # （增强在 CPU 上做，GPU 只跑 encoder 前向，单 worker GPU 利用率 5.7%），
> # **1 张卡 8 worker（15.2 条/分）> 4 张卡各 1 worker（11.6 条/分）**；且 4 卡会抢
> # 同一份 cgroup 54.4 核 CPU。
> #
> # | 章节 | 状态 |
> # |---|---|
> # | §1（闸门）、§3（环境与数据）、§8（判读）、§9（结束与成本口径） | ✅ 仍有效，照做 |
> # | §0.6（4 卡共享目录冲突） | ✅ 问题真实，但已**在代码里修掉**（tasks_meta 改成每数据集一份 + `--inflight`），不必再开 4 份工作副本 |
> # | **§0.7 成本表** | ❌ **作废**：基于"GPU 是瓶颈、R 是常数"的错误前提。真实 R 随 L 与并发数变化，见 GATE 报告 §3 |
> # | **§2（4 卡实例）、§4（4 卡调度）、§6（启动）、§7（监控/合并分片）** | ❌ **被 `docs/RUNPOD_5090_STAGE1_LAUNCH.md` 取代**（单卡定稿执行卡） |
> #
> # **照 `docs/RUNPOD_5090_STAGE1_LAUNCH.md` 操作**；本文件保留作闸门/环境/判读的参考。

> 配套 `docs/superpowers/plans/2026-07-31-route-a-cloud-execution.md` 的 Part B 阶段1。
> Part A 四件套已交付（commit `4d5061a` / `6debed3`），checklist C1–C8 全绿。
>
> **与 `docs/RUNPOD_STAGE1_RUNBOOK.md`（A100 版）的关系**：那份是同一件事的 A100 +
> Secure Cloud 版本，判读逻辑（§6/§7）通用，**硬件、存储、耗时、成本全部作废**。
> 本手册是 **4×RTX 5090 + Community Cloud + On-Demand** 的定稿版，自成一体，
> 照本手册操作即可，不必来回翻另一份。
>
> **平台配置（已定）**：RunPod / On-Demand / Community Cloud / 4×RTX 5090。
> **新增的第 0 步**：Blackwell 兼容性 smoke 闸门 —— 不过不许进阶段1。

---

## 目录与执行顺序

```
§0  开工前必读：五个阻断项 + 耗时/成本重算          ← 先读完再花钱
§1  【第0步·闸门】1×5090 兼容性 smoke（~1 小时，~$1）  ← 本次最大风险
     ├ 1.4 四检：认卡 / sm_120 / kernel 实跑 / 装完依赖后没被降级
     ├ 1.6 数值一致性：与本地 4060 的 smoke_cache 逐键对比
     ├ 1.7 SVM(GridSearchCV) + BrokenProcessPool 兜底演练
     └ 1.8 判定表：全过 → §2；任一不过 → 停，按症状处置
§2  【第1步】4×5090 实例 + 存储（Volume 规划）
§3  【第2步】环境与数据（铁律 6 物理隔离）
§4  【第3步】4 卡并行调度（本地只验过单卡，这里是新东西）
§5  【第4步】看门狗（on-demand 简化版）
§6  【第5步】启动阶段1
§7  【第6步】进度监控
§8  【第7步】阶段1 判读清单（阶段2 决策依据）
§9  【第8步】结束操作与成本
附   检查清单
```

---

# §0 开工前必读

## 0.1 【最大风险】Blackwell sm_120：你本地的 PyTorch 一定跑不了 5090

这不是"可能不兼容"，是**已经实测确认的不兼容**。刚在你本机跑的：

```
$ python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
2.10.0+cu126 12.6 ['sm_50','sm_60','sm_61','sm_70','sm_75','sm_80','sm_86','sm_90']
                                                                          ^^^^^^
                                                        到 sm_90 为止，没有 sm_120
```

RTX 5090 是 Blackwell，compute capability **12.0（arch 字符串 `sm_120`）**。你本地这个
`2.10.0+cu126` 的 wheel 里**根本没有编译 sm_120 的 kernel**——注意版本号 2.10 已经足够新，
**决定性的不是 torch 版本号而是它编译时链的 CUDA**：`cu126` 这条产品线不含 Blackwell，
必须是 **cu128 及以上**的 wheel。

所以：

- **系统 CUDA 13.2 支持 Blackwell 与本事无关**。宿主机驱动/CUDA 只决定"能不能跑"，
  真正决定"有没有 sm_120 kernel"的是 **PyTorch wheel 自带的那份 CUDA runtime**。
- 缺 sm_120 时的表现**不是启动即报错**，而是：`torch.cuda.is_available()` 返回 `True`、
  `get_device_name()` 也能打印出 "NVIDIA GeForce RTX 5090"，**直到第一个 kernel 真的下发**
  才炸 —— 典型报错见 §1.4 判据表。这就是为什么 §1.4 的第三检必须**真跑一次矩阵乘和卷积**，
  光看 `is_available()` 是骗人的。

**结论**：RunPod 上必须选官方 Blackwell 兼容镜像（CUDA 12.8+ / torch ≥2.7 的 cu128 系），
且**装完依赖后要复查 torch 没被换掉**（§0.2）。这条链任何一环断了，你会在 4 张卡计费
的情况下调试环境——这是最贵的调试方式。

## 0.2 【阻断】`requirements.txt` 会不会把 PyTorch 降级？——分三种情况

先把事实摆清楚（`requirements.txt` 全文只有 5 行）：

```
numpy / pyyaml / pytest / torch / scikit-learn      # 全部无版本 pin
```

| 情况 | 会发生什么 | 风险 |
|---|---|---|
| 镜像**已装** torch（RunPod PyTorch 模板都是） | pip 看到 `torch` 这个**无版本约束**的需求**已被满足**，**不会重装** | 🟢 torch 本身安全 |
| numpy / scikit-learn 无 pin | pip 可能升/降它们到与镜像内 torch 编译期不一致的版本 | 🟡 `import torch` 可能报 numpy ABI 错 |
| 在**没有** torch 的干净 env 里跑它 | pip 从 **PyPI 默认源**装 torch —— 默认源的 wheel **不保证**是 cu128 系 | 🔴 **可能直接装成没有 sm_120 的版本** |

还有一条更要命的：**`requirements.txt` 里根本没有 `aeon`**（`autocls/data/uea.py:120`
是惰性 `from aeon.datasets import load_classification`），所以它**本来就不足以**把环境装齐。

**规矩（照做，别偷懒）**：

1. **永远不要在 pod 上跑 `pip install -r requirements.txt`。** 它既不完整、又无 pin。
2. 只装增量包，**逐条显式指定版本**（与本机基线对齐，见 §3.4）。
3. **每一条 `pip install` 前后各打一次 torch 指纹**（§1.5 的 `tfp` 函数），指纹变了立刻停。
   变的形式包括：版本号变、`cu128` 变 `cu126`、`sm_120` 从 arch_list 里消失。

本机基线（对齐用）：`numpy 2.2.6 / scipy 1.15.3 / pandas 2.3.3 / numba 0.61.2 /
scikit-learn 1.7.2 / aeon 1.3.0 / PyYAML 6.0.3 / joblib 1.5.3`。
**唯独 torch 不对齐**——本机的 `2.10.0+cu126` 在 5090 上是废的，pod 上用镜像自带的 cu128 系。

> `aeon` 版本要对齐 1.3.0：`autocls/data/uea.py` 对 CharacterTrajectories 里 21/1422 例
> "同 case 内各通道长度不一"的 ragged 数据写了容错解析分支，是针对 aeon 的 fail-fast
> 行为写的（memory `route-a-code-adaptation-step0`）。换大版本可能改行为。

## 0.3 【阻断】Community Cloud 与 Network Volume 的冲突

你要的组合是 "Community Cloud + Network Volume 跨阶段复用"。**这个组合在 RunPod 上
历史上不成立**——Network Volume 只存在于 Secure Cloud 数据中心，Community Cloud 的
Pod 挂不上。详细取舍与当前核实结果见 **§2.1**，那里给了三个方案和推荐。

先记住一句话：**在拿到 §2.1 的结论之前，不要把任何"删了 pod 也不能丢"的东西
只存在 pod 本地盘上。**

## 0.4 【阻断】本仓库没有 git remote，pod 上不能 `git clone`

```
$ git remote -v
（空）
```

必须打包传输。**用 `git archive`**（只打包 git 跟踪的文件，543 KB），它同时是铁律 6 的
天然防护：`data/` 整个目录未被跟踪，所以打包结果里**物理上不可能**混进 Epilepsy /
FaceDetection / har.npz。见 §3.1。

## 0.5 【阻断】`probe_cls_catalog.json` 未被 git 跟踪

```
$ git ls-files results/
results/route_a_pool/REPORT.md        ← 只有这一个
```

而 `scripts/step0_snr_gate.py:92` 要读 `results/route_a_probe/probe_cls_catalog.json`
（10 个 shared 探针 CLS 的唯一真源，也是阶段2 canonical 复用的前提）。
**必须单独上传**，否则 step0 第一秒就 `FileNotFoundError`。6.8 KB。

## 0.6 【阻断】`step0_snr_gate.py` 没有 `--outdir`，4 卡共享目录会互相覆盖

`scripts/step0_snr_gate.py:87-91` 把 `OUTDIR` 硬编码成 `<repo>/results/step0_snr_gate`：

- **`tasks_meta.json` 被覆盖（致命）**：每个 worker 写的是自己那份完整字典，只含自己
  分到的数据集。4 个 worker 写同一路径 → 最后一个赢 → `--analyze` 对另外 3/4 的数据集
  报"无 tasks_meta 记录，跳过"，**天花板重探记录全丢，判读直接残废**。
- `inflight.txt` 互相覆盖 → 看门狗毒条目归因错乱（不致命，但失去保护）。
- `noise_cache.jsonl` / `rprime_cache.jsonl` 是追加写，单行实测 ≤706 B < 4096 B 的
  POSIX 原子追加边界，并发追加**安全**——但上面两条已足以否决共享目录。

**解决（零代码改动）：每张卡一份独立的仓库工作副本**，各自 `results/` 互不干扰，跑完
再合并。`ROOT` 由 `__file__` 推导，副本天然隔离。见 §4.1 与 §7.2 的合并脚本。

## 0.7 ❌【已作废】耗时与成本：5090 的倍率是**未知数**，靠 §1.9 标定确定

> **本节整节作废**（2026-08-01 标定结论）。下面这张表的四行都建立在"GPU 是瓶颈、
> 倍率 R 是一个常数"的前提上，而实测是 **CPU-bound**：R 随 **L**（2.67× @L=51 →
> 5.4× @L=1152）与**并发数**变化，不存在单一 R。真实分档吞吐见 GATE 报告 §3，
> 单卡排程与 ETA 由 `python scripts/run_stage1.py --dry-run` 现算（读
> `configs/runtime.yaml` 的实测锚点）。保留原文仅供追溯当时的推理。

阶段1 的工作量是固定的：**13 个数据集 × (2 tasks × 10 CLS × 4 seeds) = 1040 条**
`(task, cls, seed) → R'`，每条 `early_iters=290`（`configs/train.yaml` 唯一真源）。

用探针校准过的公式 `s_4060 ≈ 84.2 × L_eff/128` 秒/条做**长度加权**（不是按均值算——
长序列端占大头，平均法会严重低估），**基线 = 78.2 个 RTX-4060-Laptop-GPU-小时**：

| 数据集 | 角色 | L_eff | 秒/条(4060) | **4060-h(80条)** | 可重探轮数 |
|---|---|---|---|---|---|
| SelfRegulationSCP2 | 源 | 1152 | 757.8 | **16.84** | 0（[2,2] 触顶） |
| EthanolConcentration | reserve | 1024 | 673.6 | **14.97** | 1 |
| SelfRegulationSCP1 | 源 | 896 | 589.4 | **13.10** | 0 |
| MotorImagery | 源 | 512 | 336.8 | **7.48** | 0 |
| Heartbeat | reserve | 405 | 266.4 | **5.92** | 0 |
| HandMovementDirection | 源 | 400 | 263.1 | **5.85** | 1 |
| UWaveGestureLibrary | 源 | 315 | 207.2 | **4.60** | 2 |
| CharacterTrajectories | 源 | 182 | 119.7 | **2.66** | 2 |
| Handwriting | 源 | 152 | 100.0 | **2.22** | 2 |
| ArticularyWordRecognition | 源 | 144 | 94.7 | **2.11** | 2 |
| ERing | 源 | 65 | 42.8 | **0.95** | 2 |
| NATOPS | 源 | 51 | 33.5 | **0.75** | 2 |
| FingerMovements | reserve | 50 | 32.9 | **0.73** | 0 |
| **合计 13 个** | | | | **78.2 4060-h** | 最坏重探 +47.4 |

换成 5090 需要一个倍率 R = `4060 秒/条 ÷ 5090 秒/条`。**这个倍率不能查规格表推**——
本项目的模型很小（10 层膨胀 CNN，bs=128，L≤1152），大概率是 **kernel 启动延迟 / 访存**
受限而非 FLOPs 受限，纸面 TFLOPS 比值（5090 ≈ 9× 4060 Laptop）会**严重高估**实际加速比。
所以下面这张表是"标定完了对号入座"用的：

| 实测倍率 R | 阶段1 基线 GPU-h | **4 卡墙钟（基线）** | 预期（+25% 重探） | 最坏（重探全触发） |
|---|---|---|---|---|
| 2.5×（保守，= A100 假设值） | 31.3 | 7.8 h | **9.8 h** | 12.6 h |
| **3×** | 26.1 | 6.5 h | **8.1 h** | 10.5 h |
| **4×（我的中位预期）** | 19.5 | 4.9 h | **6.1 h** | 7.9 h |
| 6×（乐观） | 13.0 | 3.3 h | **4.1 h** | 5.2 h |

**这是 4×5090 相对 4×A100 的核心优势**：A100 版手册按 2.5× 估出 9–12 h；5090 若跑到
4× 就是 6–8 h，而单卡价格还更低。但**必须先量**——§1.9 的标定就是干这个的，20 分钟，
在闸门那张卡上顺手做掉。

**为什么 4 张 5090 对本任务够用**（不必纠结 A100）：

- **数据并行、卡间零通信**：4 个 worker 各自跑各自的数据集分片，没有 DDP、没有 NCCL、
  没有梯度同步 → NVLink / SXM 溢价对本任务**一点用没有**。
- **显存不是瓶颈**：路线 B 在 8 GB 的 4060 Laptop 上就跑得动；5090 有 32 GB，
  最长的 SCP2（L=1152, bs=128, 7 通道）连零头都占不满。
- **FP32/TF32 小卷积上 5090 不输 A100**：这类小模型吃的是时钟和 L2，不是 HBM 带宽。

---

# §1 【第0步·闸门】1×5090 兼容性 smoke

**目的**：在只花 ~$1 的前提下，确认这套代码在 Blackwell 上**能跑、跑得对、跑得快**。
**不过闸不许开 4 卡。**

**为什么值得单独走一趟**：兼容性问题是"要么全好要么全坏"的，1 张卡验证和 4 张卡验证
结论完全一样，但 4 卡调环境的烧钱速度是 4 倍。整个第 0 步 ≈ 1 小时单卡。

## 1.1 开 1×RTX 5090 实例

控制台 → **Pods** → **Deploy**：

| 项 | 选择 | 理由 |
|---|---|---|
| **GPU** | **RTX 5090 ×1** | 闸门只需 1 张；4 卡留到 §2 |
| **Cloud Type** | Community Cloud | 与正式跑一致，顺便验证 Community 的机器状况 |
| **Pricing** | On-Demand | 已定 |
| **Template** | **必须是 CUDA 12.8+ 的官方 PyTorch 模板**，见下方判据 | Blackwell 唯一硬要求 |
| **Container Disk** | 30 GB | smoke 只用几百 MB |
| **Volume / Mount Path** | `/workspace` | 习惯保持一致 |

**模板怎么选（不认名字，认版本号）**：RunPod 官方 PyTorch 模板的 tag 形如
`runpod/pytorch:<torch>-py3.<x>-cuda<ver>-cudnn-devel-ubuntu<ver>`。判据只有两条：

1. tag 里的 **cuda ≥ 12.8**（`cuda12.8.1` / `cuda12.9` / `cuda13.x` 都行；`cuda12.4`
   `cuda12.6` **一律不要**）；
2. tag 里的 **torch ≥ 2.7**。

选好之后**把模板全名抄进 §9.5 的记录表**。⚠️ **阶段2 必须用同一个镜像重开 pod**：
阶段1 产出的 canonical R' 会被阶段2 逐键复用（不重算），若两阶段 torch/cuDNN 版本不同，
缓存里的 R' 和现算的 R' 来自不同数值环境，会在同一批 label 里混进系统性偏移。
memory `iters-protocol-baseline-and-gpu-nondeterminism` 记的"GPU 重跑 ±0.005 内不算回归"
说的是**同环境**重跑，**不覆盖跨版本**。

## 1.2 传代码 + 探针目录 + 两个小数据集

闸门只需要 **NATOPS 和 ERing** 两个数据集（它们是本地已有 4060 参考结果的那两个，
见 §1.6），几十 MB，几秒钟就传完。

**本地（Windows PowerShell）**：

```powershell
cd D:\DevelopDoc\CLS

# ① 代码（543 KB；data/ 与 results/ 均未被跟踪 → 包里物理上没有任何数据集）
git archive --format=tar.gz -o "$env:TEMP\cls_repo.tar.gz" HEAD
git log --oneline -1                      # 抄下 commit，pod 上要核对

# ② 探针 CLS 目录（未被跟踪，必须单独带，见 §0.5）
Copy-Item results\route_a_probe\probe_cls_catalog.json "$env:TEMP\"

# ③ 闸门用的两个数据集
tar -czf "$env:TEMP\uea_smoke2.tar.gz" -C D:\DevelopDoc\CLS\data\uea_cache NATOPS ERing

# ④ 本地 4060 参考结果（§1.6 数值对比用）——改名，绝不能叫 smoke_cache.jsonl
Copy-Item results\step0_snr_gate\smoke_cache.jsonl   "$env:TEMP\smoke_ref_4060.jsonl"
Copy-Item results\step0_snr_gate\analysis_smoke.json "$env:TEMP\analysis_smoke_4060.json"
```

> ⚠️ **参考文件必须改名**。若把它原名放进 pod 的 `results/step0_snr_gate/`，
> `--resume` 会认为 90 条全跑完了，直接跳过 —— 你会拿到一个"完美一致"的假结果。
> 参考文件只放 `/workspace/data/`，**永远不进 `results/`**。

**pod 上接收**（`runpodctl` 在 pod 里预装；本地**从 GitHub Releases 取**——
`winget` 仓库无此包，闸门当天实证，取 v2.8.0 的 `runpodctl-windows-amd64.exe`：
<https://github.com/runpod/runpodctl/releases>）：

```bash
mkdir -p /workspace/{repo,data,logs}
cd /workspace/data
# 本地: runpodctl send %TEMP%\cls_repo.tar.gz   → 得到配对码
# pod : runpodctl receive <配对码>
#       ↑ 五个文件各来一次：cls_repo.tar.gz / probe_cls_catalog.json /
#         uea_smoke2.tar.gz / smoke_ref_4060.jsonl / analysis_smoke_4060.json

mkdir -p /workspace/data/uea_cache
tar -xzf uea_smoke2.tar.gz -C /workspace/data/uea_cache && rm -f uea_smoke2.tar.gz
ls /workspace/data/uea_cache            # → NATOPS  ERing

mkdir -p /workspace/repo/CLS && tar -xzf cls_repo.tar.gz -C /workspace/repo/CLS
cd /workspace/repo/CLS
# ⚠️ results/step0_snr_gate 必须先建：§1.6.3 的 nohup 重定向先于 cloud_run.sh 的
#    mkdir 求值，目录不存在 → 重定向失败 → 整跑静默失败（闸门当天踩过）
mkdir -p data results/route_a_probe results/step0_snr_gate results/labels_route_a
ln -sfn /workspace/data/uea_cache data/uea_cache
cp /workspace/data/probe_cls_catalog.json results/route_a_probe/
git log --oneline -1 2>/dev/null || echo "(git archive 无 .git，用 §1.2 抄下的 commit 核对文件即可)"
```

## 1.3 依赖安装：先拍指纹，再装，再复查

**先把 torch 指纹函数存好**（后面每一步都要用）：

```bash
cat > /workspace/tfp.sh <<'EOF'
tfp() {   # torch fingerprint —— 版本 | 编译CUDA | 有没有 sm_120
  python - <<'PY'
import torch
a = torch.cuda.get_arch_list()
print(f"{torch.__version__} | cu{torch.version.cuda} | "
      f"{'sm_120 ✅' if any(x.startswith('sm_120') for x in a) else 'NO sm_120 ❌'} | {a}")
PY
}
EOF
source /workspace/tfp.sh

echo "=== 装依赖之前 ==="; tfp
```

**记下这一行输出。** 然后装增量包（**注意：没有 `-r requirements.txt`**，见 §0.2）：

```bash
# --break-system-packages：Ubuntu 24.04 的 PEP 668 会拦下裸 pip install。
# 落点是 /usr/local/lib/python3.12/dist-packages（与镜像自带 torch 同处），
# 不碰发行版目录，安全。闸门当天实证必须加。
pip install --break-system-packages "aeon==1.3.0" "scikit-learn==1.7.2" "pyyaml==6.0.3" pytest
echo "=== 装依赖之后 ==="; tfp
```

**两次输出必须逐字符相同。** 不同就停下，按 §1.5 修。

`aeon` 本身不依赖 torch，所以 pip 不会去动 torch；真正的风险是 aeon 会拉
`numpy / scipy / pandas / numba`，其中 **numpy 的大版本变化能让 `import torch` 直接报
ABI 错**。所以再确认一次：

```bash
python -c "import numpy,scipy,pandas,numba,sklearn,yaml,aeon,torch;
print('numpy',numpy.__version__,'| scipy',scipy.__version__,'| pandas',pandas.__version__,
      '| numba',numba.__version__,'| sklearn',sklearn.__version__,'| aeon',aeon.__version__)"
# 本机基线：numpy 2.2.6 | scipy 1.15.3 | pandas 2.3.3 | numba 0.61.2 | sklearn 1.7.2 | aeon 1.3.0
```

numpy 大版本（2.x → 1.x 或反之）变了要警觉；小版本差异无所谓。

## 1.4 兼容性四检

存成 `/workspace/repo/CLS/compat_check.py`：

```python
"""RTX 5090 (Blackwell sm_120) 兼容性检查。全过返回 0，任一不过返回 1。"""
import sys, torch

FAILED = []
def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILED.append(name)

print("=" * 78)
print(f"torch {torch.__version__} | 编译 CUDA {torch.version.cuda} | "
      f"cuDNN {torch.backends.cudnn.version()}")
print("=" * 78)

# ── ① wheel 本身是不是 Blackwell 系 ──────────────────────────────
ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
cu = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
check("torch >= 2.7", ver >= (2, 7), f"{torch.__version__}")
check("编译 CUDA >= 12.8", cu >= (12, 8), f"cu{torch.version.cuda}（cu126 一定不行）")

# ── ② arch_list 里有没有 sm_120（缺了就注定 no kernel image）──────
arch = torch.cuda.get_arch_list()
check("arch_list 含 sm_120", any(a.startswith("sm_120") for a in arch), f"{arch}")

# ── ③ 卡认出来了吗 ──────────────────────────────────────────────
n = torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(n)]
caps = [torch.cuda.get_device_capability(i) for i in range(n)]
vram = [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1) for i in range(n)]
check("认出 RTX 5090", n > 0 and all("5090" in x for x in names), f"{n} 卡 {set(names)}")
check("compute capability 12.x", n > 0 and all(c[0] == 12 for c in caps), f"{caps}")
print(f"  [INFO] 显存: {vram} GiB/卡")

# ── ④ 真下发 kernel（前面全 PASS 也可能死在这里）────────────────
for i in range(max(n, 1)):
    d = f"cuda:{i}"
    try:
        x = torch.randn(4096, 4096, device=d)
        mm = (x @ x).sum().item()
        # 本项目真正的算子：膨胀 Conv1d + 反向 + AdamW 一步
        conv = torch.nn.Conv1d(24, 64, 3, padding=4, dilation=4).to(d)
        opt = torch.optim.AdamW(conv.parameters(), lr=1e-3)
        loss = conv(torch.randn(128, 24, 512, device=d)).pow(2).mean()
        loss.backward(); opt.step(); torch.cuda.synchronize(i)
        check(f"{d} matmul+膨胀Conv1d+反向+AdamW",
              True, f"matmul {mm:.3e} / loss {loss.item():.4f}")
    except RuntimeError as e:
        check(f"{d} matmul+膨胀Conv1d+反向+AdamW", False, f"{type(e).__name__}: {e}")

print()
if FAILED:
    print(f"❌ 不通过：{FAILED} —— 照手册 §1.5 / §1.8 处置，不要往下走")
else:
    print("✅ 兼容性四检全过")
sys.exit(1 if FAILED else 0)
```

跑：

```bash
cd /workspace/repo/CLS && python compat_check.py 2>&1 | tee /workspace/logs/compat_check.log
```

**判据表（对号入座）**：

| 现象 | 说明 | 动作 |
|---|---|---|
| 四检全 PASS | 环境正确 | 进 §1.6 |
| `编译 CUDA >= 12.8` FAIL（显示 cu126/cu124） | 镜像选错了，或被 pip 换过 | §1.5 升级 torch；或**换镜像重开 pod**（更干净） |
| `arch_list 含 sm_120` FAIL | 同上，wheel 里没有 Blackwell kernel | §1.5 |
| 前三检 PASS、第 ④ 检报 **`CUDA error: no kernel image is available for execution on the device`** | 经典架构不匹配 | §1.5 |
| 第 ④ 检报 **`sm_120 is not compatible with the current PyTorch installation`** | PyTorch 明确告诉你 wheel 太老 | §1.5 |
| `认出 RTX 5090` FAIL 但确实租的 5090 | 驱动/容器 GPU 透传问题，不是 torch 问题 | 联系 RunPod 支持 / 换一台机器，别自己修 |
| 第 ④ 检报 OOM | 不可能（32 GB 跑 4096² 矩阵） | 说明这台机器被别人占着，换机 |

## 1.5 若 PyTorch 被降级 / 本来就不对：单独升级，且不破坏其他依赖

**先诊断是哪种**：

```bash
pip index versions torch 2>/dev/null | head -3   # 可选
python -c "import torch;print(torch.__version__, torch.version.cuda)"
```

**修复命令**（从 PyTorch 官方 cu128 索引装，**只动 torch 及其 CUDA 运行时，不碰
numpy/sklearn/aeon**）：

```bash
source /workspace/tfp.sh
echo "修之前:"; tfp

# 关键三点：
#   --index-url 指向 cu128（不是默认 PyPI）
#   --upgrade --force-reinstall 确保真的换掉已有的 cu126
#   不带 torchvision/torchaudio —— 本项目不用，少装少出事
pip install --upgrade --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch

echo "修之后:"; tfp
python -c "import numpy,sklearn,aeon;print('依赖仍在:',numpy.__version__,sklearn.__version__,aeon.__version__)"
python compat_check.py
```

> **为什么不用 `--no-deps`**：Linux 上 torch 的 CUDA 运行时是一堆 `nvidia-*-cu12` pip 包，
> `--no-deps` 会把它们跳过，装出一个 import 就崩的 torch。让 pip 正常解析依赖，
> 它只会动 `nvidia-*` 和 `triton`，**不会碰 numpy/sklearn/aeon**（torch 对 numpy 无硬依赖）。
> 装完那条 `python -c "import numpy,sklearn,aeon"` 就是验证这一点。

> **如果修完还是没有 sm_120**：别再折腾了，**换镜像重开 pod** 更快更干净。
> 一小时的手工修复不值 $0.5 的重开成本，而且残留的半修状态会在阶段1 咬你。

## 1.6 数值一致性 smoke：与本地 4060 逐键对比

**这是闸门里最有价值的一检**——它不只验证"能跑"，还验证"跑出来的数是对的"。

### 1.6.1 为什么这个对比是干净的

`scripts/step0_snr_gate.py --run --smoke` 是**完全确定性**的：任务切片由
`configs/source_pool.yaml` 的固定 seed 决定，每条 R' 的权重初始化 seed 与
训练/评估划分 seed 都由 `run_seed_for(task_id, cls_key)` 派生并**写进缓存记录**。
所以本地 4060 那 90 条和 5090 上跑出的 90 条，**除了 GPU 架构以外一切相同**，
可以按 `(task_id, cls_key, seed, iters)` 逐键配对。

本地参考（已在 `results/step0_snr_gate/smoke_cache.jsonl`，90 条，`iters=30`）：

```
NATOPS_00        15 条    NATOPS+c5-6_00  15 条    NATOPS+c6-6_00  15 条   （n_eval=108）
ERing_00         15 条    ERing+c5-6_00   15 条    ERing+c6-6_00   15 条   （n_eval=90）
```

（`+cX-Y` 是天花板加难重探档的任务 id 标记 —— 所以这一跑**顺带把加难重探那条 GPU 路径
端到端验证一次**。）

### 1.6.2 容差怎么定：ACC 是量化的，"±0.005" 是不可能满足的

R' = 下游 SVM 的 ACC，NATOPS 的 `n_eval=108`、ERing 的 `n_eval=90`。**一个测试样本
翻转 = ACC 变化 1/108 = 0.0093**。也就是说 R' 只能按 ~0.01 的步长跳变，
"±0.005 容差"要么是 0 差异、要么必然超标——用它当判据只会误伤。

**正确的判据是三层**（脚本会全打出来）：

1. **逐条**：`|ΔR'|` 换算成"翻转了几个评估样本"。中位 ≤1 个样本、最大 ≤4 个样本 = 正常。
2. **系统性**：`ΔR'` 的**带符号均值**。随机散布（均值≈0）是正常的架构数值差异；
   **单边系统性偏移**（比如 5090 全体高 0.03）才是真信号，说明某个算子行为不同。
3. **决策级（最重要）**：重跑 `--analyze --smoke`，看**结论**是否一致 ——
   SNR、R̄、verdict 一样，就说明整条判读链在新架构上给出同样的决定。

本地 4060 的决策级参考（`results/step0_snr_gate/analysis_smoke.json`）：

| 数据集 | verdict | SNR | R̄ | 重探轮数 |
|---|---|---|---|---|
| NATOPS | `fail_unstable` | **1.23** | **0.9154** | 2 |
| ERing | `fail_mismatch` | **0.71** | **0.9644** | 2 |

> 注意：这两个 verdict 是 **30 iters 的 smoke 结论，不是过闸结论**，别当真结果用。
> 它们在这里只有一个用途：**当 5090 的数值指纹**。（NATOPS 的 SNR 从基础档 0.73 涨到
> 加难档 1.23 —— 天花板假设成立；ERing 两轮没救回来。真闸门是 §6 的 290 iters 全量。）

### 1.6.3 跑 + 对比

**用看门狗启动**（不是裸 `python`）——这样 §1.7 的 kill -9 演练可以直接在这一跑上做，
不用为演练额外跑一遍：

```bash
cd /workspace/repo/CLS

# ① 跑（4060 上单条 12–16 s × 90 条 ≈ 20 分钟；5090 上按倍率更短）
#    用 cloud_run.sh 起，好处：崩了自动 --resume 重拉，且 §1.7 的演练就在这一跑上做
STAGE=step0 STEP0_DATASETS="NATOPS ERing" EXTRA_ARGS="--smoke --device cuda:0" \
  MAX_RESTARTS=3 nohup bash scripts/cloud_run.sh \
  > results/step0_snr_gate/smoke_run.log 2>&1 &

#    ⏸ 此刻先去做 §1.7 的 kill -9 演练，做完回来等它跑完

# ② 等它跑完（看门狗打印 "worker 正常退出 rc=0"）
tail -f results/step0_snr_gate/smoke_run.log     # Ctrl-C 退出观察

# ③ 分析（决策级指纹）
python scripts/step0_snr_gate.py --analyze --smoke --datasets NATOPS ERing \
  2>&1 | tee /workspace/logs/smoke_analyze.log
```

对比脚本，存成 `/workspace/repo/CLS/compare_smoke.py`：

```python
"""5090 smoke 与本地 4060 参考逐键对比。用法：
   python compare_smoke.py /workspace/data/smoke_ref_4060.jsonl \
                           results/step0_snr_gate/smoke_cache.jsonl \
                           /workspace/data/analysis_smoke_4060.json \
                           results/step0_snr_gate/analysis_smoke.json
"""
import json, sys, statistics as st
from pathlib import Path

def load(p):
    out = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["task_id"], r["cls_key"], r["seed"], r["iters"])] = r
    return out

ref, new = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(ref) & set(new))
print(f"参考(4060) {len(ref)} 条 | 本机(5090) {len(new)} 条 | 可比 {len(common)} 条")
if len(common) < len(ref):
    print(f"⚠️ 本机缺 {len(ref)-len(common)} 条 —— 没跑完就别对比，先补跑")

rows, nan_new = [], []
for k in common:
    a, b = ref[k], new[k]
    if b["r_prime"] is None:
        nan_new.append((k, b.get("error")))
    elif a["r_prime"] is not None:
        rows.append((k, a["r_prime"], b["r_prime"], b["r_prime"] - a["r_prime"], b["n_eval"]))

if nan_new:
    print(f"\n❌ 5090 上有 {len(nan_new)} 条算成 NaN（4060 上是好的）—— 这是硬失败：")
    for k, e in nan_new[:5]:
        print(f"    {k}  error={e}")

ad = sorted(abs(r[3]) for r in rows)
signed = [r[3] for r in rows]
print(f"\n评估集粒度: " + " ".join(f"n_eval={n} → 1 样本={1/n:.4f}"
      for n in sorted({r[4] for r in rows})))
print(f"|ΔR'|      中位 {st.median(ad):.4f} | 均值 {sum(ad)/len(ad):.4f} | 最大 {max(ad):.4f}")
print(f"ΔR' 带符号 均值 {st.mean(signed):+.4f}  ← 接近 0 = 随机散布(正常)；"
      f"单边大 = 系统性偏移(危险)")
samp = [abs(r[3]) * r[4] for r in rows]
print(f"折算样本数 中位 {st.median(samp):.2f} 个 | 最大 {max(samp):.2f} 个")
big = [r for r in rows if abs(r[3]) * r[4] > 4]
print(f"翻转 >4 个评估样本的条数: {len(big)}/{len(rows)}")
print("\n差异最大的 5 条:")
for k, a, b, d, n in sorted(rows, key=lambda r: -abs(r[3]))[:5]:
    print(f"  {k[0]:<20}{k[1][:8]} s{k[2]}  4060 {a:.4f} → 5090 {b:.4f}  "
          f"Δ={d:+.4f} ({abs(d)*n:.1f} 样本)")

try:
    from scipy.stats import spearmanr
    rho = spearmanr([r[1] for r in rows], [r[2] for r in rows]).statistic
    print(f"\nSpearman(4060, 5090) = {rho:.4f}   ← CLS 之间的排序是否被保住")
except Exception as e:
    rho = float("nan"); print(f"\n(Spearman 跳过: {e})")

# ── 决策级：verdict / SNR / R̄ 是否一致 ─────────────────────────
print("\n" + "=" * 74)
ra = json.load(open(sys.argv[3], encoding="utf-8"))["datasets"]
na = json.load(open(sys.argv[4], encoding="utf-8"))["datasets"]
dec_ok = True
print(f"{'数据集':<12}{'verdict(4060→5090)':<34}{'SNR':<18}{'R̄':<18}")
for d in sorted(set(ra) & set(na)):
    v = (ra[d]["verdict"], na[d]["verdict"])
    s = (ra[d]["snr"] or 0, na[d]["snr"] or 0)
    m = (ra[d]["rprime_grand_mean"] or 0, na[d]["rprime_grand_mean"] or 0)
    same = v[0] == v[1] and abs(s[0]-s[1]) <= 0.30 and abs(m[0]-m[1]) <= 0.02
    dec_ok &= same
    print(f"{d:<12}{v[0]+' → '+v[1]:<34}{s[0]:.2f} → {s[1]:.2f}{'':<8}"
          f"{m[0]:.4f} → {m[1]:.4f}   {'✅' if same else '❌'}")

ok = (not nan_new and len(common) == len(ref) and st.median(samp) <= 1.0
      and max(samp) <= 4.0 and abs(st.mean(signed)) <= 0.01
      and (rho != rho or rho >= 0.90) and dec_ok)
print("\n判定：" + ("✅ 数值一致性通过" if ok else "❌ 不通过 —— 见 §1.8 症状表"))
sys.exit(0 if ok else 1)
```

```bash
python compare_smoke.py /workspace/data/smoke_ref_4060.jsonl \
       results/step0_snr_gate/smoke_cache.jsonl \
       /workspace/data/analysis_smoke_4060.json \
       results/step0_snr_gate/analysis_smoke.json 2>&1 | tee /workspace/logs/compare_smoke.log
```

## 1.7 SVM(GridSearchCV) + BrokenProcessPool 兜底演练

`autocls/downstream/linear_eval.py:68` 的 `GridSearchCV(..., n_jobs=-1)` 用 joblib/loky
起进程池。memory `loky-orphans-after-sigkill` 记的坑：**worker 被 SIGKILL 后留下的孤儿
loky 子进程，会让重启后的进程报 `BrokenProcessPool: failed to un-serialize`**，
而 `evaluate_rprime` 会把它当成这条 (task,cls) 的属性、写永久 NaN 进缓存，`--resume`
不会重试 → **这条 label 永久丢失**。`fit_svm` 已经加了串行回退白名单
（`_POOL_ERRORS` → `n_jobs=1` 重试）。云上要验的是这个兜底在新环境仍然有效。

**顺手做：这一步和"看门狗 + resume 演练"是同一个动作**（kill -9 既制造孤儿 loky，
又制造进程级崩溃）。**就在 §1.6 那一跑还在进行时做**，不用额外跑一遍：

```bash
cd /workspace/repo/CLS

# ① 先确认 SVM 本身在新环境正常（10 秒，可与 smoke 并行）
python - <<'PY'
import numpy as np, time
from autocls.downstream.linear_eval import fit_svm
rng = np.random.default_rng(0)
X = rng.normal(size=(600, 320)).astype(np.float32); y = rng.integers(0, 6, 600)
t = time.time(); clf = fit_svm(X, y, seed=0)
print(f"fit_svm OK: C={clf.C} 耗时 {time.time()-t:.1f}s（GridSearchCV n_jobs=-1 正常）")
PY

# ② §1.6 那一跑起来约 90 秒后，把它 SIGKILL 掉（SIGKILL 不可 catch，最接近真实事故）
before=$(wc -l < results/step0_snr_gate/smoke_cache.jsonl)
echo "kill 前已完成 $before 条"
pkill -9 -f "step0_snr_gate.py --run --resume --datasets NATOPS ERing"
pgrep -af loky || echo "（没有 loky 孤儿）"

sleep 25    # 看门狗 SLEEP 默认 10s，会自动带 --resume 重拉
tail -20 results/step0_snr_gate/smoke_run.log
tail -30 results/step0_snr_gate/worker_*_r1.log
```

**演练不会浪费任何已完成的条目**——被杀之前落盘的 `before` 条都在缓存里，
重拉后 `--resume` 直接跳过。演练完让它自己跑完，再回 §1.6.3 的第 ②③ 步。

**要看到的三件事**：

| 证据 | 出现在哪 | 说明 |
|---|---|---|
| `[watchdog] 非零退出 rc=137（被杀/崩溃/抢占），将 --resume 续跑` | `smoke_run.log` | 看门狗识别到被杀 |
| `worker_*_r1.log` 里 "计划 N 条，**待跑 M 条**"，且 `M ≈ N - before` | `worker_*_r1.log` | resume 从缓存无损续跑 |
| **要么**完全没有 `BrokenProcessPool`，**要么**出现 `[fit_svm] 进程池损坏（BrokenProcessPool 类），串行 n_jobs=1 重试` 且该条**照常出 R'（不是 NaN）** | `worker_*_r1.log` | 兜底有效 |

**红线**：如果重启后出现 `r_prime: null` 且 `error` 里含 `BrokenProcessPool`
（说明兜底**没**接住），**这是硬失败**——4 卡并发下会大面积丢 label，必须先修。

演练完让 smoke 自己跑到 `rc=0`，回 §1.6.3 做 `--analyze` 和逐键对比
（缓存已被 resume 补齐，90 条一条不少）。

## 1.8 判定表：全过才进 §2

| 检查 | 通过标准 | 不过怎么办 |
|---|---|---|
| §1.4 ① wheel 是 cu128+/torch≥2.7 | 两项 PASS | §1.5 升级；修不好换镜像重开 pod |
| §1.4 ② arch_list 含 sm_120 | PASS | 同上 |
| §1.4 ③ 认出 RTX 5090 / cc 12.x | PASS | 不是软件问题 → 换机 / 报 RunPod |
| §1.4 ④ matmul+膨胀Conv1d+反向+AdamW 实跑 | PASS，无 `no kernel image` | 同上；这一检失败 = 前面全白搭 |
| §1.3 装依赖后 torch 指纹不变 | 前后两行逐字符相同 | §1.5；并检查是谁动的（`pip install` 输出里找 `Uninstalling torch`） |
| §1.6 逐条 \|ΔR'\| | 中位 ≤1 样本、最大 ≤4 样本 | 见下方症状表 |
| §1.6 ΔR' 带符号均值 | \|均值\| ≤ 0.01 | 见下方症状表 |
| §1.6 Spearman | ≥ 0.90 | 见下方症状表 |
| §1.6 决策级 | 两个数据集 verdict 完全一致、SNR 差 ≤0.30、R̄ 差 ≤0.02 | 见下方症状表 |
| §1.6 NaN | 5090 上**零**条 NaN（4060 上是好的那些） | 硬失败，查 `worker_*.log` 的 traceback |
| §1.7 resume | "待跑" 显著小于"计划" | 看门狗/缓存链有问题，先修（阶段1 全靠它） |
| §1.7 SVM 兜底 | 无 BrokenProcessPool，或有但被串行回退接住且出 R' | 硬失败，先修 |

**数值不一致的症状表**：

| 症状 | 最可能的原因 | 动作 |
|---|---|---|
| 散布大但均值≈0，Spearman 仍 ≥0.9，verdict 一致 | 正常的跨架构算子/规约顺序差异 | **放行**。这就是预期行为 |
| **单边系统性偏移**（均值 \|·\|>0.01） | 某个算子在 Blackwell 上走了不同精度路径（TF32/降精度） | 停。先试 `NVIDIA_TF32_OVERRIDE=0` 或在代码里关 `torch.backends.cudnn.allow_tf32` 重跑对比；仍偏则**不要用这个镜像** |
| Spearman < 0.9 但逐条差都很小 | R' 在 30 iters 下本来就挤在一起，秩很脆 | 看**决策级**：verdict/SNR/R̄ 一致就放行 |
| verdict 变了（如 NATOPS 从 `fail_unstable` 变 `pass`） | 判读结论被架构改变 = 最危险 | **停**。这意味着阶段1 的过闸结论不可信 |
| 5090 上出现 NaN | 算子崩溃被 `evaluate_rprime` 吞成 NaN | 停，去 `worker_*.log` 找 traceback |

> **本项目没有设置 `torch.backends.cudnn.benchmark` 或 `deterministic`**（全仓库
> grep 无命中），所以 cuDNN 走默认的启发式算法选择——跨架构选到不同算法是**预期的**，
> 这正是"逐条小差异"的来源，也是为什么判据要用决策级而不是逐条精确相等。

## 1.9 顺手做完标定（决定 §0.7 用哪一行）

smoke 跑的是 30 iters，**不能直接换算 290 iters**（有固定开销）。要拿真倍率，
在同一张卡上跑一次 **NATOPS 的真 iters step0**：

```bash
cd /workspace/repo/CLS
# 真 iters（290）、真配置；产物是正式 canonical R'，不浪费
time python scripts/step0_snr_gate.py --run --resume --datasets NATOPS --device cuda:0 \
  2>&1 | tee /workspace/logs/calib_natops.log

# 实测秒/条
grep -o '([0-9.]*s' /workspace/logs/calib_natops.log | tr -d '(s' | \
  python -c "import sys,statistics as st;v=[float(x) for x in sys.stdin];\
print(f'n={len(v)} 中位 {st.median(v):.1f}s/条  均值 {st.mean(v):.1f}s')"

# 同时另开一个窗口量利用率
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 5
```

**对号入座**：

| 实测中位秒/条 | 相对 4060（33.5 s/条）的倍率 R | 查 §0.7 的行 |
|---|---|---|
| ~13.4 s | 2.5× | 9.8 h（预期） |
| ~11.2 s | 3× | 8.1 h |
| ~8.4 s | 4× | 6.1 h |
| ~5.6 s | 6× | 4.1 h |

**GPU 利用率同样重要**：

| 利用率 | 结论 | 动作 |
|---|---|---|
| 持续 **<50%** | 单进程吃不满 5090（小模型的典型症状） | **每卡挂 2 个 worker**（§4.5），墙钟可望再砍 ~40%，卡价不变 → **直接省钱**。这是本次最大的免费杠杆 |
| 50–80% | 有余量但不多 | 可试 2 worker/卡，先在闸门这张卡上验一次 |
| >80% | 已吃满 | 保持 1 worker/卡 |

NATOPS 这一跑的产物是**正式的 canonical R'**（同样的缓存键），正式启动时把 NATOPS
留在它的分片里，`--resume` 会自动跳过已完成的部分——一秒不浪费。

## 1.10 闸门这一趟的耗时与花费

| 步骤 | 耗时 | 说明 |
|---|---|---|
| 开 pod + 传三个包 | ~10 min | 数据只有 NATOPS+ERing，几十 MB |
| §1.3 装依赖 + 指纹核对 | ~5 min | |
| §1.4 兼容性四检 | <1 min | |
| §1.6 smoke 90 条 @30 iters + 对比 | 4060 上约 20 min；5090 上按倍率更短 | |
| §1.7 kill-9 演练 + 补跑 | ~5 min | |
| §1.9 NATOPS 标定 @290 iters（80 条） | 预测 0.75 4060-h ÷ R ≈ **10–18 min** | |
| **合计** | **≈ 1 小时单卡** | 费用见 §9.1 |

**跑完立刻把 smoke 产物拉回本地**（几百 KB），然后 **terminate 这个 pod**：

```bash
cd /workspace/repo/CLS
tar -czf /workspace/gate_artifacts.tar.gz \
    results/step0_snr_gate/{smoke_cache.jsonl,analysis_smoke.json,tasks_meta.json} \
    results/labels_route_a/rprime_cache.jsonl \
    /workspace/logs/
runpodctl send /workspace/gate_artifacts.tar.gz     # 本地 runpodctl receive <码>
```

> ⚠️ `results/labels_route_a/rprime_cache.jsonl` 里是 §1.9 标定跑出的 **NATOPS canonical
> R'（正式产物）**。带回本地，阶段1 正式跑之前再传上去，`--resume` 就能直接复用，
> 省掉重跑 NATOPS 的那 10–18 分钟。

---

# §2 【第1步】4×5090 实例 + 存储

> **本节的价格数字**：RunPod 的 GPU 挂牌价按机器/数据中心浮动，**我这次没能联网核实
> 5090 的实时价**（研究子代理跑到一半撞了额度上限）。所以下面凡是 `$?` 的地方，
> 请在部署页现看现填 —— §9 的成本表设计成"填一个单价就出总额"，不依赖我猜的数。
> 与硬件无关的 RunPod 存储规则（Network Volume 只在 Secure Cloud、Volume Disk 在
> terminate 时删除等）是上一轮已核实过的，注明了出处。

## 2.1 【必须先解决】Community Cloud 与 Network Volume 冲突

你要的组合是 "**Community Cloud** + **Network Volume 跨阶段复用**"。这两个凑不到一起：

> RunPod 的 Network Volume 只存在于 **Secure Cloud** 数据中心，Community Cloud 的 Pod
> **无法挂载**（[RunPod 官方博客](https://www.runpod.io/blog/network-volumes-on-runpod-secure-cloud)、
> [docs](https://docs.runpod.io/storage/network-volumes)）。而 Pod 自带的 Volume Disk 在
> **terminate 时连同数据一起删除**，只在 **stop** 时保留（stop 状态按 $0.20/GB/月计，
> 比运行时的 $0.10 还贵）。

也就是说 "阶段1 跑完删 pod 省钱、阶段2 重起 pod 挂同一 Volume 复用缓存" 这条路，
**在 Community Cloud 上不存在**。三个可选方案：

| 方案 | 计算价 | 跨阶段持久化 | 代价 |
|---|---|---|---|
| A. Secure Cloud + Network Volume | 5090 在 Secure 的价 `$?`（通常比 Community 贵 30–40%） | ✅ Volume 独立于 pod，terminate 不丢 | 计算贵；且 Secure 不一定有 4×5090 存量 |
| **C（本手册推荐）Community Cloud + 每阶段人工备份** | Community 5090 `$?`（最便宜） | ✅ 靠人工 `runpodctl send`（产物 <5 MB） | 阶段2 要重传 0.86 GB 数据（几分钟） |
| B. Community + Volume Disk，只 stop 不 terminate | 同 C | ⚠️ 误点 terminate 就全丢；stop 期间 $0.20/GB/月；Community 主机可能被回收 | 便宜但脆，**不推荐** |

**推荐走 C，理由和 A100 版手册相反，值得说清楚**：

- A100 版当时推荐 A（Secure），是因为 A100 计算贵（4 卡 10h ≈ $75），"一次误点 terminate
  重跑一整夜"的损失远大于 Secure 的溢价。
- **5090 的算力便宜得多**（§9），同时**阶段1 的全部关键产物 <5 MB**（canonical R' 260 条 +
  噪声 780 条 + 两个 json），`runpodctl send` 两分钟就备份完。**重传数据也只有 0.86 GB。**
- 换句话说：5090 上"人工备份"的代价（几分钟）比"Secure 溢价 + 可能没有 4 卡存量"更划算。

**方案 C 的硬性纪律（不遵守就等于方案 B）**：

1. 阶段1 一结束，**先做 §9.2 的备份并确认本地收到**，再碰 pod 的任何按钮。
2. 判读期（§8，可能几天）**不留 pod**——直接 terminate，反正数据已在本地。
3. 阶段2 重开 pod 时按 §3 重传一次数据 + 把备份里的 `rprime_cache.jsonl` 放回
   `results/labels_route_a/`，`--resume` 逐键复用，一条都不重算。

> 如果你改主意要走 A（Secure + Network Volume）：先在部署页确认**目标数据中心有
> 4×5090 存量**再建 Volume——Network Volume 绑定单个数据中心，
> [官方文档](https://docs.runpod.io/storage/network-volumes)明说 "Network volumes must be
> attached during Pod deployment. They cannot be attached or detached later without
> deleting the Pod."，选错数据中心 = 盘挂得上、卡租不到。

## 2.2 存储容量估算

无论走哪个方案，需要的容量是一样的（**按方案 C 就是 pod 的 Volume Disk 大小**）：

| 内容 | 大小 | 说明 |
|---|---|---|
| `uea_cache` 16 个数据集 | **0.86 GB** | 实测；MotorImagery 一个就 512 MB |
| 仓库 5 份副本（1 主 + 4 分片） | ~3 MB | `git archive` 543 KB/份 |
| 镜像自带 torch/CUDA | 不占 Volume | 在 Container Disk 上 |
| 增量 pip 包（aeon/sklearn/numba…） | ~0.6 GB | 若装到 `/workspace/venv` 才占 Volume；装到系统 site-packages 则占 Container Disk |
| **阶段1 产物** | **< 5 MB** | canonical 260 条 + 噪声 780 条 × ~700 B + json |
| 阶段2 产物（预留） | ~0.9 GB | `task_features/*.npz` 实测 ~6.4 MB/任务 × 121 任务 ≈ 780 MB + ts2vec/set_encoder ~11 MB |
| 余量 | | |

**结论：Volume 30 GB**（两个阶段都够，不必中途扩）。想省可以开 20 GB，但省下的钱
（$0.07/GB/月 量级）远不值"阶段2 中途磁盘满导致崩溃"的风险。
**Container Disk 30–50 GB**（放镜像和 pip 包，pod 删除即失，**什么产物都别往这里写**）。

## 2.3 计费口径（三种盘别搞混）

| | 何时计费 | 停止条件 | 单价 |
|---|---|---|---|
| **计算（GPU）** | pod **Running** 时**按秒**计 | **Stop 即停** | `$?`/GPU-h ×4 |
| **Container Disk** | pod 存在就计 | **Terminate** 才停（数据同时删除） | 按 GB/月 |
| **Volume Disk**（pod 自带） | pod 存在就计 | **Terminate** 才停（**数据同时删除**） | 运行 $0.10/GB/月；**stop 期间 $0.20/GB/月** |
| **Network Volume**（仅 Secure） | 独立于 pod 生死 | 手动删除 Volume | $0.07/GB/月 |

⚠️ **官方警告：账户余额归零时 pod 会停，但存储费继续累计；欠费到扣不动时数据会被
销毁。** 这是 §2.1 坚持"产物必须落本地"的第二个理由。

## 2.4 部署 4×5090 Pod

控制台 → **Pods** → **Deploy**：

| 项 | 选择 | 理由 |
|---|---|---|
| **GPU** | **RTX 5090 ×4** | 数据并行，见下 |
| **Cloud Type** | Community Cloud | 已定（走方案 C） |
| **Pricing** | **On-Demand** | 已定；不会被抢占 → 看门狗只需管进程级崩溃（§5） |
| **Template** | **与 §1.1 闸门那台完全相同的镜像**（抄 §9.5 记录表） | 跨阶段 canonical R' 复用要求数值环境一致 |
| **Container Disk** | 30–50 GB | |
| **Volume Disk / Mount Path** | **30 GB** / `/workspace` | 所有产物必须落这里 |
| vCPU | **≥16，最好 ≥32** | 见 §4.4：4 个 worker 的 `GridSearchCV(n_jobs=-1)` 会抢核 |

**不要买 NVLink / 多卡互联溢价**：本任务是"每卡一个独立单卡进程"的分片并行，
**没有 DDP、没有 NCCL、卡间零通信**（§0.7）。

**开机第一件事：把 §1.4 的兼容性四检再跑一遍**（4 张卡都要过）。闸门是在另一台机器上
过的，这台是新机器、可能是不同宿主——`compat_check.py` 会遍历所有可见卡，10 秒的事。

---

# §3 【第2步】环境与数据（铁律 6 物理隔离）

## 3.1 本地打包（Windows PowerShell）

```powershell
cd D:\DevelopDoc\CLS

# ① 代码（543 KB；data/ 与 results/ 未被跟踪 → 包里物理上没有任何数据集）
git archive --format=tar.gz -o "$env:TEMP\cls_repo.tar.gz" HEAD
git log --oneline -1                      # 抄下 commit

# ② 探针 CLS 目录（§0.5，未被跟踪）
Copy-Item results\route_a_probe\probe_cls_catalog.json "$env:TEMP\"

# ③ 数据：显式白名单打包 16 个，绝不整目录 rsync
$sets = @(
  'ArticularyWordRecognition','CharacterTrajectories','Cricket','ERing',
  'EthanolConcentration','FingerMovements','HandMovementDirection','Handwriting',
  'Heartbeat','MotorImagery','NATOPS','PhonemeSpectra','SelfRegulationSCP1',
  'SelfRegulationSCP2','SpokenArabicDigits','UWaveGestureLibrary'
)
tar -czf "$env:TEMP\uea16.tar.gz" -C D:\DevelopDoc\CLS\data\uea_cache $sets

# ④ 闸门带回来的 NATOPS canonical R'（§1.10），阶段1 直接复用
#    （解压 gate_artifacts.tar.gz 后取 results/labels_route_a/rprime_cache.jsonl）
Copy-Item <解压路径>\results\labels_route_a\rprime_cache.jsonl "$env:TEMP\gate_canonical.jsonl"

# ⑤ 上传前自检：包里绝不能有目标池
tar -tzf "$env:TEMP\uea16.tar.gz" | Select-String -CaseSensitive -Pattern 'Epilepsy|FaceDetection|UCI-HAR|har\.npz'
#    ↑ 必须无输出。有输出就停下来查。
#    ⚠️ 别用不区分大小写的 'HAR' 当模式——会误命中 C-har-acterTrajectories，白虚惊。
tar -tzf "$env:TEMP\uea16.tar.gz" | ForEach-Object { ($_ -split '/')[0] } | Sort-Object -Unique
#    ↑ 必须恰好 16 行，且与 $sets 一致
```

> **三个骨干（Cricket / PhonemeSpectra / SpokenArabicDigits，`tested: true`）阶段1 的
> step0 不跑**，但阶段2 要用。现在一起传、一次到位（方案 C 下阶段2 反正要重传，
> 但保持 16 个清单一致能让 §3.3 的自检始终是同一条断言）。

## 3.2 传到 pod

```bash
# —— pod 上 ——
mkdir -p /workspace/{repo,shards,data,logs}
cd /workspace/data
# 本地: runpodctl send %TEMP%\<文件>   → 配对码
# pod : runpodctl receive <配对码>      （四个文件各来一次）

mkdir -p /workspace/data/uea_cache
tar -xzf uea16.tar.gz -C /workspace/data/uea_cache && rm -f uea16.tar.gz
ls /workspace/data/uea_cache | wc -l          # → 16，不是 16 就先别往下走
```

> 传不动/嫌慢的备选：pod 上 `pip install aeon` 后用
> `aeon.datasets.load_classification(name, extract_path=...)` 从 UEA 官方源下载
> ——**必须逐个点名下载这 16 个**，绝不能跑"下载全部 30 个"的循环（会把 Epilepsy 和
> FaceDetection 拉下来，直接违反铁律 6 的物理隔离）。0.86 GB 的 `runpodctl send`
> 几分钟就完，不值得为此冒险。

## 3.3 上传后自检（铁律 6 双保险，逐条核对输出）

```bash
# ① 数据集清单：必须恰好 16 行
ls /workspace/data/uea_cache | sort | tee /workspace/logs/uea_manifest.txt
wc -l < /workspace/logs/uea_manifest.txt      # → 16

# ② 目标池物理不存在：全盘搜，必须零输出
find /workspace -iname '*Epilepsy*' -o -iname '*FaceDetection*' \
     -o -iname 'har.npz' -o -iname 'UCI-HAR*' | tee /workspace/logs/leak_check.txt
[ -s /workspace/logs/leak_check.txt ] && echo "!!! 泄漏：上面有文件，停下来处理 !!!" \
                                      || echo "OK：目标池零残留（铁律 6 物理层通过）"

# ③ 每个数据集都有 TRAIN+TEST 两个 .ts
for d in /workspace/data/uea_cache/*/; do
  n=$(ls "$d"*.ts 2>/dev/null | wc -l); echo "$(basename $d) .ts=$n"
done          # 每行都应是 .ts=2

# ④ 总大小 ~0.86 GB
du -sh /workspace/data/uea_cache
```

期望的 16 行（源池 13 + reserve 3）：

```
ArticularyWordRecognition  CharacterTrajectories  Cricket                ERing
EthanolConcentration       FingerMovements        HandMovementDirection  Handwriting
Heartbeat                  MotorImagery           NATOPS                 PhonemeSpectra
SelfRegulationSCP1         SelfRegulationSCP2     SpokenArabicDigits     UWaveGestureLibrary
```

## 3.4 主副本 + 依赖 + preflight

```bash
mkdir -p /workspace/repo/CLS && tar -xzf /workspace/data/cls_repo.tar.gz -C /workspace/repo/CLS
cd /workspace/repo/CLS
mkdir -p data results/route_a_probe results/step0_snr_gate results/labels_route_a
ln -sfn /workspace/data/uea_cache          data/uea_cache
cp /workspace/data/probe_cls_catalog.json  results/route_a_probe/

# 依赖：照 §0.2 的规矩，绝不 pip install -r requirements.txt
source /workspace/tfp.sh 2>/dev/null || { cat > /workspace/tfp.sh <<'EOF'
tfp() { python - <<'PY'
import torch
a = torch.cuda.get_arch_list()
print(f"{torch.__version__} | cu{torch.version.cuda} | "
      f"{'sm_120 ✅' if any(x.startswith('sm_120') for x in a) else 'NO sm_120 ❌'}")
PY
}
EOF
source /workspace/tfp.sh; }
echo "装之前:"; tfp
pip install --break-system-packages "aeon==1.3.0" "scikit-learn==1.7.2" "pyyaml==6.0.3" pytest
echo "装之后:"; tfp        # ← 必须与"装之前"逐字符相同
```

**环境自检脚本**，存成 `/workspace/repo/CLS/preflight.sh`（五项一次跑完）：

```bash
#!/usr/bin/env bash
set -e
cd /workspace/repo/CLS
echo "===== ① Blackwell 兼容性（4 卡都要过）====="
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv
python compat_check.py
python - <<'PY'
import torch
n = torch.cuda.device_count()
assert n == 4, f"!!! 期望 4 卡，实际 {n} —— 停 !!!"
print(f"4 卡可见 ✅")
PY
echo "===== ② 单测冒烟（不吃 GPU）====="
python -m pytest tests/ -q
echo "===== ③ 目标池 fail-fast（铁律 6 代码层）====="
python - <<'PY'
from autocls.data.source_pool import load_source_dataset, TargetPoolLeakageError
for name in ("Epilepsy", "FaceDetection", "UCI-HAR"):
    try:
        load_source_dataset(name)
        raise SystemExit(f"!!! {name} 竟然加载成功 —— 铁律 6 防护失效，停 !!!")
    except (TargetPoolLeakageError, ValueError) as e:
        print(f"  {name}: 正确拒绝 ({type(e).__name__})")
PY
echo "===== ④ 探针 CLS 目录到位 ====="
python - <<'PY'
from pathlib import Path
from autocls.labeling import load_probe_cls
p = Path("results/route_a_probe/probe_cls_catalog.json")
assert p.exists(), "!!! probe_cls_catalog.json 缺失（见 §0.5）—— step0 会立刻崩 !!!"
cls, rank = load_probe_cls(p, 10)
print(f"  探针 CLS {len(cls)} 个，har_rank {len(rank)} 项")
PY
echo "===== ⑤ 闸门带回的 canonical 已就位（可选，省 10-18 min）====="
wc -l results/labels_route_a/rprime_cache.jsonl 2>/dev/null || echo "  （无，NATOPS 会重跑，不影响正确性）"
echo "===== 自检全绿 ====="
```

```bash
# 把闸门带回的 NATOPS canonical 放回原位（有的话）
cp /workspace/data/gate_canonical.jsonl /workspace/repo/CLS/results/labels_route_a/rprime_cache.jsonl 2>/dev/null || true
bash preflight.sh 2>&1 | tee /workspace/logs/preflight.log
```

**自检不全绿就不要开跑**——在 4 卡计费里排环境问题是最贵的调试方式。

---

# §4 【第3步】4 卡并行调度

> 本地只在单卡 4060 上验证过。这一节是本次唯一**全新**的执行风险，
> §4.6 给了"确认真的用上 4 张卡"的硬证据检查。

## 4.1 4 个分片工作副本（绕开 §0.6 的目录冲突）

```bash
cd /workspace/shards
for g in 0 1 2 3; do
  mkdir -p g$g/CLS
  tar -xzf /workspace/data/cls_repo.tar.gz -C g$g/CLS
  mkdir -p g$g/CLS/data g$g/CLS/results/route_a_probe g$g/CLS/results/labels_route_a
  ln -sfn /workspace/data/uea_cache            g$g/CLS/data/uea_cache
  cp /workspace/data/probe_cls_catalog.json    g$g/CLS/results/route_a_probe/
  cp /workspace/repo/CLS/compat_check.py       g$g/CLS/ 2>/dev/null || true
done
# 闸门带回的 NATOPS canonical 放进 NATOPS 所在的分片（g3）
cp /workspace/data/gate_canonical.jsonl \
   /workspace/shards/g3/CLS/results/labels_route_a/rprime_cache.jsonl 2>/dev/null || true
ls -d /workspace/shards/g*/CLS      # → 4 个
```

最终布局（**全部在 `/workspace` 上**）：

```
/workspace/
├── data/
│   ├── uea_cache/                       ← 16 数据集，4 副本共享（symlink）
│   ├── probe_cls_catalog.json
│   ├── gate_canonical.jsonl             ← §1.9 标定产物
│   └── cls_repo.tar.gz
├── repo/CLS/                            ← 主副本：跑 --analyze / progress / 合并
├── shards/g{0,1,2,3}/CLS/               ← 4 个 worker 各自的工作副本
│   └── results/
│       ├── step0_snr_gate/  {noise_cache.jsonl, tasks_meta.json, inflight.txt,
│       │                     watchdog_state.tsv, watchdog.log, worker_*.log}
│       └── labels_route_a/  {rprime_cache.jsonl}   ← canonical，阶段2 复用
└── logs/                                ← preflight / 泄漏自检 / 启动日志
```

> ⚠️ **任何东西都不要写在 `/workspace` 之外**。`/root`、`/home`、镜像默认工作目录都在
> Container Disk 上，pod terminate 即消失。`pip install --target` 和 `git clone` 很容易
> 顺手写到别处。

## 4.2 分片方案（按耗时配平）

分片权重与卡型无关（都是同一个比例缩放），所以下表用 **4060-h 基线** 表示，
再 **÷R**（§1.9 标定出的倍率）就是 5090 上的小时数：

| 分片 | GPU | 数据集 | 基线 4060-h | 重探上限 | R=4 时基线 | R=4 时预期 |
|---|---|---|---|---|---|---|
| **g0** | `cuda:0` | EthanolConcentration, ArticularyWordRecognition | 17.08 | +19.19 | 4.27 h | ~5.7 h |
| **g1** | `cuda:1` | SelfRegulationSCP2, Heartbeat, ERing, FingerMovements | 24.44 | +1.90 | 6.11 h | ~6.3 h |
| **g2** | `cuda:2` | SelfRegulationSCP1, MotorImagery, Handwriting | 22.80 | +4.44 | 5.70 h | ~6.0 h |
| **g3** | `cuda:3` | UWaveGestureLibrary, HandMovementDirection, CharacterTrajectories, NATOPS | 13.86 | +21.87 | 3.47 h | ~5.1 h |
| | | **合计 13 个** | **78.18** | **+47.4** | **19.5 h** | **~23 GPU-h** |

配平思路：把"重探可能性大但便宜"的（g3 的 UWave/CharTraj/NATOPS）和"重探不可能但贵"
的（g1 的 SCP2、g2 的 SCP1）搭在一起。**别指望四张卡同时结束**——一轮重探都不触发时
g3、g0 会明显先收工；重探全触发时次序反过来。

**哪张卡先跑完，就把最慢那张卡还没开始的数据集匀一个过来**：新建
`/workspace/shards/g4/CLS`（步骤同 §4.1），用空出来的 `--device cuda:N` 起一份新的
`cloud_run.sh`，`STEP0_DATASETS` 只写那一个数据集，同时从原分片的 tmux 里 `Ctrl-C` 掉。
**前提是那个数据集尚未开跑**（`watch.sh` 的"当前"列和 `tasks_meta.json` 能看出来）。
数据集之间完全独立、缓存各自落盘，按数据集整体搬运是安全的；**切勿把同一个数据集
同时交给两张卡**——会重复算，且两份 `tasks_meta.json` 都声称拥有它，§7.2 的合并断言
会报重复并中止。

## 4.3 启动脚本

`scripts/cloud_run.sh` 已经是为此写的，**on-demand 下不需要任何改动**。只调两个参数：

- **`MAX_RESTARTS=20`**（默认 50）：on-demand 不会被抢占，"反复崩"必然是真 bug 而不是
  环境抖动。50 次重启在长数据集上能白烧好几个 GPU-h。
- **`LOKY_MAX_CPU_COUNT` / `OMP_NUM_THREADS`**：见 §4.4。

存成 `/workspace/launch_stage1.sh`：

```bash
#!/usr/bin/env bash
# 阶段1 四卡启动。用法： bash /workspace/launch_stage1.sh
set -eu
command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }

NW=4; VCPU=$(nproc); NT=$(( VCPU / NW )); [ "$NT" -ge 1 ] || NT=1   # 写成 ||，别写 &&：
                                          # set -e 下 && 链整体返回非零会直接退出
echo "vCPU=$VCPU，$NW 个 worker → 每个限 $NT 核"

declare -A SHARD=(
  [0]="EthanolConcentration ArticularyWordRecognition"
  [1]="SelfRegulationSCP2 Heartbeat ERing FingerMovements"
  [2]="SelfRegulationSCP1 MotorImagery Handwriting"
  [3]="UWaveGestureLibrary HandMovementDirection CharacterTrajectories NATOPS"
)

for g in 0 1 2 3; do
  d=/workspace/shards/g$g/CLS
  mkdir -p "$d/results/step0_snr_gate"
  # ⚠️ 环境变量必须写在命令串里，不能靠外层 export：若 tmux server 已在运行，
  #    新 session 继承的是 **server 启动时**的环境，不是你当前 shell 的
  #    ——那样 LOKY_MAX_CPU_COUNT 会静默失效，CPU 超订而你看不出来。
  tmux new -d -s "step0_g$g" "cd $d && \
    LOKY_MAX_CPU_COUNT=$NT OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT \
    PY=python MAX_RESTARTS=20 FAIL_LIMIT=3 \
    STAGE=step0 STEP0_DATASETS='${SHARD[$g]}' EXTRA_ARGS='--device cuda:$g' \
    bash scripts/cloud_run.sh > results/step0_snr_gate/watchdog.log 2>&1"
  echo "g$g (cuda:$g) ← ${SHARD[$g]}"
done
sleep 3; tmux ls
```

**为什么用 tmux 而不是 `nohup ... &`**：两者都能扛 SSH 断开，但 tmux 让你随时
`tmux attach -t step0_g2` 看实时输出、`Ctrl-B D` 再脱离。RunPod 的 Web Terminal
会话尤其容易断，tmux 是必需品。

## 4.4 SVM 在 4 卡高并发下会不会放大 loky 问题

**会放大的是 CPU 超订，不是 BrokenProcessPool 本身。** 分清楚两件事：

**① CPU 超订（4 卡比单卡严重 4 倍，必须处理）**
`autocls/downstream/linear_eval.py:68` 的 `GridSearchCV(SVC, {C:[1,10,100]}, cv=3, n_jobs=-1)`
会按 `joblib.cpu_count()` 开满核。4 个 worker 同时到评估阶段 → 4×N 个进程抢 N 个核 →
**GPU 空转等 CPU**。`joblib` 认 `LOKY_MAX_CPU_COUNT`，设成 `总vCPU/worker数` 即可
（§4.3 已内置），**不用改代码**。

起完立刻验一下限流真的生效了（那个 tmux 环境陷阱只会在事后表现为"CPU 100% 但 GPU 空转"）：

```bash
tr '\0' '\n' < /proc/$(pgrep -f 'step0_snr_gate.py' | head -1)/environ | grep -E 'LOKY|OMP_NUM'
# 应看到 LOKY_MAX_CPU_COUNT=<vCPU/4> 和 OMP_NUM_THREADS=<同值>
```

**② BrokenProcessPool（触发条件与并发数无关）**
memory `loky-orphans-after-sigkill` 记的机制是：**某个 worker 进程被 SIGKILL** → 它的
loky 子进程变孤儿 → 干扰**重启后**的 GridSearchCV。触发源是"进程被杀"，不是"并发高"。
on-demand **不会被抢占**，所以这条路径在阶段1 的触发概率**比 spot 低得多**——
真被触发时（OOM-killer / 手滑）`fit_svm` 的串行回退白名单（`_POOL_ERRORS` → `n_jobs=1`）
会接住，**这条兜底已在 §1.7 的闸门里用真 kill -9 演练验证过**。

四卡下唯一的额外注意：**孤儿 loky 是全机器共享的**，g1 被杀留下的孤儿可能干扰 g2 的
GridSearchCV。所以看门狗每次重启后，顺手清一次：

```bash
# 加进日常巡检（§7.1），或在任何一次 worker 崩溃后手动跑一次
pgrep -af 'loky' | head
pgrep -f 'loky' | xargs -r kill -9 2>/dev/null; echo "孤儿 loky 已清"
#   ⚠️ 只在**确认某个 worker 刚崩过**时才清。正常运行中的 GridSearchCV 也有 loky 子进程，
#      无差别 kill 会把健康 worker 的评估打断（那条会走串行回退，只慢不错，但没必要）。
```

## 4.5 要不要每卡 2 个 worker

由 §1.9 标定测到的 **GPU 利用率**决定：

| 利用率 | 动作 |
|---|---|
| <50% | **建 8 个分片 g0–g7**，`--device cuda:$((g%4))`，把 §4.2 的 13 个数据集拆成 8 份。墙钟可望再砍 ~40%，卡价不变 → 直接省钱。同时把 `launch_stage1.sh` 的 `NW` 改成 8（CPU 限流跟着变） |
| 50–80% | 可试，但先只在一张卡上开双 worker 观察 10 分钟，确认两个进程的秒/条没有明显退化 |
| >80% | 保持 1 worker/卡 |

⚠️ 双 worker 时**两个 worker 必须在不同的分片副本目录**（§0.6 的理由同样适用），
且**同一个数据集绝不能分给两个 worker**。

## 4.6 确认真的用上了 4 张卡（不是只用 1 张）

启动后立刻核对这四条，缺一不可：

```bash
# ① 4 个 tmux session
tmux ls                                          # → step0_g0..g3

# ② 4 个 python 进程，分别占 GPU 0/1/2/3（关键：看 nvidia-smi 的进程表）
nvidia-smi
#    下半部分的 Processes 表应有 4 行，GPU 列分别是 0,1,2,3，都是 python
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
#    → 4 行，且 gpu_uuid 互不相同（同一个 uuid 出现两次 = 两个 worker 挤在一张卡上）

# ③ 每个 worker 各自在滚自己的数据集
for g in 0 1 2 3; do
  echo "--- g$g ---"; tail -2 /workspace/shards/g$g/CLS/results/step0_snr_gate/worker_*_r0.log
done
#    每个都应在滚  [base i/N] <数据集>/<task> <cls> s<seed> R'=0.xxxx (NNs, n_ev=...)
#    且 4 行里的 <数据集> 互不相同（属于各自分片）

# ④ 四张卡都真的在算（利用率不是 0）
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

**最常见的翻车姿势**：`EXTRA_ARGS='--device cuda:$g'` 里的 `$g` 没展开（引号用错），
四个 worker 全落在 `cuda:0` 上 → 第 ② 条会看到同一个 `gpu_uuid` 出现 4 次，
另外三张卡利用率 0。发现了就 `tmux kill-server`，检查 `launch_stage1.sh` 的引号再来。

---

# §5 【第4步】看门狗（on-demand 简化版）

on-demand **不会被抢占**，所以看门狗只需要管两件事：**进程崩溃重拉** + **毒条目兜底**。
`scripts/cloud_run.sh` 现成，逻辑：

```
worker 非零退出（被杀/OOM/CUDA 崩）
   → 带 --resume 自动重拉（日志换名 worker_<时间戳>_r<N>.log，不覆盖）
   → 读 inflight.txt 记录"崩溃时在跑哪条"，累计到 watchdog_state.tsv
   → 同一条目连崩 ≥FAIL_LIMIT(3) 次 → 向 rprime_cache.jsonl 追加 r_prime=null 永久隔离
     （--resume 据此跳过，防"崩→重拉→又崩"死循环）
   → 总重启 ≥MAX_RESTARTS(20) → 停并告警（需人工介入）
```

**on-demand 下用不到的部分**（脚本注释里的 "⚠️ spot 抢占" 那套：整机回收后靠外部
重新拉起）**放着不管即可**，不需要删。

**per-(task,cls) 级别的异常不归看门狗管**：`evaluate_rprime` 内已 try/except 记 NaN
自动跳过，SNR 统计会把 NaN 剔除。看门狗只管**进程级**崩溃。

**三个参数怎么设**（`launch_stage1.sh` 已内置）：

| 参数 | 值 | 理由 |
|---|---|---|
| `MAX_RESTARTS` | **20**（默认 50） | on-demand 无抢占，反复崩 = 真 bug，早停早查 |
| `FAIL_LIMIT` | 3（默认） | 同一条目连崩 3 次才隔离，一次演练不会误伤 |
| `SLEEP` | 10（默认） | 重启间隔 |

**已知边界（照抄脚本注释，心里有数）**：step0 的噪声种子（s≥1）崩溃无法用 canonical
缓存键精确隔离（`inflight.txt` 不含 seed）——隔离记录仍写入 canonical 缓存令 s=0 跳过，
s≥1 反复崩靠 `MAX_RESTARTS` 封顶。step0 单条仅 1–2 分钟，实际风险低。

---

# §6 【第5步】启动阶段1

```bash
bash /workspace/launch_stage1.sh 2>&1 | tee /workspace/logs/launch.log
```

跑的是 **13 个数据集**：**10 个未实测源 + 3 个 reserve**。

> 📌 **口径澄清**：`configs/source_pool.yaml` 里 `tested: false` 的源池成员是 **10 个**
> （CharacterTrajectories / Handwriting / UWaveGestureLibrary / ArticularyWordRecognition /
> NATOPS / ERing / MotorImagery / SelfRegulationSCP1 / SelfRegulationSCP2 /
> HandMovementDirection），加 3 个 reserve（EthanolConcentration / Heartbeat /
> FingerMovements）= **13 个**。
> 你说的"11 未实测源"应是把某个 reserve 算进源了 —— **以 yaml 为准**。
> 三个骨干（SpokenArabicDigits / PhonemeSpectra / Cricket）`tested: true`，**step0 不跑**，
> 直接计入闸门分母。

每数据集 `2 tasks × 10 CLS × 4 seeds = 80` 条 @ `early_iters=290`
（`configs/train.yaml` 唯一真源，与阶段2 同口径 → canonical 可复用）。

**天花板自动重探 ≤2 轮已内建**在 `snr_gate.ceiling_escalation`，**不需要你干预**：
SNR<1 且 R̄≥0.90（贴顶）且 n_classes_range 未触顶 → 自动把档提到
`min(lo+2, n_classes), min(hi+2, n_classes)` 重切重探，任务 id 打 `+c<lo>-<hi>` 标防缓存
键撞；恢复到 SNR≥1 即判可用并记下恢复档。

启动后立刻做 **§4.6 的四条核对** + 这两条：

```bash
# 写的是 Volume 上的路径，不是 Container Disk
readlink -f /workspace/shards/g0/CLS/results/labels_route_a/rprime_cache.jsonl
df -h /workspace ; findmnt -no SOURCE,FSTYPE /workspace     # 不应是 overlay

# 泄漏复查（跑起来之后再查一次，确保没有谁顺手下载了目标池）
find /workspace -iname '*Epilepsy*' -o -iname '*FaceDetection*' -o -iname 'har.npz'
```

**日志位置（全在 Volume 上）**：

| 内容 | 路径 |
|---|---|
| 看门狗主日志 | `/workspace/shards/g*/CLS/results/step0_snr_gate/watchdog.log` |
| worker 逐次日志（不覆盖） | `.../step0_snr_gate/worker_<时间戳>_r<N>.log` |
| 毒条目累计 | `.../step0_snr_gate/watchdog_state.tsv` |
| 当前在跑条目 | `.../step0_snr_gate/inflight.txt` |
| **canonical R'（阶段2 复用）** | `.../results/labels_route_a/rprime_cache.jsonl` |
| 噪声 R'（只为算 σ/SNR） | `.../step0_snr_gate/noise_cache.jsonl` |
| 任务定义 + 重探轮记录 | `.../step0_snr_gate/tasks_meta.json` |

---

# §7 【第6步】进度监控（你自己 SSH 查）

## 7.1 一条命令看全局

存成 `/workspace/watch.sh`：

```bash
#!/usr/bin/env bash
# 阶段1 进度速览。用法： watch -n 300 bash /workspace/watch.sh
printf "%-4s %-9s %-8s %-8s %-7s %s\n" 分片 canonical 噪声 完成% 重启 当前
for g in 0 1 2 3; do
  d=/workspace/shards/g$g/CLS/results
  can=$(wc -l < $d/labels_route_a/rprime_cache.jsonl 2>/dev/null || echo 0)
  noi=$(wc -l < $d/step0_snr_gate/noise_cache.jsonl 2>/dev/null || echo 0)
  nds=$(python - "$g" <<'PY'
import sys, json, os
g = sys.argv[1]
p = f"/workspace/shards/g{g}/CLS/results/step0_snr_gate/tasks_meta.json"
print(len(json.load(open(p, encoding='utf-8'))) if os.path.exists(p) else 0)
PY
)
  plan=$(( nds * 80 ))                       # 不含重探轮；重探会让实际 > 计划
  pct=$([ "$plan" -gt 0 ] && echo $(( 100 * (can + noi) / plan )) || echo "?")
  # 数 watchdog.log 里的"拉起 #N"，比数 worker_*.log 文件准
  rst=$(grep -c '拉起 #' $d/step0_snr_gate/watchdog.log 2>/dev/null || echo 1)
  cur=$(cut -f1,2 $d/step0_snr_gate/inflight.txt 2>/dev/null | head -c 46)
  printf "%-4s %-9s %-8s %-8s %-7s %s\n" "g$g" "$can" "$noi" "${pct}%" \
         "$(( rst > 0 ? rst-1 : 0 ))" "${cur:-—}"
done
echo "--- 毒条目 ---"
cat /workspace/shards/g*/CLS/results/step0_snr_gate/watchdog_state.tsv 2>/dev/null \
  | sort | uniq -c | sort -rn | head
echo "--- 看门狗尾 ---"
tail -n2 -q /workspace/shards/g*/CLS/results/step0_snr_gate/watchdog.log 2>/dev/null
echo "--- GPU ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "--- 孤儿 loky（仅在有重启时才需要清，见 §4.4）---"
pgrep -cf loky
```

跑 `watch -n 300 bash /workspace/watch.sh`（5 分钟刷一次）。

**读法**：

- `canonical` 每数据集目标 **20 条**（`2 tasks × 10 cls`，s=0），`噪声` 目标 **60 条**
  （s=1,2,3）。worklist 是 **seed-major**：**canonical 先跑满，再加深噪声** ——
  所以前 1/4 时间里 `canonical` 涨、`噪声` 不动，**属正常**。
- `完成%` 不含重探轮，触发重探的数据集会**超过 100%**，这是好消息不是 bug。
- `重启` >0 就去看对应的 `worker_*_r1.log`；同一毒条目出现 3 次会被永久隔离
  （写 `r_prime=null`），这条 (task,cls) 就此从 SNR 统计里剔除。
- GPU 那一行四张卡利用率都应 >0；某张长期 0 = 那个 worker 死了或跑完了，
  对照 `tmux ls`。

## 7.2 跑完了 → 合并 4 分片 → analyze

**先确认三个条件同时满足**：

```bash
# ① 4 个看门狗都正常退出
grep -l "worker 正常退出 rc=0" \
  /workspace/shards/g*/CLS/results/step0_snr_gate/watchdog.log | wc -l   # → 必须是 4
# ② 4 个 tmux session 都退出
tmux ls        # 无 step0_g*
```

然后合并到主副本：

```bash
cd /workspace/repo/CLS
mkdir -p results/step0_snr_gate results/labels_route_a

# ① 追加型缓存直接拼（各分片数据集不重叠）
cat /workspace/shards/g*/CLS/results/labels_route_a/rprime_cache.jsonl \
    > results/labels_route_a/rprime_cache.jsonl
cat /workspace/shards/g*/CLS/results/step0_snr_gate/noise_cache.jsonl \
    > results/step0_snr_gate/noise_cache.jsonl

# ② tasks_meta.json 合并（键 = 数据集名，4 份互不重叠，见 §0.6）
python - <<'PY'
import json, glob
m = {}
for p in sorted(glob.glob('/workspace/shards/g*/CLS/results/step0_snr_gate/tasks_meta.json')):
    d = json.load(open(p, encoding='utf-8'))
    dup = set(m) & set(d)
    assert not dup, f"!!! 数据集重复出现在多个分片: {dup} —— 分片配置有误，停 !!!"
    m.update(d)
json.dump(m, open('results/step0_snr_gate/tasks_meta.json', 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
print(f"合并 {len(m)} 个数据集: {sorted(m)}")
PY
# → 必须打印 13 个

# ③ 分析（⚠️ 必须显式列全 13 个：默认只取 tested=false 的源，不含 reserve）
python scripts/step0_snr_gate.py --analyze --datasets \
  CharacterTrajectories Handwriting UWaveGestureLibrary ArticularyWordRecognition \
  NATOPS ERing MotorImagery SelfRegulationSCP1 SelfRegulationSCP2 HandMovementDirection \
  EthanolConcentration Heartbeat FingerMovements \
  2>&1 | tee /workspace/logs/analyze.log

# ④ 汇总视图
python scripts/progress_route_a.py 2>&1 | tee /workspace/logs/progress.log

# ⑤ 第三个完成条件：13 个数据集都有 verdict，且无 pending_retry
python - <<'PY'
import json
a = json.load(open('results/step0_snr_gate/analysis.json', encoding='utf-8'))
ds = a['datasets']
print(f"有结论: {len(ds)}/13")
pend = [k for k, v in ds.items() if v['pending_retry']]
print("⚠️ 重探未跑完（需回对应分片再跑一次 cloud_run.sh，--resume 只补缺的重探轮）:",
      pend or "无")
PY
```

> ⚠️ `--analyze` 在**闸门不过**时会 `sys.exit(2)`（`step0_snr_gate.py:403`）。
> 这是**设计行为，不是崩溃** —— `analysis.json` 已经写好了，照 §8 判读即可。
> `tee` 的退出码会掩盖它，想确认就看 `analysis.json` 里的 `gate.ok`。

> `progress_route_a.py` 的"③ 阶段2 label 进度"会显示 `完成 ~260/2420` 之类
> ——这不是 bug：合并后的 `rprime_cache.jsonl` 里正是 step0 的 canonical R'，
> 它们**本来就是阶段2 的一部分**（逐键复用，阶段2 不重算）。这个比例（≈8–11%）
> 就是阶段2 能白拿的复用量。

---

# §8 【第7步】阶段1 判读清单（阶段2 决策依据）

`analysis.json` 每个数据集一条记录：

| 字段 | 含义 | 判读用途 |
|---|---|---|
| `verdict` / `action` / `passed` | 终局判定 | §8.1 |
| `snr` | CLS 区分信号 / 种子噪声 | **主判据**，≥1 才过 |
| `rprime_grand_mean` (R̄) | 所有 (task,cls,seed) R' 总均值 | ≥0.90 = 贴顶 → 区分天花板/失配 |
| `rprime_std_mean` (σ) | 种子噪声 | δ 微调依据（§8.5） |
| `cls_spread_std_mean` | CLS 区分信号强度 | SNR 的分子 |
| `pair_sign_stability` | 成对方向跨种子稳定率 | <0.7 → `fail_unstable` |
| `monotonicity` | Spearman(R̄, −HAR排名) | >0 = HAR 上的 CLS 好坏在此数据集也成立（可迁移性佐证） |
| `recommended_n_classes_range` | **恢复档** | 过闸且经重探时**必须写回 yaml**（§8.2） |
| `suggested_delta` | `3σ` | 与 `registry_delta` 对比微调 δ |
| `rounds[]` | 逐轮重探记录 | 看是第几轮救回来的 |
| `pending_retry` | 重探未跑完 | true = 结论悬置，**不可当最终结论** |

## 8.1 逐数据集分类

```bash
python - <<'PY'
import json
a = json.load(open('/workspace/repo/CLS/results/step0_snr_gate/analysis.json', encoding='utf-8'))
RESERVE = {"EthanolConcentration", "Heartbeat", "FingerMovements"}
PLANNED = {"CharacterTrajectories":16,"Handwriting":12,"UWaveGestureLibrary":10,
           "ArticularyWordRecognition":8,"NATOPS":10,"ERing":6,"MotorImagery":4,
           "SelfRegulationSCP1":2,"SelfRegulationSCP2":3,"HandMovementDirection":6,
           "EthanolConcentration":8,"Heartbeat":4,"FingerMovements":4}
rows = [(not v['passed'], -(v['snr'] or 0), k, v) for k, v in a['datasets'].items()]
print(f"{'数据集':<28}{'角色':<8}{'判定':<16}{'SNR':>6}{'R̄':>7}{'σ':>7}{'符号':>6}"
      f"{'任务':>5}  恢复档 / δ建议")
for _, _, k, v in sorted(rows):
    role = "reserve" if k in RESERVE else "源"
    rng = v['recommended_n_classes_range']
    print(f"{k:<28}{role:<8}{v['verdict']:<16}{v['snr'] or 0:>6.2f}"
          f"{v['rprime_grand_mean'] or 0:>7.3f}{v['rprime_std_mean']:>7.4f}"
          f"{v['pair_sign_stability'] or 0:>6.2f}{PLANNED[k]:>5}  "
          f"{('c%d-%d' % tuple(rng)) if rng else '—':<10} "
          f"δ {v['suggested_delta']} (表 {v['registry_delta']})"
          + ("  ⚠️PENDING" if v['pending_retry'] else ""))
g = a['gate']
print(f"\n闸门: {'✅' if g['ok'] else '❌'} {g['count']}/{g['floor']}  {g['message']}")
print(f"过闸的未实测源: {g['passed_untested']}")
PY
```

| `verdict` | 含义 | 动作 |
|---|---|---|
| `pass` | SNR≥1 且符号稳定 | **入最终源池**。若 `rounds` 非空 → 靠加难救回，**必须把 `recommended_n_classes_range` 写回 yaml**（§8.2） |
| `fail_mismatch` | 加难到最难档仍 SNR<1（或 `[2,2]` 无档可加）—— 架构失配，PenDigits 式 | **踢**，不入池。写论文并入 limitation |
| `fail_unstable` | SNR 够但成对方向被噪声翻转（<0.7） | **踢**。理论上可用 k 折/大 δ 救，但那是新工作量，阶段1 不做 |
| `insufficient` | R' 记录不足，算不出 SNR | **不是结论** —— 查该分片 `worker_*.log`，多半是数据加载或反复崩溃 |
| `retry_ceiling` + `pending_retry: true` | 重探未跑完 | **结论悬置**，回去补跑（§7.2 第 ⑤ 条） |

**特别注意 `[2,2]` 的五个**（MotorImagery / SCP1 / SCP2 / Heartbeat / FingerMovements）：
它们 `is_range_maxed=True`，**一轮重探都不会跑**，SNR<1 就直接 `fail_mismatch`。
但它们是 2 类源，天花板效应本来就轻（R̄ 通常 0.6–0.8），若真判失配，多半是**真失配**。
这几个同时是源池长序列端（L 512/896/1152）—— 全被踢的话，CLAUDE.md 记的
"长序列 × EEG 域共线"limitation 会变成"源池根本没有长序列"，性质更严重。
**若它们大面积被踢，优先从 reserve 回填 Ethanol（L=1024，非 EEG）。**

## 8.2 把恢复档写回配置（漏了阶段2 白跑）

任何 `passed: true` 且 `rounds` 非空的数据集，都是**靠更难的类别子集档才救回来的**。
阶段2 造 label 必须用同一个档，否则任务定义不同 → step0 的 canonical R' 复用不上（浪费），
且 R' 会退回贴顶的无区分状态（**label 是噪声**）。

```bash
python - <<'PY'
import json
a = json.load(open('/workspace/repo/CLS/results/step0_snr_gate/analysis.json', encoding='utf-8'))
need = {k: v['recommended_n_classes_range'] for k, v in a['datasets'].items()
        if v['passed'] and v['rounds']}
if not need:
    print("无数据集靠重探恢复 —— configs/source_pool.yaml 不需要改")
for k, r in need.items():
    print(f"configs/source_pool.yaml: {k} 的 n_classes_range 改为 [{r[0]}, {r[1]}]")
PY
```

按输出**手工改 `configs/source_pool.yaml`**（池结构唯一真源），改完 `git diff` 确认。
**这是阶段2 的前置条件。**（方案 C 下在本地改、commit，阶段2 重新 `git archive` 上传。）

## 8.3 存活任务数是否 ≥100

`gate.count = 骨干 44 + Σ(过闸的未实测源 planned_tasks)`。骨干固定
（SpokenArabicDigits 16 + PhonemeSpectra 18 + Cricket 10 = 44）。
10 个未实测源满打满算贡献 77 → 上限 121。

> **⚠️ reserve 不计入 `gate.count`**：`remaining_task_count` 只遍历 `registry.source`
> （`source_pool.py:264`）。哪怕三个 reserve 全 `pass`，闸门数字也不变 ——
> 它们的 verdict 是**给你回填决策用的信息**，回填要改 yaml 把 `role` 从 reserve
> 提到 source 才算数。

**闸门算术拍平**：

- 需要 `Σ(过闸未实测源) ≥ 56`（= 100 − 44）。未实测源池总共 77 → **最多只能损失 21**。
- **承重墙：CharacterTrajectories(16) 和 Handwriting(12)**。两个同时被踢 = 损失 28 > 21，
  **必破闸**。任一单独被踢仍可过（剩 61 / 65）。
- 四个 EEG 源全踢（MotorImagery 4 + SCP1 2 + SCP2 3 + HandMovementDirection 6 = 15）
  → 剩 62 ≥ 56，**仍过闸**（但触发 §8.1 末尾的长序列问题）。
- UWave(10) + NATOPS(10) 同时踢 = 20，**刚好卡在边缘**（剩 57）。

各源任务数速查：CharacterTrajectories 16｜Handwriting 12｜UWaveGestureLibrary 10｜
NATOPS 10｜ArticularyWordRecognition 8｜ERing 6｜HandMovementDirection 6｜
MotorImagery 4｜SelfRegulationSCP2 3｜SelfRegulationSCP1 2。

## 8.4 若 `gate.ok = false`：回填决策（信息给全，选择权在你）

**① reserve 回填（+ 最多 16 任务）** —— 三个 reserve 已随阶段1 过了 step0，verdict 现成，
**不用二次上云**。按可靠性排序：

| 优先级 | reserve | 任务 | 回填理由 | 附带收益 |
|---|---|---|---|---|
| **1** | EthanolConcentration | **8** | 任务数最多；N/C=131 中等偏好 | ⭐ **顺带解掉 CLAUDE.md 记的"长序列端全是 EEG"limitation**（唯一非 EEG 长锚，Spectro/L1024）。即使不缺任务数，只要它 `pass` 就值得回填 |
| 2 | Heartbeat | 4 | N/C≈204 高，R' 相对稳 | Audio 域，与 SpokenArabicDigits/PhonemeSpectra 同域，多样性增益小 |
| 3 | FingerMovements | 4 | N/C≈208 高 | L=50 刚好卡在编码长度下限，最接近"L≥50 边界"的风险位；EEG 域已过剩 |

排序依据（按重要性）：**先看 `verdict==pass`**（不过闸的一律不回填），再比 `snr`
（越高越可靠），再比 `pair_sign_stability`，最后比域/长度的边际多样性。
回填操作 = 在 `configs/source_pool.yaml` 把该条从 `reserve:` 移到 `source_pool:`
（δ 已在 `labels.yaml` 备好，C1 已覆盖 16 个）。

**② 高 N/C 骨干加码切片（+16 任务）** —— SpokenArabicDigits `planned_tasks` 16→24、
PhonemeSpectra 18→26。这两个是探针实测 SNR 最高（2.24 / 2.03）、N/C 最高（880 / 171）的，
多切任务**不掺水**。代价：阶段2 算力 +16 任务 × 20 CLS。

**③ 天花板恢复复查** —— 有 `pending_retry: true` 的先补跑完再谈回填，
别在结论悬置时做决策。

**④ 都用尽仍 <100 → 你二选一**（手册不替你决定）：

- **(a) 接受缩减源池**：底线是骨干 44（探针实测，铁定可靠），已是路线 B（16 任务）的
  2.75×。写论文列为 limitation。
- **(b) 放宽 floor**：改 `task_count_gate(floor=...)` 到可辩护的下界（如 80）。

**算术**：骨干 44 + reserve 满额 16 + 加码 16 = **76**。所以**至少要有 24 个任务来自
过闸的未实测源**，闸门才可能在用尽所有退路后达标。若过闸源贡献 <24，(a)/(b) 是唯一出路。

**闸门不硬闯** —— 任务数不足时静默往下产噪声 label 正是路线 B 的失败教训
（memory `m3-tclsc-route-b-result`：留出 acc 0.866 但 E 消融零下降，通用排序器而非
task-aware）。宁可缩减 + 写 limitation，也不要 121 条噪声 label。

## 8.5 per-dataset δ 微调

`analysis.json` 每个数据集给 `suggested_delta = 3 × rprime_std_mean`（`snr_gate.py:123`），
对比 `labels.yaml: delta_per_dataset` 的推测初值：

| 情况 | 动作 | 理由 |
|---|---|---|
| `suggested > registry` | **上调到 suggested** | 实测噪声比推测大，原 δ 会放行噪声 label。宁可多弃样 |
| `suggested ≤ registry`，差距 <2× | **保持 registry 不动** | step0 只有 2 任务 × 10 CLS 的 σ 估计，样本量小，不足以支撑下调 |
| `suggested ≤ registry`，差距 >2× | 可下调，但**下调不超过一半差距** | 保守，避免弃样率过低 |
| 三个骨干（0.03/0.05/0.08） | **填死不动** | 探针实测（SNR>1、label 正确率 ≥95%），不受 step0 的 2 任务估计影响 |

改完在 `configs/labels.yaml` 的 `delta_per_dataset` 里注释标注"阶段1 step0 实测 σ=X 微调，
原推测 Y"。reserve 的 δ 只在实际回填时生效。

**顺带看 `monotonicity`**：>0 说明"HAR 上排名靠前的 CLS 在这个数据集上 R̄ 也高"——
这是 CLS 好坏可跨数据集迁移的直接证据，也是 T-CLSC 能学到东西的前提。
若大面积 ≈0 或为负，即使 SNR 过闸也要在阶段2 前警觉。

---

# §9 【第8步】结束操作与成本

## 9.1 确认产物齐（先确认，再碰任何按钮）

```bash
cd /workspace/repo/CLS
echo "=== 挂载确认 ==="
df -h /workspace | tail -1 ; findmnt -no SOURCE,FSTYPE /workspace

echo "=== 阶段2 必需产物 ==="
for f in results/step0_snr_gate/analysis.json \
         results/step0_snr_gate/tasks_meta.json \
         results/step0_snr_gate/noise_cache.jsonl \
         results/labels_route_a/rprime_cache.jsonl \
         configs/source_pool.yaml configs/labels.yaml \
         results/route_a_probe/probe_cls_catalog.json; do
  [ -f "$f" ] && printf "  ✅ %-52s %s\n" "$f" "$(du -h $f | cut -f1)" \
              || printf "  ❌ %-52s 缺失\n" "$f"
done

echo "=== canonical 条数（阶段2 复用量）==="
python - <<'PY'
import json, collections
c = collections.Counter()
for line in open('results/labels_route_a/rprime_cache.jsonl', encoding='utf-8'):
    if line.strip():
        c[json.loads(line).get('dataset', '?')] += 1
print(f"总计 {sum(c.values())} 条（每数据集应 ≥20，含重探则更多）")
for k, v in sorted(c.items()): print(f"  {k:<28}{v}")
PY

echo "=== 泄漏复查（判读期也要保持零）==="
find /workspace -iname '*Epilepsy*' -o -iname '*FaceDetection*' -o -iname 'har.npz'
```

> 📌 **共享 TS2Vec 阶段1 不产出，这是对的**。`shared_ts2vec.pt` 是阶段2 起步时在
> **最终源池**上 fit 的（铁律 6：源池成员要到闸门定案才确定）。阶段1 结束时它不存在
> 属正常，不要去找。

## 9.2 拉回本地（方案 C 下这是**强制**步骤，不是可选）

```bash
cd /workspace/repo/CLS
tar -czf /workspace/stage1_artifacts.tar.gz \
    results/step0_snr_gate/{analysis.json,tasks_meta.json,noise_cache.jsonl} \
    results/labels_route_a/rprime_cache.jsonl \
    configs/source_pool.yaml configs/labels.yaml \
    /workspace/logs/
du -h /workspace/stage1_artifacts.tar.gz          # → 应 <5 MB
runpodctl send /workspace/stage1_artifacts.tar.gz
# 本地： runpodctl receive <配对码>
```

**本地收到后必须验一次再删 pod**（方案 C 下 terminate 就是不可逆的）：

```powershell
# 本地 PowerShell
tar -tzf stage1_artifacts.tar.gz | Select-String 'analysis.json|rprime_cache.jsonl'
tar -xzf stage1_artifacts.tar.gz -C D:\DevelopDoc\CLS\   # 解到对应位置
```

解到本地后，§8 的判读、改 yaml、commit **全在本机做，零计算成本**。

## 9.3 停止计费

| 操作 | 计算费 | Container/Volume Disk | 适用 |
|---|---|---|---|
| **Stop Pod** | ✅ 停 | ⚠️ 继续计（Volume Disk 涨到 $0.20/GB/月） | 几小时内还要用 |
| **Terminate Pod** | ✅ 停 | ✅ 停（**方案 C 下数据同时删除**） | **判读期推荐 —— 前提是 §9.2 已确认收到** |

**方案 C 推荐流程**：§9.1 确认产物齐 → §9.2 备份并在本地验证 → **Terminate**。
判读期零成本。阶段2 重开 pod 时按 §3 重传数据（0.86 GB，几分钟）+ 把
`rprime_cache.jsonl` 放回 `results/labels_route_a/`，`--resume` 逐键复用。

**（若你改走方案 A）**：Terminate pod **不会**删 Network Volume 的数据，判读期只付
$0.07/GB/月；阶段2 必须在**同一数据中心 + 同一镜像**重开。

## 9.4 成本：填一个单价就出总额

**GPU-h 用量**（与单价无关，只取决于 §1.9 标定出的 R）：

| 项目 | R=2.5 | R=3 | **R=4** | R=6 |
|---|---|---|---|---|
| 兼容性闸门（1 卡 × ~1 h） | 1 | 1 | **1** | 1 |
| 阶段1 基线（4 卡） | 31.3 | 26.1 | **19.5** | 13.0 |
| 阶段1 预期（+25% 重探） | 39.1 | 32.6 | **24.4** | 16.3 |
| 阶段1 最坏（重探全触发） | 50.2 | 41.9 | **31.4** | 20.9 |
| 环境准备/preflight（4 卡 ~0.5 h） | 2 | 2 | **2** | 2 |
| **预期合计 GPU-h** | **42.1** | **35.6** | **27.4** | **19.3** |

**乘上你在控制台看到的 5090 Community on-demand 单价 `$P`**：

```
阶段1 预期花费 ≈ 预期合计 GPU-h × $P
```

参考几个档位（`$P` 现填，下面只是把乘法做好）：

| `$P`/GPU-h | R=2.5（42.1 h） | R=3（35.6 h） | **R=4（27.4 h）** | R=6（19.3 h） |
|---|---|---|---|---|
| $0.50 | $21 | $18 | **$14** | $10 |
| $0.70 | $29 | $25 | **$19** | $14 |
| $0.90 | $38 | $32 | **$25** | $17 |
| $1.10 | $46 | $39 | **$30** | $21 |

**闸门那一趟单独算**：1 卡 × ~1 h × `$P` ≈ **$0.5–1.1**。这是本手册里性价比最高的一块钱。

## 9.5 vs 之前的 4×A100 估算

| | 4×A100 80GB（旧估） | **4×RTX 5090（本手册）** |
|---|---|---|
| 假设/实测倍率 vs 4060 | 2.5×（**假设值，从未实测**） | **由 §1.9 标定实测**，预期 3–4× |
| 阶段1 GPU-h（预期） | ~40 | **~24–33** |
| 4 卡墙钟（预期） | 9–12 h | **6–9 h** |
| 单卡单价 | Secure $1.89 / Community $1.39 | Community `$P`（5090 是消费卡，**通常显著低于 A100**） |
| 阶段1 花费（预期） | **$75–100（Secure）/ $55–75（Community）** | **见 §9.4 表，多数档位在 $15–40** |
| 跨阶段存储 | Network Volume（Secure 才有） | 方案 C 人工备份（<5 MB） |
| 卡间通信需求 | 无 | 无（数据并行） |
| 显存余量 | 80 GB（极大浪费） | 32 GB（对 L≤1152/bs=128 仍绰绰有余） |

**结论：4×5090 在这个任务上应当既更快又更省** —— 更快是因为小卷积吃时钟不吃 HBM，
更省是因为消费卡单价低且不为用不上的 80 GB HBM / NVLink 付费。
**但"更快"这一半必须由 §1.9 的标定证实**，别在没量之前当成既定事实。

## 9.6 阶段2 要用的数（现在就记下来）

| 项 | 值 |
|---|---|
| 镜像模板完整名 + torch 版本 + cuXXX | ______（**阶段2 必须一致**，见 §1.1⚠️） |
| 兼容性四检结果 | ______（arch_list 是否含 sm_120 / 实际 torch 版本） |
| §1.6 数值一致性判定 | ______（median 样本差 / 带符号均值 / Spearman / verdict 是否一致） |
| **实测秒/条 @290 iters（NATOPS）** | ______ vs 4060 的 33.5 s |
| **实测倍率 R = 33.5 / 实测** | ______（阶段2 按同一 R 缩放：基线 42 A100-h ⇒ 用 4060 口径重算） |
| GPU 利用率 / 最终 worker 数 | ______（决定阶段2 要不要 2 worker/卡） |
| 数据中心 / 机器 | ______（方案 A 才强制一致；方案 C 随意） |
| 阶段1 实际墙钟 vs 预估 | ______ |
| 5090 Community on-demand 实价 `$P` | ______ |

---

# 附：阶段1 检查清单（打印出来逐项划）

**闸门（§1，1 张卡，~1 h，~$1）**
- [ ] §1.1 镜像 tag 里 **cuda ≥12.8 且 torch ≥2.7**，模板全名已抄进 §9.6
- [ ] §1.2 四个包传完（代码 / 探针目录 / NATOPS+ERing / **4060 参考结果已改名**）
- [ ] §1.3 装依赖前后 **torch 指纹逐字符相同**
- [ ] §1.4 兼容性四检全 PASS（尤其第 ④ 检真下发 kernel，无 `no kernel image`）
- [ ] §1.6 数值一致性通过（逐条 / 带符号均值 / Spearman / **决策级 verdict 一致**）
- [ ] §1.7 kill -9 演练：看门狗重拉 + resume 待跑数变小 + SVM 兜底接住
- [ ] §1.9 标定：实测秒/条 + GPU 利用率已记录，R 已确定
- [ ] §1.10 gate 产物（含 NATOPS canonical）已拉回本地，pod 已 terminate
- [ ] **任一不过 → 停在 §1.8 症状表，不要开 4 卡**

**开 4 卡（§2–§3）**
- [ ] §2.1 已确认走方案 C（Community + 人工备份），且接受"terminate 即删数据"
- [ ] §2.4 Pod：4×RTX 5090、On-Demand、Community、**与闸门同一镜像**、Volume 30 GB @ `/workspace`
- [ ] 开机后 `compat_check.py` 在这台新机器上**再过一遍**（4 张卡都要过）
- [ ] §3.1 上传前自检：包里无 Epilepsy/FaceDetection/UCI-HAR/har.npz
- [ ] §3.3 数据集清单 = 16 行，全盘泄漏搜索**零输出**
- [ ] §3.4 `preflight.sh` 五项全绿（兼容性 / pytest / 目标池 fail-fast / 探针目录 / canonical）

**并行调度（§4）**
- [ ] §4.1 4 个分片副本建好，`uea_cache` symlink + 探针目录各就位
- [ ] §4.1 闸门带回的 NATOPS canonical 已放进 g3 分片
- [ ] §4.5 按 §1.9 的利用率决定 1 还是 2 worker/卡（<50% → 8 分片）
- [ ] §4.6 **四条硬证据**：4 tmux / 4 个不同 gpu_uuid / 4 个不同数据集在滚 / 4 卡利用率 >0
- [ ] §4.4 `LOKY_MAX_CPU_COUNT` 在 worker 进程的 `/proc/<pid>/environ` 里真的生效

**跑起来（§6–§7）**
- [ ] §6 `bash /workspace/launch_stage1.sh`，13 个数据集分到 4 卡
- [ ] §6 写的是 Volume 路径而非 overlay；跑起来后泄漏复查仍为零
- [ ] §7.1 `watch -n 300 bash /workspace/watch.sh` 起着

**跑完（§7.2–§8）**
- [ ] §7.2 三条件：4 个 rc=0 / tmux 清空 / 无 `pending_retry`
- [ ] §7.2 合并 4 分片（`tasks_meta.json` 打印 **13 个**）+ `--analyze`（列全 13 个）
- [ ] §8.1 逐数据集判定表已看
- [ ] §8.2 恢复档已写回 `configs/source_pool.yaml`（**漏了阶段2 白跑**）
- [ ] §8.3 闸门 count/100 已判；不过则走 §8.4 回填阶梯
- [ ] §8.5 δ 已按实测 σ 微调；`monotonicity` 已扫一眼

**收尾（§9）**
- [ ] §9.1 产物齐、泄漏复查零
- [ ] §9.2 `stage1_artifacts.tar.gz` (<5 MB) 已 send 回本地**并在本地解压验证**
- [ ] §9.4 成本核对（实际墙钟 × 4 × `$P`）
- [ ] §9.6 九个"阶段2 要用的数"已填满
- [ ] §9.3 **Terminate pod**（方案 C：确认 §9.2 已验证之后）
