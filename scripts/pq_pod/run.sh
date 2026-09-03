#!/usr/bin/env bash
# MX-3-PQ 上云：按 L 分档跑 R' 真值网格。
#   主网格 45 切片 × 21 CLS（10 shared + 10 random + 1 GGS）× 3 截点 = 2835 行
#   噪声探针 8 数据集 × 2 切片 × 10 shared × 2 rep × 3 截点 = 960 行
# 一条轨迹只预训练一次（跑到 290，途中在 60/120 各评一次），故成本 ≈ 一遍 290。
#
# 用法（pod 上；本脚本**前台**跑到全部结束，务必脱离启动）：
#   nohup bash /workspace/wp/CLS/scripts/pq_pod/run.sh > /workspace/logs/pq_run.log 2>&1 &
# 全程 --resume 可续：被 spot 抢占后原样重跑本脚本即可（已完成的格直接跳过）。
set -euo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
OUT="${OUT:-results/mx3_pq}"
cd "$REPO"; mkdir -p "$LOG" "$OUT"

# 线程/分块/分配器：configs/runtime.yaml 的 env 段逐项照抄（唯一真源在那儿）。
# OMP=1 是硬要求——不设的话 54 核容器里 torch 开 192 线程，小张量同步开销压倒计算，
# 单条 R' 慢 4.2×（route-a-throughput-omp1-and-worker-scaling）。
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864
# InfoNCE 侧同款分块（tests/test_infonce_chunking.py）：本轮 12 条 OOM 跳过里 10 条
# 是 info_nce+temporal @ Ethanol/SCP2（n=2T≈2048~2304），不设的话长序列端照旧跳格。
export AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 注意不带 --tasks：tasks.json / cls_catalog.json 是 8 个进程共享的单份产物，
# 并发的 read-modify-write 存在撞窗（fix 轮 1 IMPORTANT 3）。档 0 已串行把 45 个
# 切片写全，每个数据集进程只跑造 R′ 那三个阶段。
STAGES="--fill --ggs --noise-probe --resume"

PIDS=()      # 本档已启动的后台 PID
PNAMES=()    # 与 PIDS 同序的数据集名

launch() {   # launch <dataset> —— 一个数据集一个进程，各自 --resume
  local ds="$1"
  nohup python scripts/32_pq_generate_rprime.py $STAGES \
        --datasets "$ds" --outdir "$OUT" \
        --inflight "$OUT/inflight_$ds.txt" --oom-log "$OUT/oom_failures.jsonl" \
        > "$LOG/pq_$ds.log" 2> "$LOG/pq_$ds.err" &
  local pid=$!
  PIDS+=("$pid"); PNAMES+=("$ds")
  echo "  launched $ds pid=$pid"
}

# 逐 PID 等待并**逐个**报退出码。裸 `wait` 无参数时永远返回 0（POSIX），
# 于是 `wait || echo ...` 是死代码：某个数据集进程崩了也一声不吭，要等到
# finish.sh 的 --verify 才发现缺格，而那时已经过了几个小时（Minor 9）。
wait_tier() {   # wait_tier <档名>
  local tier="$1" rc=0 bad=0 i
  if [ "${#PIDS[@]}" -gt 0 ]; then
    for i in "${!PIDS[@]}"; do
      if wait "${PIDS[$i]}"; then
        echo "  ✅ ${PNAMES[$i]} (pid ${PIDS[$i]}) 退出 0"
      else
        rc=$?
        bad=$((bad + 1))
        echo "  ⚠️ ${PNAMES[$i]} (pid ${PIDS[$i]}) 退出码 $rc" \
             "（见 $LOG/pq_${PNAMES[$i]}.err）"
      fi
    done
  fi
  [ "$bad" -eq 0 ] && echo "  $tier 全部退出 0" \
                   || echo "  ⚠️ $tier 有 $bad 个进程非零退出，继续下一档"
  PIDS=(); PNAMES=()
}
# 每进程一份 inflight（configs/runtime.yaml 头注释）：共用一份的话看门狗把毒条目
# 归因到别的进程正在跑的那条上。rprime_cache.jsonl 反过来是共享的——单行 < 4096 B
# 的 POSIX 原子追加边界内，并发追加安全。
# 打点文件**必须落在 $OUT 里**：32 号的 --verify（inflight 为空闸）与
# --redo-partial/--retry-terminal 的互锁都只扫 outdir；放到 $LOG 就对它们全部
# 失明，而互锁失明时重写会把别的进程正在追加的行整段吞掉（CRITICAL 1）。

serial() {   # serial <dataset> —— 长序列端独占整卡
  local ds="$1"
  echo "  serial $ds"
  python scripts/32_pq_generate_rprime.py $STAGES \
        --datasets "$ds" --outdir "$OUT" \
        --inflight "$OUT/inflight_$ds.txt" --oom-log "$OUT/oom_failures.jsonl" \
        > "$LOG/pq_$ds.log" 2> "$LOG/pq_$ds.err" \
    || echo "  ⚠️ $ds 非零退出（见 $LOG/pq_$ds.err），继续下一个"
}

# 档 0：任务定义 + CLS 目录，一次跑全 8 个（零 GPU，约 15 s）。
# 这是全流程里**唯一**一次 --tasks（且串行）：它写的两份产物是共享单份的，
# 并发跑会撞窗；同时这一步会跑定义漂移闸（切片 classes vs 既有落盘，不符即 exit 2）。
echo "== 档 0：tasks.json / cls_catalog.json（含定义漂移闸）"
python scripts/32_pq_generate_rprime.py --tasks --outdir "$OUT"

# 显存分档依据 configs/runtime.yaml 的峰值模型 peak(L)=0.694+0.013113·L（GiB），
# 准入预算 gpu.budget_gib=24.0（留头寸给 resize 把真实峰值放大 1.85× 的情形）。
# 档 1：短序列端 5 个并发 —— HW 152→2.69 / UM 151→2.67 / PS 217→3.54 /
#       Cricket 256→4.05 / UWave 315→4.82，合计 17.78 GiB < 24。
#
# ⚠️ **与 runtime.yaml 分档表的公开偏离**：PhonemeSpectra(217) / Cricket(256) /
#    UWaveGestureLibrary(315) 按 workers.tiers 属于 "max_l_enc: 600 → workers: 2"
#    那一档，本脚本把它们与两个 L≈150 的一起 5 并发。依据：同一份
#    runtime.yaml 的峰值模型 peak(L)=0.694+0.013113·L 给出 17.78 GiB，低于
#    gpu.budget_gib=24.0；分档表自带注释说中档 workers:2 是"保守值，**未实测**"。
#    安全网不是这个估算：并发档走 --on-oom skip（OOM 不落盘、保持"未跑"），
#    finish.sh 再以 worker=1 串行重扫。若档 1 实跑出成批 OOM，把这三个挑出来
#    单独成一档即可（改下面的 for 列表）。
echo "== 档 1：短序列端 5 并发"
for ds in Handwriting UniMiB-SHAR PhonemeSpectra UWaveGestureLibrary Cricket; do
  launch "$ds"
done
wait_tier "档 1"

# 档 2：HandMovementDirection（l_enc 400 → 5.94 GiB）。与档 1 同跑合计 23.7 GiB
#       贴着预算上沿，故独占一档。
echo "== 档 2：HandMovementDirection"
launch HandMovementDirection
wait_tier "档 2"

# 档 3：长序列端串行链（Ethanol 1024→14.1 GiB，SCP2 1152→15.8 GiB；runtime.yaml
#       §3.3 实测 2 个 SCP2 必 OOM，且 resize 最坏可把峰值推到 29 GiB）。
echo "== 档 3：长序列端串行 Ethanol → SCP2"
serial EthanolConcentration
serial SelfRegulationSCP2

nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== run done → 收尾请跑 bash scripts/pq_pod/finish.sh"
