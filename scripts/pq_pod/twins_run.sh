#!/usr/bin/env bash
# MX-4 WP6 S1b+S2 上云（5090）：源侧孪生补标（切片协议 @{60,120,290}）。
#   档 0：--tasks（串行，含定义漂移闸）
#   档 1 = S1b：HW / UWave / HMD × 切片 00/01 × shared-10 + GGS 孪生（66 条轨迹）→ 默认到此为止，
#          拉回后本地 `40 --k1` 判 K1，用户拍板后以 CONTINUE=1 重跑本脚本续跑 S2（--resume 跳过已跑格）
#   档 2+ = S2：其余孪生（shared/GGS 全切片 + random 前 2 切片），显存分档同 pq_pod/run.sh
#
#   nohup bash /workspace/wp/CLS/scripts/pq_pod/twins_run.sh > /workspace/logs/twins_run.log 2>&1 &
#   CONTINUE=1 nohup bash /workspace/wp/CLS/scripts/pq_pod/twins_run.sh > /workspace/logs/twins_run2.log 2>&1 &
# 产物目录 results/mx4_mask/source_twins（绝不写 results/mx3_pq：36/37/38 按全表做 z 标准化）。
set -euo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
OUT="${OUT:-results/mx4_mask/source_twins}"
cd "$REPO"; mkdir -p "$LOG" "$OUT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864
export AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TW="--twins input_mask=binomial --twin-only"
STAGES="--fill --ggs $TW --resume"

PIDS=(); PNAMES=()
launch() {
  local ds="$1"
  nohup python scripts/32_pq_generate_rprime.py $STAGES \
        --datasets "$ds" --outdir "$OUT" \
        --inflight "$OUT/inflight_$ds.txt" --oom-log "$OUT/oom_failures.jsonl" \
        > "$LOG/tw_$ds.log" 2> "$LOG/tw_$ds.err" &
  local pid=$!
  PIDS+=("$pid"); PNAMES+=("$ds")
  echo "  launched $ds pid=$pid"
}
wait_tier() {
  local tier="$1" rc=0 bad=0 i
  if [ "${#PIDS[@]}" -gt 0 ]; then
    for i in "${!PIDS[@]}"; do
      if wait "${PIDS[$i]}"; then echo "  ✅ ${PNAMES[$i]} 退出 0"
      else rc=$?; bad=$((bad + 1)); echo "  ⚠️ ${PNAMES[$i]} 退出码 $rc（见 $LOG/tw_${PNAMES[$i]}.err）"; fi
    done
  fi
  [ "$bad" -eq 0 ] && echo "  $tier 全部退出 0" || echo "  ⚠️ $tier 有 $bad 个进程非零退出，继续下一档"
  PIDS=(); PNAMES=()
}
serial() {
  local ds="$1"
  echo "  serial $ds"
  python scripts/32_pq_generate_rprime.py $STAGES \
        --datasets "$ds" --outdir "$OUT" \
        --inflight "$OUT/inflight_$ds.txt" --oom-log "$OUT/oom_failures.jsonl" \
        > "$LOG/tw_$ds.log" 2> "$LOG/tw_$ds.err" \
    || echo "  ⚠️ $ds 非零退出（见 $LOG/tw_$ds.err），继续下一个"
}

if [ "${CONTINUE:-0}" != "1" ]; then
  echo "== [$(date -u +%H:%M:%S)] 档 0：tasks.json / cls_catalog.json（含定义漂移闸）"
  python scripts/32_pq_generate_rprime.py --tasks $TW --outdir "$OUT"
  echo "== [$(date -u +%H:%M:%S)] 档 1 = S1b：HW / UWave / HMD × 切片 00/01 × shared-10 + GGS 孪生（66 条，串行）"
  python scripts/32_pq_generate_rprime.py $STAGES --twin-slices 2 --twin-origins shared ggs \
        --datasets Handwriting UWaveGestureLibrary HandMovementDirection --outdir "$OUT" \
        --inflight "$OUT/inflight_s1b.txt" --oom-log "$OUT/oom_failures.jsonl" \
        > "$LOG/tw_s1b.log" 2> "$LOG/tw_s1b.err" || echo "  ⚠️ S1b 非零退出（见 $LOG/tw_s1b.err）"
  python scripts/32_pq_generate_rprime.py --verify $TW --twin-slices 2 --twin-origins shared ggs \
        --datasets Handwriting UWaveGestureLibrary HandMovementDirection --outdir "$OUT" | head -5 || true
  echo "== [$(date -u +%H:%M:%S)] S1b done → 拉回 $OUT/rprime_cache.jsonl 本地跑 40 --k1；K1 过且用户拍板后 CONTINUE=1 重跑本脚本"
  exit 0
fi

echo "== [$(date -u +%H:%M:%S)] S2 档 1：短序列端 5 并发"
for ds in Handwriting UniMiB-SHAR PhonemeSpectra UWaveGestureLibrary Cricket; do launch "$ds"; done
wait_tier "档 1"
echo "== [$(date -u +%H:%M:%S)] S2 档 2：HandMovementDirection"
launch HandMovementDirection
wait_tier "档 2"
echo "== [$(date -u +%H:%M:%S)] S2 档 3：长序列端串行 Ethanol → SCP2"
serial EthanolConcentration
serial SelfRegulationSCP2
echo "== [$(date -u +%H:%M:%S)] 收尾 --verify（--on-oom record 串行重扫留给人工）"
python scripts/32_pq_generate_rprime.py --verify $TW --outdir "$OUT" | head -8 || true
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] S2 done → 拉回 $OUT/{rprime_cache.jsonl,oom_failures.jsonl,tasks.json,cls_catalog.json} 与 $LOG/tw_*.log"
