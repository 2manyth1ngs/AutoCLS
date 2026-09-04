#!/usr/bin/env bash
# WP7-UCR 上云：按 l_enc 分档并发跑 R'（84 数据集 × 20 CLS × 三截点）。
#   nohup bash /workspace/wp/CLS/scripts/ucr_pod/run.sh > /workspace/logs/ucr_run.log 2>&1 &
# 全程 --resume 可续。分档 TIERS="lo:hi:并发"（按注册表 l_enc，含 crop 后长度），每档内按 --shard k/n 轮转分片，
# 一个分片一个进程、各自 inflight 打点（必须落在 $OUT 里，32 号互锁口径）。显存模型 peak(L)=0.694+0.013·L GiB（5090 32 GB）。
set -euo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
OUT="${OUT:-results/ucr_rprime}"
TIERS="${TIERS:-0:200:6 201:600:3 601:2000:2}"
cd "$REPO"; mkdir -p "$LOG" "$OUT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864
export AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PIDS=(); PNAMES=()
launch() {   # launch <tag> <min> <max> <k/n>
  local tag="$1" lo="$2" hi="$3" sh="$4"
  nohup python scripts/41_ucr_rprime.py --fill --resume --min-len "$lo" --max-len "$hi" --shard "$sh" \
        --outdir "$OUT" --inflight "$OUT/inflight_${tag}.txt" \
        > "$LOG/ucr_${tag}.log" 2> "$LOG/ucr_${tag}.err" &
  PIDS+=("$!"); PNAMES+=("$tag")
  echo "  launched $tag pid=$!"
}
wait_tier() {
  local bad=0 i
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then echo "  ✅ ${PNAMES[$i]} 退出 0"; else echo "  ⚠️ ${PNAMES[$i]} 退出码 $?（见 $LOG/ucr_${PNAMES[$i]}.err）"; bad=$((bad+1)); fi
  done
  PIDS=(); PNAMES=()
  [ "$bad" -eq 0 ] || echo "  ⚠️ 本档 $bad 个进程非零退出，继续"
}

t=0
for spec in $TIERS; do
  IFS=: read -r lo hi par <<< "$spec"
  t=$((t+1))
  echo "== 档 $t：l_enc ∈ [$lo, $hi]，并发 $par  $(date -u +%FT%TZ)"
  for ((k=0; k<par; k++)); do launch "t${t}_s${k}" "$lo" "$hi" "$k/$par"; done
  wait_tier
done
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== run done $(date -u +%FT%TZ) → 收尾 bash scripts/ucr_pod/finish.sh"
