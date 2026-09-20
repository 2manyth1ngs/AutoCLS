#!/usr/bin/env bash
# WP17-ACTS S1″ 在实验室 8k-4090 跑一个阶段：PHASE=W（整集口径对照 160 格）或 C（主批）。
#   83 --run --phase $PHASE 按 GPUS 列表分进程（按任务估时分片，各占一张卡，共用 runs.jsonl）→ 等待 → --probe0 --phase → --verify --phase
#   （不过则单进程 --resume 补跑）→ 打包 upload/out_acts_s1pp_<PHASE>.tar.gz。不自停（非计费机器）。
#   PHASE=W GPUS="0 1" nohup bash .../scripts/pa_pod/acts_lab_run_s1pp.sh > .../logs/acts_s1pp_W.log 2>&1 &
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"
LOG="$REPO/logs"; UP="$REPO/upload"; OUT="results/acts/s1pp"
PHASE="${PHASE:?W|C}"; GPUS="${GPUS:-0 1}"
GPU0=$(echo $GPUS | cut -d" " -f1)
NP=$(echo $GPUS | wc -w)
COMMON="--profile s1pp --outdir $OUT --phase $PHASE"
cd "$REPO"; mkdir -p "$LOG" "$UP" "$OUT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in $GPUS; do
  CUDA_VISIBLE_DEVICES=$g python - <<'PY' || { echo "CUDA gate FAILED on GPU $g"; exit 5; }
import torch; assert torch.cuda.is_available(); x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA", torch.cuda.get_device_name(0))
PY
done
echo "== [$(date -u +%H:%M:%S)] phase $PHASE start ($NP procs, GPUs $GPUS)"
PIDS=""; i=0
for g in $GPUS; do
  CUDA_VISIBLE_DEVICES=$g nohup nice -n 10 python scripts/83_acts_corpus.py --run --resume --device cuda $COMMON --parts $NP --part $i > "$LOG/acts_s1pp_${PHASE}_p$i.log" 2>&1 < /dev/null &
  PIDS="$PIDS $!"; i=$((i+1)); sleep 20
done
echo "pids$PIDS"
alive=1; while [ $alive = 1 ]; do alive=0; for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && alive=1; done; [ $alive = 1 ] && sleep 60; done
echo "== [$(date -u +%H:%M:%S)] all exited; rows $(wc -l < "$OUT/runs.jsonl")"
CUDA_VISIBLE_DEVICES=$GPU0 nice -n 10 python scripts/83_acts_corpus.py --probe0 --resume --device cuda $COMMON > "$LOG/acts_s1pp_${PHASE}_probe0.log" 2>&1
python scripts/83_acts_corpus.py --verify --device cpu $COMMON > "$LOG/acts_s1pp_${PHASE}_verify.log" 2>&1; cat "$LOG/acts_s1pp_${PHASE}_verify.log"
if ! grep -q "→ OK" "$LOG/acts_s1pp_${PHASE}_verify.log"; then
  echo "== verify NOT OK → 单进程 --resume 补跑"
  CUDA_VISIBLE_DEVICES=$GPU0 nice -n 10 python scripts/83_acts_corpus.py --run --resume --device cuda $COMMON >> "$LOG/acts_s1pp_${PHASE}_fix.log" 2>&1
  CUDA_VISIBLE_DEVICES=$GPU0 nice -n 10 python scripts/83_acts_corpus.py --probe0 --resume --device cuda $COMMON >> "$LOG/acts_s1pp_${PHASE}_fix.log" 2>&1
  python scripts/83_acts_corpus.py --verify --device cpu $COMMON > "$LOG/acts_s1pp_${PHASE}_verify.log" 2>&1; cat "$LOG/acts_s1pp_${PHASE}_verify.log"
fi
tar -czf "$UP/out_acts_s1pp_${PHASE}.tar.gz" "$OUT/runs.jsonl" "$OUT/probe0.jsonl" $( [ -f "$OUT/oom_failures.jsonl" ] && echo "$OUT/oom_failures.jsonl" ) "$LOG"/acts_s1pp_*.log
ls -la "$UP/out_acts_s1pp_${PHASE}.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s1pp phase $PHASE done"
