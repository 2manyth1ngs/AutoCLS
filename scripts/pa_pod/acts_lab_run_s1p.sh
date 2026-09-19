#!/usr/bin/env bash
# WP17-ACTS S1′ 在实验室 8k-4090 跑一个阶段：PHASE=A（准入块 + Cricket 3× 消融）或 B（主批）。
#   83 --run --phase $PHASE 分 2 进程（按任务估时分片，各占一张卡，共用 runs.jsonl）→ 等待 → --probe0 --phase → --verify --phase
#   （不过则单进程 --resume 补跑）→ 打包 upload/out_acts_s1p_<PHASE>.tar.gz。不自停（非计费机器）。
#   PHASE=A GPU0=0 GPU1=1 nohup bash .../scripts/pa_pod/acts_lab_run_s1p.sh > .../logs/acts_s1p_A.log 2>&1 &
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"
LOG="$REPO/logs"; UP="$REPO/upload"; OUT="results/acts/s1p"
PHASE="${PHASE:?A|B}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
COMMON="--profile s1p --outdir $OUT --phase $PHASE"
cd "$REPO"; mkdir -p "$LOG" "$UP" "$OUT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in "$GPU0" "$GPU1"; do
  CUDA_VISIBLE_DEVICES=$g python - <<'PY' || { echo "CUDA gate FAILED on GPU $g"; exit 5; }
import torch; assert torch.cuda.is_available(); x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA", torch.cuda.get_device_name(0))
PY
done
echo "== [$(date -u +%H:%M:%S)] phase $PHASE start (2 procs, GPU $GPU0 / $GPU1)"
CUDA_VISIBLE_DEVICES=$GPU0 nohup nice -n 10 python scripts/83_acts_corpus.py --run --resume --device cuda $COMMON --parts 2 --part 0 > "$LOG/acts_s1p_${PHASE}_p0.log" 2>&1 < /dev/null &
P0=$!
sleep 20
CUDA_VISIBLE_DEVICES=$GPU1 nohup nice -n 10 python scripts/83_acts_corpus.py --run --resume --device cuda $COMMON --parts 2 --part 1 > "$LOG/acts_s1p_${PHASE}_p1.log" 2>&1 < /dev/null &
P1=$!
echo "pids $P0 $P1"
while kill -0 "$P0" 2>/dev/null || kill -0 "$P1" 2>/dev/null; do sleep 60; done
echo "== [$(date -u +%H:%M:%S)] both exited; rows $(wc -l < "$OUT/runs.jsonl")"
CUDA_VISIBLE_DEVICES=$GPU0 nice -n 10 python scripts/83_acts_corpus.py --probe0 --resume --device cuda $COMMON > "$LOG/acts_s1p_${PHASE}_probe0.log" 2>&1
python scripts/83_acts_corpus.py --verify --device cpu $COMMON > "$LOG/acts_s1p_${PHASE}_verify.log" 2>&1; cat "$LOG/acts_s1p_${PHASE}_verify.log"
if ! grep -q "→ OK" "$LOG/acts_s1p_${PHASE}_verify.log"; then
  echo "== verify NOT OK → 单进程 --resume 补跑"
  CUDA_VISIBLE_DEVICES=$GPU0 nice -n 10 python scripts/83_acts_corpus.py --run --resume --device cuda $COMMON >> "$LOG/acts_s1p_${PHASE}_fix.log" 2>&1
  CUDA_VISIBLE_DEVICES=$GPU0 nice -n 10 python scripts/83_acts_corpus.py --probe0 --resume --device cuda $COMMON >> "$LOG/acts_s1p_${PHASE}_fix.log" 2>&1
  python scripts/83_acts_corpus.py --verify --device cpu $COMMON > "$LOG/acts_s1p_${PHASE}_verify.log" 2>&1; cat "$LOG/acts_s1p_${PHASE}_verify.log"
fi
tar -czf "$UP/out_acts_s1p_${PHASE}.tar.gz" "$OUT/runs.jsonl" "$OUT/probe0.jsonl" $( [ -f "$OUT/oom_failures.jsonl" ] && echo "$OUT/oom_failures.jsonl" ) "$LOG"/acts_s1p_*.log
ls -la "$UP/out_acts_s1p_${PHASE}.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s1p phase $PHASE done"
