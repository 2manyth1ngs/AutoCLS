#!/usr/bin/env bash
# WP7-UCR 上云收尾：串行重扫（--on-oom record）→ iters=0 基线 → --verify。
#   bash /workspace/wp/CLS/scripts/ucr_pod/finish.sh
set -euo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
OUT="${OUT:-results/ucr_rprime}"
cd "$REPO"; mkdir -p "$LOG" "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864
export AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== 串行重扫（worker=1，--on-oom record）"
python scripts/41_ucr_rprime.py --fill --resume --on-oom record --outdir "$OUT" \
      --inflight "$OUT/inflight_finish.txt" >> "$LOG/ucr_finish.log" 2>> "$LOG/ucr_finish.err" || echo "  ⚠️ 重扫非零退出"
echo "== iters=0 基线 × 5 种子"
python scripts/41_ucr_rprime.py --baseline --resume --outdir "$OUT" >> "$LOG/ucr_finish.log" 2>> "$LOG/ucr_finish.err" || echo "  ⚠️ 基线非零退出"
echo "== verify"
if python scripts/41_ucr_rprime.py --verify --outdir "$OUT" 2>&1 | tee -a "$LOG/ucr_finish.log"; then
  echo "== verify 通过；拉回 $OUT/{rprime_cache.jsonl,baseline_cache.jsonl,tasks.json,verify.json,oom_failures.jsonl} 后**立刻 stop pod**"
else
  echo "== ❌ verify 未通过：看 $OUT/verify.json 的 terminal / partial 清单"
fi
