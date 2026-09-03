#!/usr/bin/env bash
# MX-4 WP3 上云：S1 目录（三截点）→ analyze（出 picks.json）→ S2 冠军 seeds {1,2} → analyze。
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/wp3_run.sh > /workspace/logs/wp3_run.log 2>&1 &
# 全程 --resume 可续（被杀后原样重跑本脚本）。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
OUT="${OUT:-results/mx4_deploy}"
cd "$REPO"; mkdir -p "$LOG" "$OUT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864
export AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() {  # stage <名字> <参数...>
  local name="$1"; shift
  echo "== [$(date -u +%H:%M:%S)] $name"
  python scripts/39_sh_deploy.py "$@" --outdir "$OUT" --joblib-backend "${JOBLIB_BACKEND:-loky}" \
      >> "$LOG/wp3_$name.log" 2>&1 || echo "  ⚠️ $name 非零退出（见 $LOG/wp3_$name.log）"
  tail -3 "$LOG/wp3_$name.log"
}

stage catalog  --catalog --resume
stage analyze1 --analyze
stage finals   --finals --resume
stage analyze2 --analyze
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] wp3 done：拉回 $OUT/{runs.jsonl,oom_failures.jsonl,analysis.json,picks.json,tables.md}"
