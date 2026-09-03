#!/usr/bin/env bash
# 全量协议对齐上云：run.sh 收官后的第二遍（REPORT §0.7）——若阶梯采纳了成分，六靶再跑
# `--targets --ladder-overrides <rung>`，然后 --analyze。无采纳成分则只 analyze 后退出。
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/run2.sh > /workspace/logs/pa_run2.log 2>&1 &
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
OUT="${OUT:-results/protocol_alignment}"
cd "$REPO"; mkdir -p "$LOG" "$OUT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864
export AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() {  # stage <名字> <参数...>
  local name="$1"; shift
  echo "== [$(date -u +%H:%M:%S)] $name"
  python scripts/35_protocol_alignment.py "$@" --resume --outdir "$OUT" \
      --joblib-backend "${JOBLIB_BACKEND:-loky}" \
      >> "$LOG/pa_$name.log" 2>&1 || echo "  ⚠️ $name 非零退出（见 $LOG/pa_$name.log）"
  tail -3 "$LOG/pa_$name.log"
}

stage analyze3 --analyze
RUNG=$(python - <<'PY'
import json; a = json.load(open("results/protocol_alignment/analysis.json", encoding="utf-8"))
print(a.get("ladder_adopted_rung") or "")
PY
)
echo "== adopted rung = '${RUNG}'"
if [ -n "$RUNG" ]; then
  stage targets_ladder --targets --ladder-overrides "$RUNG"
  stage analyze4 --analyze
else
  echo "== 阶梯无采纳成分，跳过第二遍"
fi
echo "== [$(date -u +%H:%M:%S)] run2 done"
