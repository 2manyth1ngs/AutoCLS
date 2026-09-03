#!/usr/bin/env bash
# 全量协议对齐上云：按计划 §4 顺序串行跑完全部 GPU 阶段（pod 上，务必脱离启动）：
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/run.sh > /workspace/logs/pa_run.log 2>&1 &
# 顺序：eval-audit → reproduce → bs-scan → official-ts2vec → [ladder，仅 P1 不过] → targets → analyze。
# 全程 --resume 可续（被杀后原样重跑本脚本即可）。
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

stage eval_audit --eval-audit
stage reproduce  --reproduce
stage bs_scan    --bs-scan
stage official   --official-ts2vec
stage analyze1   --analyze

P1=$(python - <<'PY'
import json; a = json.load(open("results/protocol_alignment/analysis.json", encoding="utf-8"))
print(a.get("reproduce", {}).get("P1"))
PY
)
echo "== P1 = $P1"
if [ "$P1" != "True" ]; then
  echo "== P1 不过 → 阶梯"
  stage ladder --ladder
fi

stage targets  --targets
stage analyze2 --analyze
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] run done：拉回 $OUT/{runs.jsonl,oom_failures.jsonl,analysis.json,tables.md,picks.json}"
