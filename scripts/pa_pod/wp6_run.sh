#!/usr/bin/env bash
# MX-4 WP6 S1 上云（4090）：官方口径孪生目录 125 格 → 部署分析（出第一批 picks）→ finals seeds {1,2} → 部署分析。
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/wp6_run.sh > /workspace/logs/wp6_run.log 2>&1 &
# 全程 --resume 可续（被杀后原样重跑本脚本）。掩码关半边来自 results/mx4_deploy/runs.jsonl（WP3，只读）。
# 第一批 finals 只覆盖与源侧孪生无关的臂（interleave / measured / allon / SH10）；prior20 / gbdt20 的冠军
# 若是新配置，S2 之后本地 40 --deploy 出第二批 picks 再补一次 --finals。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
OUT="${OUT:-results/mx4_mask}"
cd "$REPO"; mkdir -p "$LOG" "$OUT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864
export AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage39() {  # stage39 <名字> <参数...>
  local name="$1"; shift
  echo "== [$(date -u +%H:%M:%S)] $name"
  python scripts/39_sh_deploy.py "$@" --outdir "$OUT" --joblib-backend "${JOBLIB_BACKEND:-loky}" \
      --twins input_mask=binomial --legacy-runs results/mx4_deploy/runs.jsonl \
      >> "$LOG/wp6_$name.log" 2>&1 || echo "  ⚠️ $name 非零退出（见 $LOG/wp6_$name.log）"
  tail -3 "$LOG/wp6_$name.log"
}
stage40() {
  local name="$1"; shift
  echo "== [$(date -u +%H:%M:%S)] $name"
  python scripts/40_mask_field_sim.py "$@" --outdir "$OUT" >> "$LOG/wp6_$name.log" 2>&1 \
      || echo "  ⚠️ $name 非零退出（见 $LOG/wp6_$name.log）"
  tail -3 "$LOG/wp6_$name.log"
}

stage39 catalog  --catalog --twin-only --resume
stage40 deploy1  --deploy
stage39 finals   --finals --resume
stage40 deploy2  --deploy
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] wp6 S1 done：拉回 $OUT/{runs.jsonl,oom_failures.jsonl,analysis_deploy.json,picks.json,tables_deploy.md,c0.json} 与 $LOG/wp6_*.log"
