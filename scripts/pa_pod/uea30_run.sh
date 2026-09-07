#!/usr/bin/env bash
# WP10-UEA30 上云（4090）：官方口径 23 靶 × 22 格 目录（S1，seed 0 三截点）→ 分析（臂 / 门 / picks）→ finals seeds {1,2}（S2）
# → 分析 → table30。
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/uea30_run.sh > /workspace/logs/uea30_run.log 2>&1 &
# 全程 --resume 可续（被杀后原样重跑本脚本）。靶清单与顺序由 configs/uea30.yaml 决定（45 号默认 run: true 按 order），
# 候选目录 results/uea30/candidates.json 是协议锁产物（bundle 带上，pod 上不重建）。
# pod 上 --table30 只含 23 个新靶：6 个 LODO 靶（results/mx4_deploy + mx4_mask）与 HAR（mx4_deploy_har）不在本包里，
# 45 号自动跳过；全 30 合并表以拉回后本地 --table30 为准。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
OUT="${OUT:-results/uea30}"
cd "$REPO"; mkdir -p "$LOG" "$OUT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864
export AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage45() {  # stage45 <名字> <参数...>
  local name="$1"; shift
  echo "== [$(date -u +%H:%M:%S)] $name"
  python scripts/45_uea30_deploy.py "$@" --outdir "$OUT" --joblib-backend "${JOBLIB_BACKEND:-loky}" \
      >> "$LOG/uea30_$name.log" 2>&1 || echo "  ⚠️ $name 非零退出（见 $LOG/uea30_$name.log）"
  tail -3 "$LOG/uea30_$name.log"
}

stage45 catalog  --catalog --resume
stage45 analyze1 --analyze
stage45 finals   --finals --resume
stage45 analyze2 --analyze
stage45 table30  --table30
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] uea30 done：拉回 $OUT/{runs.jsonl,oom_failures.jsonl,analysis.json,picks.json,tables.md,table30.md,table30.json,candidates.json} 与 $LOG/uea30_*.log"
# 死规定：拉回后本地 `python scripts/45_uea30_deploy.py --analyze` 逐字节复现 analysis.json / picks.json / tables.md 之后，才 `python scripts/pa_pod/pod_ctl.py stop <pod_id>` + `python scripts/pa_pod/pod_ctl.py confirm-stopped <pod_id> 3`（memory runpod-api-and-stop-rule）
