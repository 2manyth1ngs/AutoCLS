#!/usr/bin/env bash
# WP9-P1 路径一上云（4090）：S1 随机格 297（seed 0 三截点）→ S2 模拟（全部臂 + 基线首次出数）→ 分析（门 / 预测 / picks）
# → S3 finals seeds {1,2}（复用三源已有终评行）→ 分析。
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/p1_run.sh > /workspace/logs/p1_run.log 2>&1 &
# 全程 --resume 可续（被杀后原样重跑本脚本）。靶清单与顺序由 configs/p1.yaml 决定（46 号 run_targets）；
# 候选池 results/mx4_p1/pool.json 是协议锁产物（bundle 带上，pod 上不重建）。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
OUT="${OUT:-results/mx4_p1}"
cd "$REPO"; mkdir -p "$LOG" "$OUT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864
export AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage46() {  # stage46 <名字> <参数...>
  local name="$1"; shift
  echo "== [$(date -u +%H:%M:%S)] $name"
  python scripts/46_p1_pool.py "$@" --outdir "$OUT" --joblib-backend "${JOBLIB_BACKEND:-loky}" \
      >> "$LOG/p1_$name.log" 2>&1 || echo "  ⚠️ $name 非零退出（见 $LOG/p1_$name.log）"
  tail -3 "$LOG/p1_$name.log"
}

stage46 run      --run --resume
stage46 simulate --simulate
stage46 analyze1 --analyze
stage46 finals   --finals --resume
stage46 analyze2 --analyze
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] p1 done：拉回 $OUT/{runs.jsonl,oom_failures.jsonl,simulate.json,analysis.json,picks.json,finals_reuse.json,tables.md} 与 $LOG/p1_*.log"
# 死规定：拉回后本地 `python scripts/46_p1_pool.py --simulate --analyze` 逐字节复现 simulate.json / analysis.json / picks.json 之后，
# 才 `python scripts/pa_pod/pod_ctl.py stop <pod_id>` + `python scripts/pa_pod/pod_ctl.py confirm-stopped <pod_id> 3`（memory runpod-api-and-stop-rule）
