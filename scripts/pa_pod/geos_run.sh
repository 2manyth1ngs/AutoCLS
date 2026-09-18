#!/usr/bin/env bash
# WP16 GEO-S 检验上云（5090）：82 --run 分 2 进程（按估时贪心分靶，共用 runs.jsonl）→ 等待 → --verify（不过单进程 --resume 补跑）→ 打包。分析在本地做。
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/geos_run.sh > /workspace/logs/geos_run.log 2>&1 &
# 用完必停：本地 pod_ctl.py stop + confirm-stopped 3（memory runpod-api-and-stop-rule）。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
UP="${UP:-/workspace/upload_geos}"
OUT="results/tclsc_geos"
cd "$REPO"; mkdir -p "$LOG" "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY' || { echo "CUDA gate FAILED"; exit 5; }
import torch; assert torch.cuda.is_available(); x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA", torch.cuda.get_device_name(0))
PY
echo "== [$(date -u +%H:%M:%S)] run start (2 procs)"
nohup python scripts/82_tclsc_geos.py --run --resume --device cuda --parts 2 --part 0 > "$LOG/geos_p0.log" 2>&1 < /dev/null &
P0=$!
sleep 20
nohup python scripts/82_tclsc_geos.py --run --resume --device cuda --parts 2 --part 1 > "$LOG/geos_p1.log" 2>&1 < /dev/null &
P1=$!
echo "pids $P0 $P1"
while kill -0 "$P0" 2>/dev/null || kill -0 "$P1" 2>/dev/null; do sleep 60; done
echo "== [$(date -u +%H:%M:%S)] both exited; rows $(wc -l < "$OUT/runs.jsonl")"
python scripts/82_tclsc_geos.py --verify > "$LOG/geos_verify.log" 2>&1; cat "$LOG/geos_verify.log"
if ! grep -q "→ OK" "$LOG/geos_verify.log"; then
  echo "== verify NOT OK → 单进程 --resume 补跑"
  python scripts/82_tclsc_geos.py --run --resume --device cuda >> "$LOG/geos_fix.log" 2>&1
  python scripts/82_tclsc_geos.py --verify > "$LOG/geos_verify.log" 2>&1; cat "$LOG/geos_verify.log"
fi
tar -czf "$UP/out_geos.tar.gz" "$OUT/runs.jsonl" $( [ -f "$OUT/oom_failures.jsonl" ] && echo "$OUT/oom_failures.jsonl" ) "$LOG"/geos_*.log
ls -la "$UP/out_geos.tar.gz"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] geos done"
