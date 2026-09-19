#!/usr/bin/env bash
# WP17-ACTS S1 上云（5090）：83 --run 分 2 进程（按任务估时分片，共用 runs.jsonl）→ 等待 → --probe0 → --verify（不过单进程 --resume 补跑）
# → 打包 → 自停守护（runpodctl 在 pod 上可用时停掉本 pod；GEO-S 教训：收官后空转 9.7 h）。
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/acts_run.sh > /workspace/logs/acts_run.log 2>&1 &
# 用完仍须本地 pod_ctl.py stop + confirm-stopped 3（memory runpod-api-and-stop-rule）——自停只是兜底。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
UP="${UP:-/workspace/upload_acts}"
OUT="results/acts/s1"
cd "$REPO"; mkdir -p "$LOG" "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY' || { echo "CUDA gate FAILED"; exit 5; }
import torch; assert torch.cuda.is_available(); x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA", torch.cuda.get_device_name(0))
PY
echo "== [$(date -u +%H:%M:%S)] run start (2 procs)"
nohup python scripts/83_acts_corpus.py --run --resume --device cuda --parts 2 --part 0 > "$LOG/acts_p0.log" 2>&1 < /dev/null &
P0=$!
sleep 20
nohup python scripts/83_acts_corpus.py --run --resume --device cuda --parts 2 --part 1 > "$LOG/acts_p1.log" 2>&1 < /dev/null &
P1=$!
echo "pids $P0 $P1"
while kill -0 "$P0" 2>/dev/null || kill -0 "$P1" 2>/dev/null; do sleep 60; done
echo "== [$(date -u +%H:%M:%S)] both exited; rows $(wc -l < "$OUT/runs.jsonl")"
python scripts/83_acts_corpus.py --probe0 --resume --device cuda > "$LOG/acts_probe0.log" 2>&1
python scripts/83_acts_corpus.py --verify > "$LOG/acts_verify.log" 2>&1; cat "$LOG/acts_verify.log"
if ! grep -q "→ OK" "$LOG/acts_verify.log"; then
  echo "== verify NOT OK → 单进程 --resume 补跑"
  python scripts/83_acts_corpus.py --run --resume --device cuda >> "$LOG/acts_fix.log" 2>&1
  python scripts/83_acts_corpus.py --probe0 --resume --device cuda >> "$LOG/acts_fix.log" 2>&1
  python scripts/83_acts_corpus.py --verify > "$LOG/acts_verify.log" 2>&1; cat "$LOG/acts_verify.log"
fi
tar -czf "$UP/out_acts_s1.tar.gz" "$OUT/runs.jsonl" "$OUT/probe0.jsonl" $( [ -f "$OUT/oom_failures.jsonl" ] && echo "$OUT/oom_failures.jsonl" ) "$LOG"/acts_*.log
ls -la "$UP/out_acts_s1.tar.gz"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] acts s1 done"
# 自停兜底：给本地留 30 分钟拉包窗口后停 pod（本地 pod_ctl.py stop 仍是正式收尾）
if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
  echo "== self-stop armed: runpodctl stop pod $RUNPOD_POD_ID in 30 min"
  (sleep 1800; runpodctl stop pod "$RUNPOD_POD_ID") > "$LOG/acts_selfstop.log" 2>&1 &
fi
