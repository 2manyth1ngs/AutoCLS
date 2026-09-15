#!/usr/bin/env bash
# WP16 U 检验上云（4090）：72 --features（23 靶官方 TRAIN GGS 探针 E_T，EigenWorms 截 3000）→ 72 --simulate（三名单 SH6-3-3 seed 0 + picks.json）
# → 45 --finals（新冠军 × seeds {1,2}，官方口径 ts2vec_official）→ 打包。
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/u23_run.sh > /workspace/logs/u23_run.log 2>&1 &
# 全程 --resume 可续。用完必停：本地 pod_ctl.py stop + confirm-stopped 3（memory runpod-api-and-stop-rule）。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
UP="${UP:-/workspace/upload_u23}"
OUT="results/tclsc_s1et_uea30"
cd "$REPO"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY' || { echo "CUDA gate FAILED"; exit 5; }
import torch; assert torch.cuda.is_available(); print("CUDA", torch.cuda.get_device_name(0))
PY
echo "== [$(date -u +%H:%M:%S)] features start"
python scripts/72_tclsc_s1et_uea30.py --features --resume --device cuda >> "$LOG/u23_features.log" 2>&1 || echo "  ⚠️ features 非零退出（见 $LOG/u23_features.log）"
tail -2 "$LOG/u23_features.log"
echo "== [$(date -u +%H:%M:%S)] simulate"
python scripts/72_tclsc_s1et_uea30.py --simulate >> "$LOG/u23_simulate.log" 2>&1 || echo "  ⚠️ simulate 非零退出"
tail -2 "$LOG/u23_simulate.log"
echo "== [$(date -u +%H:%M:%S)] finals start"
python scripts/45_uea30_deploy.py --finals --resume --device cuda --outdir "$OUT" --joblib-backend loky >> "$LOG/u23_finals.log" 2>&1 \
  || echo "  ⚠️ finals 非零退出（见 $LOG/u23_finals.log）"
tail -3 "$LOG/u23_finals.log"
n=$(grep -c '"stage": "s2_finals"' "$OUT/runs.jsonl" 2>/dev/null || echo 0)
echo "== [$(date -u +%H:%M:%S)] finals rows: $n"
tar -czf "$UP/out_u23.tar.gz" "$OUT/et_uea30.jsonl" "$OUT/sim.json" "$OUT/picks.json" $( [ -f "$OUT/runs.jsonl" ] && echo "$OUT/runs.jsonl" ) \
    $( [ -f "$OUT/oom_failures.jsonl" ] && echo "$OUT/oom_failures.jsonl" ) "$LOG/u23_features.log" "$LOG/u23_simulate.log" "$LOG/u23_finals.log"
ls -la "$UP/out_u23.tar.gz"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] u23 done"
