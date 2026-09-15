#!/usr/bin/env bash
# WP16 SX 检验上云（5090）：74 --rprime（42 切片 × 21 条轨迹，≤ 3 次 --resume 重试）→ --verify（不过则停，不进后续）→ --baseline → --features
# → --train S9/S12（6 种子 × 7 折 × 2 臂）→ 打包。排名 / 分析在本地做（S6 模型在本地）。
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/sx_run.sh > /workspace/logs/sx_run.log 2>&1 &
# 全程 --resume 可续。用完必停：本地 pod_ctl.py stop + confirm-stopped 3（memory runpod-api-and-stop-rule）。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
UP="${UP:-/workspace/upload_sx}"
OUT="results/tclsc_slices"
cd "$REPO"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gate() { python - <<'PY' || { echo "CUDA gate FAILED $(date -u +%H:%M:%S)"; return 5; }
import torch; assert torch.cuda.is_available(); x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA", torch.cuda.get_device_name(0))
PY
}
gate || exit 5
echo "== [$(date -u +%H:%M:%S)] rprime start"
for attempt in 1 2 3; do
  python scripts/74_tclsc_slices.py --rprime --resume --device cuda >> "$LOG/sx_rprime.log" 2>&1 && break
  echo "  ⚠️ rprime attempt $attempt 非零退出（见 $LOG/sx_rprime.log）"; tail -3 "$LOG/sx_rprime.log"; sleep 30; gate || exit 5
done
python scripts/74_tclsc_slices.py --verify > "$LOG/sx_verify.log" 2>&1; cat "$LOG/sx_verify.log"
grep -q "→ OK" "$LOG/sx_verify.log" || { echo "!! verify NOT OK：停在 rprime，不进后续（--resume / --retry 手工处理）"; exit 7; }
echo "== [$(date -u +%H:%M:%S)] baseline"
python scripts/74_tclsc_slices.py --baseline --resume --device cuda >> "$LOG/sx_baseline.log" 2>&1 || echo "  ⚠️ baseline 非零退出"
tail -1 "$LOG/sx_baseline.log"
echo "== [$(date -u +%H:%M:%S)] features"
python scripts/74_tclsc_slices.py --features --resume --device cuda >> "$LOG/sx_features.log" 2>&1 || echo "  ⚠️ features 非零退出"
tail -1 "$LOG/sx_features.log"
n_ctx=$(wc -l < "$OUT/ctx_ext.jsonl" 2>/dev/null || echo 0)
echo "== ctx_ext rows: $n_ctx / 42"
if [ "$n_ctx" = "42" ]; then
  echo "== [$(date -u +%H:%M:%S)] train S9 / S12"
  python scripts/74_tclsc_slices.py --train --resume --points 9 12 --device cuda >> "$LOG/sx_train.log" 2>&1 || echo "  ⚠️ train 非零退出"
  tail -2 "$LOG/sx_train.log"
else
  echo "!! ctx_ext 不全，跳过训练"
fi
tar -czf "$UP/out_sx.tar.gz" "$OUT/rprime_cache.jsonl" "$OUT/baseline_cache.jsonl" "$OUT/ctx_ext.jsonl" "$OUT/tasks.json" "$OUT/cls_catalog.json" \
    $( [ -f "$OUT/oom_failures.jsonl" ] && echo "$OUT/oom_failures.jsonl" ) $( [ -d "$OUT/models" ] && echo "$OUT/models" ) $( [ -d "$OUT/runs" ] && echo "$OUT/runs" ) \
    $( [ -f "$OUT/train.log" ] && echo "$OUT/train.log" ) "$LOG"/sx_*.log
ls -la "$UP/out_sx.tar.gz"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] sx done"
