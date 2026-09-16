#!/usr/bin/env bash
# WP16 K3 检验上云（5090）：75 --rprime 分 3 进程按数据集并行（各自缓存目录）→ 等待 → 合并进主目录 → --verify → 打包。排名 / 训练 / 分析在本地做。
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/k3_run.sh > /workspace/logs/k3_run.log 2>&1 &
# 全程 --resume 可续。用完必停：本地 pod_ctl.py stop + confirm-stopped 3（memory runpod-api-and-stop-rule）。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
UP="${UP:-/workspace/upload_k3}"
OUT="results/tclsc_k3"
cd "$REPO"; mkdir -p "$LOG" "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY' || { echo "CUDA gate FAILED"; exit 5; }
import torch; assert torch.cuda.is_available(); x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA", torch.cuda.get_device_name(0))
PY
echo "== [$(date -u +%H:%M:%S)] rprime start (3 procs)"
nohup python scripts/75_tclsc_k3.py --rprime --resume --device cuda --datasets EthanolConcentration > "$LOG/k3_rprime_p1.log" 2>&1 < /dev/null &
P1=$!
nohup python scripts/75_tclsc_k3.py --rprime --resume --device cuda --datasets PhonemeSpectra UniMiB-SHAR --outdir results/tclsc_k3_p2 > "$LOG/k3_rprime_p2.log" 2>&1 < /dev/null &
P2=$!
nohup python scripts/75_tclsc_k3.py --rprime --resume --device cuda --datasets Handwriting UWaveGestureLibrary Cricket HandMovementDirection SelfRegulationSCP2 --outdir results/tclsc_k3_p3 > "$LOG/k3_rprime_p3.log" 2>&1 < /dev/null &
P3=$!
echo "pids $P1 $P2 $P3"
while kill -0 "$P1" 2>/dev/null || kill -0 "$P2" 2>/dev/null || kill -0 "$P3" 2>/dev/null; do sleep 60; done
echo "== [$(date -u +%H:%M:%S)] all rprime processes exited"
for p in p2 p3; do
  f="results/tclsc_k3_$p/rprime_k3.jsonl"
  [ -f "$f" ] && { cat "$f" >> "$OUT/rprime_k3.jsonl"; echo "merged $p: $(wc -l < "$f") rows"; }
  [ -f "results/tclsc_k3_$p/oom_failures.jsonl" ] && cat "results/tclsc_k3_$p/oom_failures.jsonl" >> "$OUT/oom_failures.jsonl"
done
echo "== main rows: $(wc -l < "$OUT/rprime_k3.jsonl")"
python scripts/75_tclsc_k3.py --verify > "$LOG/k3_verify.log" 2>&1; cat "$LOG/k3_verify.log"
tar -czf "$UP/out_k3.tar.gz" "$OUT/rprime_k3.jsonl" $( [ -f "$OUT/oom_failures.jsonl" ] && echo "$OUT/oom_failures.jsonl" ) "$LOG"/k3_*.log
ls -la "$UP/out_k3.tar.gz"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] k3 done"
