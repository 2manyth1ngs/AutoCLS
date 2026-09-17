#!/usr/bin/env bash
# WP16 NB1 检验上云（5090）：76 --rprime 分 3 进程按数据集并行（各自缓存目录）→ 等待 → 合并进主目录 → --verify → 打包。训练 / 排名 / 分析在本地做。
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/nb_run.sh > /workspace/logs/nb_run.log 2>&1 &
# 全程 --resume 可续。用完必停：本地 pod_ctl.py stop + confirm-stopped 3（memory runpod-api-and-stop-rule）。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
UP="${UP:-/workspace/upload_nb}"
OUT="results/tclsc_nb"
cd "$REPO"; mkdir -p "$LOG" "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY' || { echo "CUDA gate FAILED"; exit 5; }
import torch; assert torch.cuda.is_available(); x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA", torch.cuda.get_device_name(0))
PY
echo "== [$(date -u +%H:%M:%S)] rprime start (3 procs)"
for p in p2 p3; do mkdir -p "results/tclsc_nb_$p" && cp "$OUT/nb_catalog.json" "results/tclsc_nb_$p/"; done
nohup python scripts/76_tclsc_nb.py --rprime --resume --device cuda --datasets EthanolConcentration SelfRegulationSCP2 > "$LOG/nb_rprime_p1.log" 2>&1 < /dev/null &
P1=$!
nohup python scripts/76_tclsc_nb.py --rprime --resume --device cuda --datasets PhonemeSpectra UniMiB-SHAR UWaveGestureLibrary --outdir results/tclsc_nb_p2 > "$LOG/nb_rprime_p2.log" 2>&1 < /dev/null &
P2=$!
nohup python scripts/76_tclsc_nb.py --rprime --resume --device cuda --datasets Handwriting Cricket HandMovementDirection --outdir results/tclsc_nb_p3 > "$LOG/nb_rprime_p3.log" 2>&1 < /dev/null &
P3=$!
echo "pids $P1 $P2 $P3"
while kill -0 "$P1" 2>/dev/null || kill -0 "$P2" 2>/dev/null || kill -0 "$P3" 2>/dev/null; do sleep 60; done
echo "== [$(date -u +%H:%M:%S)] all rprime processes exited"
for p in p2 p3; do
  f="results/tclsc_nb_$p/rprime_nb.jsonl"
  [ -f "$f" ] && { cat "$f" >> "$OUT/rprime_nb.jsonl"; echo "merged $p: $(wc -l < "$f") rows"; }
  [ -f "results/tclsc_nb_$p/oom_failures.jsonl" ] && cat "results/tclsc_nb_$p/oom_failures.jsonl" >> "$OUT/oom_failures.jsonl"
done
echo "== main rows: $(wc -l < "$OUT/rprime_nb.jsonl")"
python scripts/76_tclsc_nb.py --verify > "$LOG/nb_verify.log" 2>&1; cat "$LOG/nb_verify.log"
if ! grep -q "→ OK" "$LOG/nb_verify.log"; then
  echo "== verify NOT OK → 单进程 --resume 补跑缺格"
  python scripts/76_tclsc_nb.py --rprime --resume --device cuda >> "$LOG/nb_rprime_fix.log" 2>&1
  python scripts/76_tclsc_nb.py --verify > "$LOG/nb_verify.log" 2>&1; cat "$LOG/nb_verify.log"
fi
tar -czf "$UP/out_nb.tar.gz" "$OUT/rprime_nb.jsonl" $( [ -f "$OUT/oom_failures.jsonl" ] && echo "$OUT/oom_failures.jsonl" ) "$LOG"/nb_*.log
ls -la "$UP/out_nb.tar.gz"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] nb done"
