#!/usr/bin/env bash
# WP16 EV1 检验上云（5090，SetTF 段）：77 --fit-folds（7 折 TS2Vec + 87 切片特征）→ --train-es 2 进程分折（各 ~11 GB 显存）→ --evolve ev_es → --rprime ev_es → --verify → --rank-es → 打包。
#   nohup bash /workspace/wp/CLS/scripts/pa_pod/ev_run.sh > /workspace/logs/ev_run.log 2>&1 &
# 全程 --resume 可续。用完必停：本地 pod_ctl.py stop + confirm-stopped 3（memory runpod-api-and-stop-rule）。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
UP="${UP:-/workspace/upload_ev}"
OUT="results/tclsc_ev"
cd "$REPO"; mkdir -p "$LOG" "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY' || { echo "CUDA gate FAILED"; exit 5; }
import torch; assert torch.cuda.is_available(); x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA", torch.cuda.get_device_name(0))
PY
# Ethanol 段的 E_T / 任务盲 / rand_u 提名（本地 4060 上单条 1369 s = pod 的 10×）改在 pod 上并行跑（18 条 ≈ 40 min，与 fit-folds / train-es 并行，显存 ≈ 6 GB）
echo "== [$(date -u +%H:%M:%S)] rprime Ethanol (ev_et / ev_blind / rand_u) start in background"
nohup python scripts/77_tclsc_ev.py --rprime --guides ev_et ev_blind --random --datasets EthanolConcentration --device cuda > "$LOG/ev_rprime_ethanol.log" 2>&1 < /dev/null &
PE=$!
echo "== [$(date -u +%H:%M:%S)] fit-folds start"
python scripts/77_tclsc_ev.py --fit-folds --device cuda > "$LOG/ev_fitfolds.log" 2>&1 || { echo "fit-folds FAILED"; tail -20 "$LOG/ev_fitfolds.log"; exit 7; }
echo "== [$(date -u +%H:%M:%S)] fit-folds done: $(ls $OUT/folds/*/task_features/*.npz 2>/dev/null | wc -l) npz"
echo "== [$(date -u +%H:%M:%S)] train-es start (2 procs)"
nohup python scripts/77_tclsc_ev.py --train-es --resume --device cuda --folds Handwriting UniMiB-SHAR PhonemeSpectra UWaveGestureLibrary > "$LOG/ev_train_p1.log" 2>&1 < /dev/null &
P1=$!
nohup python scripts/77_tclsc_ev.py --train-es --resume --device cuda --folds EthanolConcentration Cricket HandMovementDirection > "$LOG/ev_train_p2.log" 2>&1 < /dev/null &
P2=$!
echo "pids $P1 $P2"
while kill -0 "$P1" 2>/dev/null || kill -0 "$P2" 2>/dev/null; do sleep 60; done
echo "== [$(date -u +%H:%M:%S)] train-es exited: $(ls $OUT/models/ES/*/*.pt 2>/dev/null | wc -l) models"
if [ "$(ls $OUT/models/ES/*/*.pt 2>/dev/null | wc -l)" -lt 21 ]; then
  echo "== models < 21 → 单进程 --resume 补训"
  python scripts/77_tclsc_ev.py --train-es --resume --device cuda >> "$LOG/ev_train_fix.log" 2>&1
fi
echo "== [$(date -u +%H:%M:%S)] evolve ev_es"
python scripts/77_tclsc_ev.py --evolve --guides ev_es --device cuda > "$LOG/ev_evolve_es.log" 2>&1; tail -3 "$LOG/ev_evolve_es.log"
echo "== [$(date -u +%H:%M:%S)] waiting Ethanol rprime (pid $PE)"
while kill -0 "$PE" 2>/dev/null; do sleep 30; done
tail -2 "$LOG/ev_rprime_ethanol.log"
echo "== [$(date -u +%H:%M:%S)] rprime ev_es"
python scripts/77_tclsc_ev.py --rprime --guides ev_es --device cuda > "$LOG/ev_rprime_es.log" 2>&1; tail -3 "$LOG/ev_rprime_es.log"
# pod 侧 verify 只对 pod 负责的格：ev_es 全部 + Ethanol 的 ev_et / ev_blind / rand_u（其余数据集的行在本地）
python scripts/77_tclsc_ev.py --verify --datasets EthanolConcentration --guides ev_et ev_blind --random > "$LOG/ev_verify_ethanol.log" 2>&1; cat "$LOG/ev_verify_ethanol.log"
python scripts/77_tclsc_ev.py --verify --guides ev_es > "$LOG/ev_verify.log" 2>&1; cat "$LOG/ev_verify.log"
if ! grep -q "→ OK" "$LOG/ev_verify.log" || ! grep -q "→ OK" "$LOG/ev_verify_ethanol.log"; then
  echo "== verify NOT OK → 补跑"
  python scripts/77_tclsc_ev.py --rprime --guides ev_es --device cuda >> "$LOG/ev_rprime_fix.log" 2>&1
  python scripts/77_tclsc_ev.py --rprime --guides ev_et ev_blind --random --datasets EthanolConcentration --device cuda >> "$LOG/ev_rprime_fix.log" 2>&1
  python scripts/77_tclsc_ev.py --verify --guides ev_es > "$LOG/ev_verify.log" 2>&1; cat "$LOG/ev_verify.log"
  python scripts/77_tclsc_ev.py --verify --datasets EthanolConcentration --guides ev_et ev_blind --random > "$LOG/ev_verify_ethanol.log" 2>&1; cat "$LOG/ev_verify_ethanol.log"
fi
echo "== [$(date -u +%H:%M:%S)] rank-es"
python scripts/77_tclsc_ev.py --rank-es --device cuda > "$LOG/ev_rank_es.log" 2>&1; tail -3 "$LOG/ev_rank_es.log"
tar -czf "$UP/out_ev.tar.gz" "$OUT/proposals_ev_es.json" "$OUT/rprime_ev.jsonl" "$OUT/rankings_es.json" "$OUT/runs" "$OUT/models/ES" "$OUT/train.log" "$OUT/set_encoder_init.pt" \
  $( [ -f "$OUT/oom_failures.jsonl" ] && echo "$OUT/oom_failures.jsonl" ) $OUT/folds/*/shared_ts2vec.meta.json "$LOG"/ev_*.log
ls -la "$UP/out_ev.tar.gz"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
echo "== [$(date -u +%H:%M:%S)] ev done"
