#!/usr/bin/env bash
# WP16-TCLSC 阶段 1 真 E 臂：pod 主流程（全程 --resume 可续；被杀后原样重跑）。
#   MODE=binned nohup bash /workspace/wp/CLS/scripts/pa_pod/s1e_run.sh > /workspace/logs/s1e_run.log 2>&1 &
# 1) 33 --fit-folds（7 折共享 TS2Vec + 45 切片窗口编码；all8 不需要，只跑 S1 折）
# 2) 33 --train 两进程分折并行（每进程 ~10 GB 显存）：e_concat_hard / e_diff_hard × 3 种子
# 3) 62 --rank --device cuda（E 消融 + 诊断）→ 打包 /workspace/upload_s1e/out_s1e.tar.gz
# 所有 python 调用显式 --device cuda + 启动前 CUDA 闸（memory runpod-nvml-dropout-mid-run）。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
OUT="${OUT:-results/tclsc_s1e/binned}"
MODE="${MODE:-binned}"
GRAPH_ARGS="${GRAPH_ARGS:-}"
cd "$REPO"; mkdir -p "$LOG" "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gate() { python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)" || { echo "!! CUDA 闸失败 $(date -u +%T)"; exit 3; }; }
stamp() { echo "== [$(date -u +%H:%M:%S)] $*"; }

gate
stamp "fit-folds（S1 7 折）"
python scripts/33_pq_comparator_loo.py --fit-folds --folds Handwriting UniMiB-SHAR PhonemeSpectra UWaveGestureLibrary \
    EthanolConcentration Cricket HandMovementDirection --device cuda --outdir "$OUT" > "$LOG/s1e_fit_folds.log" 2>&1 \
    || echo "  ⚠️ fit-folds 非零退出（见 $LOG/s1e_fit_folds.log）"
tail -2 "$LOG/s1e_fit_folds.log"
for t in Handwriting UniMiB-SHAR PhonemeSpectra UWaveGestureLibrary EthanolConcentration Cricket HandMovementDirection; do
  n=$(ls "$OUT/folds/$t/task_features"/*.npz 2>/dev/null | wc -l); echo "  折 $t 特征 $n 个"
done

gate
stamp "train ×2 并行（MODE=$MODE）"
python scripts/33_pq_comparator_loo.py --train --arms e_concat_hard e_diff_hard --seeds 0 1 2 --resume --device cuda \
    --encoder-mode "$MODE" $GRAPH_ARGS --outdir "$OUT" --folds Handwriting UniMiB-SHAR PhonemeSpectra UWaveGestureLibrary \
    > "$LOG/s1e_train_A.log" 2>&1 &
PA=$!
python scripts/33_pq_comparator_loo.py --train --arms e_concat_hard e_diff_hard --seeds 0 1 2 --resume --device cuda \
    --encoder-mode "$MODE" $GRAPH_ARGS --outdir "$OUT" --folds EthanolConcentration Cricket HandMovementDirection \
    > "$LOG/s1e_train_B.log" 2>&1 &
PB=$!
wait $PA; echo "  train A exit $?"; wait $PB; echo "  train B exit $?"
grep -h "^\[" "$LOG/s1e_train_A.log" "$LOG/s1e_train_B.log" | wc -l
n_models=$(ls "$OUT"/models/*/e_*_s*.pt 2>/dev/null | wc -l); echo "  模型 $n_models / 42"

gate
stamp "rank（E 消融 + 诊断）"
python scripts/62_tclsc_s1e.py --rank --device cuda --outdir "$OUT" > "$LOG/s1e_rank.log" 2>&1 || echo "  ⚠️ rank 非零退出"
tail -9 "$LOG/s1e_rank.log"

stamp "pack"
mkdir -p /workspace/upload_s1e
tar -czf /workspace/upload_s1e/out_s1e.tar.gz "$OUT/runs" "$OUT/models" "$OUT/rankings_s1e.json" "$OUT"/folds/*/shared_ts2vec.meta.json \
    "$OUT/set_encoder_init.pt" "$LOG"/s1e_*.log 2>/dev/null
ls -la /workspace/upload_s1e/out_s1e.tar.gz
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
stamp "s1e done：拉回 out_s1e.tar.gz → 本地 62 --analyze → pod_ctl.py stop + confirm-stopped 3"
