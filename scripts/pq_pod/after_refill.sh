#!/usr/bin/env bash
# MX-3-PQ 上云后半程（R' 网格补齐之后一条龙）：
#   verify → reproduce-check（§8 失守已诊断，显式 --allow-reproduction-breach 放行，见
#   results/mx3_pq/reproduction_diagnosis.md）→ 33 --train 三组并行（GPU 很闲：标定 1 epoch
#   0.8 s）→ 串行 rescan 兜底（补并行时的漏/skip 竞态）→ 33 --analyze → 34 rank/analyze →
#   34 evolve + evaluate-evo（GPU 真评估）→ 34 analyze。
#
# 用法：PARTIAL="--allow-partial-cache" INC="--allow-incomplete" bash scripts/pq_pod/after_refill.sh
#   两个变量只在网格仍有缺格时才设（analysis.json 会标 partial_cache/incomplete）。
set -uo pipefail
REPO="${REPO:-/workspace/wp/CLS}"; LOG="${LOG:-/workspace/logs}"; OUT="${OUT:-results/mx3_pq}"
PARTIAL="${PARTIAL:-}"; INC="${INC:-}"
cd "$REPO"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "== $1 $(date -u +%H:%M:%S)"; }

step verify
python scripts/32_pq_generate_rprime.py --verify --outdir "$OUT" > "$LOG/pq_verify.log" 2>&1
echo "verify exit=$?"
step reproduce-check
python scripts/32_pq_generate_rprime.py --reproduce-check --allow-reproduction-breach \
       --outdir "$OUT" > "$LOG/pq_repro.log" 2>&1
echo "reproduce exit=$?"

step train-parallel
python scripts/33_pq_comparator_loo.py --train --resume --arms main $PARTIAL \
       --device cuda --outdir "$OUT" > "$LOG/pq_train_A.log" 2>&1 &
python scripts/33_pq_comparator_loo.py --train --resume --arms autocts_style wo_settf cls_only cls_b \
       $PARTIAL --device cuda --outdir "$OUT" > "$LOG/pq_train_B.log" 2>&1 &
python scripts/33_pq_comparator_loo.py --train --resume --arms wo_ts2vec wo_shared soft onehot_oracle main_s2 \
       $PARTIAL --device cuda --outdir "$OUT" > "$LOG/pq_train_C.log" 2>&1 &
wait
step train-rescan
python scripts/33_pq_comparator_loo.py --train --resume $PARTIAL --device cuda --outdir "$OUT" \
       > "$LOG/pq_train_rescan.log" 2>&1
echo "train exit=$?  runs=$(ls $OUT/runs/*/*.json 2>/dev/null | wc -l)"
step analyze-33
python scripts/33_pq_comparator_loo.py --analyze $PARTIAL --outdir "$OUT" > "$LOG/pq_analyze33.log" 2>&1
echo "analyze33 exit=$?"

step rank-34
python scripts/34_pq_search.py --rank --outdir "$OUT" > "$LOG/pq_34_rank.log" 2>&1
echo "rank exit=$?"
step evolve-34
python scripts/34_pq_search.py --evolve --evaluate-evo --resume --on-oom record \
       --inflight "$OUT/inflight_evo.txt" --oom-log "$OUT/oom_failures_evo.jsonl" \
       --device cuda --outdir "$OUT" > "$LOG/pq_34_evo.log" 2>&1
echo "evolve exit=$?"
step analyze-34
python scripts/34_pq_search.py --analyze $INC --outdir "$OUT" > "$LOG/pq_34_analyze.log" 2>&1
echo "analyze34 exit=$?"
echo "CHAIN_DONE $(date -u +%H:%M:%S)"
