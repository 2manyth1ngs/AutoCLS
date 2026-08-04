#!/usr/bin/env bash
# 解耦实验 §3.1 判据先行验证：旧探针 10 条 CLS × 4 数据集 × 2 任务 × 4 种子
# = 320 条 iters=0 配对基线（PREREGISTRATION.md §3.1）。
# 每数据集一个进程（数据集为并行粒度，惯例同阶段1），共用一个 baseline_cache.jsonl
# （行追加原子安全）；各自 inflight。--resume 幂等，重跑只补缺。
set -euo pipefail
cd /workspace/repo/CLS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864

OUT=results/decouple_random_cls/validation_baseline
LOGS=/workspace/logs/decouple
mkdir -p "$OUT" "$LOGS"

for ds in ArticularyWordRecognition CharacterTrajectories ERing NATOPS; do
  nohup python -u scripts/step0_snr_gate.py --run --resume --baseline --no-escalate \
    --catalog results/route_a_probe/probe_cls_catalog.json --n-cls 10 \
    --cache-dir "$OUT" --datasets "$ds" --device cuda \
    --inflight "$OUT/inflight_$ds.txt" \
    --oom-log "$OUT/oom_failures.jsonl" \
    > "$LOGS/val_$ds.log" 2>&1 &
  echo "  起 $ds (pid $!)"
done
echo "4 进程已脱离启动；进度：wc -l $OUT/baseline_cache.jsonl（目标 320）"
