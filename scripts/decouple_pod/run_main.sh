#!/usr/bin/env bash
# 解耦实验主跑批（PREREGISTRATION.md §4）：随机 20 条 CLS × 4 数据集 × 2 任务 ×
# 4 种子 = 640 条 R'（early_iters=290）+ 640 条 iters=0 配对基线。
# 每数据集一个进程，进程内先 R' 后基线串行（同数据集共享加载/切片路径，且基线
# 极快）；4 进程并发。--resume 幂等。
set -euo pipefail
cd /workspace/repo/CLS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864

CAT=results/decouple_random_cls/random_cls_catalog.json
OUT=results/decouple_random_cls/main
LOGS=/workspace/logs/decouple
mkdir -p "$OUT" "$LOGS"

for ds in ArticularyWordRecognition CharacterTrajectories ERing NATOPS; do
  nohup bash -c "
    python -u scripts/step0_snr_gate.py --run --resume --no-escalate \
      --catalog $CAT --n-cls 20 --cache-dir $OUT --datasets $ds --device cuda \
      --inflight $OUT/inflight_rprime_$ds.txt --oom-log $OUT/oom_failures.jsonl \
    && python -u scripts/step0_snr_gate.py --run --resume --baseline --no-escalate \
      --catalog $CAT --n-cls 20 --cache-dir $OUT --datasets $ds --device cuda \
      --inflight $OUT/inflight_base_$ds.txt --oom-log $OUT/oom_failures.jsonl
  " > "$LOGS/main_$ds.log" 2>&1 &
  echo "  起 $ds (pid $!)"
done
echo "4 进程已脱离启动。进度："
echo "  R'：cat $OUT/rprime_cache.jsonl $OUT/noise_cache.jsonl | wc -l（目标 640）"
echo "  基线：wc -l $OUT/baseline_cache.jsonl（目标 640）"
