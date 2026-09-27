#!/usr/bin/env bash
# WP17-ACTS S3-O 第一阶段：多卡并行计算 160 任务的 E_T（每任务单变量 TS2Vec，50 次 / 原码总次数两个截点的 rep_profile）→ et.npz → upload/out_acts_s3o_et.tar.gz。
#   GPUS="0 1 2 3 4 5 6 7" setsid nohup bash .../scripts/pa_pod/acts_lab_run_s3o_et.sh > /tmp/acts_s3o_out/logs/run.log 2>&1 < /dev/null &
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_s3o_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3o"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; SLOT_MB="${SLOT_MB:-6000}"; RETRIES="${RETRIES:-10}"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_s3o_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD.txt")"
mkdir -p "$OUT" "$T/et"; [ -L "$OUT/et" ] || { rm -rf "$OUT/et"; ln -s "$T/et" "$OUT/et"; }
python -m pytest tests/test_acts_s3o.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
SLOTS=()
for g in $GPUS; do
  free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $g | awk -F', ' '{print $1-$2}')
  n=$(( (free - 1500) / SLOT_MB )); [ $n -gt 2 ] && n=2; [ $n -lt 0 ] && n=0
  for ((k=0; k<n; k++)); do SLOTS+=($g); done
  echo "== GPU $g free ${free} MiB → $n 槽"
done
NS=${#SLOTS[@]}; echo "== 共 $NS 槽：${SLOTS[*]}"; [ $NS -ge 1 ] || { echo "!! 无可用槽"; exit 5; }
echo "== [$(date -u +%H:%M:%S)] E_T $NS 份"
pids=()
for ((s=0; s<NS; s++)); do g=${SLOTS[$s]}
  ( for ((r=0; r<=RETRIES; r++)); do CUDA_VISIBLE_DEVICES=$g nice -n 10 python scripts/102_acts_s3o.py --compute-et --parts $NS --part $s --device cuda --outdir $OUT >> "$LOG/et_p$s.log" 2>&1 < /dev/null && break; echo "== retry #$((r+1))" >> "$LOG/et_p$s.log"; sleep 60; done ) & pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "== [$(date -u +%H:%M:%S)] E_T 完：$(cat $OUT/et/part*.jsonl | wc -l) / 160"
python scripts/102_acts_s3o.py --merge-et --outdir $OUT --device cpu 2>&1 | tee "$LOG/merge.log"
tar -chzf "$UP/out_acts_s3o_et.tar.gz" $OUT/et.npz $OUT/et "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_s3o_et.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3o et done"
