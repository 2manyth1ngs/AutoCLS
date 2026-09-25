#!/usr/bin/env bash
# WP17-ACTS S3-C 在实验室 8k-4090 跑（输出 symlink 在根盘 /tmp/acts_s3c_out）：
#   T 段：24 个比较器（6 折 × 臂 A/B/C/D）轮转分到 GPUS × JOBS_PER_GPU 个槽，失败按 --resume 重试
#   S 段：24 组搜索（同上分槽）
#   P 段：--plan-runs（提名去重 → runs/plan_X.json）
#   R 段：83 --run --phase X 每卡一片（--parts），失败重试；最后单进程 --resume 补跑
#   打包 upload/out_acts_s3c.tar.gz（train/*.npz、search/*.json、runs/{plan_X.json,runs.jsonl}、日志；不含模型权重）。不自停（非计费机器）。
#   GPUS="0 1 2 3" JOBS_PER_GPU=2 RETRIES=3 setsid nohup bash .../scripts/pa_pod/acts_lab_run_s3c.sh > /tmp/acts_s3c_out/logs/run.log 2>&1 < /dev/null &
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_s3c_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3c"
GPUS="${GPUS:-0 1}"; RETRIES="${RETRIES:-3}"; JPG="${JOBS_PER_GPU:-2}"; NP=$(echo $GPUS | wc -w)
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="python scripts/89_acts_s3c.py --outdir $OUT --device cuda --resume"
retry() {  # retry <gpu> <logfile> <cmd...>
  local g=$1 lf=$2; shift 2
  for ((r=0; r<=RETRIES; r++)); do
    CUDA_VISIBLE_DEVICES=$g nice -n 10 "$@" >> "$lf" 2>&1 < /dev/null && return 0
    echo "== [$(date -u +%H:%M:%S)] GPU $g retry #$((r+1)): $*" >> "$lf"; sleep 30
  done
  echo "== FAILED after $RETRIES retries: $*" >> "$lf"; return 1
}
run_jobs() {  # run_jobs <stage> <flag>：24 个 (arm, fold) 作业轮转到 NP×JPG 个槽；重活（A/B）排在前面
  local stage=$1 flag=$2
  local JOBS=()
  for arm in A B C D; do for f in Handwriting PhonemeSpectra UWaveGestureLibrary Cricket EthanolConcentration SelfRegulationSCP1; do JOBS+=("$arm $f"); done; done
  local NS=$((NP * JPG)); local pids=()
  for ((s=0; s<NS; s++)); do
    local g=$(echo $GPUS | cut -d" " -f$(( s % NP + 1 )))
    ( for ((j=s; j<${#JOBS[@]}; j+=NS)); do set -- ${JOBS[$j]}; retry $g "$LOG/${stage}_s$s.log" $P $flag --arm $1 --fold $2; done ) &
    pids+=($!); sleep 5
  done
  for p in "${pids[@]}"; do wait $p; done
}
echo "== [$(date -u +%H:%M:%S)] T 段：比较器训练（GPUs $GPUS × $JPG）"
run_jobs train --train
echo "== [$(date -u +%H:%M:%S)] T 段完：$(ls $OUT/train/*.npz 2>/dev/null | wc -l) / 24"
echo "== [$(date -u +%H:%M:%S)] S 段：搜索"
run_jobs search --search
echo "== [$(date -u +%H:%M:%S)] S 段完：$(ls $OUT/search/*.json 2>/dev/null | wc -l) / 24"
python scripts/89_acts_s3c.py --outdir $OUT --plan-runs --device cpu > "$LOG/plan_runs.log" 2>&1; tail -1 "$LOG/plan_runs.log"
echo "== [$(date -u +%H:%M:%S)] R 段：提名真跑"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
R83="python scripts/83_acts_corpus.py --run --resume --device cuda --profile s2 --outdir $OUT/runs --phase X"
i=0; pids=()
for g in $GPUS; do ( retry $g "$LOG/runs_p$i.log" $R83 --parts $NP --part $i ) & pids+=($!); i=$((i+1)); sleep 20; done
for p in "${pids[@]}"; do wait $p; done
G0=$(echo $GPUS | cut -d" " -f1)
retry $G0 "$LOG/runs_fix.log" $R83
python - <<PY | tee "$LOG/runs_verify.log"
import json
p = json.load(open("$OUT/runs/plan_X.json")); keys = {c["key"] for c in p["cells"]}
done = set(); err = 0
for l in open("$OUT/runs/runs.jsonl"):
    r = json.loads(l); done.add(r["key"]); err += bool(r.get("error"))
print(f"verify: plan {len(keys)} / done {len(keys & done)} / error {err} → {'OK' if keys <= done else 'MISSING ' + str(len(keys - done))}")
PY
tar -chzf "$UP/out_acts_s3c.tar.gz" $OUT/train $OUT/search $OUT/runs/plan_X.json $OUT/runs/runs.jsonl "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_s3c.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3c done"
