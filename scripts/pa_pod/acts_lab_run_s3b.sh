#!/usr/bin/env bash
# WP17-ACTS S3-B 在实验室 8k-4090 跑（输出经 symlink 落根盘 /tmp/acts_s3b_out）：
#   阶段 1（每卡一个进程）：E_pt 第 i 片（176 任务按 --parts 分）→ E_sh 第 i 片（17 折 + ALL 按 --parts 分），失败按 --resume 重试
#   阶段 2：--pack-means（均值 E + 秒数汇总，回传用）
#   阶段 3：SetTF 72 个作业（18 折 × {pt, sh} × {E, E ⊕ ④}，每作业 3 seed）轮转分到 GPUS，每卡 JOBS_PER_GPU 并发，失败重试
#   阶段 4：打包 upload/out_acts_s3b.tar.gz（settf/*.json、_means.npz、E_sh/*/meta.json、日志；不含 windows_repr）。不自停（非计费机器）。
#   GPUS="0 1 2 3" JOBS_PER_GPU=2 RETRIES=3 nohup bash .../scripts/pa_pod/acts_lab_run_s3b.sh > /tmp/acts_s3b_out/logs/run.log 2>&1 &
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"
T="/tmp/acts_s3b_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3b"
GPUS="${GPUS:-0 1}"; RETRIES="${RETRIES:-3}"; JPG="${JOBS_PER_GPU:-2}"
NP=$(echo $GPUS | wc -w)
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="python scripts/86_acts_s3b.py --outdir $OUT --device cuda --resume"
retry() {  # retry <gpu> <logfile> <args...>
  local g=$1 lf=$2; shift 2
  for ((r=0; r<=RETRIES; r++)); do
    CUDA_VISIBLE_DEVICES=$g nice -n 10 $P "$@" >> "$lf" 2>&1 < /dev/null && return 0
    echo "== [$(date -u +%H:%M:%S)] GPU $g retry #$((r+1)): $*" >> "$lf"; sleep 30
  done
  echo "== FAILED after $RETRIES retries: $*" >> "$lf"; return 1
}
echo "== [$(date -u +%H:%M:%S)] stage 1 E extraction ($NP GPUs: $GPUS)"
i=0; pids=()
for g in $GPUS; do
  ( retry $g "$LOG/pt_p$i.log" --extract-pt --parts $NP --part $i; retry $g "$LOG/sh_p$i.log" --fit-shared --parts $NP --part $i ) &
  pids+=($!); i=$((i+1)); sleep 10
done
for p in "${pids[@]}"; do wait $p; done
echo "== [$(date -u +%H:%M:%S)] stage 1 done: E_pt $(ls $OUT/E_pt/*.npz 2>/dev/null | wc -l) / 176；E_sh 折 $(ls -d $OUT/E_sh/*/ 2>/dev/null | wc -l) / 18"
python scripts/86_acts_s3b.py --outdir $OUT --pack-means --device cpu > "$LOG/pack.log" 2>&1; tail -3 "$LOG/pack.log"
echo "== [$(date -u +%H:%M:%S)] stage 3 SetTF"
FOLDS=$(python -c "import numpy as np; z=np.load('$OUT/targets.npz'); print(' '.join(list(z['order'].astype(str))+['ALL']))")
JOBS=()
for f in $FOLDS; do for arm in pt sh; do JOBS+=("$arm $f 0" "$arm $f 1"); done; done
NS=$((NP * JPG)); pids=()
for ((s=0; s<NS; s++)); do
  g=$(echo $GPUS | cut -d" " -f$(( s % NP + 1 )))
  (
    for ((j=s; j<${#JOBS[@]}; j+=NS)); do
      set -- ${JOBS[$j]}
      extra=""; [ "$3" = 1 ] && extra="--combo"
      retry $g "$LOG/settf_s$s.log" --train-settf --arm $1 --fold $2 $extra
    done
  ) &
  pids+=($!); sleep 5
done
for p in "${pids[@]}"; do wait $p; done
echo "== [$(date -u +%H:%M:%S)] stage 3 done: settf json $(ls $OUT/settf/*/*.json 2>/dev/null | wc -l) / 72"
cd "$REPO"
tar -chzf "$UP/out_acts_s3b.tar.gz" $OUT/settf $OUT/E_pt/_means.npz $OUT/E_sh/*/_means.npz $OUT/E_sh/*/meta.json "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_s3b.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3b done"
