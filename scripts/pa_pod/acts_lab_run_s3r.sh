#!/usr/bin/env bash
# WP17-ACTS S3-R 全自动流水（实验室 8k-4090；输出 symlink 到根盘 /tmp/acts_s3r_out）：
#   解包 → 核对 S3-C 数据 / E_pt 窗口 / labels.npz → symlink → pytest → 按各卡空闲显存分槽（每卡 ≤ 2 槽）→
#   36 次训练（E 臂 18 个先排、B 臂 18 个后排；6 折 × 3 种子）→ 打包 upload/out_acts_s3r.tar.gz（train，不含权重）。
#   GPUS="0 1 2 3 4 5 6 7" setsid nohup bash .../scripts/pa_pod/acts_lab_run_s3r.sh > /tmp/acts_s3r_out/logs/run.log 2>&1 < /dev/null &
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_s3r_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3r"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; RETRIES="${RETRIES:-20}"; SLOT_MB="${SLOT_MB:-6000}"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_s3r_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD.txt")"
[ -f results/acts/s3c/data.npz ] && [ -f results/acts/s3c/data.json ] || { echo "!! S3-C 数据缺"; exit 3; }
npt=$(ls results/acts/s3b/E_pt/*.npz | grep -vc _means); echo "== E_pt $npt / 176"; [ "$npt" -ge 176 ] || exit 3
[ -f "$OUT/labels.npz" ] || { echo "!! labels.npz 缺"; exit 3; }
for d in models train; do mkdir -p "$T/$d"; if [ -e "$OUT/$d" ] && [ ! -L "$OUT/$d" ]; then cp -r "$OUT/$d/." "$T/$d/" && rm -rf "$OUT/$d"; fi; [ -L "$OUT/$d" ] || ln -s "$T/$d" "$OUT/$d"; done
python -m pytest tests/test_acts_s3r.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
# 分槽
SLOTS=()
for g in $GPUS; do
  free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $g | awk -F', ' '{print $1-$2}')
  n=$(( (free - 1500) / SLOT_MB )); [ $n -gt 2 ] && n=2; [ $n -lt 0 ] && n=0
  for ((k=0; k<n; k++)); do SLOTS+=($g); done
  echo "== GPU $g free ${free} MiB → $n 槽"
done
NS=${#SLOTS[@]}; echo "== 共 $NS 槽：${SLOTS[*]}"; [ $NS -ge 1 ] || { echo "!! 无可用槽"; exit 5; }
retry() { local g=$1 lf=$2; shift 2
  for ((r=0; r<=RETRIES; r++)); do CUDA_VISIBLE_DEVICES=$g nice -n 10 "$@" >> "$lf" 2>&1 < /dev/null && return 0; echo "== [$(date -u +%H:%M:%S)] GPU $g retry #$((r+1)): $*" >> "$lf"; sleep 60; done
  echo "== FAILED after $RETRIES retries: $*" >> "$lf"; return 1; }
P="python scripts/105_acts_s3r.py --outdir $OUT --device cuda --resume"
run_jobs() {  # run_jobs <stage> <job 参数串...>：作业轮转到 NS 个槽
  local stage=$1; shift; local JOBS=("$@"); local pids=()
  for ((s=0; s<NS; s++)); do local g=${SLOTS[$s]}
    ( for ((j=s; j<${#JOBS[@]}; j+=NS)); do retry $g "$LOG/${stage}_s$s.log" $P ${JOBS[$j]}; done ) & pids+=($!); sleep 2; done
  for p in "${pids[@]}"; do wait $p; done; }
FOLDS="Handwriting PhonemeSpectra UWaveGestureLibrary Cricket EthanolConcentration SelfRegulationSCP1"
echo "== [$(date -u +%H:%M:%S)] 训练 36 次"
J=(); for arm in E B; do for sd in 0 1 2; do for f in $FOLDS; do J+=("--train --arm $arm --fold $f --seed $sd"); done; done; done; run_jobs train "${J[@]}"
echo "== [$(date -u +%H:%M:%S)] 训练完：$(ls $OUT/train/*.npz 2>/dev/null | wc -l) / 36"
tar -chzf "$UP/out_acts_s3r.tar.gz" $OUT/train "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_s3r.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3r done"
